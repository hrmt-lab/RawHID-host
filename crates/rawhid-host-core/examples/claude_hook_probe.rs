use std::{
    collections::BTreeMap,
    env,
    fs::{self, File, OpenOptions},
    io::{self, BufRead, BufReader, Read, Write},
    net::SocketAddr,
    path::{Path, PathBuf},
    process::{Command, ExitCode, Stdio},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};

use chrono::{Local, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::{mpsc, watch},
    time,
};

const MAX_HEADER_BYTES: usize = 32 * 1024;
const MAX_BODY_BYTES: usize = 1024 * 1024;
const DEFAULT_QUEUE_CAPACITY: usize = 128;
const DEFAULT_DRAIN_MS: u64 = 1_500;

#[derive(Debug)]
struct RunOptions {
    claude: PathBuf,
    project: PathBuf,
    prompt: Option<String>,
    output_root: PathBuf,
    queue_capacity: usize,
    response_delay_ms: u64,
    drop_response: bool,
    refuse_connections: bool,
    flood_events: usize,
    writer_delay_ms: u64,
    drain_ms: u64,
    with_mcp: bool,
    claude_args: Vec<String>,
    permission_decision: PermissionDecisionMode,
    permission_updates_key: String,
}

/// What the probe should answer a `PermissionRequest` hook with. `None` is the
/// historical behavior (always 204, byte-for-byte unchanged) so existing
/// captures stay comparable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PermissionDecisionMode {
    None,
    Allow,
    AllowWithSuggestions,
    Deny,
}

impl PermissionDecisionMode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "none" => Ok(Self::None),
            "allow" => Ok(Self::Allow),
            "allow-with-suggestions" => Ok(Self::AllowWithSuggestions),
            "deny" => Ok(Self::Deny),
            other => Err(format!(
                "--permission-decision must be one of none, allow, allow-with-suggestions, deny (got {other})"
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Allow => "allow",
            Self::AllowWithSuggestions => "allow-with-suggestions",
            Self::Deny => "deny",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ObserverConfig {
    endpoint: String,
    bearer_token: String,
    timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
struct EvidenceRecord {
    received_at: String,
    peer: String,
    event: Option<String>,
    session_id: Option<String>,
    prompt_id: Option<String>,
    body_sha256: String,
    body: Value,
    /// The decision body actually returned to Claude for this request, or
    /// `None` when nothing but the plain 204 went back (every non
    /// `PermissionRequest` event, and `PermissionRequest` under `--permission-decision none`
    /// or `--drop-response`). This is the only place events.jsonl records what
    /// was answered, as opposed to what was received.
    responded: Option<Value>,
}

#[derive(Debug, Default)]
struct Counters {
    received: AtomicU64,
    accepted: AtomicU64,
    unauthorized: AtomicU64,
    malformed: AtomicU64,
    oversized: AtomicU64,
    normal_overflow: AtomicU64,
    priority_overflow: AtomicU64,
    dropped_responses: AtomicU64,
    permission_decisions_sent: AtomicU64,
}

#[derive(Clone)]
struct ReceiverState {
    token: Arc<String>,
    normal_tx: mpsc::Sender<EvidenceRecord>,
    priority_tx: mpsc::Sender<EvidenceRecord>,
    counters: Arc<Counters>,
    response_delay: Duration,
    drop_response: bool,
    permission_decision: PermissionDecisionMode,
    permission_updates_key: Arc<String>,
}

#[derive(Debug, Serialize)]
struct RunSummary {
    claude_version: String,
    project: String,
    run_directory: String,
    exit_code: Option<i32>,
    response_delay_ms: u64,
    drop_response: bool,
    refuse_connections: bool,
    flood_events: usize,
    writer_delay_ms: u64,
    queue_capacity: usize,
    received: u64,
    accepted: u64,
    unauthorized: u64,
    malformed: u64,
    oversized: u64,
    normal_overflow: u64,
    priority_overflow: u64,
    dropped_responses: u64,
    permission_decision: String,
    permission_updates_key: String,
    permission_decisions_sent: u64,
    events: BTreeMap<String, u64>,
}

#[derive(Debug)]
struct HttpRequest {
    authorization: Option<String>,
    body: Vec<u8>,
}

#[derive(Debug)]
enum RequestError {
    Malformed,
    Oversized,
    Io(io::Error),
}

fn main() -> ExitCode {
    let mut args = env::args_os();
    let _program = args.next();
    let Some(command) = args.next() else {
        print_usage();
        return ExitCode::from(2);
    };
    let remaining = args.collect::<Vec<_>>();

    match command.to_string_lossy().as_ref() {
        "run" => match parse_run_options(&remaining) {
            Ok(options) => run_probe(options),
            Err(error) => {
                eprintln!("{error}");
                print_usage();
                ExitCode::from(2)
            }
        },
        "forward" => forward_session_start(&remaining),
        "mcp-fixture" => run_mcp_fixture(),
        "help" | "--help" | "-h" => {
            print_usage();
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("unknown command: {other}");
            print_usage();
            ExitCode::from(2)
        }
    }
}

fn print_usage() {
    eprintln!(
        "Usage:\n  claude_hook_probe run --project <path> [options]\n\
         Options: --claude <path> --print <prompt> --with-mcp --queue-capacity <n>\n\
         --response-delay-ms <n> --drop-response --refuse-connections\n\
         --flood-events <n> --writer-delay-ms <n> --drain-ms <n>\n\
         --output-root <path> --claude-arg <value>\n\
         --permission-decision <none|allow|allow-with-suggestions|deny> (default: none)\n\
         --permission-updates-key <name> (default: updatedPermissions;\n\
         only used by allow-with-suggestions)\n\
         claude_hook_probe forward --observer <path>\n\
         claude_hook_probe mcp-fixture"
    );
}

fn parse_run_options(args: &[std::ffi::OsString]) -> Result<RunOptions, String> {
    let mut claude = PathBuf::from("claude");
    let mut project = None;
    let mut prompt = None;
    let mut output_root = PathBuf::from("target").join("claude-gate-c");
    let mut queue_capacity = DEFAULT_QUEUE_CAPACITY;
    let mut response_delay_ms = 0;
    let mut drop_response = false;
    let mut refuse_connections = false;
    let mut flood_events = 0;
    let mut writer_delay_ms = 0;
    let mut drain_ms = DEFAULT_DRAIN_MS;
    let mut with_mcp = false;
    let mut claude_args = Vec::new();
    let mut permission_decision = PermissionDecisionMode::None;
    let mut permission_updates_key = "updatedPermissions".to_string();
    let mut index = 0;

    while index < args.len() {
        let flag = args[index].to_string_lossy();
        match flag.as_ref() {
            "--claude" => claude = PathBuf::from(next_value(args, &mut index, "--claude")?),
            "--project" => {
                project = Some(PathBuf::from(next_value(args, &mut index, "--project")?))
            }
            "--print" => prompt = Some(next_value(args, &mut index, "--print")?),
            "--output-root" => {
                output_root = PathBuf::from(next_value(args, &mut index, "--output-root")?)
            }
            "--queue-capacity" => {
                queue_capacity = parse_positive_usize(
                    &next_value(args, &mut index, "--queue-capacity")?,
                    "--queue-capacity",
                )?
            }
            "--response-delay-ms" => {
                response_delay_ms = next_value(args, &mut index, "--response-delay-ms")?
                    .parse::<u64>()
                    .map_err(|_| "--response-delay-ms must be an integer".to_string())?;
            }
            "--drain-ms" => {
                drain_ms = next_value(args, &mut index, "--drain-ms")?
                    .parse::<u64>()
                    .map_err(|_| "--drain-ms must be an integer".to_string())?;
            }
            "--drop-response" => drop_response = true,
            "--refuse-connections" => refuse_connections = true,
            "--flood-events" => {
                flood_events = next_value(args, &mut index, "--flood-events")?
                    .parse::<usize>()
                    .map_err(|_| "--flood-events must be an integer".to_string())?;
            }
            "--writer-delay-ms" => {
                writer_delay_ms = next_value(args, &mut index, "--writer-delay-ms")?
                    .parse::<u64>()
                    .map_err(|_| "--writer-delay-ms must be an integer".to_string())?;
            }
            "--with-mcp" => with_mcp = true,
            "--claude-arg" => claude_args.push(next_value(args, &mut index, "--claude-arg")?),
            "--permission-decision" => {
                permission_decision = PermissionDecisionMode::parse(&next_value(
                    args,
                    &mut index,
                    "--permission-decision",
                )?)?;
            }
            "--permission-updates-key" => {
                permission_updates_key = next_value(args, &mut index, "--permission-updates-key")?;
            }
            _ => return Err(format!("unknown run option: {flag}")),
        }
        index += 1;
    }

    let project = project.ok_or_else(|| "--project is required".to_string())?;
    if !project.is_dir() {
        return Err(format!(
            "project directory does not exist: {}",
            project.display()
        ));
    }

    Ok(RunOptions {
        claude,
        project,
        prompt,
        output_root,
        queue_capacity,
        response_delay_ms,
        drop_response,
        refuse_connections,
        flood_events,
        writer_delay_ms,
        drain_ms,
        with_mcp,
        claude_args,
        permission_decision,
        permission_updates_key,
    })
}

fn next_value(
    args: &[std::ffi::OsString],
    index: &mut usize,
    flag: &str,
) -> Result<String, String> {
    *index += 1;
    args.get(*index)
        .map(|value| value.to_string_lossy().into_owned())
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_positive_usize(value: &str, flag: &str) -> Result<usize, String> {
    let value = value
        .parse::<usize>()
        .map_err(|_| format!("{flag} must be a positive integer"))?;
    if value == 0 {
        return Err(format!("{flag} must be greater than zero"));
    }
    Ok(value)
}

fn run_probe(options: RunOptions) -> ExitCode {
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("failed to create runtime: {error}");
            return ExitCode::FAILURE;
        }
    };

    match runtime.block_on(run_probe_async(options)) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("Gate C probe failed: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn run_probe_async(options: RunOptions) -> Result<ExitCode, String> {
    let run_name = format!(
        "{}-{}",
        Local::now().format("%Y%m%d-%H%M%S"),
        std::process::id()
    );
    let run_dir = absolute_path(&options.output_root)?.join(run_name);
    let plugin_dir = run_dir.join("plugin");
    fs::create_dir_all(plugin_dir.join(".claude-plugin"))
        .and_then(|_| fs::create_dir_all(plugin_dir.join("hooks")))
        .map_err(|error| format!("failed to create run directory: {error}"))?;

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .map_err(|error| format!("failed to bind receiver: {error}"))?;
    let address = listener
        .local_addr()
        .map_err(|error| format!("failed to obtain receiver address: {error}"))?;
    let token = random_token()?;
    let observer = ObserverConfig {
        endpoint: format!("http://{address}/hooks"),
        bearer_token: token.clone(),
        timeout_ms: 2_000,
    };
    let observer_path = plugin_dir.join("observer.json");
    write_json(&observer_path, &observer)?;

    let executable = env::current_exe()
        .map_err(|error| format!("failed to resolve probe executable: {error}"))?;
    write_plugin(&plugin_dir, &executable, &observer_path, &observer)?;

    let mcp_config_path = if options.with_mcp {
        let path = run_dir.join("mcp.json");
        write_mcp_config(&path, &executable)?;
        Some(path)
    } else {
        None
    };

    let counters = Arc::new(Counters::default());
    let event_counts = Arc::new(Mutex::new(BTreeMap::new()));
    let (normal_tx, normal_rx) = mpsc::channel(options.queue_capacity);
    let (priority_tx, priority_rx) = mpsc::channel(options.queue_capacity.max(8));
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let evidence_path = run_dir.join("events.jsonl");
    let writer = tokio::spawn(evidence_writer(
        evidence_path,
        normal_rx,
        priority_rx,
        shutdown_rx.clone(),
        event_counts.clone(),
        Duration::from_millis(options.writer_delay_ms),
    ));
    let state = ReceiverState {
        token: Arc::new(token),
        normal_tx,
        priority_tx,
        counters: counters.clone(),
        response_delay: Duration::from_millis(options.response_delay_ms),
        drop_response: options.drop_response,
        permission_decision: options.permission_decision,
        permission_updates_key: Arc::new(options.permission_updates_key.clone()),
    };
    let receiver = if options.refuse_connections {
        drop(listener);
        None
    } else {
        Some(tokio::spawn(receiver_loop(listener, state, shutdown_rx)))
    };

    if options.flood_events > 0 && !options.refuse_connections {
        flood_receiver(&observer, options.flood_events).await;
    }

    let claude_version = command_output(&options.claude, &["--version"])
        .unwrap_or_else(|error| format!("unavailable: {error}"));
    let mut command = Command::new(&options.claude);
    command
        .current_dir(&options.project)
        .arg("--plugin-dir")
        .arg(&plugin_dir)
        .args(&options.claude_args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    if let Some(mcp_config_path) = &mcp_config_path {
        command
            .arg("--mcp-config")
            .arg(mcp_config_path)
            .arg("--strict-mcp-config");
    }
    if let Some(prompt) = &options.prompt {
        command.args(["--print", prompt, "--output-format", "text"]);
    }

    let status = command
        .status()
        .map_err(|error| format!("failed to launch Claude Code: {error}"))?;
    time::sleep(Duration::from_millis(options.drain_ms)).await;
    let _ = shutdown_tx.send(true);
    if let Some(receiver) = receiver {
        let _ = receiver.await;
    }
    let _ = writer.await;

    let summary = RunSummary {
        claude_version: claude_version.trim().to_string(),
        project: options.project.display().to_string(),
        run_directory: run_dir.display().to_string(),
        exit_code: status.code(),
        response_delay_ms: options.response_delay_ms,
        drop_response: options.drop_response,
        refuse_connections: options.refuse_connections,
        flood_events: options.flood_events,
        writer_delay_ms: options.writer_delay_ms,
        queue_capacity: options.queue_capacity,
        received: counters.received.load(Ordering::Relaxed),
        accepted: counters.accepted.load(Ordering::Relaxed),
        unauthorized: counters.unauthorized.load(Ordering::Relaxed),
        malformed: counters.malformed.load(Ordering::Relaxed),
        oversized: counters.oversized.load(Ordering::Relaxed),
        normal_overflow: counters.normal_overflow.load(Ordering::Relaxed),
        priority_overflow: counters.priority_overflow.load(Ordering::Relaxed),
        dropped_responses: counters.dropped_responses.load(Ordering::Relaxed),
        permission_decision: options.permission_decision.as_str().to_string(),
        permission_updates_key: options.permission_updates_key.clone(),
        permission_decisions_sent: counters.permission_decisions_sent.load(Ordering::Relaxed),
        events: event_counts.lock().unwrap().clone(),
    };
    write_json(&run_dir.join("summary.json"), &summary)?;
    println!("{}", run_dir.display());

    Ok(match status.code() {
        Some(0) => ExitCode::SUCCESS,
        Some(code) if (1..=255).contains(&code) => ExitCode::from(code as u8),
        _ => ExitCode::FAILURE,
    })
}

async fn receiver_loop(
    listener: TcpListener,
    state: ReceiverState,
    mut shutdown: watch::Receiver<bool>,
) {
    loop {
        tokio::select! {
            changed = shutdown.changed() => {
                if changed.is_err() || *shutdown.borrow() {
                    break;
                }
            }
            accepted = listener.accept() => {
                match accepted {
                    Ok((stream, peer)) => {
                        let state = state.clone();
                        tokio::spawn(async move {
                            handle_connection(stream, peer, state).await;
                        });
                    }
                    Err(_) => break,
                }
            }
        }
    }
}

async fn handle_connection(mut stream: TcpStream, peer: SocketAddr, state: ReceiverState) {
    state.counters.received.fetch_add(1, Ordering::Relaxed);
    let request = match read_http_request(&mut stream).await {
        Ok(request) => request,
        Err(RequestError::Malformed) => {
            state.counters.malformed.fetch_add(1, Ordering::Relaxed);
            let _ = write_response(&mut stream, 400).await;
            return;
        }
        Err(RequestError::Oversized) => {
            state.counters.oversized.fetch_add(1, Ordering::Relaxed);
            let _ = write_response(&mut stream, 413).await;
            return;
        }
        Err(RequestError::Io(error)) => {
            let _ = error.kind();
            return;
        }
    };

    let expected = format!("Bearer {}", state.token);
    let authorized = request
        .authorization
        .as_deref()
        .map(|value| constant_time_equal(value.as_bytes(), expected.as_bytes()))
        .unwrap_or(false);
    if !authorized {
        state.counters.unauthorized.fetch_add(1, Ordering::Relaxed);
        let _ = write_response(&mut stream, 401).await;
        return;
    }

    let body: Value = match serde_json::from_slice(&request.body) {
        Ok(body) => body,
        Err(_) => {
            state.counters.malformed.fetch_add(1, Ordering::Relaxed);
            let _ = write_response(&mut stream, 400).await;
            return;
        }
    };
    let event = body
        .get("hook_event_name")
        .and_then(Value::as_str)
        .map(str::to_string);

    // Only PermissionRequest gets a decision body, and only if the drop-response
    // flag doesn't already mean nothing goes back over the wire at all.
    let decision_body = if event.as_deref() == Some("PermissionRequest") && !state.drop_response {
        permission_decision_body(
            state.permission_decision,
            body.get("permission_suggestions"),
            &state.permission_updates_key,
        )
    } else {
        None
    };

    let record = EvidenceRecord {
        received_at: Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Micros, true),
        peer: peer.to_string(),
        event: event.clone(),
        session_id: body
            .get("session_id")
            .and_then(Value::as_str)
            .map(str::to_string),
        prompt_id: body
            .get("prompt_id")
            .and_then(Value::as_str)
            .map(str::to_string),
        body_sha256: hex::encode(Sha256::digest(&request.body)),
        body,
        responded: decision_body.clone(),
    };

    let priority = event.as_deref().map(is_priority_event).unwrap_or(false);
    let queued = if priority {
        state.priority_tx.try_send(record).map_err(|_| {
            state
                .counters
                .priority_overflow
                .fetch_add(1, Ordering::Relaxed)
        })
    } else {
        state.normal_tx.try_send(record).map_err(|_| {
            state
                .counters
                .normal_overflow
                .fetch_add(1, Ordering::Relaxed)
        })
    };
    if queued.is_ok() {
        state.counters.accepted.fetch_add(1, Ordering::Relaxed);
    }

    if !state.response_delay.is_zero() {
        time::sleep(state.response_delay).await;
    }
    if state.drop_response {
        state
            .counters
            .dropped_responses
            .fetch_add(1, Ordering::Relaxed);
        return;
    }
    match &decision_body {
        Some(body) => {
            state
                .counters
                .permission_decisions_sent
                .fetch_add(1, Ordering::Relaxed);
            let _ = write_json_response(&mut stream, 200, body).await;
        }
        None => {
            let _ = write_response(&mut stream, 204).await;
        }
    }
}

/// Builds the JSON body to answer a `PermissionRequest` hook with, or `None`
/// when the probe should fall back to the plain 204 (mode `none`, or
/// `allow-with-suggestions` with nothing to suggest). Pure and side-effect
/// free so it can be reasoned about (and eyeballed) without spinning up a
/// server: it only assembles the fixed shape the hook protocol expects, it
/// never inspects or restructures `suggestions` beyond checking for presence.
fn permission_decision_body(
    mode: PermissionDecisionMode,
    suggestions: Option<&Value>,
    updates_key: &str,
) -> Option<Value> {
    let decision = match mode {
        PermissionDecisionMode::None => return None,
        PermissionDecisionMode::Allow => json!({"behavior": "allow"}),
        PermissionDecisionMode::Deny => json!({
            "behavior": "deny",
            "message": "probe denied"
        }),
        PermissionDecisionMode::AllowWithSuggestions => match suggestions {
            Some(value) if is_present_and_nonempty(value) => {
                // updates_key is a CLI flag (default "updatedPermissions") because
                // the correct key for surfacing permission_suggestions back to
                // Claude Code has not been confirmed yet; this lets the exact
                // key be swapped at the command line without a rebuild.
                let mut object = serde_json::Map::new();
                object.insert("behavior".to_string(), json!("allow"));
                object.insert(updates_key.to_string(), value.clone());
                Value::Object(object)
            }
            // No suggestions on this request: fall back to a plain allow rather
            // than emitting an empty/absent updates_key.
            _ => json!({"behavior": "allow"}),
        },
    };
    Some(json!({
        "hookSpecificOutput": {
            "hookEventName": "PermissionRequest",
            "decision": decision
        }
    }))
}

/// Whether a `permission_suggestions` value should be treated as carrying
/// something to forward, as opposed to being absent or an empty array.
fn is_present_and_nonempty(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Array(items) => !items.is_empty(),
        _ => true,
    }
}

fn is_priority_event(event: &str) -> bool {
    matches!(
        event,
        "SessionStart"
            | "UserPromptSubmit"
            | "Stop"
            | "StopFailure"
            | "SessionEnd"
            | "PreCompact"
            | "PostCompact"
    )
}

async fn read_http_request(stream: &mut TcpStream) -> Result<HttpRequest, RequestError> {
    let mut bytes = Vec::with_capacity(4096);
    let header_end = loop {
        if bytes.len() > MAX_HEADER_BYTES {
            return Err(RequestError::Oversized);
        }
        if let Some(position) = find_subslice(&bytes, b"\r\n\r\n") {
            break position + 4;
        }
        let mut chunk = [0_u8; 4096];
        let read = stream.read(&mut chunk).await.map_err(RequestError::Io)?;
        if read == 0 {
            return Err(RequestError::Malformed);
        }
        bytes.extend_from_slice(&chunk[..read]);
    };

    let mut headers = [httparse::EMPTY_HEADER; 64];
    let mut parsed = httparse::Request::new(&mut headers);
    let status = parsed
        .parse(&bytes[..header_end])
        .map_err(|_| RequestError::Malformed)?;
    if !status.is_complete() || parsed.method != Some("POST") {
        return Err(RequestError::Malformed);
    }
    let mut content_length = None;
    let mut authorization = None;
    for header in parsed.headers.iter() {
        if header.name.eq_ignore_ascii_case("content-length") {
            let value = std::str::from_utf8(header.value).map_err(|_| RequestError::Malformed)?;
            content_length = Some(
                value
                    .parse::<usize>()
                    .map_err(|_| RequestError::Malformed)?,
            );
        } else if header.name.eq_ignore_ascii_case("authorization") {
            authorization = Some(
                std::str::from_utf8(header.value)
                    .map_err(|_| RequestError::Malformed)?
                    .to_string(),
            );
        }
    }
    let content_length = content_length.ok_or(RequestError::Malformed)?;
    if content_length > MAX_BODY_BYTES {
        return Err(RequestError::Oversized);
    }
    while bytes.len() < header_end + content_length {
        let remaining = header_end + content_length - bytes.len();
        let mut chunk = vec![0_u8; remaining.min(8192)];
        let read = stream.read(&mut chunk).await.map_err(RequestError::Io)?;
        if read == 0 {
            return Err(RequestError::Malformed);
        }
        bytes.extend_from_slice(&chunk[..read]);
    }
    Ok(HttpRequest {
        authorization,
        body: bytes[header_end..header_end + content_length].to_vec(),
    })
}

fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        413 => "Payload Too Large",
        _ => "Error",
    }
}

async fn write_response(stream: &mut TcpStream, status: u16) -> io::Result<()> {
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        reason = reason_phrase(status)
    );
    stream.write_all(response.as_bytes()).await?;
    stream.shutdown().await
}

async fn write_json_response(stream: &mut TcpStream, status: u16, body: &Value) -> io::Result<()> {
    let payload = serde_json::to_vec(body).unwrap_or_default();
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {length}\r\nConnection: close\r\n\r\n",
        reason = reason_phrase(status),
        length = payload.len()
    );
    stream.write_all(response.as_bytes()).await?;
    stream.write_all(&payload).await?;
    stream.shutdown().await
}

async fn evidence_writer(
    path: PathBuf,
    mut normal_rx: mpsc::Receiver<EvidenceRecord>,
    mut priority_rx: mpsc::Receiver<EvidenceRecord>,
    mut shutdown: watch::Receiver<bool>,
    counts: Arc<Mutex<BTreeMap<String, u64>>>,
    writer_delay: Duration,
) {
    let mut file = match OpenOptions::new().create(true).append(true).open(path) {
        Ok(file) => file,
        Err(_) => return,
    };
    loop {
        tokio::select! {
            biased;
            Some(record) = priority_rx.recv() => {
                write_record(&mut file, &counts, record);
                if !writer_delay.is_zero() {
                    time::sleep(writer_delay).await;
                }
            },
            Some(record) = normal_rx.recv() => {
                write_record(&mut file, &counts, record);
                if !writer_delay.is_zero() {
                    time::sleep(writer_delay).await;
                }
            },
            changed = shutdown.changed() => {
                if changed.is_err() || *shutdown.borrow() {
                    while let Ok(record) = priority_rx.try_recv() {
                        write_record(&mut file, &counts, record);
                    }
                    while let Ok(record) = normal_rx.try_recv() {
                        write_record(&mut file, &counts, record);
                    }
                    let _ = file.flush();
                    break;
                }
            }
        }
    }
}

async fn flood_receiver(observer: &ObserverConfig, count: usize) {
    let Some(address) = observer
        .endpoint
        .strip_prefix("http://")
        .and_then(|value| value.strip_suffix("/hooks"))
        .and_then(|value| value.parse::<SocketAddr>().ok())
    else {
        return;
    };
    let mut tasks = Vec::with_capacity(count);
    for index in 0..count {
        let token = observer.bearer_token.clone();
        tasks.push(tokio::spawn(async move {
            let Ok(mut stream) = TcpStream::connect(address).await else {
                return;
            };
            let body = json!({
                "session_id": "gate-c-flood-session",
                "hook_event_name": "PreToolUse",
                "tool_name": "GateCFlood",
                "tool_use_id": format!("gate-c-flood-{index}"),
                "tool_input": {"index": index}
            })
            .to_string();
            let request = format!(
                "POST /hooks HTTP/1.1\r\nHost: {address}\r\nAuthorization: Bearer {token}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(request.as_bytes()).await;
            let mut response = Vec::new();
            let _ = stream.read_to_end(&mut response).await;
        }));
    }
    for task in tasks {
        let _ = task.await;
    }
}

fn write_record(
    file: &mut File,
    counts: &Arc<Mutex<BTreeMap<String, u64>>>,
    record: EvidenceRecord,
) {
    if let Some(event) = &record.event {
        *counts.lock().unwrap().entry(event.clone()).or_default() += 1;
    }
    if serde_json::to_writer(&mut *file, &record).is_ok() {
        let _ = file.write_all(b"\n");
        let _ = file.flush();
    }
}

fn write_plugin(
    plugin_dir: &Path,
    executable: &Path,
    observer_path: &Path,
    observer: &ObserverConfig,
) -> Result<(), String> {
    write_json(
        &plugin_dir.join(".claude-plugin").join("plugin.json"),
        &json!({
            "name": "keylink-gate-c-probe",
            "description": "Temporary observation-only Claude Code hook probe",
            "version": "0.0.0"
        }),
    )?;

    let session_start = json!({
        "hooks": [{
            "type": "command",
            "command": executable.display().to_string(),
            "args": ["forward", "--observer", observer_path.display().to_string()],
            "timeout": 2
        }]
    });
    let http_hook = |timeout: u64| {
        json!({
            "hooks": [{
                "type": "http",
                "url": observer.endpoint,
                "headers": {"Authorization": format!("Bearer {}", observer.bearer_token)},
                "timeout": timeout
            }]
        })
    };
    let matched_http_hook = |matcher: &str, timeout: u64| {
        json!({
            "matcher": matcher,
            "hooks": [{
                "type": "http",
                "url": observer.endpoint,
                "headers": {"Authorization": format!("Bearer {}", observer.bearer_token)},
                "timeout": timeout
            }]
        })
    };

    let hooks = json!({
        "hooks": {
            "SessionStart": [session_start],
            "UserPromptSubmit": [http_hook(2)],
            "PreToolUse": [matched_http_hook("*", 1)],
            "PermissionRequest": [matched_http_hook("*", 1)],
            "PermissionDenied": [matched_http_hook("*", 1)],
            "PostToolUse": [matched_http_hook("*", 1)],
            "PostToolUseFailure": [matched_http_hook("*", 1)],
            "PostToolBatch": [http_hook(1)],
            "Notification": [matched_http_hook("*", 1)],
            "Stop": [http_hook(3)],
            "StopFailure": [matched_http_hook("*", 3)],
            "PreCompact": [matched_http_hook("*", 2)],
            "PostCompact": [matched_http_hook("*", 2)],
            "SessionEnd": [matched_http_hook("*", 1)],
            "Elicitation": [matched_http_hook("*", 1)],
            "ElicitationResult": [matched_http_hook("*", 1)]
        }
    });
    write_json(&plugin_dir.join("hooks").join("hooks.json"), &hooks)
}

fn write_mcp_config(path: &Path, executable: &Path) -> Result<(), String> {
    write_json(
        path,
        &json!({
            "mcpServers": {
                "gate-c-fixture": {
                    "command": executable.display().to_string(),
                    "args": ["mcp-fixture"]
                }
            }
        }),
    )
}

fn forward_session_start(args: &[std::ffi::OsString]) -> ExitCode {
    let mut observer_path = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].to_string_lossy().as_ref() {
            "--observer" => match next_value(args, &mut index, "--observer") {
                Ok(value) => observer_path = Some(PathBuf::from(value)),
                Err(_) => return ExitCode::SUCCESS,
            },
            _ => return ExitCode::SUCCESS,
        }
        index += 1;
    }
    let Some(observer_path) = observer_path else {
        return ExitCode::SUCCESS;
    };
    let Ok(observer) = read_json::<ObserverConfig>(&observer_path) else {
        return ExitCode::SUCCESS;
    };
    let mut body = Vec::new();
    if io::stdin().read_to_end(&mut body).is_err() || body.len() > MAX_BODY_BYTES {
        return ExitCode::SUCCESS;
    }
    let client = match reqwest::blocking::Client::builder()
        .timeout(Duration::from_millis(observer.timeout_ms))
        .build()
    {
        Ok(client) => client,
        Err(_) => return ExitCode::SUCCESS,
    };
    let _ = client
        .post(observer.endpoint)
        .header(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {}", observer.bearer_token),
        )
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(body)
        .send();
    ExitCode::SUCCESS
}

fn run_mcp_fixture() -> ExitCode {
    let stdin = io::stdin();
    let mut reader = BufReader::new(stdin.lock());
    let mut stdout = io::stdout().lock();
    let mut line = String::new();
    let mut client_supports_elicitation = false;

    loop {
        line.clear();
        let Ok(read) = reader.read_line(&mut line) else {
            return ExitCode::SUCCESS;
        };
        if read == 0 {
            return ExitCode::SUCCESS;
        }
        let Ok(message) = serde_json::from_str::<Value>(line.trim()) else {
            continue;
        };
        let method = message.get("method").and_then(Value::as_str);
        let id = message.get("id").cloned();
        match method {
            Some("initialize") => {
                client_supports_elicitation = message
                    .pointer("/params/capabilities/elicitation")
                    .is_some();
                let protocol = message
                    .pointer("/params/protocolVersion")
                    .cloned()
                    .unwrap_or_else(|| json!("2025-06-18"));
                write_rpc(
                    &mut stdout,
                    json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": {
                            "protocolVersion": protocol,
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "keylink-gate-c-fixture", "version": "0.0.0"}
                        }
                    }),
                );
            }
            Some("tools/list") => write_rpc(
                &mut stdout,
                json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "tools": [{
                            "name": "request_gate_c_input",
                            "description": "Requests one non-sensitive Gate C fixture value from the user",
                            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": false}
                        }]
                    }
                }),
            ),
            Some("tools/call") => {
                if !client_supports_elicitation {
                    write_rpc(
                        &mut stdout,
                        tool_result(
                            id,
                            "Claude Code did not advertise elicitation support",
                            true,
                        ),
                    );
                    continue;
                }
                let elicitation_id = "gate-c-elicitation-1";
                write_rpc(
                    &mut stdout,
                    json!({
                        "jsonrpc": "2.0",
                        "id": elicitation_id,
                        "method": "elicitation/create",
                        "params": {
                            "message": "Gate C fixture: enter a non-sensitive test label",
                            "requestedSchema": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string", "title": "Test label", "minLength": 1, "maxLength": 32}
                                },
                                "required": ["label"]
                            }
                        }
                    }),
                );
                let outcome = wait_for_rpc_response(&mut reader, &mut stdout, elicitation_id);
                write_rpc(
                    &mut stdout,
                    tool_result(id, &format!("elicitation outcome: {outcome}"), false),
                );
            }
            Some("ping") => write_rpc(
                &mut stdout,
                json!({"jsonrpc": "2.0", "id": id, "result": {}}),
            ),
            Some(method) if id.is_some() => write_rpc(
                &mut stdout,
                json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {"code": -32601, "message": format!("unsupported method: {method}")}
                }),
            ),
            _ => {}
        }
    }
}

fn wait_for_rpc_response<R: BufRead, W: Write>(
    reader: &mut R,
    writer: &mut W,
    expected_id: &str,
) -> String {
    let mut line = String::new();
    loop {
        line.clear();
        let Ok(read) = reader.read_line(&mut line) else {
            return "read-error".to_string();
        };
        if read == 0 {
            return "eof".to_string();
        }
        let Ok(message) = serde_json::from_str::<Value>(line.trim()) else {
            continue;
        };
        if message.get("method").and_then(Value::as_str) == Some("ping") {
            write_rpc(
                writer,
                json!({"jsonrpc": "2.0", "id": message.get("id"), "result": {}}),
            );
            continue;
        }
        if message.get("id").and_then(Value::as_str) == Some(expected_id) {
            return message
                .pointer("/result/action")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string();
        }
    }
}

fn tool_result(id: Option<Value>, text: &str, is_error: bool) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "content": [{"type": "text", "text": text}],
            "isError": is_error
        }
    })
}

fn write_rpc<W: Write>(writer: &mut W, value: Value) {
    let _ = serde_json::to_writer(&mut *writer, &value);
    let _ = writer.write_all(b"\n");
    let _ = writer.flush();
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let file = File::create(path)
        .map_err(|error| format!("failed to create {}: {error}", path.display()))?;
    serde_json::to_writer_pretty(file, value)
        .map_err(|error| format!("failed to write {}: {error}", path.display()))
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let file =
        File::open(path).map_err(|error| format!("failed to open {}: {error}", path.display()))?;
    serde_json::from_reader(file)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))
}

fn random_token() -> Result<String, String> {
    let mut bytes = [0_u8; 32];
    getrandom::fill(&mut bytes).map_err(|error| format!("failed to generate token: {error}"))?;
    Ok(hex::encode(bytes))
}

fn constant_time_equal(left: &[u8], right: &[u8]) -> bool {
    left.len() == right.len() && bool::from(left.ct_eq(right))
}

fn absolute_path(path: &Path) -> Result<PathBuf, String> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        env::current_dir()
            .map(|cwd| cwd.join(path))
            .map_err(|error| format!("failed to resolve output root: {error}"))
    }
}

fn command_output(program: &Path, args: &[&str]) -> Result<String, String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|error| error.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_time_token_check_rejects_length_and_value_mismatches() {
        assert!(constant_time_equal(b"same", b"same"));
        assert!(!constant_time_equal(b"same", b"diff"));
        assert!(!constant_time_equal(b"same", b"shorter"));
    }

    #[test]
    fn priority_events_preserve_lifecycle_boundaries() {
        assert!(is_priority_event("SessionStart"));
        assert!(is_priority_event("Stop"));
        assert!(is_priority_event("SessionEnd"));
        assert!(!is_priority_event("PreToolUse"));
        assert!(!is_priority_event("Notification"));
    }

    #[test]
    fn run_options_require_a_project_and_positive_capacity() {
        let missing = vec!["--queue-capacity".into(), "1".into()];
        assert!(parse_run_options(&missing).is_err());

        let invalid = vec![
            "--project".into(),
            ".".into(),
            "--queue-capacity".into(),
            "0".into(),
        ];
        assert!(parse_run_options(&invalid).is_err());
    }

    #[test]
    fn generated_plugin_uses_command_only_for_session_start() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir_all(root.path().join(".claude-plugin")).unwrap();
        fs::create_dir_all(root.path().join("hooks")).unwrap();
        let observer = ObserverConfig {
            endpoint: "http://127.0.0.1:12345/hooks".to_string(),
            bearer_token: "test-token".to_string(),
            timeout_ms: 2_000,
        };
        write_plugin(
            root.path(),
            Path::new(r"C:\Program Files\Keylink\probe.exe"),
            Path::new(r"C:\Temp\observer.json"),
            &observer,
        )
        .unwrap();

        let hooks: Value = read_json(&root.path().join("hooks/hooks.json")).unwrap();
        assert_eq!(
            hooks.pointer("/hooks/SessionStart/0/hooks/0/type"),
            Some(&json!("command"))
        );
        assert_eq!(
            hooks.pointer("/hooks/UserPromptSubmit/0/hooks/0/type"),
            Some(&json!("http"))
        );
        let serialized = serde_json::to_string(&hooks).unwrap();
        assert!(serialized.contains("test-token"));
        assert!(serialized.contains("ElicitationResult"));
    }

    #[test]
    fn mcp_tool_result_is_non_sensitive_and_structured() {
        let value = tool_result(Some(json!(7)), "elicitation outcome: accept", false);
        assert_eq!(value.pointer("/id"), Some(&json!(7)));
        assert_eq!(value.pointer("/result/isError"), Some(&json!(false)));
    }
}
