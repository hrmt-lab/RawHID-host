use std::{
    io,
    net::SocketAddr,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use subtle::ConstantTimeEq;
use thiserror::Error;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::{mpsc, oneshot, watch},
    task::JoinHandle,
};

use crate::{
    claude_decision::{ClaudePermissionGate, UnansweredReason, CLAUDE_PERMISSION_DECISION_TIMEOUT},
    claude_hook_event::{ClaudeHookEvent, ClaudeObserverEvent, ClaudeWrapperExited},
    pending_approval::claude_key,
};

const MAX_HEADER_BYTES: usize = 32 * 1024;
const MAX_BODY_BYTES: usize = 1024 * 1024;

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ClaudeObserverConfig {
    pub endpoint: String,
    pub wrapper_exit_endpoint: String,
    pub bearer_token: String,
    pub launch_id: String,
    pub request_timeout_ms: u64,
}

impl std::fmt::Debug for ClaudeObserverConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ClaudeObserverConfig")
            .field("endpoint", &self.endpoint)
            .field("wrapper_exit_endpoint", &self.wrapper_exit_endpoint)
            .field("bearer_token", &"<redacted>")
            .field("launch_id", &self.launch_id)
            .field("request_timeout_ms", &self.request_timeout_ms)
            .finish()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ClaudeObserverCounters {
    pub received: u64,
    pub accepted: u64,
    pub unauthorized: u64,
    pub malformed: u64,
    pub oversized: u64,
    pub normal_overflow: u64,
    pub priority_overflow: u64,
    /// How many `PermissionRequest` hooks got back an actual decision
    /// (200 + `hook_response_body()`) rather than the 204 every other hook
    /// -- and a `PermissionRequest` that lost the race, timed out, or
    /// failed to queue -- gets. See `handle_permission_request`.
    pub permission_decided: u64,
}

#[derive(Default)]
struct AtomicCounters {
    received: AtomicU64,
    accepted: AtomicU64,
    unauthorized: AtomicU64,
    malformed: AtomicU64,
    oversized: AtomicU64,
    normal_overflow: AtomicU64,
    priority_overflow: AtomicU64,
    permission_decided: AtomicU64,
}

impl AtomicCounters {
    fn snapshot(&self) -> ClaudeObserverCounters {
        ClaudeObserverCounters {
            received: self.received.load(Ordering::Relaxed),
            accepted: self.accepted.load(Ordering::Relaxed),
            unauthorized: self.unauthorized.load(Ordering::Relaxed),
            malformed: self.malformed.load(Ordering::Relaxed),
            oversized: self.oversized.load(Ordering::Relaxed),
            normal_overflow: self.normal_overflow.load(Ordering::Relaxed),
            priority_overflow: self.priority_overflow.load(Ordering::Relaxed),
            permission_decided: self.permission_decided.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ClaudeObserverReceiverOptions {
    pub bind_address: SocketAddr,
    pub launch_id: String,
    pub bearer_token: String,
    pub normal_queue_capacity: usize,
    pub priority_queue_capacity: usize,
    pub helper_request_timeout_ms: u64,
    /// Shared with `rawhid-host-tauri`'s `AppState` so a physical HUD
    /// answer (dispatched from the Tauri monitor thread) and this
    /// connection's held-open `PermissionRequest` hook (running here, on
    /// the Tokio receiver task) can rendezvous on the same waiter. See
    /// `claude_decision.rs`'s module doc for the full handoff.
    pub permission_gate: Arc<ClaudePermissionGate>,
    /// How long `handle_permission_request` waits on the gate before
    /// degrading to 204. See `CLAUDE_PERMISSION_DECISION_TIMEOUT`'s doc
    /// comment for why this is shorter than the hook's own `timeout`.
    pub permission_decision_timeout: Duration,
}

impl ClaudeObserverReceiverOptions {
    pub fn loopback(launch_id: impl Into<String>, bearer_token: impl Into<String>) -> Self {
        Self {
            bind_address: SocketAddr::from(([127, 0, 0, 1], 0)),
            launch_id: launch_id.into(),
            bearer_token: bearer_token.into(),
            normal_queue_capacity: 128,
            priority_queue_capacity: 16,
            helper_request_timeout_ms: 500,
            permission_gate: Arc::new(ClaudePermissionGate::default()),
            permission_decision_timeout: CLAUDE_PERMISSION_DECISION_TIMEOUT,
        }
    }
}

#[derive(Debug, Error)]
pub enum ClaudeObserverError {
    #[error("invalid Claude observer configuration: {0}")]
    InvalidConfig(String),
    #[error("failed to bind Claude observer receiver: {0}")]
    Bind(#[source] io::Error),
    #[error("Claude observer receiver task failed: {0}")]
    Task(#[from] tokio::task::JoinError),
}

pub struct ClaudeObserverEvents {
    normal_rx: mpsc::Receiver<ClaudeObserverEvent>,
    priority_rx: mpsc::Receiver<ClaudeObserverEvent>,
}

impl ClaudeObserverEvents {
    pub async fn recv(&mut self) -> Option<ClaudeObserverEvent> {
        tokio::select! {
            biased;
            event = self.priority_rx.recv() => event,
            event = self.normal_rx.recv() => event,
        }
    }

    pub fn try_recv(&mut self) -> Result<ClaudeObserverEvent, mpsc::error::TryRecvError> {
        match self.priority_rx.try_recv() {
            Ok(event) => Ok(event),
            Err(mpsc::error::TryRecvError::Empty) => self.normal_rx.try_recv(),
            Err(mpsc::error::TryRecvError::Disconnected) => self.normal_rx.try_recv(),
        }
    }
}

pub struct ClaudeObserverReceiver {
    config: ClaudeObserverConfig,
    counters: Arc<AtomicCounters>,
    shutdown_tx: watch::Sender<bool>,
    task: JoinHandle<()>,
}

impl ClaudeObserverReceiver {
    pub async fn bind(
        options: ClaudeObserverReceiverOptions,
    ) -> Result<(Self, ClaudeObserverEvents), ClaudeObserverError> {
        if options.launch_id.is_empty() {
            return Err(ClaudeObserverError::InvalidConfig(
                "launch_id must not be empty".to_string(),
            ));
        }
        if options.bearer_token.len() < 32 {
            return Err(ClaudeObserverError::InvalidConfig(
                "bearer token must contain at least 32 characters".to_string(),
            ));
        }
        if options.normal_queue_capacity == 0 || options.priority_queue_capacity == 0 {
            return Err(ClaudeObserverError::InvalidConfig(
                "queue capacities must be greater than zero".to_string(),
            ));
        }
        if options.helper_request_timeout_ms == 0 {
            return Err(ClaudeObserverError::InvalidConfig(
                "helper timeout must be greater than zero".to_string(),
            ));
        }

        let listener = TcpListener::bind(options.bind_address)
            .await
            .map_err(ClaudeObserverError::Bind)?;
        let address = listener.local_addr().map_err(ClaudeObserverError::Bind)?;
        let base = format!("http://{address}");
        let config = ClaudeObserverConfig {
            endpoint: format!("{base}/hooks"),
            wrapper_exit_endpoint: format!("{base}/wrapper-exit"),
            bearer_token: options.bearer_token.clone(),
            launch_id: options.launch_id.clone(),
            request_timeout_ms: options.helper_request_timeout_ms,
        };
        let (normal_tx, normal_rx) = mpsc::channel(options.normal_queue_capacity);
        let (priority_tx, priority_rx) = mpsc::channel(options.priority_queue_capacity);
        let counters = Arc::new(AtomicCounters::default());
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let state = ReceiverState {
            launch_id: Arc::new(options.launch_id),
            bearer_token: Arc::new(options.bearer_token),
            normal_tx,
            priority_tx,
            counters: counters.clone(),
            permission_gate: options.permission_gate,
            permission_decision_timeout: options.permission_decision_timeout,
        };
        let task = tokio::spawn(receiver_loop(listener, state, shutdown_rx));

        Ok((
            Self {
                config,
                counters,
                shutdown_tx,
                task,
            },
            ClaudeObserverEvents {
                normal_rx,
                priority_rx,
            },
        ))
    }

    pub fn config(&self) -> &ClaudeObserverConfig {
        &self.config
    }

    pub fn counters(&self) -> ClaudeObserverCounters {
        self.counters.snapshot()
    }

    pub async fn shutdown(self) -> Result<(), ClaudeObserverError> {
        let _ = self.shutdown_tx.send(true);
        self.task.await?;
        Ok(())
    }
}

#[derive(Clone)]
struct ReceiverState {
    launch_id: Arc<String>,
    bearer_token: Arc<String>,
    normal_tx: mpsc::Sender<ClaudeObserverEvent>,
    priority_tx: mpsc::Sender<ClaudeObserverEvent>,
    counters: Arc<AtomicCounters>,
    permission_gate: Arc<ClaudePermissionGate>,
    permission_decision_timeout: Duration,
}

async fn receiver_loop(
    listener: TcpListener,
    state: ReceiverState,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    loop {
        tokio::select! {
            changed = shutdown_rx.changed() => {
                if changed.is_err() || *shutdown_rx.borrow() {
                    return;
                }
            }
            accepted = listener.accept() => {
                let Ok((stream, _)) = accepted else {
                    return;
                };
                tokio::spawn(handle_connection(stream, state.clone()));
            }
        }
    }
}

async fn handle_connection(mut stream: TcpStream, state: ReceiverState) {
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
        Err(RequestError::Io) => return,
    };

    if request.method != "POST" {
        state.counters.malformed.fetch_add(1, Ordering::Relaxed);
        let _ = write_response(&mut stream, 405).await;
        return;
    }
    let expected = format!("Bearer {}", state.bearer_token);
    let authorized = request
        .authorization
        .as_deref()
        .map(|actual| constant_time_equal(actual.as_bytes(), expected.as_bytes()))
        .unwrap_or(false);
    if !authorized {
        state.counters.unauthorized.fetch_add(1, Ordering::Relaxed);
        let _ = write_response(&mut stream, 401).await;
        return;
    }

    let event = match request.path.as_str() {
        "/hooks" => parse_hook_event(&request.body, &state.launch_id),
        "/wrapper-exit" => parse_wrapper_exit(&request.body, &state.launch_id),
        _ => {
            let _ = write_response(&mut stream, 404).await;
            return;
        }
    };
    let Some(event) = event else {
        state.counters.malformed.fetch_add(1, Ordering::Relaxed);
        let _ = write_response(&mut stream, 400).await;
        return;
    };

    // `PermissionRequest` with a known `session_id` is the one hook this
    // receiver ever answers with a decision instead of a bare 204
    // (`docs/ai-approval-hud-design.md` §9.2). Every other hook -- and a
    // sessionless `PermissionRequest`, which cannot be correlated with a
    // `PendingApprovalStore` entry (`claude_key` needs both ids) -- keeps
    // the original observation-only behavior below unchanged.
    let is_decidable_permission_request = matches!(
        &event,
        ClaudeObserverEvent::Hook(hook)
            if hook.hook_event_name == "PermissionRequest" && hook.session_id.is_some()
    );
    if is_decidable_permission_request {
        handle_permission_request(&mut stream, &state, event).await;
        return;
    }

    let queued = if event.is_priority() {
        state.priority_tx.try_send(event).map_err(|_| {
            state
                .counters
                .priority_overflow
                .fetch_add(1, Ordering::Relaxed)
        })
    } else {
        state.normal_tx.try_send(event).map_err(|_| {
            state
                .counters
                .normal_overflow
                .fetch_add(1, Ordering::Relaxed)
        })
    };
    if queued.is_ok() {
        state.counters.accepted.fetch_add(1, Ordering::Relaxed);
    }
    let _ = write_response(&mut stream, 204).await;
}

/// Answers one `PermissionRequest` hook connection with an actual decision
/// when one arrives in time, degrading to 204 (observation-only) in every
/// other case. Call order matters here and is the crux of stage 3's
/// correctness: the gate is registered *before* the event is queued for
/// `ClaudeApprovalBodyConsumer`/`ClaudeSessionRegistry` to consume
/// (`rawhid-host-tauri`'s `drain_claude_state_changes`), because that
/// consumption is what makes the request visible to a HUD in the first
/// place -- if a HUD answer could somehow race in before `register` ran,
/// it would find no waiter and silently do nothing.
async fn handle_permission_request(
    stream: &mut TcpStream,
    state: &ReceiverState,
    event: ClaudeObserverEvent,
) {
    let ClaudeObserverEvent::Hook(hook) = &event else {
        // Unreachable given the caller's `is_decidable_permission_request`
        // guard; degrade safely rather than unwrap on a refactor that
        // breaks that invariant.
        let _ = write_response(stream, 204).await;
        return;
    };
    let session_id = match hook.session_id.as_deref() {
        Some(session_id) => session_id.to_string(),
        None => {
            let _ = write_response(stream, 204).await;
            return;
        }
    };
    // Captured now, before `event` (which `hook` borrows from) is moved into
    // `try_send` below -- needed later for `note_unanswerable`, which wants
    // the `(launch_id, session_id)` pair alongside the token so its caller
    // can also withdraw the session's own approval request.
    let launch_id = hook.launch_id.clone();
    let token = claude_key(&launch_id, &session_id).token().to_string();

    // Register before queuing -- see this function's own doc comment.
    let receiver = state.permission_gate.register(token.clone());

    let queued = if event.is_priority() {
        state.priority_tx.try_send(event).map_err(|_| {
            state
                .counters
                .priority_overflow
                .fetch_add(1, Ordering::Relaxed)
        })
    } else {
        state.normal_tx.try_send(event).map_err(|_| {
            state
                .counters
                .normal_overflow
                .fetch_add(1, Ordering::Relaxed)
        })
    };
    if queued.is_err() {
        state.permission_gate.cancel(&token);
        let _ = write_response(stream, 204).await;
        return;
    }
    state.counters.accepted.fetch_add(1, Ordering::Relaxed);

    // Below, the decision wait (`receiver`, timeout-guarded via `sleep`) runs
    // concurrently with a single-byte read probe on the same `stream` that
    // is otherwise sitting idle while we wait. Two real-machine
    // `PermissionRequest`s compared on 2026-09-06 confirmed what that probe
    // firing actually means: a request rejected from the terminal at
    // 14:59:22 had this connection's read side close 10.7s later, while a
    // request left completely untouched, received at 15:02:51, still had it
    // open three and a half minutes later. So a close here reliably means
    // "this request was settled somewhere other than Studio" and is now
    // this wait's own end condition -- see the probe arm's own comment below
    // for how it decides whether that is actually true for *this*
    // connection's own waiter before ending the wait.
    tokio::pin!(receiver);
    let sleep = tokio::time::sleep(state.permission_decision_timeout);
    tokio::pin!(sleep);
    let mut probe_byte = [0_u8; 1];

    // Every arm below ends this connection, so this is one `select!` rather
    // than a loop. In particular the read probe never "keeps waiting": a
    // closed peer would return `Ok(0)` immediately and forever, so anything
    // that polled it twice would spin hot.
    //
    // Two of the three arms have to answer the same question before acting:
    // is this connection's own waiter still the live one for `token`? A
    // retried `PermissionRequest` for the same session registers again under
    // the same token (`ClaudePermissionGate::register`'s own doc comment),
    // which drops this connection's sender and makes the *newer*
    // registration the owner of both the gate waiter and the
    // `PendingApprovalStore` entry. Acting on `token` after that point would
    // reach into the newer request's state: `note_unanswerable` would delete
    // its still-live store entry, and `cancel` would drop the waiter it is
    // relying on to receive a HUD answer. So both arms triage with
    // `try_recv` first, and only the `Empty` case -- still registered, still
    // unanswered -- records or cancels anything.
    tokio::select! {
        decision = &mut receiver => {
            match decision {
                Ok(decision) => {
                    state
                        .counters
                        .permission_decided
                        .fetch_add(1, Ordering::Relaxed);
                    let _ = write_json_response(stream, 200, &decision.hook_response_body()).await;
                }
                // The sender was dropped without ever sending: either a
                // later `register` for the same token replaced this waiter,
                // or `ClaudePermissionGate::cancel`/`cancel_launch` removed
                // it because the terminal answered first or the
                // session/launch ended. Either way this connection's own
                // registration is already gone, so there is nothing here to
                // record as unanswerable and nothing of ours left to cancel
                // -- deliberately no `cancel(&token)` call, which would
                // otherwise drop a newer registration's waiter.
                Err(_) => {
                    let _ = write_response(stream, 204).await;
                }
            }
        }
        () = &mut sleep => {
            // The decision timeout elapsed. Whether that leaves anything to
            // clean up depends on whether this connection still owns the
            // token -- see this block's own comment above.
            match receiver.as_mut().try_recv() {
                // Superseded or already cancelled: not ours to touch.
                Err(oneshot::error::TryRecvError::Closed) => {
                    let _ = write_response(stream, 204).await;
                }
                // A decision landed in the same instant the timeout fired.
                // Deliver it rather than dropping an answer the gate
                // believes it handed off.
                Ok(decision) => {
                    state
                        .counters
                        .permission_decided
                        .fetch_add(1, Ordering::Relaxed);
                    let _ = write_json_response(stream, 200, &decision.hook_response_body()).await;
                }
                // Still the live waiter, and nobody -- neither a HUD action
                // nor the terminal -- ever answered. Record it so Host code
                // can drop the matching `PendingApprovalStore` entry and
                // withdraw the session's own approval request (see
                // `ClaudePermissionGate::note_unanswerable`'s doc comment).
                Err(oneshot::error::TryRecvError::Empty) => {
                    state.permission_gate.note_unanswerable(
                        &token,
                        &launch_id,
                        &session_id,
                        UnansweredReason::DecisionTimeout,
                    );
                    state.permission_gate.cancel(&token);
                    let _ = write_response(stream, 204).await;
                }
            }
        }
        read_result = stream.read(&mut probe_byte) => {
            // `Ok(0)` (EOF), `Ok(n >= 1)` (unexpected data -- this
            // connection is not supposed to send anything more once its
            // request finished), and `Err(_)` (a read error, e.g. a reset)
            // are all the same thing here: the read side of this connection
            // ended while we were still waiting, which the 2026-09-06
            // comparison above established means the request was settled
            // from the terminal.
            let _ = read_result;
            match receiver.as_mut().try_recv() {
                // Superseded or already cancelled: not ours to touch. No
                // response either -- the peer closed its read side.
                Err(oneshot::error::TryRecvError::Closed) => {}
                // A decision landed in the same instant as the close, and
                // this probe observed it first. Deliver it exactly as the
                // decision arm would have.
                Ok(decision) => {
                    state
                        .counters
                        .permission_decided
                        .fetch_add(1, Ordering::Relaxed);
                    let _ = write_json_response(stream, 200, &decision.hook_response_body()).await;
                }
                // Still the live, unanswered waiter. The peer that closed
                // this connection is already gone, so there is nothing to
                // write a response to. Record the close and stop waiting.
                Err(oneshot::error::TryRecvError::Empty) => {
                    state.permission_gate.note_unanswerable(
                        &token,
                        &launch_id,
                        &session_id,
                        UnansweredReason::ConnectionClosed,
                    );
                    state.permission_gate.cancel(&token);
                }
            }
        }
    }
}

fn parse_hook_event(body: &[u8], launch_id: &str) -> Option<ClaudeObserverEvent> {
    let body: Value = serde_json::from_slice(body).ok()?;
    let hook_event_name = body.get("hook_event_name")?.as_str()?.to_string();
    let session_id = body
        .get("session_id")
        .and_then(Value::as_str)
        .map(str::to_string);
    Some(ClaudeObserverEvent::Hook(ClaudeHookEvent {
        launch_id: launch_id.to_string(),
        hook_event_name,
        session_id,
        body,
    }))
}

#[derive(Deserialize)]
struct WrapperExitBody {
    launch_id: String,
    exit_code: i32,
}

fn parse_wrapper_exit(body: &[u8], launch_id: &str) -> Option<ClaudeObserverEvent> {
    let body: WrapperExitBody = serde_json::from_slice(body).ok()?;
    if body.launch_id != launch_id {
        return None;
    }
    Some(ClaudeObserverEvent::WrapperExited(ClaudeWrapperExited {
        launch_id: body.launch_id,
        exit_code: body.exit_code,
    }))
}

struct HttpRequest {
    method: String,
    path: String,
    authorization: Option<String>,
    body: Vec<u8>,
}

enum RequestError {
    Malformed,
    Oversized,
    Io,
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
        let read = stream
            .read(&mut chunk)
            .await
            .map_err(|_| RequestError::Io)?;
        if read == 0 {
            return Err(RequestError::Malformed);
        }
        bytes.extend_from_slice(&chunk[..read]);
    };

    let mut headers = [httparse::EMPTY_HEADER; 64];
    let mut parsed = httparse::Request::new(&mut headers);
    parsed
        .parse(&bytes[..header_end])
        .map_err(|_| RequestError::Malformed)?;
    let method = parsed.method.ok_or(RequestError::Malformed)?.to_string();
    let path = parsed.path.ok_or(RequestError::Malformed)?.to_string();
    let mut authorization = None;
    let mut content_length = None;
    for header in parsed.headers {
        if header.name.eq_ignore_ascii_case("authorization") {
            authorization = Some(
                std::str::from_utf8(header.value)
                    .map_err(|_| RequestError::Malformed)?
                    .to_string(),
            );
        } else if header.name.eq_ignore_ascii_case("content-length") {
            content_length = Some(
                std::str::from_utf8(header.value)
                    .map_err(|_| RequestError::Malformed)?
                    .parse::<usize>()
                    .map_err(|_| RequestError::Malformed)?,
            );
        }
    }
    let content_length = content_length.ok_or(RequestError::Malformed)?;
    if content_length > MAX_BODY_BYTES {
        return Err(RequestError::Oversized);
    }
    let required = header_end + content_length;
    while bytes.len() < required {
        let mut chunk = [0_u8; 4096];
        let read = stream
            .read(&mut chunk)
            .await
            .map_err(|_| RequestError::Io)?;
        if read == 0 {
            return Err(RequestError::Malformed);
        }
        bytes.extend_from_slice(&chunk[..read]);
        if bytes.len() > required + MAX_HEADER_BYTES {
            return Err(RequestError::Oversized);
        }
    }
    Ok(HttpRequest {
        method,
        path,
        authorization,
        body: bytes[header_end..required].to_vec(),
    })
}

async fn write_response(stream: &mut TcpStream, status: u16) -> io::Result<()> {
    let reason = match status {
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        _ => "Internal Server Error",
    };
    let response =
        format!("HTTP/1.1 {status} {reason}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
    stream.write_all(response.as_bytes()).await
}

/// Like [`write_response`], but with a JSON body -- used only for the
/// `hookSpecificOutput` decision body `handle_permission_request` writes
/// back to a `PermissionRequest` connection. Every other response on this
/// receiver has an empty body, hence `write_response` staying separate
/// rather than folding a `None` body into it.
async fn write_json_response(stream: &mut TcpStream, status: u16, body: &Value) -> io::Result<()> {
    let reason = match status {
        200 => "OK",
        _ => "Internal Server Error",
    };
    let payload = serde_json::to_vec(body).unwrap_or_default();
    let mut response = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        payload.len()
    )
    .into_bytes();
    response.extend_from_slice(&payload);
    stream.write_all(&response).await
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn constant_time_equal(left: &[u8], right: &[u8]) -> bool {
    left.len() == right.len() && left.ct_eq(right).into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::claude_decision::{ClaudeDecision, CLAUDE_HUD_DENY_MESSAGE};

    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Runtime::new().unwrap()
    }

    #[test]
    fn config_debug_redacts_token() {
        let config = ClaudeObserverConfig {
            endpoint: "http://127.0.0.1/hooks".to_string(),
            wrapper_exit_endpoint: "http://127.0.0.1/wrapper-exit".to_string(),
            bearer_token: "secret-token".to_string(),
            launch_id: "launch".to_string(),
            request_timeout_ms: 500,
        };
        let rendered = format!("{config:?}");
        assert!(!rendered.contains("secret-token"));
        assert!(rendered.contains("<redacted>"));
    }

    #[test]
    fn receiver_accepts_hook_and_wrapper_events() {
        runtime().block_on(async {
            let token = "0123456789abcdef0123456789abcdef";
            let (receiver, mut events) = ClaudeObserverReceiver::bind(
                ClaudeObserverReceiverOptions::loopback("launch-1", token),
            )
            .await
            .unwrap();
            let client = reqwest::Client::new();
            let response = client
                .post(&receiver.config().endpoint)
                .bearer_auth(token)
                .json(&serde_json::json!({
                    "hook_event_name": "SessionStart",
                    "session_id": "session-1",
                    "prompt": "must stay in memory only"
                }))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::NO_CONTENT);
            let event = events.recv().await.unwrap();
            let ClaudeObserverEvent::Hook(event) = event else {
                panic!("expected hook event");
            };
            assert_eq!(event.launch_id, "launch-1");
            assert_eq!(event.session_id.as_deref(), Some("session-1"));

            let response = client
                .post(&receiver.config().wrapper_exit_endpoint)
                .bearer_auth(token)
                .json(&serde_json::json!({"launch_id": "launch-1", "exit_code": 7}))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::NO_CONTENT);
            assert_eq!(
                events.recv().await,
                Some(ClaudeObserverEvent::WrapperExited(ClaudeWrapperExited {
                    launch_id: "launch-1".to_string(),
                    exit_code: 7,
                }))
            );
            assert_eq!(receiver.counters().accepted, 2);
            receiver.shutdown().await.unwrap();
        });
    }

    /// Exercises the same path `rawhid-host-tauri`'s
    /// `drain_claude_state_changes` wires up: a `PermissionRequest` hook
    /// delivered over the real HTTP receiver (not a hand-built event) ends
    /// up in `PendingApprovalStore` once `ClaudeApprovalBodyConsumer`
    /// ingests it. Uses the real captured body shape from
    /// `docs/claude-permission-hook-gate-results.md` §4 -- no
    /// `tool_use_id` -- to confirm the whole pipeline, not just the
    /// consumer in isolation, accumulates a body for it.
    #[test]
    fn claude_approval_consumer_accumulates_a_body_delivered_through_the_real_receiver() {
        use crate::{
            claude_activity::ClaudeApprovalBodyConsumer,
            pending_approval::{claude_key, PendingApprovalContent, PendingApprovalStore},
        };

        runtime().block_on(async {
            let token = "0123456789abcdef0123456789abcdef";
            // Nobody answers this request in this test -- it only checks
            // that the body reaches `PendingApprovalStore`, not the
            // decision path -- so shorten the wait far below the 55s
            // production default rather than let the request block on it.
            let mut options = ClaudeObserverReceiverOptions::loopback("launch-1", token);
            options.permission_decision_timeout = std::time::Duration::from_millis(50);
            let (receiver, mut events) = ClaudeObserverReceiver::bind(options).await.unwrap();
            let client = reqwest::Client::new();
            let response = client
                .post(&receiver.config().endpoint)
                .bearer_auth(token)
                .json(&serde_json::json!({
                    "hook_event_name": "PermissionRequest",
                    "session_id": "session-1",
                    "cwd": "C:\\work\\keylink-claude-permission-probe-8",
                    "tool_name": "PowerShell",
                    "tool_input": {
                        "command": "New-Item -ItemType Directory ko3-test8",
                        "description": "Create ko3-test8 directory"
                    }
                }))
                .send()
                .await
                .unwrap();
            // Nobody answered the gate, so this degrades to 204 -- same
            // response code as before stage 3, just after the short timeout
            // above instead of immediately.
            assert_eq!(response.status(), reqwest::StatusCode::NO_CONTENT);
            let event = events.recv().await.unwrap();

            let store = PendingApprovalStore::new();
            let gate = crate::claude_decision::ClaudePermissionGate::default();
            ClaudeApprovalBodyConsumer.ingest(&store, &gate, &event);

            let key = claude_key("launch-1", "session-1");
            let snapshot = store.get(&key).expect("body accumulated in the store");
            match snapshot.content {
                PendingApprovalContent::Body(body) => {
                    assert_eq!(
                        body.full_command.as_deref(),
                        Some("New-Item -ItemType Directory ko3-test8")
                    );
                    assert_eq!(body.tool_use_id, None);
                }
                PendingApprovalContent::Oversized => panic!("unexpected oversized marker"),
            }

            receiver.shutdown().await.unwrap();
        });
    }

    #[test]
    fn receiver_rejects_wrong_token_without_queueing() {
        runtime().block_on(async {
            let (receiver, mut events) =
                ClaudeObserverReceiver::bind(ClaudeObserverReceiverOptions::loopback(
                    "launch-1",
                    "0123456789abcdef0123456789abcdef",
                ))
                .await
                .unwrap();
            let response = reqwest::Client::new()
                .post(&receiver.config().endpoint)
                .bearer_auth("wrong-token")
                .json(&serde_json::json!({"hook_event_name": "Stop"}))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::UNAUTHORIZED);
            assert_eq!(events.try_recv(), Err(mpsc::error::TryRecvError::Empty));
            assert_eq!(receiver.counters().unauthorized, 1);
            receiver.shutdown().await.unwrap();
        });
    }

    #[test]
    fn receiver_drops_overflowing_detail_without_blocking_response() {
        runtime().block_on(async {
            let token = "0123456789abcdef0123456789abcdef";
            let mut options = ClaudeObserverReceiverOptions::loopback("launch-1", token);
            options.normal_queue_capacity = 1;
            let (receiver, mut events) = ClaudeObserverReceiver::bind(options).await.unwrap();
            let client = reqwest::Client::new();
            for tool_id in ["tool-1", "tool-2"] {
                let response = client
                    .post(&receiver.config().endpoint)
                    .bearer_auth(token)
                    .json(&serde_json::json!({
                        "hook_event_name": "PreToolUse",
                        "session_id": "session-1",
                        "tool_use_id": tool_id
                    }))
                    .send()
                    .await
                    .unwrap();
                assert_eq!(response.status(), reqwest::StatusCode::NO_CONTENT);
            }
            assert_eq!(receiver.counters().accepted, 1);
            assert_eq!(receiver.counters().normal_overflow, 1);
            assert!(matches!(
                events.recv().await,
                Some(ClaudeObserverEvent::Hook(_))
            ));
            receiver.shutdown().await.unwrap();
        });
    }

    fn permission_request_body(session_id: &str) -> serde_json::Value {
        serde_json::json!({
            "hook_event_name": "PermissionRequest",
            "session_id": session_id,
            "cwd": "C:\\work",
            "tool_name": "PowerShell",
            "tool_input": {"command": "mkdir foo"}
        })
    }

    /// KO-3's confirmed success path (`docs/claude-permission-hook-gate-results.md`
    /// §Q3): answering the gate with `Allow` while the hook connection is
    /// still open returns 200 with exactly the `hookSpecificOutput` body
    /// Claude Code accepted in the real-machine gate.
    #[test]
    fn permission_request_returns_200_and_the_allow_body_when_answered() {
        runtime().block_on(async {
            let token = "0123456789abcdef0123456789abcdef";
            let gate = Arc::new(ClaudePermissionGate::default());
            let mut options = ClaudeObserverReceiverOptions::loopback("launch-1", token);
            options.permission_gate = gate.clone();
            options.permission_decision_timeout = Duration::from_secs(5);
            let (receiver, mut events) = ClaudeObserverReceiver::bind(options).await.unwrap();
            let endpoint = receiver.config().endpoint.clone();
            let request = tokio::spawn(async move {
                reqwest::Client::new()
                    .post(&endpoint)
                    .bearer_auth(token)
                    .json(&permission_request_body("session-1"))
                    .send()
                    .await
                    .unwrap()
            });

            // The event only reaches the queue *after* `handle_permission_request`
            // registers with the gate, so this also proves the answer below
            // cannot race ahead of registration.
            let _event = events.recv().await.unwrap();
            let token_str = claude_key("launch-1", "session-1").token().to_string();
            assert!(gate.answer(&token_str, ClaudeDecision::Allow));

            let response = request.await.unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::OK);
            assert_eq!(
                response.headers().get("content-type").unwrap(),
                "application/json"
            );
            let body: Value = response.json().await.unwrap();
            assert_eq!(
                body,
                serde_json::json!({
                    "hookSpecificOutput": {
                        "hookEventName": "PermissionRequest",
                        "decision": {"behavior": "allow"}
                    }
                })
            );
            assert_eq!(receiver.counters().permission_decided, 1);
            receiver.shutdown().await.unwrap();
        });
    }

    /// KO-3's confirmed denial path (§Q4): a `Deny` decision returns 200
    /// with `behavior: "deny"` and the fixed `CLAUDE_HUD_DENY_MESSAGE` --
    /// see `ClaudeDecision`'s own doc comment for why that string is fixed
    /// and carries no action instruction.
    #[test]
    fn permission_request_returns_200_and_the_deny_body_with_the_fixed_message_when_answered() {
        runtime().block_on(async {
            let token = "0123456789abcdef0123456789abcdef";
            let gate = Arc::new(ClaudePermissionGate::default());
            let mut options = ClaudeObserverReceiverOptions::loopback("launch-1", token);
            options.permission_gate = gate.clone();
            options.permission_decision_timeout = Duration::from_secs(5);
            let (receiver, mut events) = ClaudeObserverReceiver::bind(options).await.unwrap();
            let endpoint = receiver.config().endpoint.clone();
            let request = tokio::spawn(async move {
                reqwest::Client::new()
                    .post(&endpoint)
                    .bearer_auth(token)
                    .json(&permission_request_body("session-1"))
                    .send()
                    .await
                    .unwrap()
            });

            let _event = events.recv().await.unwrap();
            let token_str = claude_key("launch-1", "session-1").token().to_string();
            assert!(gate.answer(
                &token_str,
                ClaudeDecision::Deny {
                    message: CLAUDE_HUD_DENY_MESSAGE.to_string()
                }
            ));

            let response = request.await.unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::OK);
            let body: Value = response.json().await.unwrap();
            assert_eq!(
                body,
                serde_json::json!({
                    "hookSpecificOutput": {
                        "hookEventName": "PermissionRequest",
                        "decision": {
                            "behavior": "deny",
                            "message": CLAUDE_HUD_DENY_MESSAGE
                        }
                    }
                })
            );
            assert_eq!(receiver.counters().permission_decided, 1);
            receiver.shutdown().await.unwrap();
        });
    }

    /// Q6's confirmed fallback (`docs/claude-permission-hook-gate-results.md`
    /// §Q6): with nobody -- neither a HUD nor the consumer -- ever answering
    /// the gate, the connection degrades to the same 204 every other hook
    /// gets, once its (here, deliberately short) decision timeout elapses.
    #[test]
    fn permission_request_degrades_to_204_when_nobody_answers() {
        runtime().block_on(async {
            let token = "0123456789abcdef0123456789abcdef";
            let mut options = ClaudeObserverReceiverOptions::loopback("launch-1", token);
            options.permission_decision_timeout = Duration::from_millis(50);
            let (receiver, mut events) = ClaudeObserverReceiver::bind(options).await.unwrap();
            let response = reqwest::Client::new()
                .post(&receiver.config().endpoint)
                .bearer_auth(token)
                .json(&permission_request_body("session-1"))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::NO_CONTENT);
            assert_eq!(receiver.counters().permission_decided, 0);
            // The request is still queued for observation even though it
            // was never decided -- 204 only changes the hook's HTTP
            // response, never the normal ingestion path.
            assert!(matches!(
                events.recv().await,
                Some(ClaudeObserverEvent::Hook(_))
            ));
            receiver.shutdown().await.unwrap();
        });
    }

    /// The decision-timeout branch (`Err(_)` from `tokio::time::timeout`)
    /// must record the token as unanswerable, so Host code can withdraw the
    /// matching `PendingApprovalStore` entry -- see
    /// `ClaudePermissionGate::note_unanswerable`'s doc comment for why.
    #[test]
    fn permission_request_records_the_token_as_unanswerable_when_the_decision_timeout_elapses() {
        runtime().block_on(async {
            let token = "0123456789abcdef0123456789abcdef";
            let gate = Arc::new(ClaudePermissionGate::default());
            let mut options = ClaudeObserverReceiverOptions::loopback("launch-1", token);
            options.permission_gate = gate.clone();
            options.permission_decision_timeout = Duration::from_millis(50);
            let (receiver, mut events) = ClaudeObserverReceiver::bind(options).await.unwrap();
            let response = reqwest::Client::new()
                .post(&receiver.config().endpoint)
                .bearer_auth(token)
                .json(&permission_request_body("session-1"))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::NO_CONTENT);
            let token_str = claude_key("launch-1", "session-1").token().to_string();
            assert_eq!(
                gate.drain_unanswerable(),
                vec![crate::claude_decision::UnansweredApproval {
                    token: token_str,
                    launch_id: "launch-1".to_string(),
                    session_id: "session-1".to_string(),
                    reason: crate::claude_decision::UnansweredReason::DecisionTimeout,
                }]
            );
            assert!(matches!(
                events.recv().await,
                Some(ClaudeObserverEvent::Hook(_))
            ));
            receiver.shutdown().await.unwrap();
        });
    }

    /// The other losing branch (`Ok(Err(_))`, a dropped sender) must NOT
    /// record the token -- here simulated the same way
    /// `ClaudeApprovalBodyConsumer` would when the terminal answers first
    /// (§9.4's first-wins rule), by calling `cancel` directly while the
    /// connection is still open and well before its own decision timeout
    /// would fire. Recording in this branch would risk deleting a *newer*
    /// registration's live entry on a retried request -- see the call
    /// site's own comment in `handle_permission_request`.
    #[test]
    fn permission_request_does_not_record_the_token_as_unanswerable_when_cancelled_first() {
        runtime().block_on(async {
            let token = "0123456789abcdef0123456789abcdef";
            let gate = Arc::new(ClaudePermissionGate::default());
            let mut options = ClaudeObserverReceiverOptions::loopback("launch-1", token);
            options.permission_gate = gate.clone();
            options.permission_decision_timeout = Duration::from_secs(5);
            let (receiver, mut events) = ClaudeObserverReceiver::bind(options).await.unwrap();
            let endpoint = receiver.config().endpoint.clone();
            let request = tokio::spawn(async move {
                reqwest::Client::new()
                    .post(&endpoint)
                    .bearer_auth(token)
                    .json(&permission_request_body("session-1"))
                    .send()
                    .await
                    .unwrap()
            });

            let _event = events.recv().await.unwrap();
            let token_str = claude_key("launch-1", "session-1").token().to_string();
            gate.cancel(&token_str);

            let response = request.await.unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::NO_CONTENT);
            assert!(
                gate.drain_unanswerable().is_empty(),
                "cancel (terminal-first / superseded) must not be recorded as a decision timeout"
            );
            receiver.shutdown().await.unwrap();
        });
    }

    /// Polls `condition` until it is true or a bounded number of attempts
    /// elapses, for asserting on state a background task (here, the
    /// receiver's connection task) updates asynchronously with no other
    /// signal to await. Never used to paper over a flaky assertion --
    /// see its two call sites, both of which wait on a genuinely
    /// concurrent background write.
    async fn wait_for(mut condition: impl FnMut() -> bool) -> bool {
        for _ in 0..200 {
            if condition() {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        condition()
    }

    /// Stage 4's connecting of the observation to a real end: a client that
    /// closes its side of a still-waiting `PermissionRequest` connection (no
    /// HTTP response ever read) while its waiter is still the live one for
    /// `token` must be recorded via `note_unanswerable`'s `ConnectionClosed`
    /// reason, and -- unlike the earlier observation-only version -- must
    /// end the handler's wait immediately rather than merely being logged
    /// alongside a wait that keeps running. Exercised through the real
    /// receiver and a raw `TcpStream` (rather than `reqwest`, which does not
    /// expose a way to close the write side without ever reading a
    /// response). The decision timeout is set far longer than this test's
    /// own poll budget so that a pass here can only mean the close itself
    /// ended the wait, not a timeout that happened to also elapse.
    #[test]
    fn client_closing_the_connection_while_waiting_is_recorded_and_ends_the_wait() {
        runtime().block_on(async {
            let token = "0123456789abcdef0123456789abcdef";
            let gate = Arc::new(ClaudePermissionGate::default());
            let mut options = ClaudeObserverReceiverOptions::loopback("launch-1", token);
            options.permission_gate = gate.clone();
            options.permission_decision_timeout = Duration::from_secs(600);
            let (receiver, mut events) = ClaudeObserverReceiver::bind(options).await.unwrap();
            let endpoint = receiver.config().endpoint.clone();
            let addr = endpoint
                .trim_start_matches("http://")
                .trim_end_matches("/hooks")
                .to_string();

            let body = serde_json::to_vec(&permission_request_body("session-1")).unwrap();
            let request = format!(
                "POST /hooks HTTP/1.1\r\nHost: {addr}\r\nAuthorization: Bearer {token}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let mut stream = TcpStream::connect(&addr).await.unwrap();
            stream.write_all(request.as_bytes()).await.unwrap();
            stream.write_all(&body).await.unwrap();

            // `handle_permission_request` registers with the gate and queues
            // the event before it ever waits on a decision (see that
            // function's own doc comment), so once the event is observable
            // here, the connection is already sitting in its `select!` loop.
            let _event = events.recv().await.unwrap();
            // Close our side without ever reading a response -- the server's
            // read-probe arm should see this as an EOF.
            drop(stream);

            let recorded = wait_for(|| !gate.drain_unanswerable().is_empty()).await;
            assert!(
                recorded,
                "the read-probe arm must observe the client-side close and record it"
            );

            // The wait actually ended (the waiter was removed), well before
            // the 600s decision timeout above could ever have fired on its
            // own -- `wait_for`'s own bound (200 * 5ms = 1s) proves that.
            assert_eq!(
                gate.waiting_count(),
                0,
                "the connection close must cancel the now-dead waiter, not merely log the close"
            );
            // The recorded entry's exact shape -- token, launch/session pair,
            // and `ConnectionClosed` reason -- is pinned separately by
            // `client_closing_the_connection_while_waiting_records_the_connection_closed_reason`
            // below. This test is only about the wait ending on the close.

            receiver.shutdown().await.unwrap();
        });
    }

    /// Pins the exact recorded entry's shape (token, launch/session pair,
    /// and `ConnectionClosed` reason) rather than only checking "something
    /// was recorded" -- a regression that dropped the reason or mixed up the
    /// token would slip past a looser assertion.
    #[test]
    fn client_closing_the_connection_while_waiting_records_the_connection_closed_reason() {
        runtime().block_on(async {
            let token = "0123456789abcdef0123456789abcdef";
            let gate = Arc::new(ClaudePermissionGate::default());
            let mut options = ClaudeObserverReceiverOptions::loopback("launch-1", token);
            options.permission_gate = gate.clone();
            options.permission_decision_timeout = Duration::from_secs(600);
            let (receiver, mut events) = ClaudeObserverReceiver::bind(options).await.unwrap();
            let endpoint = receiver.config().endpoint.clone();
            let addr = endpoint
                .trim_start_matches("http://")
                .trim_end_matches("/hooks")
                .to_string();

            let body = serde_json::to_vec(&permission_request_body("session-1")).unwrap();
            let request = format!(
                "POST /hooks HTTP/1.1\r\nHost: {addr}\r\nAuthorization: Bearer {token}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let mut stream = TcpStream::connect(&addr).await.unwrap();
            stream.write_all(request.as_bytes()).await.unwrap();
            stream.write_all(&body).await.unwrap();

            let _event = events.recv().await.unwrap();
            drop(stream);

            let token_str = claude_key("launch-1", "session-1").token().to_string();
            let mut recorded = Vec::new();
            let found = wait_for(|| {
                recorded = gate.drain_unanswerable();
                !recorded.is_empty()
            })
            .await;
            assert!(found, "the close must be recorded");
            assert_eq!(
                recorded,
                vec![crate::claude_decision::UnansweredApproval {
                    token: token_str,
                    launch_id: "launch-1".to_string(),
                    session_id: "session-1".to_string(),
                    reason: crate::claude_decision::UnansweredReason::ConnectionClosed,
                }]
            );

            receiver.shutdown().await.unwrap();
        });
    }

    /// Once a connection close has ended the wait, the waiter is gone for
    /// good -- unlike the pre-stage-4 observation-only behavior, a decision
    /// arriving afterward for the same token no longer has anyone to deliver
    /// it to. This pins that the cleanup is real (`cancel`, not just a log
    /// line): `gate.answer` returns `false` because nothing is registered
    /// any more.
    #[test]
    fn a_decision_arriving_after_the_connection_close_finds_no_waiter() {
        runtime().block_on(async {
            let token = "0123456789abcdef0123456789abcdef";
            let gate = Arc::new(ClaudePermissionGate::default());
            let mut options = ClaudeObserverReceiverOptions::loopback("launch-1", token);
            options.permission_gate = gate.clone();
            options.permission_decision_timeout = Duration::from_secs(600);
            let (receiver, mut events) = ClaudeObserverReceiver::bind(options).await.unwrap();
            let endpoint = receiver.config().endpoint.clone();
            let addr = endpoint
                .trim_start_matches("http://")
                .trim_end_matches("/hooks")
                .to_string();

            let body = serde_json::to_vec(&permission_request_body("session-1")).unwrap();
            let request = format!(
                "POST /hooks HTTP/1.1\r\nHost: {addr}\r\nAuthorization: Bearer {token}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let mut stream = TcpStream::connect(&addr).await.unwrap();
            stream.write_all(request.as_bytes()).await.unwrap();
            stream.write_all(&body).await.unwrap();

            let _event = events.recv().await.unwrap();
            drop(stream);

            let token_str = claude_key("launch-1", "session-1").token().to_string();
            assert!(
                wait_for(|| !gate.drain_unanswerable().is_empty()).await,
                "the close must be recorded before we proceed"
            );

            assert!(
                !gate.answer(&token_str, ClaudeDecision::Allow),
                "the waiter was already removed by the connection-close cleanup"
            );
            assert_eq!(receiver.counters().permission_decided, 0);

            receiver.shutdown().await.unwrap();
        });
    }

    /// The hazard the read-probe's `try_recv` guards against: a retried
    /// `PermissionRequest` for the same session registers a *new* waiter for
    /// the same token, replacing the first connection's. That first
    /// connection's own eventual close (or its decision-arm `Err(_)` losing
    /// branch, whichever the scheduler happens to observe first -- both
    /// share the same "do not record" contract) must never record an
    /// unanswerable entry for a token that a newer, still-live registration
    /// now owns; doing so would let the HUD's live offer for the *new*
    /// request disappear out from under it.
    #[test]
    fn a_retried_request_superseding_an_older_connections_waiter_never_records_it_as_unanswerable()
    {
        runtime().block_on(async {
            let token = "0123456789abcdef0123456789abcdef";
            let gate = Arc::new(ClaudePermissionGate::default());
            let mut options = ClaudeObserverReceiverOptions::loopback("launch-1", token);
            options.permission_gate = gate.clone();
            options.permission_decision_timeout = Duration::from_secs(600);
            let (receiver, mut events) = ClaudeObserverReceiver::bind(options).await.unwrap();
            let endpoint = receiver.config().endpoint.clone();

            let client = reqwest::Client::new();
            let first_endpoint = endpoint.clone();
            let first_request = tokio::spawn(async move {
                client
                    .post(&first_endpoint)
                    .bearer_auth(token)
                    .json(&permission_request_body("session-1"))
                    .send()
                    .await
                    .unwrap()
            });
            let _first_event = events.recv().await.unwrap();

            // The retry: same launch/session, a brand-new connection. Its
            // internal `register` call replaces the first connection's
            // waiter, dropping that first connection's sender.
            let second_client = reqwest::Client::new();
            let second_endpoint = endpoint.clone();
            let second_request = tokio::spawn(async move {
                second_client
                    .post(&second_endpoint)
                    .bearer_auth(token)
                    .json(&permission_request_body("session-1"))
                    .send()
                    .await
                    .unwrap()
            });
            let _second_event = events.recv().await.unwrap();

            // The first connection has nothing left to win with -- it
            // degrades on its own (either the decision arm's dropped-sender
            // branch, or the read-probe's `Closed` branch if this test's own
            // client closing its socket happens to race ahead of that;
            // either is an acceptable outcome here).
            let first_response = first_request.await.unwrap();
            assert_eq!(first_response.status(), reqwest::StatusCode::NO_CONTENT);

            // Whichever branch handled it, nothing must have been recorded
            // as unanswerable for this token.
            assert!(
                gate.drain_unanswerable().is_empty(),
                "a superseded connection's own end must never record the token \
                 a newer registration now owns"
            );

            // And the new registration must still be fully answerable.
            let token_str = claude_key("launch-1", "session-1").token().to_string();
            assert!(gate.answer(&token_str, ClaudeDecision::Allow));
            let second_response = second_request.await.unwrap();
            assert_eq!(second_response.status(), reqwest::StatusCode::OK);
            assert_eq!(receiver.counters().permission_decided, 1);

            receiver.shutdown().await.unwrap();
        });
    }

    /// Every hook other than `PermissionRequest` must be completely
    /// unaffected by stage 3: still an immediate 204, and never touching
    /// `permission_decided`.
    #[test]
    fn non_permission_request_hooks_stay_204_and_never_count_as_decided() {
        runtime().block_on(async {
            let token = "0123456789abcdef0123456789abcdef";
            let (receiver, mut events) = ClaudeObserverReceiver::bind(
                ClaudeObserverReceiverOptions::loopback("launch-1", token),
            )
            .await
            .unwrap();
            let response = reqwest::Client::new()
                .post(&receiver.config().endpoint)
                .bearer_auth(token)
                .json(&serde_json::json!({
                    "hook_event_name": "PreToolUse",
                    "session_id": "session-1",
                    "tool_use_id": "tool-a"
                }))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::NO_CONTENT);
            assert_eq!(receiver.counters().permission_decided, 0);
            assert!(matches!(
                events.recv().await,
                Some(ClaudeObserverEvent::Hook(_))
            ));
            receiver.shutdown().await.unwrap();
        });
    }

    /// A `PermissionRequest` without a `session_id` cannot be correlated
    /// with a `claude_key`, so it is never routed through the decision
    /// path at all -- immediate 204, exactly like before stage 3.
    #[test]
    fn permission_request_without_session_id_is_immediate_204() {
        runtime().block_on(async {
            let token = "0123456789abcdef0123456789abcdef";
            let mut options = ClaudeObserverReceiverOptions::loopback("launch-1", token);
            // Long enough that a slow test box could never mistake "it
            // returned immediately" for "it happened to already time out".
            options.permission_decision_timeout = Duration::from_secs(5);
            let (receiver, mut events) = ClaudeObserverReceiver::bind(options).await.unwrap();
            let started = std::time::Instant::now();
            let response = reqwest::Client::new()
                .post(&receiver.config().endpoint)
                .bearer_auth(token)
                .json(&serde_json::json!({
                    "hook_event_name": "PermissionRequest",
                    "tool_name": "PowerShell",
                    "tool_input": {"command": "mkdir foo"}
                }))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::NO_CONTENT);
            assert!(started.elapsed() < Duration::from_secs(1));
            assert_eq!(receiver.counters().permission_decided, 0);
            assert!(matches!(
                events.recv().await,
                Some(ClaudeObserverEvent::Hook(_))
            ));
            receiver.shutdown().await.unwrap();
        });
    }
}
