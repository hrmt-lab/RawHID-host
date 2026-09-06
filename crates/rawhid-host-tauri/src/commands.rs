use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use directories::ProjectDirs;
use getrandom::fill as fill_random;
#[cfg(debug_assertions)]
use rawhid_host_core::hid::DeviceInfo;
use rawhid_host_core::{
    active_app::SystemActiveAppProvider,
    ai_usage::{AiUsageRefreshError, AiUsageRuntime, AiUsageShared},
    claude_activity::{ClaudeApprovalBodyConsumer, ClaudeSessionSnapshot, ClaudeStateChange},
    claude_decision::ClaudePermissionGate,
    claude_hook_event::ClaudeObserverEvent,
    claude_hooks::{write_claude_observer_plugin, ClaudePluginOptions},
    claude_observer::{ClaudeObserverReceiver, ClaudeObserverReceiverOptions},
    codex_activity::{
        AiClientStateChange, AiClientStateSnapshot, CodexActivityRuntime, CodexSessionSnapshot,
        CodexStateChange,
    },
    codex_broker::{
        CodexAppServerRuntime, CodexApprovalResponseOutcome, CodexBrokerConfig, CodexBrokerManager,
        CodexBrokerPhase, CodexBrokerStatus,
    },
    config::{
        load_config, ActionsConfig, AppConfig, ClaudeLauncherConfig, CodexLaunchEnvironment,
        CodexLauncherConfig, ConfigPaths,
    },
    hid::{HidDeviceManager, HidError, ProbeResult},
    packet::{
        AiActivityState, AiClientStatePacket, AiClientType, AiClientVariant, AiScreenKeyState,
        AiWorkPhase, ComboBinding, ComboInfo, ComboItem, ConfigStatus, EncoderBinding,
        EncoderBindingFlags, EncoderBindingSource, EncoderGetBindings, EncoderGetInfo,
        UplinkPacket, CAPABILITY_AI_CLIENT_DISPLAY_SLOT,
    },
    pending_approval::{ApprovalKey, PendingApprovalStore},
    runner::{uplink_device_key, RunEvent, Runner},
    stats::{KeyStatsSummary, SharedKeyStatsStore, StatsPeriod},
    studio::{
        key_catalog, keymap_backup_from_snapshot, parse_keymap_backup, plan_combo_restore,
        plan_encoder_restore, plan_keymap_restore,
        probe_studio_devices as probe_all_studio_devices, read_keymap_for_device,
        resolve_behavior_labels_for_device, serialize_keymap_backup, BackupCombo, BackupCombos,
        BackupEncoderBinding, BackupEncoderOverride, BackupEncoders, ComboRestoreWrite,
        EditBehavior, EncoderResolveError, EncoderRestoreWrite, KeyCatalogEntry, KeymapFileError,
        RestoreApplyStatus, RestoreChangedKey, RestoreIssue, RestoreReport,
        StudioBindingLabelPatch, StudioDeviceStatus, StudioEditSession, StudioError,
        StudioKeymapSnapshot, StudioRawBinding, KEYMAP_BACKUP_MAX_BYTES,
    },
};
use tauri::{AppHandle, Emitter, State};

use crate::debug_log::AI_DISPLAY_LOG_TARGET;
use crate::foreground::ForegroundWatcher;
use crate::hud_coordinator::{HudCoordinator, HudTargetSession};
use crate::state::{
    add_log, AiDisplayCandidate, AiDisplaySelection, AiDisplayTarget, AppState, ClaudeIntegration,
    ClaudeLaunchIntegration, HostLinkCall, HostLinkRequest, HostLinkResponse, LogEntry,
    MonitorCommand, MonitorStatus,
};
use crate::{actions, claude_launcher, codex_launcher, icon, startup};

type MonitorRunner = Runner<SystemActiveAppProvider, rawhid_host_core::hid::RealHidTransport>;

/// Shared handles the monitor loop needs beyond its own status/log arcs,
/// mainly for executing HOST_ACTION packets and persisting key stats.
pub struct MonitorExtras {
    pub config: Arc<std::sync::Mutex<AppConfig>>,
    pub ai_usage_runtime: Arc<std::sync::Mutex<Option<AiUsageRuntime>>>,
    pub ai_usage_refreshing: Arc<AtomicBool>,
    pub ai_terminal_focusing: Arc<AtomicBool>,
    pub key_stats: SharedKeyStatsStore,
    pub codex_activity: Arc<CodexActivityRuntime>,
    pub codex_broker: CodexBrokerManager,
    pub claude_integration: Arc<Mutex<Option<ClaudeIntegration>>>,
    /// Cloned from `AppState::claude_permission_gate` -- see that field's
    /// doc comment for why it is one process-wide instance.
    pub claude_permission_gate: Arc<ClaudePermissionGate>,
    pub ai_display_slots: Arc<Mutex<AiDisplaySelection>>,
    pub hud: Arc<Mutex<Option<HudCoordinator>>>,
    /// The ScreenKey slot that most recently dispatched a HUD answer, and
    /// when. Read by the monitor loop's per-slot ScreenKey state calculation
    /// so that slot's display state is held at
    /// `AiScreenKeyState::Responded` for `HUD_RESPONDED_HOLD` after the
    /// answer, even though the underlying activity state has already moved
    /// on (e.g. `WaitingApproval` -> `Working`). Set from
    /// `actions::ActionOutcome::HudResponded` in `handle_uplink_events`.
    pub hud_responded: Arc<Mutex<Option<(u8, Instant)>>>,
}

#[derive(Default)]
struct AiClientStateSendTracker {
    last_device_generation: Option<u64>,
    last_sent_at: Option<Instant>,
    last_slot_epoch: Option<u64>,
    /// The `AiScreenKeyState` last actually sent for this slot. A mismatch
    /// against the freshly computed value forces a full send on the current
    /// tick, the same way `last_device_generation` does -- otherwise a
    /// target switch or a `HUD_RESPONDED_HOLD` expiry between two sessions
    /// that are both still `WaitingApproval` (so `activity_state` never
    /// changes) would sit unsent until the next 5s heartbeat. See
    /// `sync_ai_client_state_slot`.
    last_screenkey_state: Option<AiScreenKeyState>,
    /// De-dup key for the last wire-send diagnostic actually recorded for
    /// this slot (see `AiWireSend`), so a heartbeat resend that repeats the
    /// exact same content is not logged again. A change-caused send, and any
    /// `DeviceGeneration`-caused send, is always recorded regardless of
    /// this -- a device-generation bump means a device was lost/regained
    /// (`hid.rs`'s `forget_device` increments it on a failed write), which is
    /// exactly the kind of transition this diagnostic exists to catch, even
    /// when the state/phase/revision otherwise look unchanged. `devices` is
    /// part of the key so a send that reaches a different number of devices
    /// is never treated as a repeat. `screenkey_state` is part of the key too,
    /// so a hold expiring or a HUD target switching -- neither of which
    /// bumps `revision` -- still gets logged. See `push_wire_send`.
    last_logged_send: Option<(
        AiClientType,
        AiActivityState,
        AiWorkPhase,
        u16,
        bool,
        AiScreenKeyState,
        usize,
    )>,
}

impl AiClientStateSendTracker {
    /// Appends `send` to `sends` unless it repeats -- same client type,
    /// activity state, work phase, revision, partial-send flag, ScreenKey
    /// display state, and device count -- the last diagnostic entry recorded
    /// for this slot AND its cause is `Heartbeat`. Change-caused and
    /// `DeviceGeneration`-caused sends are always appended: a
    /// device-generation bump signals a device was lost or regained, which
    /// must never be swallowed by de-dup. This only gates the diagnostic
    /// record: the wire send itself already happened before this is called.
    /// Keeps the shared, unexported, 200-entry app log (`MAX_LOG_ENTRIES`)
    /// usable across a session even though heartbeats fire on every slot
    /// every 5s.
    fn push_wire_send(&mut self, sends: &mut Vec<AiWireSend>, send: AiWireSend) {
        let key = (
            send.client_type,
            send.activity_state,
            send.work_phase,
            send.revision,
            send.work_phase_only,
            send.screenkey_state,
            send.devices,
        );
        let is_periodic = matches!(send.cause, AiWireSendCause::Heartbeat);
        if is_periodic && self.last_logged_send == Some(key) {
            return;
        }
        self.last_logged_send = Some(key);
        sends.push(send);
    }
}

const AI_CLIENT_STATE_RESEND_INTERVAL: Duration = Duration::from_secs(5);

trait AiClientStateTransport {
    fn device_generation(&self) -> u64;
    fn has_ai_client_state_device(&self, client_type: AiClientType) -> bool;
    fn send_ai_client_state(
        &mut self,
        packet: AiClientStatePacket,
        work_phase_only: bool,
    ) -> Result<usize, String>;
}

impl AiClientStateTransport for MonitorRunner {
    fn device_generation(&self) -> u64 {
        self.device_generation()
    }

    fn has_ai_client_state_device(&self, client_type: AiClientType) -> bool {
        self.has_ai_client_state_device(client_type)
    }

    fn send_ai_client_state(
        &mut self,
        packet: AiClientStatePacket,
        work_phase_only: bool,
    ) -> Result<usize, String> {
        self.send_ai_client_state(packet, work_phase_only)
            .map_err(|error| error.to_string())
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RunningApp {
    pub exe: String,
    pub path: Option<String>,
    pub display_name: String,
    pub titles: Vec<String>,
}

#[tauri::command]
pub fn get_running_apps() -> Vec<RunningApp> {
    #[cfg(windows)]
    {
        windows_running_apps()
    }
    #[cfg(not(windows))]
    {
        vec![]
    }
}

#[cfg(windows)]
fn windows_running_apps() -> Vec<RunningApp> {
    use std::collections::HashMap;
    use std::ffi::OsString;
    use std::os::windows::ffi::OsStringExt;
    use windows::Win32::{
        Foundation::{BOOL, HWND, LPARAM},
        System::Threading::{
            OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_WIN32,
            PROCESS_QUERY_LIMITED_INFORMATION,
        },
        UI::WindowsAndMessaging::{
            EnumWindows, GetWindowLongW, GetWindowTextLengthW, GetWindowTextW,
            GetWindowThreadProcessId, IsWindowVisible, GWL_EXSTYLE, WS_EX_TOOLWINDOW,
        },
    };

    struct CollectData {
        entries: HashMap<String, RunningApp>,
    }

    unsafe extern "system" fn enum_proc(hwnd: HWND, lparam: LPARAM) -> BOOL {
        let data = &mut *(lparam.0 as *mut CollectData);

        if !IsWindowVisible(hwnd).as_bool() {
            return BOOL(1);
        }

        let ex_style = GetWindowLongW(hwnd, GWL_EXSTYLE) as u32;
        if ex_style & WS_EX_TOOLWINDOW.0 != 0 {
            return BOOL(1);
        }

        let title_len = GetWindowTextLengthW(hwnd);
        if title_len == 0 {
            return BOOL(1);
        }
        let mut title_buf = vec![0u16; title_len as usize + 1];
        let copied = GetWindowTextW(hwnd, &mut title_buf);
        if copied == 0 {
            return BOOL(1);
        }
        title_buf.truncate(copied as usize);
        let title = OsString::from_wide(&title_buf)
            .to_string_lossy()
            .to_string();

        let mut pid = 0u32;
        GetWindowThreadProcessId(hwnd, Some(&mut pid));
        if pid == 0 {
            return BOOL(1);
        }

        let Ok(handle) = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) else {
            return BOOL(1);
        };
        let mut path_buf = vec![0u16; 32768];
        let mut path_len = path_buf.len() as u32;
        let path_str = if QueryFullProcessImageNameW(
            handle,
            PROCESS_NAME_WIN32,
            windows::core::PWSTR(path_buf.as_mut_ptr()),
            &mut path_len,
        )
        .is_ok()
            && path_len > 0
        {
            path_buf.truncate(path_len as usize);
            Some(OsString::from_wide(&path_buf).to_string_lossy().to_string())
        } else {
            None
        };
        let _ = windows::Win32::Foundation::CloseHandle(handle);

        let exe = path_str
            .as_ref()
            .and_then(|p| std::path::Path::new(p).file_name())
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_default();

        if exe.is_empty() {
            return BOOL(1);
        }

        let exe_lower = exe.to_lowercase();
        let system_procs = [
            "explorer.exe",
            "searchhost.exe",
            "searchapp.exe",
            "shellexperiencehost.exe",
            "startmenuexperiencehost.exe",
            "applicationframehost.exe",
            "systemsettings.exe",
            "textinputhost.exe",
            "lockapp.exe",
            "dwm.exe",
            "taskhostw.exe",
            "sihost.exe",
            "ctfmon.exe",
        ];
        if system_procs.contains(&exe_lower.as_str()) {
            return BOOL(1);
        }

        let entry = data.entries.entry(exe.clone()).or_insert_with(|| {
            let stem = std::path::Path::new(&exe)
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| exe.clone());
            RunningApp {
                exe: exe.clone(),
                path: path_str,
                display_name: stem,
                titles: Vec::new(),
            }
        });

        if !entry.titles.contains(&title) && entry.titles.len() < 3 {
            entry.titles.push(title);
        }

        BOOL(1)
    }

    let mut data = CollectData {
        entries: HashMap::new(),
    };
    unsafe {
        let _ = EnumWindows(
            Some(enum_proc),
            LPARAM(&mut data as *mut CollectData as isize),
        );
    }

    let mut list: Vec<RunningApp> = data.entries.into_values().collect();
    list.sort_by(|a, b| {
        a.display_name
            .to_lowercase()
            .cmp(&b.display_name.to_lowercase())
    });
    list
}

pub fn load_initial_config() -> (AppConfig, Option<PathBuf>) {
    load_config(preferred_existing_config_path()).unwrap_or_default()
}

fn preferred_config_path() -> Option<PathBuf> {
    workspace_config_path().or_else(|| ConfigPaths::discover(None).selected_path())
}

fn preferred_existing_config_path() -> Option<PathBuf> {
    workspace_config_path().filter(|path| path.exists())
}

#[cfg(debug_assertions)]
fn workspace_config_path() -> Option<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest_dir.parent()?.parent()?;
    Some(workspace.join("keylink-studio.toml"))
}

#[cfg(not(debug_assertions))]
fn workspace_config_path() -> Option<PathBuf> {
    None
}

#[tauri::command]
pub fn get_config(state: State<AppState>) -> AppConfig {
    state.config.lock().unwrap().clone()
}

#[tauri::command]
pub fn get_config_path(state: State<AppState>) -> Option<String> {
    state
        .config_path
        .lock()
        .unwrap()
        .as_ref()
        .map(|p| p.to_string_lossy().to_string())
}

#[tauri::command]
pub fn save_config(config: AppConfig, state: State<AppState>) -> Result<(), String> {
    ensure_codex_config_editable(&state, &config)?;
    persist_config(config, state.inner())
}

fn persist_config(config: AppConfig, state: &AppState) -> Result<(), String> {
    // Reflect a `debug_log.enabled` change onto the running file sink before
    // writing anything to disk. If enabling fails (e.g. the exe's directory
    // isn't writable), the whole save is rejected and `enabled` stays at its
    // previous value in both the config file and `state.config` — it never
    // becomes `true` without the sink actually being active.
    if state.debug_log.is_enabled() != config.debug_log.enabled {
        state.debug_log.set_enabled(config.debug_log.enabled)?;
    }

    let toml_str = toml::to_string_pretty(&config).map_err(|e| e.to_string())?;

    let path = {
        let config_path = state.config_path.lock().unwrap();
        match config_path.clone() {
            Some(p) => p,
            None => preferred_config_path().unwrap_or_else(|| PathBuf::from("keylink-studio.toml")),
        }
    };

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(&path, &toml_str).map_err(|e| e.to_string())?;

    *state.config.lock().unwrap() = config.clone();
    *state.config_path.lock().unwrap() = Some(path);

    let ai_usage_shared = restart_ai_usage_runtime(&state, &config);

    if let Some(tx) = state.monitor_tx.lock().unwrap().as_ref() {
        let _ = tx.send(MonitorCommand::UpdateConfig(config, ai_usage_shared));
    } else {
        update_ai_usage_status(&state);
    }

    Ok(())
}

#[tauri::command]
pub fn reload_config(app: AppHandle, state: State<AppState>) -> Result<AppConfig, String> {
    let (mut config, path) =
        load_config(preferred_existing_config_path()).map_err(|e| e.to_string())?;
    ensure_codex_config_editable(&state, &config)?;

    if state.debug_log.is_enabled() != config.debug_log.enabled {
        if let Err(error) = state.debug_log.set_enabled(config.debug_log.enabled) {
            // Same handling as a `debug_log::init` startup failure: log the
            // error to the UI log and fall back to disabled in memory rather
            // than failing the whole reload. The on-disk toml (just read
            // above) is left as-is; the next save reconciles it.
            let message = format!("Debug file log could not be enabled: {error}");
            let entry = add_log(&state.log_entries, &state.log_counter, "error", &message);
            let _ = app.emit("log-added", entry);
            config.debug_log.enabled = false;
        }
    }

    *state.config.lock().unwrap() = config.clone();
    *state.config_path.lock().unwrap() = path;
    restart_ai_usage_runtime(&state, &config);
    update_ai_usage_status(&state);
    Ok(config)
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ConfigLocationResult {
    pub path: String,
    pub revealed: bool,
}

#[tauri::command]
pub fn show_config_file_location(state: State<AppState>) -> Result<ConfigLocationResult, String> {
    let path = {
        let config_path = state.config_path.lock().unwrap();
        match config_path.clone() {
            Some(p) => p,
            None => preferred_config_path().unwrap_or_else(|| PathBuf::from("keylink-studio.toml")),
        }
    };
    let revealed = reveal_config_path(&path);
    Ok(ConfigLocationResult {
        path: path.to_string_lossy().to_string(),
        revealed,
    })
}

#[cfg(windows)]
fn reveal_config_path(path: &std::path::Path) -> bool {
    std::process::Command::new("explorer.exe")
        .arg(format!("/select,{}", path.to_string_lossy()))
        .spawn()
        .is_ok()
}

#[cfg(not(windows))]
fn reveal_config_path(_path: &std::path::Path) -> bool {
    false
}

#[tauri::command]
pub fn get_status(state: State<AppState>) -> MonitorStatus {
    update_ai_usage_status(&state);
    state.status.lock().unwrap().clone()
}

#[tauri::command]
pub fn get_log_entries(state: State<AppState>) -> Vec<LogEntry> {
    state.log_entries.lock().unwrap().iter().cloned().collect()
}

#[tauri::command]
pub fn get_codex_integration_status(state: State<AppState>) -> CodexBrokerStatus {
    state.codex_broker.status()
}

/// Answers the exact Codex approval currently represented by
/// `request_key`. `decision_index` is resolved against the retained opaque
/// `availableDecisions` array; the frontend cannot fabricate a decision.
#[tauri::command]
pub fn respond_to_codex_approval(
    request_key: String,
    decision_index: usize,
    state: State<AppState>,
) -> Result<bool, String> {
    let pending = state.codex_activity.pending_approvals();
    respond_to_codex_approval_internal(&pending, &state.codex_broker, &request_key, decision_index)
}

/// Shared by the Tauri command and physical HUD actions. It always retrieves
/// the opaque decision from `PendingApprovalStore` immediately before using
/// the Broker's existing per-connection first-wins path.
pub(crate) fn respond_to_codex_approval_internal(
    pending: &PendingApprovalStore,
    broker: &CodexBrokerManager,
    request_key: &str,
    decision_index: usize,
) -> Result<bool, String> {
    let response = pending
        .codex_response(&request_key, decision_index)
        .ok_or_else(|| "The approval is no longer available".to_string())?;
    let outcome = broker
        .respond_to_approval(
            &response.connection_id,
            response.request_id,
            response.decision,
        )
        .map_err(|error| error.to_string())?;
    match outcome {
        CodexApprovalResponseOutcome::Accepted => {
            pending.resolve(&response.key);
            Ok(true)
        }
        CodexApprovalResponseOutcome::AlreadyResolved
        | CodexApprovalResponseOutcome::RequestNotFound
        | CodexApprovalResponseOutcome::ConnectionNotFound => {
            pending.resolve(&response.key);
            Ok(false)
        }
    }
}

/// Answers the exact Claude Code `PermissionRequest` currently represented
/// by `request_key`, the Claude Code counterpart to
/// `respond_to_codex_approval`. `decision_index` is resolved against the
/// Host-normalized `available_decisions` array -- `[CLAUDE_DECISION_ALLOW,
/// CLAUDE_DECISION_DENY]`, or, when the request also offered a
/// "remember this choice" option, the three-element `[CLAUDE_DECISION_ALLOW,
/// CLAUDE_DECISION_ALLOW_WITH_PERMISSIONS, CLAUDE_DECISION_DENY]` (see
/// `claude_activity.rs`'s `claude_approval_body`); the frontend cannot
/// fabricate a decision, exactly as with Codex.
#[tauri::command]
pub fn respond_to_claude_approval(
    request_key: String,
    decision_index: usize,
    state: State<AppState>,
) -> Result<bool, String> {
    let pending = state.codex_activity.pending_approvals();
    respond_to_claude_approval_internal(
        &pending,
        &state.claude_permission_gate,
        &request_key,
        decision_index,
    )
}

/// Shared by the Tauri command and physical HUD actions, mirroring
/// `respond_to_codex_approval_internal`'s shape. Unlike Codex's Broker
/// round trip, `ClaudePermissionGate::answer` is synchronous -- there is no
/// App Server to await -- but the lookup-then-answer order is exactly the
/// same "check the live state immediately before answering" discipline: by
/// the time this runs, `ClaudeApprovalBodyConsumer` may have already
/// canceled the gate waiter because the terminal answered first (§9.4's
/// first-wins rule), in which case `answer` below simply returns `false`
/// the same way Codex's `AlreadyResolved` does.
pub(crate) fn respond_to_claude_approval_internal(
    pending: &PendingApprovalStore,
    gate: &ClaudePermissionGate,
    request_key: &str,
    decision_index: usize,
) -> Result<bool, String> {
    let response = pending
        .claude_response(request_key, decision_index)
        .ok_or_else(|| "The approval is no longer available".to_string())?;
    // `diagnostic_label()`, not `behavior()`: the wire `behavior` string
    // collapses `Allow` and `AllowWithPermissions` to the same `"allow"`
    // (that collapse is what the real hook protocol expects -- see
    // `ClaudeDecision::behavior`'s own doc comment), which would make this
    // log unable to tell a plain allow from an allow-that-writes-permanent-
    // rules-to-disk. The diagnostic label keeps that distinction without
    // exposing anything about *what* rules were written.
    let behavior = response.decision.diagnostic_label();
    let answered = gate.answer(response.key.token(), response.decision);
    // Diagnostics only, at the same target/level as the other AI display
    // logging (`AI_DISPLAY_LOG_TARGET`) so a real-HID run that presses the
    // HUD button and sees nothing happen can be traced without turning on
    // full tracing. Deliberately just these three fields -- `behavior`,
    // `answered`, `waiting` -- and nothing else: no token, request key,
    // launch/session id, command text, tool name, hook body content, or
    // `updatedPermissions` contents (which, for an `AllowWithPermissions`
    // decision, can itself carry a full command string via `ruleContent`).
    // Those identify *what* was asked and *who* asked it, which this log
    // has no reason to know; `behavior`/`answered`/`waiting` are enough to
    // tell whether the press reached a waiting hook connection at all.
    tracing::debug!(
        target: AI_DISPLAY_LOG_TARGET,
        behavior,
        answered,
        waiting = gate.waiting_count(),
        "claude approval decision dispatched"
    );
    pending.resolve(&response.key);
    Ok(answered)
}

#[tauri::command]
pub fn get_ai_client_state(state: State<AppState>) -> AiClientStateSnapshot {
    state
        .ai_display_slots
        .lock()
        .expect("AI display selection lock poisoned")
        .selected_snapshot()
        .unwrap_or(AiClientStateSnapshot {
            client_type: AiClientType::Codex,
            client_variant: AiClientVariant::Cli,
            session_active: false,
            activity_state: AiActivityState::None,
            work_phase: AiWorkPhase::Unspecified,
            revision: 0,
        })
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AiDisplaySlotDto {
    pub slot: u8,
    pub mode: crate::state::AiDisplaySlotMode,
    pub target: Option<AiDisplayTarget>,
    pub snapshot: AiClientStateSnapshot,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AiDisplaySlotsDto {
    pub slots: Vec<AiDisplaySlotDto>,
    pub candidates: Vec<AiDisplayCandidateDto>,
    pub slot_capable_device_count: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AiDisplayCandidateDto {
    pub target: AiDisplayTarget,
    pub label: String,
    pub snapshot: AiClientStateSnapshot,
}

fn inactive_ai_snapshot() -> AiClientStateSnapshot {
    AiClientStateSnapshot {
        client_type: AiClientType::Codex,
        client_variant: AiClientVariant::Cli,
        session_active: false,
        activity_state: AiActivityState::None,
        work_phase: AiWorkPhase::Unspecified,
        revision: 0,
    }
}

#[tauri::command]
pub fn get_ai_display_slots(state: State<AppState>) -> AiDisplaySlotsDto {
    let slots = state
        .ai_display_slots
        .lock()
        .expect("AI display slots lock poisoned");
    let slot_capable_device_count = state
        .status
        .lock()
        .expect("monitor status lock poisoned")
        .host_link_devices
        .iter()
        .filter(|device| {
            device.capabilities
                & (CAPABILITY_AI_CLIENT_DISPLAY_SLOT
                    | rawhid_host_core::packet::CAPABILITY_AI_CLIENT_WORK_PHASE)
                == (CAPABILITY_AI_CLIENT_DISPLAY_SLOT
                    | rawhid_host_core::packet::CAPABILITY_AI_CLIENT_WORK_PHASE)
        })
        .count();
    AiDisplaySlotsDto {
        slots: slots
            .slots()
            .iter()
            .map(|entry| AiDisplaySlotDto {
                slot: entry.slot,
                mode: entry.mode.clone(),
                target: entry.assigned.clone(),
                snapshot: slots
                    .slot_snapshot(entry.slot)
                    .unwrap_or_else(inactive_ai_snapshot),
            })
            .collect(),
        candidates: slots
            .all_candidates()
            .iter()
            .map(|candidate| AiDisplayCandidateDto {
                target: candidate.target.clone(),
                label: candidate.target.label(),
                snapshot: candidate.snapshot,
            })
            .collect(),
        slot_capable_device_count,
    }
}

#[tauri::command]
pub fn pin_ai_display_slot(
    slot: u8,
    target: AiDisplayTarget,
    state: State<AppState>,
) -> Result<(), String> {
    state
        .ai_display_slots
        .lock()
        .expect("AI display slots lock poisoned")
        .pin(slot, target)
}

#[tauri::command]
pub fn set_ai_display_slot_auto(slot: u8, state: State<AppState>) -> Result<(), String> {
    state
        .ai_display_slots
        .lock()
        .expect("AI display slots lock poisoned")
        .set_auto(slot)
}

#[tauri::command]
pub async fn start_codex_integration(
    state: State<'_, AppState>,
) -> Result<CodexBrokerStatus, String> {
    let configured = state.config.lock().unwrap().clone();
    let manager = state.codex_broker.clone();
    let config = codex_broker_config(&configured)?;
    tauri::async_runtime::spawn_blocking(move || manager.start(config).map_err(|e| e.to_string()))
        .await
        .map_err(|error| error.to_string())?
}

#[tauri::command]
pub async fn launch_codex_cli(
    launcher: CodexLauncherConfig,
    state: State<'_, AppState>,
) -> Result<CodexLaunchCommandResult, String> {
    let project_for_title = match launcher.environment {
        CodexLaunchEnvironment::Windows => launcher.windows_project_directory.as_deref(),
        CodexLaunchEnvironment::Wsl => launcher.wsl_project_directory.as_deref(),
    }
    .ok_or_else(|| "Codex project directory is required".to_string())?;
    let terminal_target_id = new_terminal_target_id("codex")?;
    let display_name = terminal_display_name("Codex", project_for_title, &terminal_target_id);
    let manager = state.codex_broker.clone();
    match manager.status().phase {
        CodexBrokerPhase::Stopped
        | CodexBrokerPhase::Error
        | CodexBrokerPhase::WaitingForClient
        | CodexBrokerPhase::Connected
        | CodexBrokerPhase::Reconnecting => {}
        CodexBrokerPhase::Starting | CodexBrokerPhase::Stopping => {
            return Err("Codex連携の状態遷移が完了してから、もう一度起動してください".to_string());
        }
    }

    let launcher_to_validate = launcher.clone();
    tauri::async_runtime::spawn_blocking(move || codex_launcher::validate(&launcher_to_validate))
        .await
        .map_err(|error| error.to_string())??;

    let persisted_config =
        config_with_codex_launcher(state.config.lock().unwrap().clone(), launcher.clone());
    let response_config = persisted_config.clone();
    persist_config(persisted_config.clone(), state.inner())?;

    let launched = tauri::async_runtime::spawn_blocking(move || {
        let status = manager.status();
        match status.phase {
            CodexBrokerPhase::Stopped | CodexBrokerPhase::Error => {
                manager
                    .start(codex_broker_config(&persisted_config)?)
                    .map_err(|error| error.to_string())?;
            }
            CodexBrokerPhase::WaitingForClient
            | CodexBrokerPhase::Connected
            | CodexBrokerPhase::Reconnecting => {}
            CodexBrokerPhase::Starting | CodexBrokerPhase::Stopping => {
                return Err(
                    "Codex連携の状態遷移が完了してから、もう一度起動してください".to_string(),
                );
            }
        }
        let connection = manager
            .client_launch_info(terminal_target_id.clone(), display_name.clone())
            .map_err(|error| error.to_string())?;
        match codex_launcher::launch(&launcher, &connection) {
            Ok(launched) => Ok(launched),
            Err(error) => {
                manager.cancel_managed_launch(connection.terminal_target_id);
                Err(error)
            }
        }
    })
    .await
    .map_err(|error| error.to_string())??;

    Ok(CodexLaunchCommandResult {
        environment: launched.environment,
        project_directory: launched.project_directory,
        terminal_target_id: launched.terminal_target_id,
        display_name: launched.display_name,
        config: response_config,
    })
}

#[tauri::command]
pub async fn launch_claude_code(
    launcher: ClaudeLauncherConfig,
    state: State<'_, AppState>,
) -> Result<claude_launcher::ClaudeLaunchResult, String> {
    let to_validate = launcher.clone();
    tauri::async_runtime::spawn_blocking(move || claude_launcher::validate(&to_validate))
        .await
        .map_err(|error| error.to_string())??;
    let mut token = [0_u8; 32];
    fill_random(&mut token)
        .map_err(|error| format!("Claude Code連携用tokenを生成できません: {error}"))?;
    let terminal_target_id = new_terminal_target_id("claude")?;
    let project_for_title = launcher
        .project_directory
        .as_deref()
        .ok_or_else(|| "Claude Code project directory is required".to_string())?;
    let display_name = terminal_display_name("Claude Code", project_for_title, &terminal_target_id);
    let launch_id = format!("launch-{}", &terminal_target_id[7..]);
    // Share the process-wide gate rather than `loopback`'s own fresh one:
    // a physical HUD answer (routed through `AppState::claude_permission_gate`
    // via `MonitorExtras`) must reach the exact hook connection this launch's
    // receiver holds open, and vice versa for
    // `ClaudeApprovalBodyConsumer`'s first-wins cancellation.
    let mut observer_options =
        ClaudeObserverReceiverOptions::loopback(launch_id, hex::encode(token));
    observer_options.permission_gate = Arc::clone(&state.claude_permission_gate);
    let (receiver, events) = ClaudeObserverReceiver::bind(observer_options)
        .await
        .map_err(|error| error.to_string())?;
    let plugin_root = claude_plugin_root(receiver.config().launch_id.as_str())?;
    let helper_executable = claude_helper_executable()?;
    let options = ClaudePluginOptions {
        plugin_root: plugin_root.clone(),
        helper_executable: helper_executable.clone(),
        observer: receiver.config().clone(),
    };
    let artifacts =
        match tauri::async_runtime::spawn_blocking(move || write_claude_observer_plugin(&options))
            .await
        {
            Ok(Ok(artifacts)) => artifacts,
            Ok(Err(error)) => {
                let _ = receiver.shutdown().await;
                return Err(error.to_string());
            }
            Err(error) => {
                let _ = receiver.shutdown().await;
                return Err(error.to_string());
            }
        };
    let launcher_for_spawn = launcher.clone();
    let artifacts_for_spawn = artifacts.clone();
    let helper_for_spawn = helper_executable.clone();
    let terminal_target_id_for_spawn = terminal_target_id.clone();
    let display_name_for_spawn = display_name.clone();
    let launched = match tauri::async_runtime::spawn_blocking(move || {
        claude_launcher::launch(
            &launcher_for_spawn,
            &artifacts_for_spawn,
            &helper_for_spawn,
            &terminal_target_id_for_spawn,
            &display_name_for_spawn,
        )
    })
    .await
    {
        Ok(Ok(result)) => result,
        Ok(Err(error)) => {
            let _ = receiver.shutdown().await;
            let _ = std::fs::remove_dir_all(&plugin_root);
            return Err(error);
        }
        Err(error) => {
            let _ = receiver.shutdown().await;
            let _ = std::fs::remove_dir_all(&plugin_root);
            return Err(error.to_string());
        }
    };

    let updated = config_with_claude_launcher(state.config.lock().unwrap().clone(), launcher);
    persist_config(updated, state.inner())?;
    let launch_id = receiver.config().launch_id.clone();
    let launch = ClaudeLaunchIntegration {
        receiver,
        events,
        last_counters: Default::default(),
        plugin_root,
        terminal_target_id,
        display_name,
        timed_out_at: Some(Instant::now() + Duration::from_secs(30)),
        remove_at: None,
    };
    let mut guard = state.claude_integration.lock().unwrap();
    let integration = guard.get_or_insert_with(|| ClaudeIntegration {
        launches: Default::default(),
        registry: Default::default(),
    });
    integration.launches.insert(launch_id, launch);
    drop(guard);
    Ok(launched)
}

#[tauri::command]
pub fn get_claude_sessions(state: State<AppState>) -> Vec<ClaudeSessionSnapshot> {
    state
        .claude_integration
        .lock()
        .unwrap()
        .as_ref()
        .map(|integration| integration.registry.snapshots())
        .unwrap_or_default()
}

#[tauri::command]
pub async fn stop_claude_code(state: State<'_, AppState>) -> Result<(), String> {
    let integration = state.claude_integration.lock().unwrap().take();
    let Some(integration) = integration else {
        return Ok(());
    };
    let mut errors = Vec::new();
    for launch in integration.launches.into_values() {
        let root = launch.plugin_root.clone();
        if let Err(error) = launch.receiver.shutdown().await {
            errors.push(error.to_string());
        }
        match tauri::async_runtime::spawn_blocking(move || {
            std::fs::remove_dir_all(root).map_err(|error| error.to_string())
        })
        .await
        {
            Ok(Ok(())) => {}
            Ok(Err(error)) => errors.push(error),
            Err(error) => errors.push(error.to_string()),
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

fn config_with_codex_launcher(mut config: AppConfig, launcher: CodexLauncherConfig) -> AppConfig {
    config.ai_client.codex_launcher = launcher;
    config
}

fn config_with_claude_launcher(mut config: AppConfig, launcher: ClaudeLauncherConfig) -> AppConfig {
    config.ai_client.claude_launcher = launcher;
    config
}

fn new_terminal_target_id(provider: &str) -> Result<String, String> {
    let mut bytes = [0_u8; 16];
    fill_random(&mut bytes)
        .map_err(|error| format!("terminal target ID generation failed: {error}"))?;
    Ok(format!("{provider}-{}", hex::encode(bytes)))
}

pub(crate) fn terminal_display_name(
    provider: &str,
    project: &str,
    terminal_target_id: &str,
) -> String {
    let project = project
        .trim_end_matches(['/', '\\'])
        .rsplit(['/', '\\'])
        .next()
        .filter(|name| !name.is_empty())
        .unwrap_or("project");
    let suffix = terminal_target_id
        .rsplit('-')
        .next()
        .unwrap_or_default()
        .chars()
        .take(8)
        .collect::<String>()
        .to_ascii_uppercase();
    format!("{provider} · {project} · {suffix}")
}

fn claude_plugin_root(launch_id: &str) -> Result<PathBuf, String> {
    let data = ProjectDirs::from("", "", "Keylink Studio")
        .ok_or_else(|| "Keylink Studioのデータディレクトリを決定できません".to_string())?
        .data_local_dir()
        .join("claude-observer");
    Ok(data.join(launch_id))
}

fn claude_helper_executable() -> Result<PathBuf, String> {
    let executable = std::env::current_exe().map_err(|error| error.to_string())?;
    let directory = executable
        .parent()
        .ok_or_else(|| "Keylink Studio実行ファイルの親ディレクトリを取得できません".to_string())?;
    let name = if cfg!(windows) {
        "keylink-claude-hook.exe"
    } else {
        "keylink-claude-hook"
    };
    Ok(directory.join(name))
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CodexLaunchCommandResult {
    pub environment: CodexLaunchEnvironment,
    pub project_directory: String,
    pub terminal_target_id: String,
    pub display_name: String,
    pub config: AppConfig,
}

#[tauri::command]
pub async fn list_wsl_distributions() -> Result<Vec<codex_launcher::WslDistribution>, String> {
    tauri::async_runtime::spawn_blocking(codex_launcher::list_wsl_distributions)
        .await
        .map_err(|error| error.to_string())?
}

#[tauri::command]
pub async fn stop_codex_integration(
    state: State<'_, AppState>,
) -> Result<CodexBrokerStatus, String> {
    let manager = state.codex_broker.clone();
    tauri::async_runtime::spawn_blocking(move || manager.stop().map_err(|e| e.to_string()))
        .await
        .map_err(|error| error.to_string())?
}

pub fn shutdown_codex_integration(state: &AppState) {
    let _ = state.codex_broker.stop();
}

fn codex_broker_config(config: &AppConfig) -> Result<CodexBrokerConfig, String> {
    let configured = &config.ai_client.codex;
    let launcher = &config.ai_client.codex_launcher;
    let runtime = match launcher.environment {
        CodexLaunchEnvironment::Windows => CodexAppServerRuntime::Windows,
        CodexLaunchEnvironment::Wsl => CodexAppServerRuntime::Wsl {
            distribution: launcher
                .wsl_distribution
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| "WSLディストリビューションを設定してください".to_string())?
                .to_string(),
            executable: launcher.wsl_executable.trim().to_string(),
        },
    };
    Ok(CodexBrokerConfig {
        codex_executable: configured.executable_path.as_deref().map(PathBuf::from),
        runtime,
        version_check_enabled: configured.version_check_enabled,
        app_server_port: configured.app_server_port,
        broker_port: configured.broker_port,
        ..CodexBrokerConfig::default()
    })
}

fn ensure_codex_config_editable(state: &AppState, next: &AppConfig) -> Result<(), String> {
    let current = state.config.lock().unwrap().ai_client.codex.clone();
    let phase = state.codex_broker.status().phase;
    if current != next.ai_client.codex
        && !matches!(phase, CodexBrokerPhase::Stopped | CodexBrokerPhase::Error)
    {
        return Err(
            "Codex integration settings cannot be changed while integration is running".to_string(),
        );
    }
    Ok(())
}

#[tauri::command]
pub async fn probe_devices(state: State<'_, AppState>) -> Result<Vec<ProbeResult>, String> {
    let tx = host_link_sender(state.inner())?;
    let timeout = host_link_reply_timeout(&state.config.lock().unwrap());
    tauri::async_runtime::spawn_blocking(move || {
        let (reply, receiver) = mpsc::channel();
        tx.send(MonitorCommand::Probe(reply))
            .map_err(|_| "hostlink_worker_unavailable".to_string())?;
        receiver
            .recv_timeout(timeout)
            .map_err(|_| "hostlink_result_unknown".to_string())?
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
pub async fn probe_studio_devices(
    state: State<'_, AppState>,
) -> Result<Vec<StudioDeviceStatus>, String> {
    let edit = Arc::clone(&state.studio_edit);
    let studio_config = state.config.lock().unwrap().studio.clone();
    tauri::async_runtime::spawn_blocking(move || {
        {
            let mut guard = edit.lock().unwrap();
            if let Some(session) = guard.as_mut() {
                match session.has_unsaved() {
                    Ok(_) => return Err("port_busy".to_string()),
                    Err(error) if studio_session_connection_lost(&error) => {
                        tracing::info!(
                            device_id = %session.device_id,
                            "released disconnected Studio edit session before scan"
                        );
                        *guard = None;
                    }
                    Err(_) => return Err("port_busy".to_string()),
                }
            }
        }
        Ok(probe_all_studio_devices(&studio_config))
    })
    .await
    .map_err(|_| "studio_probe_failed".to_string())?
}

fn studio_session_connection_lost(error: &StudioError) -> bool {
    matches!(error, StudioError::Disconnected | StudioError::Timeout)
}

#[tauri::command]
pub async fn read_studio_keymap(
    device_id: String,
    state: State<'_, AppState>,
) -> Result<StudioKeymapSnapshot, String> {
    let edit = Arc::clone(&state.studio_edit);
    let studio_config = state.config.lock().unwrap().studio.clone();
    tauri::async_runtime::spawn_blocking(move || {
        {
            let mut guard = edit.lock().unwrap();
            if let Some(session) = guard.as_mut() {
                if session.device_id == device_id {
                    return session.snapshot().map_err(studio_error_code);
                }
                return Err("port_busy".to_string());
            }
        }
        read_keymap_for_device(&device_id, &studio_config).map_err(studio_error_code)
    })
    .await
    .map_err(|_| "studio_read_failed".to_string())?
}

#[derive(Debug, Clone, serde::Serialize, PartialEq, Eq)]
pub struct KeymapExportReport {
    pub warnings: Vec<RestoreIssue>,
}

#[tauri::command]
pub async fn studio_export_keymap(
    device_id: String,
    path: String,
    host_link_uid: Option<String>,
    state: State<'_, AppState>,
) -> Result<KeymapExportReport, String> {
    let edit = Arc::clone(&state.studio_edit);
    let studio_config = state.config.lock().unwrap().studio.clone();
    let host_link_tx = host_link_sender(state.inner())?;
    let host_link_timeout = host_link_reply_timeout(&state.config.lock().unwrap());
    tauri::async_runtime::spawn_blocking(move || {
        let prepared = (|| {
            let snapshot = {
                let mut guard = edit.lock().unwrap();
                if let Some(session) = guard.as_mut() {
                    if session.device_id == device_id {
                        session.snapshot().map_err(studio_error_code)?
                    } else {
                        return Err(studio_error_code(StudioError::PortBusy));
                    }
                } else {
                    read_keymap_for_device(&device_id, &studio_config).map_err(studio_error_code)?
                }
            };

            let encoders = match host_link_uid.as_deref() {
                None => None,
                Some(uid) => Some(export_encoder_overrides(
                    &device_id,
                    uid,
                    &edit,
                    &studio_config,
                    host_link_tx.clone(),
                    host_link_timeout,
                    &snapshot,
                )?),
            };

            let (combos, warnings) = match host_link_uid.as_deref() {
                None => (
                    None,
                    vec![RestoreIssue {
                        code: "combo_export_unavailable".to_string(),
                        layer_index: None,
                        position: None,
                        message: "Host Link is not connected; Combo settings were not exported"
                            .to_string(),
                    }],
                ),
                Some(uid) => match export_combo_table(
                    &device_id,
                    uid,
                    &edit,
                    &studio_config,
                    host_link_tx,
                    host_link_timeout,
                ) {
                    Ok(combos) => (Some(combos), Vec::new()),
                    Err(message) => (
                        None,
                        vec![RestoreIssue {
                            code: "combo_export_unavailable".to_string(),
                            layer_index: None,
                            position: None,
                            message,
                        }],
                    ),
                },
            };

            let backup = keymap_backup_from_snapshot(
                &snapshot,
                Default::default(),
                env!("CARGO_PKG_VERSION"),
                encoders,
                combos,
            );
            let text = serialize_keymap_backup(&backup).map_err(keymap_file_error_code)?;
            Ok((text, KeymapExportReport { warnings }))
        })();

        write_completed_keymap_export(&PathBuf::from(path), prepared)
    })
    .await
    .map_err(|_| "keymap_export_failed".to_string())?
}

/// Writes an export only after every data source has completed successfully.
/// Keeping the preparation result explicit prevents a future partial-export
/// path from truncating or replacing the destination before encoder reads end.
fn write_completed_keymap_export(
    path: &std::path::Path,
    prepared: Result<(String, KeymapExportReport), String>,
) -> Result<KeymapExportReport, String> {
    let (text, report) = prepared?;
    write_keymap_backup_file(path, &text)?;
    Ok(report)
}

/// Reads every runtime encoder override for `device_id` over Host Link Config
/// RPC and labels each one exactly like normal keys, so the export mirrors the
/// key backup format 1:1. Any failure here (uid parse, device lookup, GET_INFO,
/// GET_BINDINGS, label resolution) fails the whole export -- callers must never
/// write a backup file that silently dropped the encoder section.
fn export_encoder_overrides(
    device_id: &str,
    host_link_uid: &str,
    edit: &Arc<Mutex<Option<StudioEditSession>>>,
    studio_config: &rawhid_host_core::config::StudioConfig,
    host_link_tx: mpsc::Sender<MonitorCommand>,
    host_link_timeout: Duration,
    snapshot: &StudioKeymapSnapshot,
) -> Result<BackupEncoders, String> {
    let target_uid = parse_host_link_uid(host_link_uid)?;
    let info = match host_link_call(
        &host_link_tx,
        host_link_timeout,
        target_uid,
        HostLinkRequest::EncoderGetInfo,
    )? {
        HostLinkResponse::EncoderInfo(info) => info,
        _ => return Err("hostlink_invalid_response".to_string()),
    };
    let timeout = Duration::from_millis(studio_config.keymap_read_timeout_ms.max(1));

    let mut overrides = Vec::new();
    for layer in &snapshot.layers {
        for encoder_id in 0..info.encoder_count {
            let bindings = match host_link_call(
                &host_link_tx,
                host_link_timeout,
                target_uid,
                HostLinkRequest::EncoderGetBindings {
                    layer_id: layer.id,
                    encoder_id,
                },
            )? {
                HostLinkResponse::EncoderBindings(bindings) => bindings,
                _ => return Err("hostlink_invalid_response".to_string()),
            };
            if bindings.source != EncoderBindingSource::Override {
                continue;
            }
            let [cw_label, ccw_label] = label_encoder_bindings_for_device(
                device_id,
                edit,
                studio_config,
                bindings.source,
                bindings.cw_binding,
                bindings.ccw_binding,
                timeout,
            )?;
            overrides.push(BackupEncoderOverride {
                layer_index: layer.index,
                layer_id: layer.id,
                encoder_id,
                cw: backup_encoder_binding(bindings.cw_binding, cw_label),
                ccw: backup_encoder_binding(bindings.ccw_binding, ccw_label),
            });
        }
    }

    Ok(BackupEncoders {
        encoder_count: info.encoder_count,
        overrides,
    })
}

fn backup_encoder_binding(
    binding: EncoderBinding,
    label: Option<StudioBindingLabelPatch>,
) -> BackupEncoderBinding {
    match label {
        Some(patch) => BackupEncoderBinding {
            behavior_id: binding.behavior_id,
            param1: binding.param1,
            param2: binding.param2,
            behavior: patch.behavior,
            label: patch.full_label,
        },
        None => BackupEncoderBinding {
            behavior_id: binding.behavior_id,
            param1: binding.param1,
            param2: binding.param2,
            behavior: format!("behavior {}", binding.behavior_id),
            label: String::new(),
        },
    }
}

fn export_combo_table(
    device_id: &str,
    host_link_uid: &str,
    edit: &Arc<Mutex<Option<StudioEditSession>>>,
    studio_config: &rawhid_host_core::config::StudioConfig,
    host_link_tx: mpsc::Sender<MonitorCommand>,
    host_link_timeout: Duration,
) -> Result<BackupCombos, String> {
    let uid = parse_host_link_uid(host_link_uid)?;
    let info = match host_link_call(
        &host_link_tx,
        host_link_timeout,
        uid,
        HostLinkRequest::ComboGetInfo,
    )? {
        HostLinkResponse::ComboInfo(info) => info,
        _ => return Err("hostlink_invalid_response".to_string()),
    };
    let mut entries = Vec::new();
    for slot in 0..info.max_combos {
        if info.occupied_slots & (1u32 << slot) == 0 {
            continue;
        }
        let item = match host_link_call(
            &host_link_tx,
            host_link_timeout,
            uid,
            HostLinkRequest::ComboGet { slot },
        )? {
            HostLinkResponse::ComboItem(item) => item,
            _ => return Err("hostlink_invalid_response".to_string()),
        };
        let label = label_combo_binding_for_device(
            device_id,
            edit,
            studio_config,
            item.binding,
            host_link_timeout,
        );
        entries.push(BackupCombo {
            name: item.name.as_str().to_string(),
            key_positions: item.key_positions[..usize::from(item.key_count)].to_vec(),
            slow_release: item.flags.slow_release(),
            binding: backup_encoder_binding(
                EncoderBinding {
                    behavior_id: item.binding.behavior_id,
                    param1: item.binding.param1,
                    param2: item.binding.param2,
                },
                label,
            ),
            layer_mask: item.layer_mask,
            timeout_ms: item.timeout_ms,
            require_prior_idle_ms: item.require_prior_idle_ms,
        });
    }
    Ok(BackupCombos { entries })
}

#[tauri::command]
pub async fn studio_preview_keymap_restore(
    device_id: String,
    path: String,
    host_link_uid: Option<String>,
    state: State<'_, AppState>,
) -> Result<RestoreReport, String> {
    let edit = Arc::clone(&state.studio_edit);
    let studio_config = state.config.lock().unwrap().studio.clone();
    let host_link_tx = host_link_sender(state.inner())?;
    let host_link_timeout = host_link_reply_timeout(&state.config.lock().unwrap());
    tauri::async_runtime::spawn_blocking(move || {
        let backup = read_keymap_backup_file(&PathBuf::from(path))?;

        let snapshot = {
            let mut guard = edit.lock().unwrap();
            let session = guard
                .as_mut()
                .ok_or_else(|| studio_error_code(StudioError::NoEditSession))?;
            if session.device_id != device_id {
                return Err(studio_error_code(StudioError::SessionDeviceMismatch));
            }
            session.snapshot().map_err(studio_error_code)?
        };

        let mut ids = backup_behavior_ids(&backup);
        let encoder_ctx = prepare_encoder_restore_context(
            &backup,
            host_link_uid.as_deref(),
            host_link_tx.clone(),
            host_link_timeout,
            &snapshot,
            &mut ids,
        )?;
        let combo_ctx = prepare_combo_restore_context(
            &backup,
            host_link_uid.as_deref(),
            host_link_tx.clone(),
            host_link_timeout,
            &mut ids,
        )
        .ok()
        .flatten();

        let target_names = {
            let mut guard = edit.lock().unwrap();
            let session = guard
                .as_mut()
                .ok_or_else(|| studio_error_code(StudioError::NoEditSession))?;
            if session.device_id != device_id {
                return Err(studio_error_code(StudioError::SessionDeviceMismatch));
            }
            session
                .resolve_behavior_names(
                    &ids,
                    Duration::from_millis(studio_config.keymap_read_timeout_ms.max(1)),
                )
                .map_err(studio_error_code)?
        };

        let mut report = plan_keymap_restore(&snapshot, target_names.as_ref(), &backup).report;
        merge_encoder_restore_into_report(
            &mut report,
            &backup,
            &snapshot,
            encoder_ctx.as_ref(),
            target_names.as_ref(),
            host_link_uid.is_some(),
        );
        merge_combo_restore_into_report(
            &mut report,
            &backup,
            &snapshot,
            combo_ctx.as_ref(),
            target_names.as_ref(),
            host_link_uid.is_some(),
        );

        Ok(report)
    })
    .await
    .map_err(|_| "keymap_restore_preview_failed".to_string())?
}

#[tauri::command]
pub async fn studio_apply_keymap_restore(
    device_id: String,
    path: String,
    host_link_uid: Option<String>,
    state: State<'_, AppState>,
) -> Result<(StudioKeymapSnapshot, RestoreReport), String> {
    let edit = Arc::clone(&state.studio_edit);
    let studio_config = state.config.lock().unwrap().studio.clone();
    let host_link_tx = host_link_sender(state.inner())?;
    let host_link_timeout = host_link_reply_timeout(&state.config.lock().unwrap());
    let encoder_rollbacks = Arc::clone(&state.encoder_restore_rollbacks);
    tauri::async_runtime::spawn_blocking(move || {
        let backup = read_keymap_backup_file(&PathBuf::from(path))?;

        let snapshot = {
            let mut guard = edit.lock().unwrap();
            let session = guard
                .as_mut()
                .ok_or_else(|| studio_error_code(StudioError::NoEditSession))?;
            if session.device_id != device_id {
                return Err(studio_error_code(StudioError::SessionDeviceMismatch));
            }
            session.snapshot().map_err(studio_error_code)?
        };

        let mut ids = backup_behavior_ids(&backup);
        let mut encoder_ctx = prepare_encoder_restore_context(
            &backup,
            host_link_uid.as_deref(),
            host_link_tx.clone(),
            host_link_timeout,
            &snapshot,
            &mut ids,
        )?;
        let mut combo_ctx = prepare_combo_restore_context(
            &backup,
            host_link_uid.as_deref(),
            host_link_tx.clone(),
            host_link_timeout,
            &mut ids,
        )
        .ok()
        .flatten();

        let (mut next, mut report, encoder_writes, combo_writes, key_failed) = {
            let mut guard = edit.lock().unwrap();
            let session = guard
                .as_mut()
                .ok_or_else(|| studio_error_code(StudioError::NoEditSession))?;
            if session.device_id != device_id {
                return Err(studio_error_code(StudioError::SessionDeviceMismatch));
            }
            let target_names = session
                .resolve_behavior_names(
                    &ids,
                    Duration::from_millis(studio_config.keymap_read_timeout_ms.max(1)),
                )
                .map_err(studio_error_code)?;

            let key_plan = plan_keymap_restore(&snapshot, target_names.as_ref(), &backup);
            let mut report = key_plan.report;
            let encoder_writes = merge_encoder_restore_into_report(
                &mut report,
                &backup,
                &snapshot,
                encoder_ctx.as_ref(),
                target_names.as_ref(),
                host_link_uid.is_some(),
            );
            let combo_writes = merge_combo_restore_into_report(
                &mut report,
                &backup,
                &snapshot,
                combo_ctx.as_ref(),
                target_names.as_ref(),
                host_link_uid.is_some(),
            );

            if !report.can_apply {
                return Err("restore_structure_mismatch".to_string());
            }

            session.begin_backup_restore(&snapshot);
            let mut next = snapshot.clone();
            let mut key_failed = false;
            for write in &key_plan.writes {
                match session.apply_raw_writes(std::slice::from_ref(write)) {
                    Ok(updated) => {
                        next = updated;
                        report.applied_keys.push(RestoreChangedKey {
                            layer_index: write.layer_index,
                            position: write.position.max(0) as usize,
                        });
                    }
                    Err(error) => {
                        key_failed = true;
                        report.apply_status = RestoreApplyStatus::Partial;
                        report.errors.push(RestoreIssue {
                            code: "key_apply_failed".to_string(),
                            layer_index: Some(write.layer_index),
                            position: Some(write.position.max(0) as usize),
                            message: studio_error_code(error),
                        });
                        break;
                    }
                }
            }
            (next, report, encoder_writes, combo_writes, key_failed)
        };

        if !key_failed {
            if let Some(ctx) = encoder_ctx.as_mut() {
                if !encoder_writes.is_empty() {
                    let mut rollbacks = encoder_rollbacks.lock().unwrap();
                    let rollback = rollbacks.entry((device_id.clone(), ctx.uid)).or_default();
                    for write in &encoder_writes {
                        if let Some(current) = ctx
                            .current_bindings
                            .get(&(write.layer_id, write.encoder_id))
                        {
                            rollback
                                .entry((write.layer_id, write.encoder_id))
                                .or_insert(*current);
                        }
                    }
                }
                for write in &encoder_writes {
                    let result = host_link_call(
                        &ctx.tx,
                        ctx.timeout,
                        ctx.uid,
                        HostLinkRequest::EncoderSetBindings {
                            layer_id: write.layer_id,
                            encoder_id: write.encoder_id,
                            cw: write.cw,
                            ccw: write.ccw,
                        },
                    );
                    match result {
                        Ok(HostLinkResponse::Done) => {
                            if let Some(changed) = report.changed_encoders.iter().find(|changed| {
                                changed.encoder_id == write.encoder_id
                                    && snapshot
                                        .layers
                                        .get(changed.layer_index)
                                        .is_some_and(|layer| layer.id == write.layer_id)
                            }) {
                                report.applied_encoders.push(changed.clone());
                            }
                        }
                        other => {
                            let message = match other {
                                Err(error) => error,
                                Ok(_) => "hostlink_invalid_response".to_string(),
                            };
                            report.apply_status = RestoreApplyStatus::Partial;
                            report.errors.push(RestoreIssue {
                                code: "encoder_apply_failed".to_string(),
                                layer_index: snapshot
                                    .layers
                                    .iter()
                                    .position(|layer| layer.id == write.layer_id),
                                position: None,
                                message,
                            });
                            break;
                        }
                    }
                }
            }
        }

        if let Some(ctx) = combo_ctx.as_mut() {
            for write in &combo_writes {
                let result = host_link_call(
                    &ctx.tx,
                    ctx.timeout,
                    ctx.uid,
                    HostLinkRequest::ComboSet { item: write.item },
                );
                match result {
                    Ok(HostLinkResponse::Done) => {
                        report.applied_combos.push(write.changed.clone());
                    }
                    other => {
                        let message = match other {
                            Err(error) => error,
                            Ok(_) => "hostlink_invalid_response".to_string(),
                        };
                        report.apply_status = RestoreApplyStatus::Partial;
                        report.errors.push(RestoreIssue {
                            code: "combo_apply_failed".to_string(),
                            layer_index: None,
                            position: None,
                            message,
                        });
                        break;
                    }
                }
            }
        }

        if report.apply_status != RestoreApplyStatus::Partial {
            report.apply_status = RestoreApplyStatus::Complete;
        } else {
            let refresh = {
                let mut guard = edit.lock().unwrap();
                guard.as_mut().and_then(|session| session.snapshot().ok())
            };
            if let Some(refreshed) = refresh {
                next = refreshed;
            } else {
                report.errors.push(RestoreIssue {
                    code: "state_refresh_failed".to_string(),
                    layer_index: None,
                    position: None,
                    message: "failed to refresh Studio keymap after partial restore".to_string(),
                });
            }
        }

        Ok((next, report))
    })
    .await
    .map_err(|_| "keymap_restore_apply_failed".to_string())?
}

fn backup_behavior_ids(backup: &rawhid_host_core::studio::KeymapBackup) -> BTreeSet<i32> {
    backup
        .layers
        .iter()
        .flat_map(|layer| layer.bindings.iter().map(|binding| binding.behavior_id))
        .collect()
}

/// Holds an already-open Config RPC connection plus the current encoder state
/// relevant to the backup being restored, gathered *before* the Studio edit
/// session lock is (re-)acquired. `manager`/`device` are kept open so `apply`
/// can reuse them for the SET calls after the key-side writes succeed, without
/// ever touching Host Link while `state.studio_edit` is locked.
struct EncoderRestoreContext {
    tx: mpsc::Sender<MonitorCommand>,
    timeout: Duration,
    uid: u64,
    encoder_count: u8,
    current_bindings: BTreeMap<(u32, u8), EncoderGetBindings>,
}

struct ComboRestoreContext {
    tx: mpsc::Sender<MonitorCommand>,
    timeout: Duration,
    uid: u64,
    max_combos: u8,
    max_keys_per_combo: u8,
    current: Vec<ComboItem>,
}

fn prepare_combo_restore_context(
    backup: &rawhid_host_core::studio::KeymapBackup,
    host_link_uid: Option<&str>,
    host_link_tx: mpsc::Sender<MonitorCommand>,
    host_link_timeout: Duration,
    ids: &mut BTreeSet<i32>,
) -> Result<Option<ComboRestoreContext>, String> {
    let Some(backup_combos) = backup.combos.as_ref() else {
        return Ok(None);
    };
    for combo in &backup_combos.entries {
        ids.insert(i32::from(combo.binding.behavior_id));
    }
    if backup_combos.entries.is_empty() {
        return Ok(None);
    }
    let Some(uid) = host_link_uid else {
        return Ok(None);
    };
    let target_uid = parse_host_link_uid(uid)?;
    let info = match host_link_call(
        &host_link_tx,
        host_link_timeout,
        target_uid,
        HostLinkRequest::ComboGetInfo,
    )? {
        HostLinkResponse::ComboInfo(info) => info,
        _ => return Err("hostlink_invalid_response".to_string()),
    };
    let mut current = Vec::new();
    for slot in 0..info.max_combos {
        if info.occupied_slots & (1u32 << slot) == 0 {
            continue;
        }
        let item = match host_link_call(
            &host_link_tx,
            host_link_timeout,
            target_uid,
            HostLinkRequest::ComboGet { slot },
        )? {
            HostLinkResponse::ComboItem(item) => item,
            _ => return Err("hostlink_invalid_response".to_string()),
        };
        current.push(item);
    }
    Ok(Some(ComboRestoreContext {
        tx: host_link_tx,
        timeout: host_link_timeout,
        uid: target_uid,
        max_combos: info.max_combos,
        max_keys_per_combo: info.max_keys_per_combo,
        current,
    }))
}

/// Builds the encoder restore context for a backup, if applicable, and folds
/// every encoder behavior id referenced by the backup into `ids` so a single
/// `resolve_behavior_names` call (done later, under the session lock) covers
/// both keys and encoders. Returns `Ok(None)` when the backup has no encoder
/// section, or when it does but no Host Link uid was supplied (handled later
/// by `merge_encoder_restore_into_report` as an all-blocked warning).
fn prepare_encoder_restore_context(
    backup: &rawhid_host_core::studio::KeymapBackup,
    host_link_uid: Option<&str>,
    host_link_tx: mpsc::Sender<MonitorCommand>,
    host_link_timeout: Duration,
    snapshot: &StudioKeymapSnapshot,
    ids: &mut BTreeSet<i32>,
) -> Result<Option<EncoderRestoreContext>, String> {
    let (Some(backup_encoders), Some(uid)) = (backup.encoders.as_ref(), host_link_uid) else {
        return Ok(None);
    };

    let target_uid = parse_host_link_uid(uid)?;
    let info = match host_link_call(
        &host_link_tx,
        host_link_timeout,
        target_uid,
        HostLinkRequest::EncoderGetInfo,
    )? {
        HostLinkResponse::EncoderInfo(info) => info,
        _ => return Err("hostlink_invalid_response".to_string()),
    };
    let encoder_count = info.encoder_count;

    let mut current_bindings = BTreeMap::new();
    for override_ in &backup_encoders.overrides {
        ids.insert(i32::from(override_.cw.behavior_id));
        ids.insert(i32::from(override_.ccw.behavior_id));
        let Some(layer) = resolve_encoder_restore_layer(snapshot, backup, override_) else {
            continue;
        };
        if override_.encoder_id >= encoder_count {
            continue;
        }
        let key = (layer.id, override_.encoder_id);
        if current_bindings.contains_key(&key) {
            continue;
        }
        let bindings = match host_link_call(
            &host_link_tx,
            host_link_timeout,
            target_uid,
            HostLinkRequest::EncoderGetBindings {
                layer_id: layer.id,
                encoder_id: override_.encoder_id,
            },
        )? {
            HostLinkResponse::EncoderBindings(bindings) => bindings,
            _ => return Err("hostlink_invalid_response".to_string()),
        };
        current_bindings.insert(key, bindings);
    }

    Ok(Some(EncoderRestoreContext {
        tx: host_link_tx,
        timeout: host_link_timeout,
        uid: target_uid,
        encoder_count,
        current_bindings,
    }))
}

fn resolve_encoder_restore_layer<'a>(
    snapshot: &'a StudioKeymapSnapshot,
    backup: &rawhid_host_core::studio::KeymapBackup,
    override_: &BackupEncoderOverride,
) -> Option<&'a rawhid_host_core::studio::StudioLayer> {
    if let Some(layer) = snapshot
        .layers
        .iter()
        .find(|layer| layer.id == override_.layer_id)
    {
        return Some(layer);
    }
    let source = backup
        .layers
        .iter()
        .find(|layer| layer.id == override_.layer_id)
        .or_else(|| backup.layers.get(override_.layer_index))?;
    let candidate = snapshot.layers.get(override_.layer_index)?;
    (normalize_restore_name(&source.name) == normalize_restore_name(&candidate.name))
        .then_some(candidate)
}

fn normalize_restore_name(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Merges the encoder side of a restore plan into `report` and returns the
/// writes to apply (empty unless called from `studio_apply_keymap_restore`
/// with `can_apply` about to be checked by the caller). When the backup has
/// an encoder section but no Host Link uid was supplied, every override is
/// counted as blocked with a dedicated warning instead of being silently
/// dropped.
fn merge_encoder_restore_into_report(
    report: &mut RestoreReport,
    backup: &rawhid_host_core::studio::KeymapBackup,
    snapshot: &StudioKeymapSnapshot,
    encoder_ctx: Option<&EncoderRestoreContext>,
    target_names: Option<&BTreeMap<i32, String>>,
    host_link_uid_present: bool,
) -> Vec<EncoderRestoreWrite> {
    let Some(backup_encoders) = backup.encoders.as_ref() else {
        return Vec::new();
    };

    match encoder_ctx {
        Some(ctx) => {
            let encoder_plan = plan_encoder_restore(
                &snapshot.layers,
                &backup.layers,
                ctx.encoder_count,
                &ctx.current_bindings,
                target_names,
                backup_encoders,
            );
            report.encoder_will_write = encoder_plan.will_write;
            report.encoder_unchanged_skipped = encoder_plan.unchanged_skipped;
            report.encoder_blocked = encoder_plan.blocked;
            report.changed_encoders = encoder_plan.changed_encoders;
            report.warnings.extend(encoder_plan.warnings);
            encoder_plan.writes
        }
        None => {
            if !host_link_uid_present {
                report.encoder_blocked = backup_encoders.overrides.len();
                report.warnings.push(RestoreIssue {
                    code: "encoder_hostlink_missing".to_string(),
                    layer_index: None,
                    position: None,
                    message: "Host Link is not connected; encoder overrides in the backup were not restored".to_string(),
                });
            }
            Vec::new()
        }
    }
}

fn merge_combo_restore_into_report(
    report: &mut RestoreReport,
    backup: &rawhid_host_core::studio::KeymapBackup,
    snapshot: &StudioKeymapSnapshot,
    combo_ctx: Option<&ComboRestoreContext>,
    target_names: Option<&BTreeMap<i32, String>>,
    host_link_uid_present: bool,
) -> Vec<ComboRestoreWrite> {
    let Some(backup_combos) = backup.combos.as_ref() else {
        return Vec::new();
    };
    if backup_combos.entries.is_empty() {
        return Vec::new();
    }
    let Some(ctx) = combo_ctx else {
        report.combo_blocked = backup_combos.entries.len();
        report.warnings.push(RestoreIssue {
            code: "combo_unavailable".to_string(),
            layer_index: None,
            position: None,
            message: if host_link_uid_present {
                "Combo configuration is not supported or could not be read".to_string()
            } else {
                "Host Link is not connected; Combos in the backup were not restored".to_string()
            },
        });
        return Vec::new();
    };
    let positions = snapshot
        .selected_layout_keys
        .iter()
        .filter_map(|key| u16::try_from(key.position).ok())
        .collect();
    let plan = plan_combo_restore(
        &ctx.current,
        ctx.max_combos,
        ctx.max_keys_per_combo,
        &positions,
        snapshot.layers.len(),
        target_names,
        backup_combos,
    );
    report.combo_added = plan.added;
    report.combo_updated = plan.updated;
    report.combo_unchanged_skipped = plan.unchanged_skipped;
    report.combo_blocked = plan.blocked;
    report.changed_combos = plan.changed_combos;
    report.warnings.extend(plan.warnings);
    plan.writes
}

fn read_keymap_backup_file(
    path: &std::path::Path,
) -> Result<rawhid_host_core::studio::KeymapBackup, String> {
    let metadata = std::fs::metadata(path).map_err(|_| "keymap_invalid_path".to_string())?;
    if !metadata.is_file() {
        return Err("keymap_invalid_path".to_string());
    }
    if metadata.len() > KEYMAP_BACKUP_MAX_BYTES as u64 {
        return Err("keymap_file_too_large".to_string());
    }
    let text = std::fs::read_to_string(path).map_err(|_| "keymap_invalid_file".to_string())?;
    parse_keymap_backup(&text).map_err(keymap_file_error_code)
}

fn write_keymap_backup_file(path: &std::path::Path, text: &str) -> Result<(), String> {
    let Some(parent) = path.parent() else {
        return Err("keymap_invalid_path".to_string());
    };
    if !parent.is_dir() {
        return Err("keymap_invalid_path".to_string());
    }
    std::fs::write(path, text).map_err(|_| "keymap_invalid_path".to_string())
}

fn keymap_file_error_code(error: KeymapFileError) -> String {
    match error {
        KeymapFileError::InvalidFile => "keymap_invalid_file",
        KeymapFileError::UnsupportedVersion => "keymap_unsupported_version",
        KeymapFileError::FileTooLarge => "keymap_file_too_large",
    }
    .to_string()
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EditBehaviorDto {
    KeyPress {
        hid_usage: u32,
    },
    Transparent,
    None,
    MomentaryLayer {
        target_layer_index: u32,
    },
    ToggleLayer {
        target_layer_index: u32,
    },
    ToLayer {
        target_layer_index: u32,
    },
    ModTap {
        hold_hid_usage: u32,
        tap_hid_usage: u32,
    },
    LayerTap {
        target_layer_index: u32,
        tap_hid_usage: u32,
    },
    StickyKey {
        hid_usage: u32,
    },
    StickyLayer {
        target_layer_index: u32,
    },
    Bluetooth {
        command: u32,
        value: u32,
    },
    OutputSelection {
        value: u32,
    },
    MouseKeyPress {
        value: u32,
    },
    MouseMove {
        value: u32,
    },
    MouseScroll {
        value: u32,
    },
    CapsWord,
    KeyRepeat,
    Reset,
    Bootloader,
    StudioUnlock,
    GraveEscape,
    HostAction {
        action_id: u32,
        value: u32,
    },
}

impl From<EditBehaviorDto> for EditBehavior {
    fn from(value: EditBehaviorDto) -> Self {
        match value {
            EditBehaviorDto::KeyPress { hid_usage } => EditBehavior::KeyPress(hid_usage),
            EditBehaviorDto::Transparent => EditBehavior::Transparent,
            EditBehaviorDto::None => EditBehavior::None,
            EditBehaviorDto::MomentaryLayer { target_layer_index } => {
                EditBehavior::MomentaryLayer(target_layer_index)
            }
            EditBehaviorDto::ToggleLayer { target_layer_index } => {
                EditBehavior::ToggleLayer(target_layer_index)
            }
            EditBehaviorDto::ToLayer { target_layer_index } => {
                EditBehavior::ToLayer(target_layer_index)
            }
            EditBehaviorDto::ModTap {
                hold_hid_usage,
                tap_hid_usage,
            } => EditBehavior::ModTap {
                hold: hold_hid_usage,
                tap: tap_hid_usage,
            },
            EditBehaviorDto::LayerTap {
                target_layer_index,
                tap_hid_usage,
            } => EditBehavior::LayerTap {
                target_layer_index,
                tap: tap_hid_usage,
            },
            EditBehaviorDto::StickyKey { hid_usage } => EditBehavior::StickyKey(hid_usage),
            EditBehaviorDto::StickyLayer { target_layer_index } => {
                EditBehavior::StickyLayer(target_layer_index)
            }
            EditBehaviorDto::Bluetooth { command, value } => {
                EditBehavior::Bluetooth { command, value }
            }
            EditBehaviorDto::OutputSelection { value } => EditBehavior::OutputSelection(value),
            EditBehaviorDto::MouseKeyPress { value } => EditBehavior::MouseKeyPress(value),
            EditBehaviorDto::MouseMove { value } => EditBehavior::MouseMove(value),
            EditBehaviorDto::MouseScroll { value } => EditBehavior::MouseScroll(value),
            EditBehaviorDto::CapsWord => EditBehavior::CapsWord,
            EditBehaviorDto::KeyRepeat => EditBehavior::KeyRepeat,
            EditBehaviorDto::Reset => EditBehavior::Reset,
            EditBehaviorDto::Bootloader => EditBehavior::Bootloader,
            EditBehaviorDto::StudioUnlock => EditBehavior::StudioUnlock,
            EditBehaviorDto::GraveEscape => EditBehavior::GraveEscape,
            EditBehaviorDto::HostAction { action_id, value } => {
                EditBehavior::HostAction { action_id, value }
            }
        }
    }
}

#[tauri::command]
pub fn studio_key_catalog() -> Vec<KeyCatalogEntry> {
    key_catalog()
}

#[tauri::command]
pub async fn resolve_studio_behavior_labels(
    device_id: String,
    raw_bindings: Vec<StudioRawBinding>,
    state: State<'_, AppState>,
) -> Result<Vec<StudioBindingLabelPatch>, String> {
    let studio_config = state.config.lock().unwrap().studio.clone();
    let edit = Arc::clone(&state.studio_edit);
    tauri::async_runtime::spawn_blocking(move || {
        {
            let mut guard = edit.lock().unwrap();
            if let Some(session) = guard.as_mut() {
                if session.device_id == device_id {
                    return session
                        .resolve_behavior_labels(
                            raw_bindings,
                            Duration::from_millis(studio_config.keymap_read_timeout_ms.max(1)),
                        )
                        .map_err(studio_error_code);
                }
                return Err(studio_error_code(StudioError::PortBusy));
            }
        }
        resolve_behavior_labels_for_device(&device_id, raw_bindings, &studio_config)
            .map_err(studio_error_code)
    })
    .await
    .map_err(|_| "studio_behavior_resolve_failed".to_string())?
}

#[tauri::command]
pub async fn studio_begin_edit(
    device_id: String,
    force_discard: bool,
    label_patches: Vec<StudioBindingLabelPatch>,
    state: State<'_, AppState>,
) -> Result<StudioKeymapSnapshot, String> {
    let studio_config = state.config.lock().unwrap().studio.clone();
    let edit = Arc::clone(&state.studio_edit);
    tauri::async_runtime::spawn_blocking(move || {
        let mut guard = edit.lock().unwrap();
        if let Some(session) = guard.as_mut() {
            if session.device_id == device_id {
                session.seed_behavior_labels(&label_patches);
                if force_discard {
                    return session.discard().map_err(studio_error_code);
                }
                return session.snapshot().map_err(studio_error_code);
            }
            if force_discard {
                let _ = session.discard();
                *guard = None;
            } else if session.has_unsaved().map_err(studio_error_code)? {
                return Err(studio_error_code(StudioError::UnsavedChangesExist));
            } else {
                *guard = None;
            }
        }

        let (session, snapshot) = StudioEditSession::open_with_snapshot(&device_id, &studio_config)
            .map_err(studio_error_code)?;
        let mut session = session;
        session.seed_behavior_labels(&label_patches);
        *guard = Some(session);
        Ok(snapshot)
    })
    .await
    .map_err(|_| "studio_begin_edit_failed".to_string())?
}

#[tauri::command]
pub async fn studio_set_key(
    device_id: String,
    layer_id: u32,
    position: i32,
    behavior: EditBehaviorDto,
    state: State<'_, AppState>,
) -> Result<StudioKeymapSnapshot, String> {
    let edit = Arc::clone(&state.studio_edit);
    tauri::async_runtime::spawn_blocking(move || {
        let mut guard = edit.lock().unwrap();
        let session = guard
            .as_mut()
            .ok_or_else(|| studio_error_code(StudioError::NoEditSession))?;
        if session.device_id != device_id {
            return Err(studio_error_code(StudioError::SessionDeviceMismatch));
        }
        session
            .set_binding(layer_id, position, behavior.into())
            .map_err(studio_error_code)
    })
    .await
    .map_err(|_| "studio_set_key_failed".to_string())?
}

#[tauri::command]
pub async fn studio_add_layer(
    device_id: String,
    name: String,
    state: State<'_, AppState>,
) -> Result<StudioKeymapSnapshot, String> {
    let edit = Arc::clone(&state.studio_edit);
    tauri::async_runtime::spawn_blocking(move || {
        let mut guard = edit.lock().unwrap();
        let session = guard
            .as_mut()
            .ok_or_else(|| studio_error_code(StudioError::NoEditSession))?;
        if session.device_id != device_id {
            return Err(studio_error_code(StudioError::SessionDeviceMismatch));
        }
        session.add_layer(name).map_err(studio_error_code)
    })
    .await
    .map_err(|_| "studio_add_layer_failed".to_string())?
}

#[tauri::command]
pub async fn studio_rename_layer(
    device_id: String,
    layer_id: u32,
    name: String,
    state: State<'_, AppState>,
) -> Result<StudioKeymapSnapshot, String> {
    let edit = Arc::clone(&state.studio_edit);
    tauri::async_runtime::spawn_blocking(move || {
        let mut guard = edit.lock().unwrap();
        let session = guard
            .as_mut()
            .ok_or_else(|| studio_error_code(StudioError::NoEditSession))?;
        if session.device_id != device_id {
            return Err(studio_error_code(StudioError::SessionDeviceMismatch));
        }
        session
            .rename_layer(layer_id, name)
            .map_err(studio_error_code)
    })
    .await
    .map_err(|_| "studio_rename_layer_failed".to_string())?
}

#[tauri::command]
pub async fn studio_remove_layer(
    device_id: String,
    layer_index: u32,
    state: State<'_, AppState>,
) -> Result<StudioKeymapSnapshot, String> {
    let edit = Arc::clone(&state.studio_edit);
    tauri::async_runtime::spawn_blocking(move || {
        let mut guard = edit.lock().unwrap();
        let session = guard
            .as_mut()
            .ok_or_else(|| studio_error_code(StudioError::NoEditSession))?;
        if session.device_id != device_id {
            return Err(studio_error_code(StudioError::SessionDeviceMismatch));
        }
        session.remove_layer(layer_index).map_err(studio_error_code)
    })
    .await
    .map_err(|_| "studio_remove_layer_failed".to_string())?
}

/// Outcome of attempting a save-or-discard operation on a single target
/// (Studio RPC keys, or a Config RPC feature). `skipped` implies `success:
/// true`: there was nothing to do (no dirty state) or the target was not
/// connected at all. `attempted` means the underlying RPC call actually ran.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SaveOrDiscardTargetDto {
    pub attempted: bool,
    pub skipped: bool,
    pub success: bool,
    pub error: Option<String>,
}

/// Per-feature Config RPC result (currently only "ENCODER"; COMBO/TAP_DANCE
/// will add more entries to `ConfigSaveOrDiscardDto::results` without changing
/// this shape).
#[derive(Debug, Clone, serde::Serialize)]
pub struct ConfigFeatureResultDto {
    pub feature: String,
    pub attempted: bool,
    pub skipped: bool,
    pub success: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ConfigSaveOrDiscardDto {
    pub attempted: bool,
    pub skipped: bool,
    pub success: bool,
    pub results: Vec<ConfigFeatureResultDto>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SaveOrDiscardResultDto {
    pub overall_success: bool,
    pub studio: SaveOrDiscardTargetDto,
    pub config: ConfigSaveOrDiscardDto,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct DiscardChangesDto {
    pub result: SaveOrDiscardResultDto,
    /// Present when the studio-side discard ran successfully (the re-read snapshot).
    pub snapshot: Option<StudioKeymapSnapshot>,
}

/// Result of restoring the editable Studio and Keylink encoder state to the
/// firmware `.keymap`. The two transports are independent, so callers must
/// expose a partial success rather than flattening it into one error.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ResetToKeymapDto {
    pub overall_success: bool,
    pub studio: SaveOrDiscardTargetDto,
    pub config: ConfigSaveOrDiscardDto,
    pub snapshot: Option<StudioKeymapSnapshot>,
    pub refresh_error: Option<String>,
}

fn config_save_or_discard_from_features(
    results: Vec<ConfigFeatureResultDto>,
) -> ConfigSaveOrDiscardDto {
    let attempted = results.iter().any(|feature| feature.attempted);
    let skipped = results.iter().all(|feature| feature.skipped);
    let success = results.iter().all(|feature| feature.success);
    ConfigSaveOrDiscardDto {
        attempted,
        skipped,
        success,
        results,
    }
}

fn config_save_or_discard_from_feature(feature: ConfigFeatureResultDto) -> ConfigSaveOrDiscardDto {
    config_save_or_discard_from_features(vec![feature])
}

fn skipped_config_feature(feature: &str) -> ConfigFeatureResultDto {
    ConfigFeatureResultDto {
        feature: feature.to_string(),
        attempted: false,
        skipped: true,
        success: true,
        error: None,
    }
}

fn failed_config_feature(feature: &str, error: impl Into<String>) -> ConfigFeatureResultDto {
    ConfigFeatureResultDto {
        feature: feature.to_string(),
        attempted: true,
        skipped: false,
        success: false,
        error: Some(error.into()),
    }
}

fn failed_encoder_feature(error: impl Into<String>) -> ConfigFeatureResultDto {
    failed_config_feature("ENCODER", error)
}

enum EncoderFeatureOp {
    Save,
    Discard,
}

enum ComboFeatureOp {
    Save,
    Discard,
}

fn run_combo_feature(
    tx: mpsc::Sender<MonitorCommand>,
    timeout: Duration,
    host_link_uid: String,
    op: ComboFeatureOp,
) -> ConfigFeatureResultDto {
    let fail = |error: String| failed_config_feature("COMBO", error);
    let target_uid = match parse_host_link_uid(&host_link_uid) {
        Ok(uid) => uid,
        Err(error) => return fail(error),
    };
    let dirty = match host_link_call(&tx, timeout, target_uid, HostLinkRequest::ComboGetDirty) {
        Ok(HostLinkResponse::Dirty(dirty)) => dirty,
        Ok(_) => return fail("hostlink_invalid_response".to_string()),
        Err(error) => return fail(error),
    };
    if !dirty {
        return skipped_config_feature("COMBO");
    }
    let request = match op {
        ComboFeatureOp::Save => HostLinkRequest::ComboSave,
        ComboFeatureOp::Discard => HostLinkRequest::ComboDiscard,
    };
    match host_link_call(&tx, timeout, target_uid, request) {
        Ok(HostLinkResponse::Done) => ConfigFeatureResultDto {
            feature: "COMBO".to_string(),
            attempted: true,
            skipped: false,
            success: true,
            error: None,
        },
        Ok(_) => fail("hostlink_invalid_response".to_string()),
        Err(error) => fail(error),
    }
}

fn run_combo_reset_to_keymap(
    tx: mpsc::Sender<MonitorCommand>,
    timeout: Duration,
    host_link_uid: String,
) -> ConfigFeatureResultDto {
    let fail = |error: String| failed_config_feature("COMBO", error);
    let target_uid = match parse_host_link_uid(&host_link_uid) {
        Ok(uid) => uid,
        Err(error) => return fail(error),
    };
    match host_link_call(
        &tx,
        timeout,
        target_uid,
        HostLinkRequest::ComboResetToKeymap,
    ) {
        Ok(HostLinkResponse::Done) => {}
        Ok(_) => return fail("hostlink_invalid_response".to_string()),
        Err(error) => return fail(error),
    }
    match host_link_call(&tx, timeout, target_uid, HostLinkRequest::ComboSave) {
        Ok(HostLinkResponse::Done) => ConfigFeatureResultDto {
            feature: "COMBO".to_string(),
            attempted: true,
            skipped: false,
            success: true,
            error: None,
        },
        Ok(_) => fail("hostlink_invalid_response".to_string()),
        Err(error) => fail(error),
    }
}

fn run_encoder_feature(
    tx: mpsc::Sender<MonitorCommand>,
    timeout: Duration,
    host_link_uid: String,
    op: EncoderFeatureOp,
) -> ConfigFeatureResultDto {
    let fail = |error: String| ConfigFeatureResultDto {
        feature: "ENCODER".to_string(),
        attempted: true,
        skipped: false,
        success: false,
        error: Some(error),
    };
    let target_uid = match parse_host_link_uid(&host_link_uid) {
        Ok(uid) => uid,
        Err(err) => return fail(err),
    };
    let dirty = match host_link_call(&tx, timeout, target_uid, HostLinkRequest::EncoderGetDirty) {
        Ok(HostLinkResponse::Dirty(dirty)) => dirty,
        Ok(_) => return fail("hostlink_invalid_response".to_string()),
        Err(err) => return fail(err),
    };
    if !dirty {
        return ConfigFeatureResultDto {
            feature: "ENCODER".to_string(),
            attempted: false,
            skipped: true,
            success: true,
            error: None,
        };
    }
    let request = match op {
        EncoderFeatureOp::Save => HostLinkRequest::EncoderSave,
        EncoderFeatureOp::Discard => HostLinkRequest::EncoderDiscard,
    };
    let result = host_link_call(&tx, timeout, target_uid, request);
    match result {
        Ok(HostLinkResponse::Done) => ConfigFeatureResultDto {
            feature: "ENCODER".to_string(),
            attempted: true,
            skipped: false,
            success: true,
            error: None,
        },
        Ok(_) => fail("hostlink_invalid_response".to_string()),
        Err(err) => fail(err),
    }
}

fn run_encoder_reset_to_keymap(
    tx: mpsc::Sender<MonitorCommand>,
    timeout: Duration,
    host_link_uid: String,
    encoder_info: EncoderGetInfo,
    snapshot: &StudioKeymapSnapshot,
) -> ConfigFeatureResultDto {
    let fail = |error: String| ConfigFeatureResultDto {
        feature: "ENCODER".to_string(),
        attempted: true,
        skipped: false,
        success: false,
        error: Some(error),
    };
    let target_uid = match parse_host_link_uid(&host_link_uid) {
        Ok(uid) => uid,
        Err(err) => return fail(err),
    };

    // `Layer.id` is the stable firmware ID. It is explicitly not the UI index
    // nor the 0..layer_count range reported by GET_INFO.
    let layer_ids: BTreeSet<u32> = snapshot.layers.iter().map(|layer| layer.id).collect();
    for layer_id in layer_ids {
        for encoder_id in 0..encoder_info.encoder_count {
            match host_link_call(
                &tx,
                timeout,
                target_uid,
                HostLinkRequest::EncoderClearOverride {
                    layer_id,
                    encoder_id,
                },
            ) {
                Ok(HostLinkResponse::Done) => {}
                Ok(_) => {
                    let _ =
                        host_link_call(&tx, timeout, target_uid, HostLinkRequest::EncoderDiscard);
                    return fail("hostlink_invalid_response".to_string());
                }
                Err(error) => {
                    let _ =
                        host_link_call(&tx, timeout, target_uid, HostLinkRequest::EncoderDiscard);
                    return fail(error);
                }
            }
        }
    }

    // SAVE also removes stale/orphan persisted entries that no longer have a
    // layer in the reset stock keymap, even when encoder_count is zero.
    match host_link_call(&tx, timeout, target_uid, HostLinkRequest::EncoderSave) {
        Ok(HostLinkResponse::Done) => ConfigFeatureResultDto {
            feature: "ENCODER".to_string(),
            attempted: true,
            skipped: false,
            success: true,
            error: None,
        },
        Ok(_) => fail("hostlink_invalid_response".to_string()),
        Err(error) => fail(error),
    }
}

fn same_encoder_binding_content(left: &EncoderGetBindings, right: &EncoderGetBindings) -> bool {
    left.layer_id == right.layer_id
        && left.encoder_id == right.encoder_id
        && left.source == right.source
        && left.cw_binding == right.cw_binding
        && left.ccw_binding == right.ccw_binding
}

fn run_encoder_discard_with_rollback(
    tx: mpsc::Sender<MonitorCommand>,
    timeout: Duration,
    host_link_uid: String,
    rollback: BTreeMap<(u32, u8), EncoderGetBindings>,
) -> ConfigFeatureResultDto {
    let fail = |error: String| ConfigFeatureResultDto {
        feature: "ENCODER".to_string(),
        attempted: true,
        skipped: false,
        success: false,
        error: Some(error),
    };
    let target_uid = match parse_host_link_uid(&host_link_uid) {
        Ok(uid) => uid,
        Err(err) => return fail(err),
    };
    let dirty = match host_link_call(&tx, timeout, target_uid, HostLinkRequest::EncoderGetDirty) {
        Ok(HostLinkResponse::Dirty(dirty)) => dirty,
        Ok(_) => return fail("hostlink_invalid_response".to_string()),
        Err(err) => return fail(err),
    };

    if dirty {
        match host_link_call(&tx, timeout, target_uid, HostLinkRequest::EncoderDiscard) {
            Ok(HostLinkResponse::Done) => {}
            Ok(_) => return fail("hostlink_invalid_response".to_string()),
            Err(err) => return fail(err),
        }
    }

    let mut repaired = false;
    for baseline in rollback.values() {
        let current = match host_link_call(
            &tx,
            timeout,
            target_uid,
            HostLinkRequest::EncoderGetBindings {
                layer_id: baseline.layer_id,
                encoder_id: baseline.encoder_id,
            },
        ) {
            Ok(HostLinkResponse::EncoderBindings(bindings)) => bindings,
            Ok(_) => return fail("hostlink_invalid_response".to_string()),
            Err(err) => return fail(err),
        };
        if same_encoder_binding_content(&current, baseline) {
            continue;
        }

        let request = match baseline.source {
            EncoderBindingSource::Override => HostLinkRequest::EncoderSetBindings {
                layer_id: baseline.layer_id,
                encoder_id: baseline.encoder_id,
                cw: baseline.cw_binding,
                ccw: baseline.ccw_binding,
            },
            EncoderBindingSource::Keymap => HostLinkRequest::EncoderClearOverride {
                layer_id: baseline.layer_id,
                encoder_id: baseline.encoder_id,
            },
        };
        match host_link_call(&tx, timeout, target_uid, request) {
            Ok(HostLinkResponse::Done) => repaired = true,
            Ok(_) => return fail("hostlink_invalid_response".to_string()),
            Err(err) => return fail(err),
        }
    }

    if repaired {
        match host_link_call(&tx, timeout, target_uid, HostLinkRequest::EncoderSave) {
            Ok(HostLinkResponse::Done) => {}
            Ok(_) => return fail("hostlink_invalid_response".to_string()),
            Err(err) => return fail(err),
        }
    }

    for baseline in rollback.values() {
        let current = match host_link_call(
            &tx,
            timeout,
            target_uid,
            HostLinkRequest::EncoderGetBindings {
                layer_id: baseline.layer_id,
                encoder_id: baseline.encoder_id,
            },
        ) {
            Ok(HostLinkResponse::EncoderBindings(bindings)) => bindings,
            Ok(_) => return fail("hostlink_invalid_response".to_string()),
            Err(err) => return fail(err),
        };
        if !same_encoder_binding_content(&current, baseline) {
            return fail("encoder_discard_restore_mismatch".to_string());
        }
    }

    ConfigFeatureResultDto {
        feature: "ENCODER".to_string(),
        // A rollback baseline means the device was re-read even when no write
        // was necessary. Report it as attempted so the UI refreshes values
        // that may still be showing the imported backup.
        attempted: true,
        skipped: false,
        success: true,
        error: None,
    }
}

#[tauri::command]
pub async fn studio_save_changes(
    device_id: String,
    host_link_uid: Option<String>,
    combo_host_link_uid: Option<String>,
    state: State<'_, AppState>,
) -> Result<SaveOrDiscardResultDto, String> {
    let edit = Arc::clone(&state.studio_edit);
    let edit_device_id = device_id.clone();
    let studio = tauri::async_runtime::spawn_blocking(move || {
        let mut guard = edit.lock().unwrap();
        let session = match guard.as_mut() {
            Some(session) => session,
            None => {
                return SaveOrDiscardTargetDto {
                    attempted: true,
                    skipped: false,
                    success: false,
                    error: Some(studio_error_code(StudioError::NoEditSession)),
                };
            }
        };
        if session.device_id != edit_device_id {
            return SaveOrDiscardTargetDto {
                attempted: true,
                skipped: false,
                success: false,
                error: Some(studio_error_code(StudioError::SessionDeviceMismatch)),
            };
        }
        let has_unsaved = match session.has_unsaved() {
            Ok(has_unsaved) => has_unsaved,
            Err(err) => {
                return SaveOrDiscardTargetDto {
                    attempted: true,
                    skipped: false,
                    success: false,
                    error: Some(studio_error_code(err)),
                };
            }
        };
        if !has_unsaved {
            return SaveOrDiscardTargetDto {
                attempted: false,
                skipped: true,
                success: true,
                error: None,
            };
        }
        match session.save() {
            Ok(()) => SaveOrDiscardTargetDto {
                attempted: true,
                skipped: false,
                success: true,
                error: None,
            },
            Err(err) => SaveOrDiscardTargetDto {
                attempted: true,
                skipped: false,
                success: false,
                error: Some(studio_error_code(err)),
            },
        }
    })
    .await
    .map_err(|_| "studio_save_failed".to_string())?;

    let saved_uid = host_link_uid.clone();
    let encoder_feature = match host_link_uid {
        None => skipped_config_feature("ENCODER"),
        Some(uid) => match host_link_sender(state.inner()) {
            Ok(tx) => {
                let timeout = host_link_reply_timeout(&state.config.lock().unwrap());
                match tauri::async_runtime::spawn_blocking(move || {
                    run_encoder_feature(tx, timeout, uid, EncoderFeatureOp::Save)
                })
                .await
                {
                    Ok(feature) => feature,
                    Err(_) => failed_encoder_feature("studio_encoder_save_failed"),
                }
            }
            Err(error) => failed_encoder_feature(error),
        },
    };
    let combo_feature = match combo_host_link_uid {
        None => skipped_config_feature("COMBO"),
        Some(uid) => match host_link_sender(state.inner()) {
            Ok(tx) => {
                let timeout = host_link_reply_timeout(&state.config.lock().unwrap());
                tauri::async_runtime::spawn_blocking(move || {
                    run_combo_feature(tx, timeout, uid, ComboFeatureOp::Save)
                })
                .await
                .unwrap_or_else(|_| failed_config_feature("COMBO", "studio_combo_save_failed"))
            }
            Err(error) => failed_config_feature("COMBO", error),
        },
    };
    let config = config_save_or_discard_from_features(vec![encoder_feature, combo_feature]);

    if config.success {
        if let Some(uid) = saved_uid {
            if let Ok(uid) = parse_host_link_uid(&uid) {
                state
                    .encoder_restore_rollbacks
                    .lock()
                    .unwrap()
                    .remove(&(device_id, uid));
            }
        }
    }

    // `skipped` always carries `success: true`, so this is exactly "no
    // attempted target failed".
    let overall_success = studio.success && config.success;
    Ok(SaveOrDiscardResultDto {
        overall_success,
        studio,
        config,
    })
}

#[tauri::command]
pub async fn studio_discard_changes(
    device_id: String,
    host_link_uid: Option<String>,
    combo_host_link_uid: Option<String>,
    state: State<'_, AppState>,
) -> Result<DiscardChangesDto, String> {
    let edit = Arc::clone(&state.studio_edit);
    let edit_device_id = device_id.clone();
    let (studio, snapshot) = tauri::async_runtime::spawn_blocking(move || {
        let mut guard = edit.lock().unwrap();
        let session = match guard.as_mut() {
            Some(session) => session,
            None => {
                return (
                    SaveOrDiscardTargetDto {
                        attempted: true,
                        skipped: false,
                        success: false,
                        error: Some(studio_error_code(StudioError::NoEditSession)),
                    },
                    None,
                );
            }
        };
        if session.device_id != edit_device_id {
            return (
                SaveOrDiscardTargetDto {
                    attempted: true,
                    skipped: false,
                    success: false,
                    error: Some(studio_error_code(StudioError::SessionDeviceMismatch)),
                },
                None,
            );
        }
        // Preserving the current implementation's behavior: discard runs
        // unconditionally (there is no "nothing to discard" skip path in the
        // pre-refactor command either).
        match session.discard() {
            Ok(snapshot) => (
                SaveOrDiscardTargetDto {
                    attempted: true,
                    skipped: false,
                    success: true,
                    error: None,
                },
                Some(snapshot),
            ),
            Err(err) => (
                SaveOrDiscardTargetDto {
                    attempted: true,
                    skipped: false,
                    success: false,
                    error: Some(studio_error_code(err)),
                },
                None,
            ),
        }
    })
    .await
    .map_err(|_| "studio_discard_failed".to_string())?;

    let discarded_uid = host_link_uid.clone();
    let rollback = discarded_uid
        .as_deref()
        .and_then(|uid| parse_host_link_uid(uid).ok())
        .and_then(|uid| {
            state
                .encoder_restore_rollbacks
                .lock()
                .unwrap()
                .get(&(device_id.clone(), uid))
                .cloned()
        });
    let encoder_feature = match host_link_uid {
        None => skipped_config_feature("ENCODER"),
        Some(uid) => match host_link_sender(state.inner()) {
            Ok(tx) => {
                let timeout = host_link_reply_timeout(&state.config.lock().unwrap());
                match tauri::async_runtime::spawn_blocking(move || match rollback {
                    Some(rollback) => run_encoder_discard_with_rollback(tx, timeout, uid, rollback),
                    None => run_encoder_feature(tx, timeout, uid, EncoderFeatureOp::Discard),
                })
                .await
                {
                    Ok(feature) => feature,
                    Err(_) => failed_encoder_feature("studio_encoder_discard_failed"),
                }
            }
            Err(error) => failed_encoder_feature(error),
        },
    };
    let combo_feature = match combo_host_link_uid {
        None => skipped_config_feature("COMBO"),
        Some(uid) => match host_link_sender(state.inner()) {
            Ok(tx) => {
                let timeout = host_link_reply_timeout(&state.config.lock().unwrap());
                tauri::async_runtime::spawn_blocking(move || {
                    run_combo_feature(tx, timeout, uid, ComboFeatureOp::Discard)
                })
                .await
                .unwrap_or_else(|_| failed_config_feature("COMBO", "studio_combo_discard_failed"))
            }
            Err(error) => failed_config_feature("COMBO", error),
        },
    };
    let config = config_save_or_discard_from_features(vec![encoder_feature, combo_feature]);

    if config.success {
        if let Some(uid) = discarded_uid {
            if let Ok(uid) = parse_host_link_uid(&uid) {
                state
                    .encoder_restore_rollbacks
                    .lock()
                    .unwrap()
                    .remove(&(device_id, uid));
            }
        }
    }

    let overall_success = studio.success && config.success;
    Ok(DiscardChangesDto {
        result: SaveOrDiscardResultDto {
            overall_success,
            studio,
            config,
        },
        snapshot,
    })
}

#[tauri::command]
pub async fn studio_reset_to_keymap(
    device_id: String,
    host_link_uid: Option<String>,
    combo_host_link_uid: Option<String>,
    state: State<'_, AppState>,
) -> Result<ResetToKeymapDto, String> {
    // Verify the optional encoder transport before touching the Studio state.
    // A paired Host Link device means this command promises to reset both
    // editable surfaces, so a preflight failure must be fail-closed.
    let preflight = match host_link_uid.clone() {
        None => Ok(None),
        Some(uid) => match host_link_sender(state.inner()) {
            Err(error) => Err(failed_encoder_feature(error)),
            Ok(tx) => {
                let timeout = host_link_reply_timeout(&state.config.lock().unwrap());
                match tauri::async_runtime::spawn_blocking(move || {
                    let target_uid = parse_host_link_uid(&uid)?;
                    match host_link_call(&tx, timeout, target_uid, HostLinkRequest::EncoderGetInfo)?
                    {
                        HostLinkResponse::EncoderInfo(info) => Ok((uid, info)),
                        _ => Err("hostlink_invalid_response".to_string()),
                    }
                })
                .await
                {
                    Ok(Ok(value)) => Ok(Some(value)),
                    Ok(Err(error)) => Err(failed_encoder_feature(error)),
                    Err(_) => Err(failed_encoder_feature(
                        "studio_encoder_reset_preflight_failed",
                    )),
                }
            }
        },
    };

    let skipped_studio = || SaveOrDiscardTargetDto {
        attempted: false,
        skipped: true,
        success: true,
        error: None,
    };
    let preflight = match preflight {
        Ok(value) => value,
        Err(feature) => {
            let config = config_save_or_discard_from_feature(feature);
            return Ok(ResetToKeymapDto {
                overall_success: false,
                studio: skipped_studio(),
                config,
                snapshot: None,
                refresh_error: None,
            });
        }
    };

    let edit = Arc::clone(&state.studio_edit);
    let edit_device_id = device_id.clone();
    let (studio, snapshot, refresh_error) = tauri::async_runtime::spawn_blocking(move || {
        let mut guard = edit.lock().unwrap();
        let session = match guard.as_mut() {
            Some(session) => session,
            None => {
                return (
                    SaveOrDiscardTargetDto {
                        attempted: true,
                        skipped: false,
                        success: false,
                        error: Some(studio_error_code(StudioError::NoEditSession)),
                    },
                    None,
                    None,
                );
            }
        };
        if session.device_id != edit_device_id {
            return (
                SaveOrDiscardTargetDto {
                    attempted: true,
                    skipped: false,
                    success: false,
                    error: Some(studio_error_code(StudioError::SessionDeviceMismatch)),
                },
                None,
                None,
            );
        }

        match session.reset_settings() {
            Err(error) => (
                SaveOrDiscardTargetDto {
                    attempted: true,
                    skipped: false,
                    success: false,
                    error: Some(studio_error_code(error)),
                },
                None,
                None,
            ),
            Ok(()) => match session.refresh_after_reset() {
                Ok(snapshot) => (
                    SaveOrDiscardTargetDto {
                        attempted: true,
                        skipped: false,
                        success: true,
                        error: None,
                    },
                    Some(snapshot),
                    None,
                ),
                Err(error) => (
                    // The device acknowledged reset_settings. Preserve that
                    // fact and ask the UI to re-read rather than claiming the
                    // destructive action itself failed.
                    SaveOrDiscardTargetDto {
                        attempted: true,
                        skipped: false,
                        success: true,
                        error: None,
                    },
                    None,
                    Some(studio_error_code(error)),
                ),
            },
        }
    })
    .await
    .map_err(|_| "studio_reset_to_keymap_failed".to_string())?;

    let encoder_config = if !studio.success || refresh_error.is_some() {
        config_save_or_discard_from_feature(ConfigFeatureResultDto {
            feature: "ENCODER".to_string(),
            attempted: false,
            skipped: true,
            success: true,
            error: None,
        })
    } else if let (Some((uid, encoder_info)), Some(snapshot)) = (preflight, snapshot.clone()) {
        match host_link_sender(state.inner()) {
            Ok(tx) => {
                let timeout = host_link_reply_timeout(&state.config.lock().unwrap());
                let feature = tauri::async_runtime::spawn_blocking(move || {
                    run_encoder_reset_to_keymap(tx, timeout, uid, encoder_info, &snapshot)
                })
                .await
                .unwrap_or_else(|_| failed_encoder_feature("studio_encoder_reset_failed"));
                config_save_or_discard_from_feature(feature)
            }
            Err(error) => config_save_or_discard_from_feature(failed_encoder_feature(error)),
        }
    } else {
        config_save_or_discard_from_feature(ConfigFeatureResultDto {
            feature: "ENCODER".to_string(),
            attempted: false,
            skipped: true,
            success: true,
            error: None,
        })
    };

    let combo_feature = if !studio.success || refresh_error.is_some() {
        skipped_config_feature("COMBO")
    } else if let Some(uid) = combo_host_link_uid {
        match host_link_sender(state.inner()) {
            Ok(tx) => {
                let timeout = host_link_reply_timeout(&state.config.lock().unwrap());
                tauri::async_runtime::spawn_blocking(move || {
                    run_combo_reset_to_keymap(tx, timeout, uid)
                })
                .await
                .unwrap_or_else(|_| failed_config_feature("COMBO", "studio_combo_reset_failed"))
            }
            Err(error) => failed_config_feature("COMBO", error),
        }
    } else {
        skipped_config_feature("COMBO")
    };
    let mut config_results = encoder_config.results;
    config_results.push(combo_feature);
    let config = config_save_or_discard_from_features(config_results);

    let overall_success = studio.success && refresh_error.is_none() && config.success;
    if overall_success {
        if let Some(uid) = host_link_uid.and_then(|uid| parse_host_link_uid(&uid).ok()) {
            state
                .encoder_restore_rollbacks
                .lock()
                .unwrap()
                .remove(&(device_id, uid));
        }
    }

    Ok(ResetToKeymapDto {
        overall_success,
        studio,
        config,
        snapshot,
        refresh_error,
    })
}

#[tauri::command]
pub async fn studio_has_unsaved(
    device_id: String,
    state: State<'_, AppState>,
) -> Result<bool, String> {
    let edit = Arc::clone(&state.studio_edit);
    tauri::async_runtime::spawn_blocking(move || {
        let mut guard = edit.lock().unwrap();
        let session = guard
            .as_mut()
            .ok_or_else(|| studio_error_code(StudioError::NoEditSession))?;
        if session.device_id != device_id {
            return Err(studio_error_code(StudioError::SessionDeviceMismatch));
        }
        session.has_unsaved().map_err(studio_error_code)
    })
    .await
    .map_err(|_| "studio_has_unsaved_failed".to_string())?
}

/// A consistent view of the active Studio edit session after a write whose
/// outcome may be unknown. This command only reads device state: it never
/// saves, discards, or clears pending edits.
#[derive(Debug, Clone, serde::Serialize)]
pub struct StudioEditResyncDto {
    pub snapshot: StudioKeymapSnapshot,
    pub has_unsaved: bool,
}

#[tauri::command]
pub async fn studio_resync_edit_state(
    device_id: String,
    state: State<'_, AppState>,
) -> Result<StudioEditResyncDto, String> {
    let edit = Arc::clone(&state.studio_edit);
    tauri::async_runtime::spawn_blocking(move || {
        let mut guard = edit.lock().unwrap();
        let session = guard
            .as_mut()
            .ok_or_else(|| studio_error_code(StudioError::NoEditSession))?;
        if session.device_id != device_id {
            return Err(studio_error_code(StudioError::SessionDeviceMismatch));
        }

        let snapshot = session.snapshot().map_err(studio_error_code)?;
        let has_unsaved = session.has_unsaved().map_err(studio_error_code)?;
        Ok(StudioEditResyncDto {
            snapshot,
            has_unsaved,
        })
    })
    .await
    .map_err(|_| "studio_resync_failed".to_string())?
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct EncoderBindingDto {
    pub behavior_id: u16,
    pub param1: u32,
    pub param2: u32,
    /// Only populated when `source == "override"`. A `source == "keymap"` encoder
    /// must never reveal its concrete `.keymap` binding (see keymap-encoder-editing-plan.md).
    pub label: Option<StudioBindingLabelPatch>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct EncoderBindingsDto {
    pub layer_id: u32,
    pub encoder_id: u8,
    pub source: String,
    pub stale_saved_exists: bool,
    pub saved_exists: bool,
    pub runtime_dirty: bool,
    pub invalid_saved_exists: bool,
    pub cw: EncoderBindingDto,
    pub ccw: EncoderBindingDto,
}

fn hid_error_code(error: HidError) -> String {
    match error {
        HidError::Hid(_) => "hid_error",
        HidError::Packet(_) => "packet_error",
        HidError::InvalidDevicePath => "invalid_device_path",
        HidError::ConfigRpcUnsupported => "config_rpc_unsupported",
        HidError::DeviceNotFound => "device_not_found",
        HidError::ConfigRpcTimeout => "config_rpc_timeout",
        HidError::ConfigRpcStatus(status) => match status {
            ConfigStatus::Ok => "config_rpc_status_ok",
            ConfigStatus::BadPacket => "config_rpc_status_bad_packet",
            ConfigStatus::UnsupportedFeature => "config_rpc_status_unsupported_feature",
            ConfigStatus::UnsupportedOp => "config_rpc_status_unsupported_op",
            ConfigStatus::InvalidArgument => "config_rpc_status_invalid_argument",
            ConfigStatus::Busy => "config_rpc_status_busy",
            ConfigStatus::NotFound => "config_rpc_status_not_found",
            ConfigStatus::StorageError => "config_rpc_status_storage_error",
            ConfigStatus::InternalError => "config_rpc_status_internal_error",
        },
        HidError::ConfigRpcMissingPayload => "config_rpc_missing_payload",
    }
    .to_string()
}

fn encoder_resolve_error_code(error: EncoderResolveError) -> String {
    match error {
        EncoderResolveError::Ineligible => "encoder_behavior_ineligible".to_string(),
        EncoderResolveError::UnsupportedByFirmware => {
            "encoder_behavior_unsupported_by_firmware".to_string()
        }
        EncoderResolveError::Studio(err) => studio_error_code(err),
    }
}

/// Parses the `device_uid_hash` string the UI already carries (see
/// `ui/src/lib/deviceCards.ts` `uidStringsMatch`), e.g. `"uid:7a91c3e4d102ab55"`.
fn parse_host_link_uid(value: &str) -> Result<u64, String> {
    let hex = value.strip_prefix("uid:").unwrap_or(value);
    u64::from_str_radix(hex, 16).map_err(|_| "invalid_host_link_uid".to_string())
}

fn host_link_reply_timeout(config: &AppConfig) -> Duration {
    let rpc_budget = config.hid.hello_timeout_ms.max(1) as u64 * 2 + 1_000;
    Duration::from_millis(rpc_budget.max(5_000))
}

fn host_link_call(
    tx: &mpsc::Sender<MonitorCommand>,
    timeout: Duration,
    uid: u64,
    request: HostLinkRequest,
) -> Result<HostLinkResponse, String> {
    let (reply, receiver) = mpsc::channel();
    tx.send(MonitorCommand::Config(HostLinkCall {
        uid,
        request,
        deadline: Instant::now() + timeout,
        reply,
    }))
    .map_err(|_| "hostlink_worker_unavailable".to_string())?;
    receiver
        .recv_timeout(timeout)
        .map_err(|_| "hostlink_result_unknown".to_string())?
}

fn host_link_sender(state: &AppState) -> Result<mpsc::Sender<MonitorCommand>, String> {
    state
        .monitor_tx
        .lock()
        .unwrap()
        .as_ref()
        .cloned()
        .ok_or_else(|| "hostlink_worker_unavailable".to_string())
}

fn encoder_binding_dto(
    binding: EncoderBinding,
    label: Option<StudioBindingLabelPatch>,
) -> EncoderBindingDto {
    EncoderBindingDto {
        behavior_id: binding.behavior_id,
        param1: binding.param1,
        param2: binding.param2,
        label,
    }
}

/// Labels CW/CCW bindings for passive display (viewing, not editing). Reuses an
/// already-open edit session for this device if one exists; otherwise opens a
/// short-lived Studio RPC connection just for the label lookup, exactly like
/// `read_studio_keymap` falls back to `read_keymap_for_device` when not editing.
/// Encoder display must work without entering edit mode.
fn label_encoder_bindings_for_device(
    device_id: &str,
    edit: &Arc<Mutex<Option<StudioEditSession>>>,
    studio_config: &rawhid_host_core::config::StudioConfig,
    source: EncoderBindingSource,
    cw: EncoderBinding,
    ccw: EncoderBinding,
    timeout: Duration,
) -> Result<[Option<StudioBindingLabelPatch>; 2], String> {
    if source == EncoderBindingSource::Keymap {
        return Ok([None, None]);
    }
    let raw = vec![
        StudioRawBinding {
            behavior_id: i32::from(cw.behavior_id),
            param1: cw.param1,
            param2: cw.param2,
        },
        StudioRawBinding {
            behavior_id: i32::from(ccw.behavior_id),
            param1: ccw.param1,
            param2: ccw.param2,
        },
    ];
    let patches = {
        let mut guard = edit.lock().unwrap();
        match guard.as_mut() {
            Some(session) if session.device_id == device_id => {
                session.resolve_behavior_labels(raw, timeout)
            }
            _ => {
                drop(guard);
                resolve_behavior_labels_for_device(device_id, raw, studio_config)
            }
        }
    }
    .map_err(studio_error_code)?;
    let mut patches = patches.into_iter();
    Ok([patches.next(), patches.next()])
}

fn label_encoder_bindings_batch_for_device(
    device_id: &str,
    edit: &Arc<Mutex<Option<StudioEditSession>>>,
    studio_config: &rawhid_host_core::config::StudioConfig,
    bindings: &[EncoderGetBindings],
    timeout: Duration,
) -> Result<Vec<[Option<StudioBindingLabelPatch>; 2]>, String> {
    let raw: Vec<StudioRawBinding> = bindings
        .iter()
        .filter(|binding| binding.source == EncoderBindingSource::Override)
        .flat_map(|binding| {
            [binding.cw_binding, binding.ccw_binding].map(|side| StudioRawBinding {
                behavior_id: i32::from(side.behavior_id),
                param1: side.param1,
                param2: side.param2,
            })
        })
        .collect();
    let patches = if raw.is_empty() {
        Vec::new()
    } else {
        let mut guard = edit.lock().unwrap();
        match guard.as_mut() {
            Some(session) if session.device_id == device_id => {
                session.resolve_behavior_labels(raw, timeout)
            }
            _ => {
                drop(guard);
                resolve_behavior_labels_for_device(device_id, raw, studio_config)
            }
        }
        .map_err(studio_error_code)?
    };
    let mut patches = patches.into_iter();
    Ok(bindings
        .iter()
        .map(|binding| {
            if binding.source == EncoderBindingSource::Keymap {
                [None, None]
            } else {
                [patches.next(), patches.next()]
            }
        })
        .collect())
}

fn encoder_bindings_dto_from(
    bindings: EncoderGetBindings,
    labels: [Option<StudioBindingLabelPatch>; 2],
) -> EncoderBindingsDto {
    let [cw_label, ccw_label] = labels;
    EncoderBindingsDto {
        layer_id: bindings.layer_id,
        encoder_id: bindings.encoder_id,
        source: match bindings.source {
            EncoderBindingSource::Keymap => "keymap".to_string(),
            EncoderBindingSource::Override => "override".to_string(),
        },
        stale_saved_exists: bindings.flags.bits() & EncoderBindingFlags::STALE_SAVED_EXISTS != 0,
        saved_exists: bindings.flags.bits() & EncoderBindingFlags::SAVED_EXISTS != 0,
        runtime_dirty: bindings.flags.bits() & EncoderBindingFlags::RUNTIME_DIRTY != 0,
        invalid_saved_exists: bindings.flags.bits() & EncoderBindingFlags::INVALID_SAVED_EXISTS
            != 0,
        cw: encoder_binding_dto(bindings.cw_binding, cw_label),
        ccw: encoder_binding_dto(bindings.ccw_binding, ccw_label),
    }
}

/// Labels CW/CCW bindings by routing them through the same `StudioRawBinding` label
/// pipeline used for normal keys, so encoder overrides render with identical text.
/// Returns `[None, None]` for `source == Keymap`, since that state must never reveal
/// the underlying `.keymap` binding to the UI.
fn label_encoder_bindings(
    session: &mut StudioEditSession,
    source: EncoderBindingSource,
    cw: EncoderBinding,
    ccw: EncoderBinding,
    timeout: Duration,
) -> Result<[Option<StudioBindingLabelPatch>; 2], String> {
    if source == EncoderBindingSource::Keymap {
        return Ok([None, None]);
    }
    let raw = vec![
        StudioRawBinding {
            behavior_id: i32::from(cw.behavior_id),
            param1: cw.param1,
            param2: cw.param2,
        },
        StudioRawBinding {
            behavior_id: i32::from(ccw.behavior_id),
            param1: ccw.param1,
            param2: ccw.param2,
        },
    ];
    let mut patches = session
        .resolve_behavior_labels(raw, timeout)
        .map_err(studio_error_code)?
        .into_iter();
    Ok([patches.next(), patches.next()])
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct EncoderInfoDto {
    pub layer_count: u8,
    pub encoder_count: u8,
    pub capabilities: u8,
    pub scroll_value: Option<u16>,
    pub encoder_tap_ms: Option<u16>,
}

#[tauri::command]
pub async fn read_encoder_info(
    host_link_uid: String,
    state: State<'_, AppState>,
) -> Result<EncoderInfoDto, String> {
    let target_uid = parse_host_link_uid(&host_link_uid)?;
    let tx = host_link_sender(state.inner())?;
    let timeout = host_link_reply_timeout(&state.config.lock().unwrap());
    tauri::async_runtime::spawn_blocking(move || {
        let info = match host_link_call(&tx, timeout, target_uid, HostLinkRequest::EncoderGetInfo)?
        {
            HostLinkResponse::EncoderInfo(info) => info,
            _ => return Err("hostlink_invalid_response".to_string()),
        };
        Ok(EncoderInfoDto {
            layer_count: info.layer_count,
            encoder_count: info.encoder_count,
            capabilities: info.capabilities,
            scroll_value: info.scroll_value,
            encoder_tap_ms: info.encoder_tap_ms,
        })
    })
    .await
    .map_err(|_| "read_encoder_info_failed".to_string())?
}

#[tauri::command]
pub async fn read_encoder_bindings(
    device_id: String,
    host_link_uid: String,
    layer_id: u32,
    encoder_id: u8,
    state: State<'_, AppState>,
) -> Result<EncoderBindingsDto, String> {
    let target_uid = parse_host_link_uid(&host_link_uid)?;
    let tx = host_link_sender(state.inner())?;
    let host_link_timeout = host_link_reply_timeout(&state.config.lock().unwrap());
    let studio_timeout = Duration::from_millis(
        state
            .config
            .lock()
            .unwrap()
            .studio
            .keymap_read_timeout_ms
            .max(1),
    );
    let studio_config = state.config.lock().unwrap().studio.clone();
    let edit = Arc::clone(&state.studio_edit);
    tauri::async_runtime::spawn_blocking(move || {
        let bindings = match host_link_call(
            &tx,
            host_link_timeout,
            target_uid,
            HostLinkRequest::EncoderGetBindings {
                layer_id,
                encoder_id,
            },
        )? {
            HostLinkResponse::EncoderBindings(bindings) => bindings,
            _ => return Err("hostlink_invalid_response".to_string()),
        };

        let [cw_label, ccw_label] = label_encoder_bindings_for_device(
            &device_id,
            &edit,
            &studio_config,
            bindings.source,
            bindings.cw_binding,
            bindings.ccw_binding,
            studio_timeout,
        )?;

        Ok(encoder_bindings_dto_from(bindings, [cw_label, ccw_label]))
    })
    .await
    .map_err(|_| "read_encoder_bindings_failed".to_string())?
}

#[tauri::command]
pub async fn read_encoder_layer_bindings(
    device_id: String,
    host_link_uid: String,
    layer_id: u32,
    encoder_count: u8,
    state: State<'_, AppState>,
) -> Result<Vec<EncoderBindingsDto>, String> {
    let target_uid = parse_host_link_uid(&host_link_uid)?;
    let tx = host_link_sender(state.inner())?;
    let host_link_timeout = host_link_reply_timeout(&state.config.lock().unwrap());
    let studio_config = state.config.lock().unwrap().studio.clone();
    let studio_timeout = Duration::from_millis(studio_config.keymap_read_timeout_ms.max(1));
    let edit = Arc::clone(&state.studio_edit);
    tauri::async_runtime::spawn_blocking(move || {
        let mut bindings = Vec::with_capacity(usize::from(encoder_count));
        for encoder_id in 0..encoder_count {
            let binding = match host_link_call(
                &tx,
                host_link_timeout,
                target_uid,
                HostLinkRequest::EncoderGetBindings {
                    layer_id,
                    encoder_id,
                },
            )? {
                HostLinkResponse::EncoderBindings(binding) => binding,
                _ => return Err("hostlink_invalid_response".to_string()),
            };
            bindings.push(binding);
        }
        let labels = label_encoder_bindings_batch_for_device(
            &device_id,
            &edit,
            &studio_config,
            &bindings,
            studio_timeout,
        )?;
        Ok(bindings
            .into_iter()
            .zip(labels)
            .map(|(binding, labels)| encoder_bindings_dto_from(binding, labels))
            .collect())
    })
    .await
    .map_err(|_| "read_encoder_bindings_failed".to_string())?
}

/// `cw` / `ccw` may each be omitted; an omitted side keeps the current runtime
/// override binding unchanged. Omitting a side is only valid while the target
/// encoder already has a runtime override (`source == override`): during the
/// first edit of a `source == keymap` encoder both sides must be sent together,
/// because the host must never fabricate a binding the user did not choose
/// (see keymap-encoder-editing-plan.md).
#[tauri::command]
pub async fn studio_set_encoder_bindings(
    device_id: String,
    host_link_uid: String,
    layer_id: u32,
    encoder_id: u8,
    cw: Option<EditBehaviorDto>,
    ccw: Option<EditBehaviorDto>,
    state: State<'_, AppState>,
) -> Result<EncoderBindingsDto, String> {
    let target_uid = parse_host_link_uid(&host_link_uid)?;
    let tx = host_link_sender(state.inner())?;
    let host_link_timeout = host_link_reply_timeout(&state.config.lock().unwrap());
    let studio_timeout = Duration::from_millis(
        state
            .config
            .lock()
            .unwrap()
            .studio
            .keymap_read_timeout_ms
            .max(1),
    );
    let edit = Arc::clone(&state.studio_edit);
    tauri::async_runtime::spawn_blocking(move || {
        let resolve_side = |dto: EditBehaviorDto| -> Result<EncoderBinding, String> {
            let mut guard = edit.lock().unwrap();
            let session = guard
                .as_mut()
                .ok_or_else(|| studio_error_code(StudioError::NoEditSession))?;
            if session.device_id != device_id {
                return Err(studio_error_code(StudioError::SessionDeviceMismatch));
            }
            session
                .resolve_encoder_binding(&EditBehavior::from(dto))
                .map_err(encoder_resolve_error_code)
        };
        let cw_resolved = cw.map(&resolve_side).transpose()?;
        let ccw_resolved = ccw.map(&resolve_side).transpose()?;

        let (cw_binding, ccw_binding) = match (cw_resolved, ccw_resolved) {
            (Some(cw_binding), Some(ccw_binding)) => (cw_binding, ccw_binding),
            (None, None) => return Err("encoder_bindings_incomplete".to_string()),
            (cw_resolved, ccw_resolved) => {
                let current = match host_link_call(
                    &tx,
                    host_link_timeout,
                    target_uid,
                    HostLinkRequest::EncoderGetBindings {
                        layer_id,
                        encoder_id,
                    },
                )? {
                    HostLinkResponse::EncoderBindings(bindings) => bindings,
                    _ => return Err("hostlink_invalid_response".to_string()),
                };
                if current.source != EncoderBindingSource::Override {
                    return Err("encoder_bindings_incomplete".to_string());
                }
                (
                    cw_resolved.unwrap_or(current.cw_binding),
                    ccw_resolved.unwrap_or(current.ccw_binding),
                )
            }
        };
        match host_link_call(
            &tx,
            host_link_timeout,
            target_uid,
            HostLinkRequest::EncoderSetBindings {
                layer_id,
                encoder_id,
                cw: cw_binding,
                ccw: ccw_binding,
            },
        )? {
            HostLinkResponse::Done => {}
            _ => return Err("hostlink_invalid_response".to_string()),
        }
        let bindings = match host_link_call(
            &tx,
            host_link_timeout,
            target_uid,
            HostLinkRequest::EncoderGetBindings {
                layer_id,
                encoder_id,
            },
        )? {
            HostLinkResponse::EncoderBindings(bindings) => bindings,
            _ => return Err("hostlink_invalid_response".to_string()),
        };

        let mut guard = edit.lock().unwrap();
        let session = guard
            .as_mut()
            .ok_or_else(|| studio_error_code(StudioError::NoEditSession))?;
        let [cw_label, ccw_label] = label_encoder_bindings(
            session,
            bindings.source,
            bindings.cw_binding,
            bindings.ccw_binding,
            studio_timeout,
        )?;

        Ok(EncoderBindingsDto {
            layer_id: bindings.layer_id,
            encoder_id: bindings.encoder_id,
            source: match bindings.source {
                EncoderBindingSource::Keymap => "keymap".to_string(),
                EncoderBindingSource::Override => "override".to_string(),
            },
            stale_saved_exists: bindings.flags.bits() & EncoderBindingFlags::STALE_SAVED_EXISTS
                != 0,
            saved_exists: bindings.flags.bits() & EncoderBindingFlags::SAVED_EXISTS != 0,
            runtime_dirty: bindings.flags.bits() & EncoderBindingFlags::RUNTIME_DIRTY != 0,
            invalid_saved_exists: bindings.flags.bits() & EncoderBindingFlags::INVALID_SAVED_EXISTS
                != 0,
            cw: encoder_binding_dto(bindings.cw_binding, cw_label),
            ccw: encoder_binding_dto(bindings.ccw_binding, ccw_label),
        })
    })
    .await
    .map_err(|_| "studio_set_encoder_bindings_failed".to_string())?
}

#[tauri::command]
pub async fn studio_encoder_has_unsaved(
    host_link_uid: String,
    state: State<'_, AppState>,
) -> Result<bool, String> {
    let target_uid = parse_host_link_uid(&host_link_uid)?;
    let tx = host_link_sender(state.inner())?;
    let timeout = host_link_reply_timeout(&state.config.lock().unwrap());
    tauri::async_runtime::spawn_blocking(move || {
        match host_link_call(&tx, timeout, target_uid, HostLinkRequest::EncoderGetDirty)? {
            HostLinkResponse::Dirty(dirty) => Ok(dirty),
            _ => Err("hostlink_invalid_response".to_string()),
        }
    })
    .await
    .map_err(|_| "studio_encoder_has_unsaved_failed".to_string())?
}

#[tauri::command]
pub async fn studio_encoder_save(
    host_link_uid: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    let target_uid = parse_host_link_uid(&host_link_uid)?;
    let tx = host_link_sender(state.inner())?;
    let timeout = host_link_reply_timeout(&state.config.lock().unwrap());
    tauri::async_runtime::spawn_blocking(move || {
        match host_link_call(&tx, timeout, target_uid, HostLinkRequest::EncoderSave)? {
            HostLinkResponse::Done => Ok(()),
            _ => Err("hostlink_invalid_response".to_string()),
        }
    })
    .await
    .map_err(|_| "studio_encoder_save_failed".to_string())?
}

#[tauri::command]
pub async fn studio_encoder_discard(
    host_link_uid: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    let target_uid = parse_host_link_uid(&host_link_uid)?;
    let tx = host_link_sender(state.inner())?;
    let timeout = host_link_reply_timeout(&state.config.lock().unwrap());
    tauri::async_runtime::spawn_blocking(move || {
        match host_link_call(&tx, timeout, target_uid, HostLinkRequest::EncoderDiscard)? {
            HostLinkResponse::Done => Ok(()),
            _ => Err("hostlink_invalid_response".to_string()),
        }
    })
    .await
    .map_err(|_| "studio_encoder_discard_failed".to_string())?
}

#[tauri::command]
pub async fn studio_encoder_clear_override(
    host_link_uid: String,
    layer_id: u32,
    encoder_id: u8,
    state: State<'_, AppState>,
) -> Result<(), String> {
    let target_uid = parse_host_link_uid(&host_link_uid)?;
    let tx = host_link_sender(state.inner())?;
    let timeout = host_link_reply_timeout(&state.config.lock().unwrap());
    tauri::async_runtime::spawn_blocking(move || {
        match host_link_call(
            &tx,
            timeout,
            target_uid,
            HostLinkRequest::EncoderClearOverride {
                layer_id,
                encoder_id,
            },
        )? {
            HostLinkResponse::Done => Ok(()),
            _ => Err("hostlink_invalid_response".to_string()),
        }
    })
    .await
    .map_err(|_| "studio_encoder_clear_override_failed".to_string())?
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ComboInfoDto {
    pub max_combos: u8,
    pub max_keys_per_combo: u8,
    pub combo_count: u8,
    pub flags: u8,
    pub occupied_slots: u32,
    pub stale_slots: u32,
    pub invalid_slots: u32,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ComboBindingDto {
    pub behavior_id: u16,
    pub param1: u32,
    pub param2: u32,
    #[serde(default)]
    pub label: Option<StudioBindingLabelPatch>,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ComboItemDto {
    pub slot: u8,
    pub name: String,
    pub key_positions: Vec<u16>,
    pub slow_release: bool,
    pub binding: ComboBindingDto,
    pub layer_mask: u32,
    pub timeout_ms: u16,
    pub require_prior_idle_ms: Option<u16>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ComboItemInputDto {
    pub slot: u8,
    pub name: String,
    pub key_positions: Vec<u16>,
    pub slow_release: bool,
    /// Kept when an existing Combo is edited without changing its behavior.
    pub binding: ComboBindingDto,
    /// A newly selected behavior is resolved through the connected firmware's
    /// Studio catalog. This is deliberately separate from the wire DTO because
    /// behavior ids are build-specific.
    pub behavior: Option<EditBehaviorDto>,
    pub layer_mask: u32,
    pub timeout_ms: u16,
    pub require_prior_idle_ms: Option<u16>,
}

fn combo_info_dto(info: ComboInfo) -> ComboInfoDto {
    ComboInfoDto {
        max_combos: info.max_combos,
        max_keys_per_combo: info.max_keys_per_combo,
        combo_count: info.combo_count,
        flags: info.flags.bits(),
        occupied_slots: info.occupied_slots,
        stale_slots: info.stale_slots,
        invalid_slots: info.invalid_slots,
    }
}

fn combo_item_dto(item: ComboItem, label: Option<StudioBindingLabelPatch>) -> ComboItemDto {
    ComboItemDto {
        slot: item.slot,
        name: item.name.as_str().to_string(),
        key_positions: item.key_positions[..usize::from(item.key_count)].to_vec(),
        slow_release: item.flags.slow_release(),
        binding: ComboBindingDto {
            behavior_id: item.binding.behavior_id,
            param1: item.binding.param1,
            param2: item.binding.param2,
            label,
        },
        layer_mask: item.layer_mask,
        timeout_ms: item.timeout_ms,
        require_prior_idle_ms: item.require_prior_idle_ms,
    }
}

/// Resolve the build-specific behavior id for passive Combo display.  A
/// Studio lookup failure must not prevent Config RPC reads, so callers retain
/// the raw binding and use a UI fallback when this returns `None`.
fn label_combo_binding_for_device(
    device_id: &str,
    edit: &Arc<Mutex<Option<StudioEditSession>>>,
    studio_config: &rawhid_host_core::config::StudioConfig,
    binding: ComboBinding,
    timeout: Duration,
) -> Option<StudioBindingLabelPatch> {
    let raw = vec![StudioRawBinding {
        behavior_id: i32::from(binding.behavior_id),
        param1: binding.param1,
        param2: binding.param2,
    }];
    let labels = {
        let mut guard = edit.lock().unwrap();
        match guard.as_mut() {
            Some(session) if session.device_id == device_id => {
                session.resolve_behavior_labels(raw, timeout)
            }
            _ => {
                drop(guard);
                resolve_behavior_labels_for_device(device_id, raw, studio_config)
            }
        }
    };
    labels.ok().and_then(|mut labels| labels.pop())
}

fn combo_item_from_dto(
    item: ComboItemInputDto,
    binding: ComboBinding,
) -> Result<ComboItem, String> {
    ComboItem::new(
        item.slot,
        &item.name,
        &item.key_positions,
        item.slow_release,
        binding,
        item.layer_mask,
        item.timeout_ms,
        item.require_prior_idle_ms,
    )
    .map_err(|error| format!("combo_invalid_argument:{error}"))
}

#[tauri::command]
pub async fn read_combo_info(
    host_link_uid: String,
    state: State<'_, AppState>,
) -> Result<ComboInfoDto, String> {
    let uid = parse_host_link_uid(&host_link_uid)?;
    let tx = host_link_sender(state.inner())?;
    let timeout = host_link_reply_timeout(&state.config.lock().unwrap());
    tauri::async_runtime::spawn_blocking(move || {
        match host_link_call(&tx, timeout, uid, HostLinkRequest::ComboGetInfo)? {
            HostLinkResponse::ComboInfo(info) => Ok(combo_info_dto(info)),
            _ => Err("hostlink_invalid_response".to_string()),
        }
    })
    .await
    .map_err(|_| "read_combo_info_failed".to_string())?
}

#[tauri::command]
pub async fn read_combo(
    device_id: String,
    host_link_uid: String,
    slot: u8,
    state: State<'_, AppState>,
) -> Result<ComboItemDto, String> {
    let uid = parse_host_link_uid(&host_link_uid)?;
    let tx = host_link_sender(state.inner())?;
    let timeout = host_link_reply_timeout(&state.config.lock().unwrap());
    let studio_config = state.config.lock().unwrap().studio.clone();
    let edit = Arc::clone(&state.studio_edit);
    tauri::async_runtime::spawn_blocking(move || {
        match host_link_call(&tx, timeout, uid, HostLinkRequest::ComboGet { slot })? {
            HostLinkResponse::ComboItem(item) => {
                let label = label_combo_binding_for_device(
                    &device_id,
                    &edit,
                    &studio_config,
                    item.binding,
                    timeout,
                );
                Ok(combo_item_dto(item, label))
            }
            _ => Err("hostlink_invalid_response".to_string()),
        }
    })
    .await
    .map_err(|_| "read_combo_failed".to_string())?
}

#[tauri::command]
pub async fn studio_set_combo(
    device_id: String,
    host_link_uid: String,
    item: ComboItemInputDto,
    state: State<'_, AppState>,
) -> Result<ComboItemDto, String> {
    let uid = parse_host_link_uid(&host_link_uid)?;
    let tx = host_link_sender(state.inner())?;
    let timeout = host_link_reply_timeout(&state.config.lock().unwrap());
    let studio_config = state.config.lock().unwrap().studio.clone();
    let edit = Arc::clone(&state.studio_edit);
    tauri::async_runtime::spawn_blocking(move || {
        let binding = match item.behavior.clone() {
            Some(behavior) => {
                let mut guard = edit.lock().unwrap();
                let session = guard
                    .as_mut()
                    .ok_or_else(|| studio_error_code(StudioError::NoEditSession))?;
                if session.device_id != device_id {
                    return Err(studio_error_code(StudioError::SessionDeviceMismatch));
                }
                session
                    .resolve_combo_binding(&EditBehavior::from(behavior))
                    .map_err(encoder_resolve_error_code)?
            }
            None => ComboBinding {
                behavior_id: item.binding.behavior_id,
                param1: item.binding.param1,
                param2: item.binding.param2,
            },
        };
        let item = combo_item_from_dto(item, binding)?;
        match host_link_call(&tx, timeout, uid, HostLinkRequest::ComboSet { item })? {
            HostLinkResponse::Done => {
                let label = label_combo_binding_for_device(
                    &device_id,
                    &edit,
                    &studio_config,
                    item.binding,
                    timeout,
                );
                Ok(combo_item_dto(item, label))
            }
            _ => Err("hostlink_invalid_response".to_string()),
        }
    })
    .await
    .map_err(|_| "studio_set_combo_failed".to_string())?
}

#[tauri::command]
pub async fn studio_combo_has_unsaved(
    host_link_uid: String,
    state: State<'_, AppState>,
) -> Result<bool, String> {
    let uid = parse_host_link_uid(&host_link_uid)?;
    let tx = host_link_sender(state.inner())?;
    let timeout = host_link_reply_timeout(&state.config.lock().unwrap());
    tauri::async_runtime::spawn_blocking(move || {
        match host_link_call(&tx, timeout, uid, HostLinkRequest::ComboGetDirty)? {
            HostLinkResponse::Dirty(dirty) => Ok(dirty),
            _ => Err("hostlink_invalid_response".to_string()),
        }
    })
    .await
    .map_err(|_| "studio_combo_has_unsaved_failed".to_string())?
}

async fn combo_simple_operation(
    host_link_uid: String,
    state: &AppState,
    request: HostLinkRequest,
    error: &'static str,
) -> Result<(), String> {
    let uid = parse_host_link_uid(&host_link_uid)?;
    let tx = host_link_sender(state)?;
    let timeout = host_link_reply_timeout(&state.config.lock().unwrap());
    tauri::async_runtime::spawn_blocking(move || {
        match host_link_call(&tx, timeout, uid, request)? {
            HostLinkResponse::Done => Ok(()),
            _ => Err("hostlink_invalid_response".to_string()),
        }
    })
    .await
    .map_err(|_| error.to_string())?
}

#[tauri::command]
pub async fn studio_combo_save(
    host_link_uid: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    combo_simple_operation(
        host_link_uid,
        state.inner(),
        HostLinkRequest::ComboSave,
        "studio_combo_save_failed",
    )
    .await
}

#[tauri::command]
pub async fn studio_combo_discard(
    host_link_uid: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    combo_simple_operation(
        host_link_uid,
        state.inner(),
        HostLinkRequest::ComboDiscard,
        "studio_combo_discard_failed",
    )
    .await
}

#[tauri::command]
pub async fn studio_combo_delete(
    host_link_uid: String,
    slot: u8,
    state: State<'_, AppState>,
) -> Result<(), String> {
    combo_simple_operation(
        host_link_uid,
        state.inner(),
        HostLinkRequest::ComboDelete { slot },
        "studio_combo_delete_failed",
    )
    .await
}

#[tauri::command]
pub async fn studio_combo_reset_to_keymap(
    host_link_uid: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    combo_simple_operation(
        host_link_uid,
        state.inner(),
        HostLinkRequest::ComboResetToKeymap,
        "studio_combo_reset_failed",
    )
    .await
}

#[tauri::command]
pub async fn studio_end_edit(device_id: String, state: State<'_, AppState>) -> Result<(), String> {
    let edit = Arc::clone(&state.studio_edit);
    tauri::async_runtime::spawn_blocking(move || {
        let mut guard = edit.lock().unwrap();
        let Some(session) = guard.as_mut() else {
            return Ok(());
        };
        if session.device_id != device_id {
            tracing::debug!(
                requested_device_id = %device_id,
                active_device_id = %session.device_id,
                "ignored stale studio edit cleanup"
            );
            return Ok(());
        }
        match session.has_unsaved() {
            Ok(true) => return Err(studio_error_code(StudioError::UnsavedChangesExist)),
            Ok(false) => {}
            Err(error) if studio_session_connection_lost(&error) => {
                tracing::info!(
                    device_id = %session.device_id,
                    "released disconnected Studio edit session while ending edit"
                );
            }
            Err(error) => return Err(studio_error_code(error)),
        }
        *guard = None;
        Ok(())
    })
    .await
    .map_err(|_| "studio_end_edit_failed".to_string())?
}

#[tauri::command]
pub async fn studio_abort_edit(
    device_id: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    let edit = Arc::clone(&state.studio_edit);
    tauri::async_runtime::spawn_blocking(move || {
        let mut guard = edit.lock().unwrap();
        let Some(session) = guard.as_ref() else {
            return Ok(());
        };
        if session.device_id != device_id {
            tracing::debug!(
                requested_device_id = %device_id,
                active_device_id = %session.device_id,
                "ignored stale studio edit abort"
            );
            return Ok(());
        }
        *guard = None;
        Ok(())
    })
    .await
    .map_err(|_| "studio_abort_edit_failed".to_string())?
}

fn studio_error_code(error: StudioError) -> String {
    match error {
        StudioError::DeviceNotFound => "device_not_found",
        StudioError::Locked => "locked",
        StudioError::Timeout => "timeout",
        StudioError::Disconnected => "disconnected",
        StudioError::InvalidLocation => "invalid_location",
        StudioError::InvalidBehavior => "invalid_behavior",
        StudioError::InvalidParameters => "invalid_parameters",
        StudioError::MissingBehaviorRole => "missing_behavior_role",
        StudioError::SaveFailed => "save_failed",
        StudioError::SaveNotSupported => "save_not_supported",
        StudioError::SaveNoSpace => "save_no_space",
        StudioError::SaveResultUnknown => "save_result_unknown",
        StudioError::ResetSettingsRejected => "reset_settings_rejected",
        StudioError::NoEditSession => "no_edit_session",
        StudioError::EditSessionExists => "edit_session_exists",
        StudioError::UnsavedChangesExist => "unsaved_changes_exist",
        StudioError::SessionDeviceMismatch => "session_device_mismatch",
        StudioError::PortBusy => "port_busy",
        StudioError::EditingUnsupportedForBle => "editing_unsupported_for_ble",
        StudioError::AddLayerFailed => "add_layer_failed",
        StudioError::AddLayerNoSpace => "add_layer_no_space",
        StudioError::RemoveLayerFailed => "remove_layer_failed",
        StudioError::InvalidLayer => "invalid_layer",
        StudioError::RenameLayerFailed => "rename_layer_failed",
        StudioError::RpcFailed => "rpc_failed",
    }
    .to_string()
}
#[tauri::command]
pub fn start_monitoring(app: AppHandle, state: State<AppState>) -> Result<(), String> {
    begin_monitoring(app, state.inner())
}

/// Shared monitoring-start logic, callable from the command, the tray menu, and
/// the auto-start-on-launch path.
pub fn begin_monitoring(app: AppHandle, state: &AppState) -> Result<(), String> {
    let _ = app;
    if state.status.lock().unwrap().running {
        return Err("Already monitoring".to_string());
    }
    set_automation_enabled(state, true)
}

/// Starts the single Host Link owner for the lifetime of the application.
pub fn start_host_link_worker(
    app: AppHandle,
    state: &AppState,
    automation_enabled: bool,
) -> Result<(), String> {
    if state.monitor_tx.lock().unwrap().is_some() {
        return Ok(());
    }
    let config = state.config.lock().unwrap().clone();
    let status = Arc::clone(&state.status);
    let log_entries = Arc::clone(&state.log_entries);
    let log_counter = Arc::clone(&state.log_counter);
    let monitor_tx_arc = Arc::clone(&state.monitor_tx);
    let ai_usage_shared = state
        .ai_usage_runtime
        .lock()
        .unwrap()
        .as_ref()
        .map(AiUsageRuntime::shared);
    let extras = MonitorExtras {
        config: Arc::clone(&state.config),
        ai_usage_runtime: Arc::clone(&state.ai_usage_runtime),
        ai_usage_refreshing: Arc::clone(&state.ai_usage_refreshing),
        ai_terminal_focusing: Arc::clone(&state.ai_terminal_focusing),
        key_stats: Arc::clone(&state.key_stats),
        codex_activity: Arc::clone(&state.codex_activity),
        codex_broker: state.codex_broker.clone(),
        claude_integration: Arc::clone(&state.claude_integration),
        claude_permission_gate: Arc::clone(&state.claude_permission_gate),
        ai_display_slots: Arc::clone(&state.ai_display_slots),
        hud: Arc::clone(&state.hud),
        hud_responded: Arc::new(Mutex::new(None)),
    };

    let (tx, rx) = mpsc::channel();
    let watcher_tx = tx.clone();
    *monitor_tx_arc.lock().unwrap() = Some(tx);

    thread::spawn(move || {
        // The foreground watcher delivers instant wake-ups; polling stays as a
        // fallback. It is unhooked when this scope ends (monitoring stops).
        let _foreground_watcher = ForegroundWatcher::new(watcher_tx);
        run_monitor_loop(
            app,
            config,
            status,
            log_entries,
            log_counter,
            rx,
            ai_usage_shared,
            extras,
            automation_enabled,
        );
        *monitor_tx_arc.lock().unwrap() = None;
    });

    Ok(())
}

#[tauri::command]
pub fn stop_monitoring(state: State<AppState>) -> Result<(), String> {
    stop_monitoring_internal(state.inner())
}

/// Shared monitoring-stop logic, callable from the command and the tray menu.
pub fn stop_monitoring_internal(state: &AppState) -> Result<(), String> {
    if !state.status.lock().unwrap().running {
        return Err("Not monitoring".to_string());
    }
    set_automation_enabled(state, false)
}

fn set_automation_enabled(state: &AppState, enabled: bool) -> Result<(), String> {
    let tx = state
        .monitor_tx
        .lock()
        .unwrap()
        .as_ref()
        .cloned()
        .ok_or_else(|| "hostlink_worker_unavailable".to_string())?;
    let (reply_tx, reply_rx) = mpsc::channel();
    tx.send(MonitorCommand::SetAutomationEnabled(enabled, reply_tx))
        .map_err(|_| "hostlink_worker_unavailable".to_string())?;
    let timeout = host_link_reply_timeout(&state.config.lock().unwrap());
    reply_rx
        .recv_timeout(timeout)
        .map_err(|_| "hostlink_result_unknown".to_string())?
}

pub fn shutdown_host_link_worker(state: &AppState) {
    if let Some(tx) = state.monitor_tx.lock().unwrap().take() {
        let _ = tx.send(MonitorCommand::Shutdown);
    }
}

#[tauri::command]
pub fn refresh_ai_usage(app: AppHandle, state: State<AppState>) -> Result<(), String> {
    if state
        .ai_usage_refreshing
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return Err("refresh_in_progress".to_string());
    }

    // Capture the snapshot generation before requesting a refresh so the watcher
    // thread can detect when the worker has produced fresh data.
    let baseline = {
        let runtime = state.ai_usage_runtime.lock().unwrap();
        let Some(runtime) = runtime.as_ref() else {
            state.ai_usage_refreshing.store(false, Ordering::SeqCst);
            return Err("source_disabled".to_string());
        };
        let generation = runtime.shared().generation();
        if let Err(error) = runtime.refresh() {
            state.ai_usage_refreshing.store(false, Ordering::SeqCst);
            return Err(refresh_error_code(error));
        }
        generation
    };

    // Wait for completion off the command thread, then push a status-update so
    // the UI reflects new data even while monitoring is stopped (no tick loop).
    spawn_ai_refresh_watcher(
        app,
        Arc::clone(&state.config),
        Arc::clone(&state.ai_usage_runtime),
        Arc::clone(&state.status),
        Arc::clone(&state.ai_usage_refreshing),
        baseline,
    );
    Ok(())
}

pub(crate) fn spawn_ai_refresh_watcher(
    app: AppHandle,
    config: Arc<std::sync::Mutex<AppConfig>>,
    runtime: Arc<std::sync::Mutex<Option<AiUsageRuntime>>>,
    status: Arc<std::sync::Mutex<MonitorStatus>>,
    refreshing: Arc<AtomicBool>,
    baseline: u64,
) {
    thread::spawn(move || {
        let _guard = RefreshGuard(refreshing);
        // Poll for the worker to publish a new snapshot generation (best effort).
        for _ in 0..100 {
            let changed = runtime
                .lock()
                .unwrap()
                .as_ref()
                .map(|runtime| runtime.shared().generation() != baseline)
                .unwrap_or(true);
            if changed {
                break;
            }
            thread::sleep(Duration::from_millis(100));
        }

        let stale_after_sec = config.lock().unwrap().ai_usage.stale_after_sec;
        let statuses = runtime
            .lock()
            .unwrap()
            .as_ref()
            .map(|runtime| runtime.statuses(stale_after_sec))
            .unwrap_or_default();
        update_status(&app, &status, |s| s.ai_usage = statuses);
    });
}

fn restart_ai_usage_runtime(state: &AppState, config: &AppConfig) -> Option<AiUsageShared> {
    let runtime = AiUsageRuntime::start(config.ai_usage.clone());
    let shared = runtime.as_ref().map(AiUsageRuntime::shared);
    *state.ai_usage_runtime.lock().unwrap() = runtime;
    shared
}

fn update_ai_usage_status(state: &AppState) {
    let config = state.config.lock().unwrap().clone();
    let ai_usage = state
        .ai_usage_runtime
        .lock()
        .unwrap()
        .as_ref()
        .map(|runtime| runtime.statuses(config.ai_usage.stale_after_sec))
        .unwrap_or_default();
    state.status.lock().unwrap().ai_usage = ai_usage;
}

struct RefreshGuard(Arc<AtomicBool>);

impl Drop for RefreshGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

fn rebuild_runner(
    config: &AppConfig,
    ai_usage_shared: Option<AiUsageShared>,
) -> Result<MonitorRunner, String> {
    let hid = HidDeviceManager::real(config.hid.clone()).map_err(|e| e.to_string())?;
    Ok(Runner::new_with_ai_usage_shared(
        config.clone(),
        SystemActiveAppProvider,
        hid,
        rawhid_host_core::time::SystemClock,
        ai_usage_shared,
    ))
}

/// Mutate the shared monitor status and publish the result.
///
/// The status lock must be released before publishing. `TrayIcon::set_tooltip`
/// blocks until the main thread runs it, and the main thread takes this same
/// lock from IPC commands such as `get_status` and `get_ai_display_slots`, so
/// publishing while holding the guard deadlocks the two against each other.
/// Locking is kept inside this helper so no caller can reintroduce that.
fn update_status<F>(app: &AppHandle, status: &Arc<std::sync::Mutex<MonitorStatus>>, mutate: F)
where
    F: FnOnce(&mut MonitorStatus),
{
    update_status_if(app, status, |s| {
        mutate(s);
        true
    });
}

/// [`update_status`] that publishes only when `mutate` reports a real change.
fn update_status_if<F>(app: &AppHandle, status: &Arc<std::sync::Mutex<MonitorStatus>>, mutate: F)
where
    F: FnOnce(&mut MonitorStatus) -> bool,
{
    let snapshot = {
        let mut s = status.lock().unwrap();
        mutate(&mut s).then(|| s.clone())
    };
    if let Some(snapshot) = snapshot {
        emit_status(app, &snapshot);
    }
}

/// Emit the status snapshot to the UI and refresh the tray tooltip together so
/// the tray always reflects the latest device/battery state.
///
/// Never call this while holding the status lock; go through [`update_status`].
fn emit_status(app: &AppHandle, status: &MonitorStatus) {
    let _ = app.emit("status-update", status);
    update_tray_tooltip(app, status);
}

fn update_tray_tooltip(app: &AppHandle, status: &MonitorStatus) {
    if let Some(tray) = app.tray_by_id("main") {
        let _ = tray.set_tooltip(Some(build_tray_tooltip(status)));
    }
}

/// Build the tray tooltip: "Keylink Studio" plus one line per battery-reporting
/// device. Capped to ~128 chars (Windows tray tooltip limit).
fn build_tray_tooltip(status: &MonitorStatus) -> String {
    let mut text = String::from("Keylink Studio");
    for dev in &status.device_battery {
        let name = dev
            .product
            .as_deref()
            .or(dev.serial_number.as_deref())
            .unwrap_or(&dev.device_key);
        let sources: Vec<String> = dev
            .sources
            .iter()
            .filter_map(|src| {
                let level = src.level?;
                let label = if src.source == 0 {
                    "C".to_string()
                } else {
                    format!("P{}", src.source)
                };
                Some(format!("{}:{}%", label, level))
            })
            .collect();
        text.push('\n');
        let summary = if sources.is_empty() {
            "?".to_string()
        } else {
            sources.join(" ")
        };
        text.push_str(&format!("{}: {}", name, summary));
    }
    if text.chars().count() > 127 {
        text = text.chars().take(126).collect::<String>() + "…";
    }
    text
}

/// Copy the runner's device/AI-usage view into the shared status.
fn apply_runner_view(s: &mut MonitorStatus, runner: &MonitorRunner) {
    s.connected_devices = runner.verified_device_count();
    s.connected_device_names = runner.verified_device_names();
    s.host_link_devices = runner.verified_devices();
    s.ai_usage = runner.ai_usage_statuses();
    s.device_battery = runner.battery_statuses();
    s.device_layers = runner.layer_states();
}

fn drain_codex_state_changes(activity: &CodexActivityRuntime) -> Vec<CodexStateChange> {
    let mut changes = Vec::new();
    while let Some(change) = activity.try_recv_session_change() {
        changes.push(change);
    }
    changes
}

fn claude_as_ai_snapshot(snapshot: Option<ClaudeSessionSnapshot>) -> AiClientStateSnapshot {
    match snapshot {
        Some(snapshot) => AiClientStateSnapshot {
            client_type: AiClientType::ClaudeCode,
            client_variant: AiClientVariant::Cli,
            session_active: snapshot.session_active,
            activity_state: snapshot.activity_state,
            work_phase: snapshot.work_phase,
            revision: snapshot.revision,
        },
        None => AiClientStateSnapshot {
            client_type: AiClientType::ClaudeCode,
            client_variant: AiClientVariant::Cli,
            session_active: false,
            activity_state: AiActivityState::None,
            work_phase: AiWorkPhase::Unspecified,
            revision: 0,
        },
    }
}

fn drain_claude_state_changes(
    integration: &Arc<Mutex<Option<ClaudeIntegration>>>,
    pending_approvals: &PendingApprovalStore,
    permission_gate: &ClaudePermissionGate,
    now: Instant,
) -> (
    Vec<ClaudeSessionSnapshot>,
    Vec<ClaudeStateChange>,
    BTreeMap<String, String>,
) {
    let mut guard = integration.lock().unwrap();
    let Some(integration) = guard.as_mut() else {
        return (Vec::new(), Vec::new(), BTreeMap::new());
    };
    let mut claude_changes: Vec<ClaudeStateChange> = Vec::new();
    // Wiring only: this reads each event before handing it to the registry
    // below, exactly like the existing loop already does with counters and
    // `mark_launch_desynchronized`. It does not alter what the registry
    // receives or how it reacts.
    let approval_consumer = ClaudeApprovalBodyConsumer;
    for launch in integration.launches.values_mut() {
        while let Ok(event) = launch.events.try_recv() {
            // Diagnostic only: this exists to tell apart two failure modes
            // behind "terminal answered deny but the HUD never went away" --
            // (a) the hook never arrived at all, versus (b) it arrived but
            // didn't meet `ClaudeApprovalBodyConsumer::ingest`'s withdrawal
            // condition (e.g. no `session_id`, which it silently drops). The
            // before/after pending count says whether this event actually
            // withdrew anything. Same restraint as `claude approval decision
            // dispatched` above: only the protocol event name, a presence
            // boolean, and counts -- never the hook body, tool_use_id,
            // session_id/launch_id, or approval content.
            let name = match &event {
                ClaudeObserverEvent::Hook(hook) => hook.hook_event_name.as_str(),
                ClaudeObserverEvent::WrapperExited(_) => "WrapperExited",
            };
            let has_session_id = matches!(
                &event,
                ClaudeObserverEvent::Hook(hook) if hook.session_id.is_some()
            );
            let pending_before = pending_approvals.len();
            approval_consumer.ingest(pending_approvals, permission_gate, &event);
            let pending_after = pending_approvals.len();
            tracing::debug!(
                target: AI_DISPLAY_LOG_TARGET,
                name,
                has_session_id,
                pending_before,
                pending_after,
                "claude hook observed"
            );
            if let Ok(mut changes) = integration.registry.apply(event, now) {
                claude_changes.append(&mut changes);
            }
        }
        let counters = launch.receiver.counters();
        if counters.normal_overflow > launch.last_counters.normal_overflow
            || counters.priority_overflow > launch.last_counters.priority_overflow
        {
            let launch_id = launch.receiver.config().launch_id.clone();
            claude_changes.extend(integration.registry.mark_launch_desynchronized(&launch_id));
        }
        launch.last_counters = counters;
    }
    claude_changes.extend(integration.registry.tick(now));
    // A `PermissionRequest` hook connection stops being answerable in one of
    // two ways, both reported here by
    // `ClaudePermissionGate::drain_unanswerable` (see `UnansweredReason`):
    // Claude Code closed the connection because the request was settled from
    // the terminal, or the Host's own decision wait
    // (`claude_decision::CLAUDE_PERMISSION_DECISION_TIMEOUT`) elapsed with
    // nobody answering anywhere. Left alone, either one leaves the HUD
    // showing a request as something it can still answer -- pressing a key
    // just does nothing, because the hook connection it would have answered
    // is already gone. This withdraws the HUD's offer to answer it.
    //
    // It also withdraws the session's own approval request via
    // `withdraw_approval_requests`, which stops the ScreenKey's yellow
    // blink. What the session falls back to depends on the reason, so that
    // method takes it: a closed connection ends the turn (the request was
    // answered on the terminal and Claude Code is idle again), while a
    // timeout only drops the request and recomputes, since Studio genuinely
    // does not know the outcome there. See that method's own doc comment.
    //
    // Both paths are otherwise the same cleanup, so they run through one
    // loop and differ in the diagnostic `reason=`.
    for unanswered in permission_gate.drain_unanswerable() {
        let reason = unanswered.reason.diagnostic_label();
        let key = ApprovalKey::new(unanswered.token);
        let pending_before = pending_approvals.len();
        pending_approvals.resolve(&key);
        let pending_after = pending_approvals.len();
        let session_changes = integration.registry.withdraw_approval_requests(
            &unanswered.launch_id,
            &unanswered.session_id,
            unanswered.reason,
            now,
        );
        let session_withdrawn = !session_changes.is_empty();
        let session_changes_count = session_changes.len();
        claude_changes.extend(session_changes);
        tracing::debug!(
            target: AI_DISPLAY_LOG_TARGET,
            reason,
            removed = pending_after < pending_before,
            pending_after,
            session_withdrawn,
            session_changes_count,
            "claude approval offer withdrawn"
        );
    }

    let terminal_targets = integration
        .launches
        .iter()
        .map(|(launch_id, launch)| (launch_id.clone(), launch.terminal_target_id.clone()))
        .collect();
    (
        integration.registry.snapshots(),
        claude_changes,
        terminal_targets,
    )
}

/// Reads the same two Claude Code display inputs `drain_claude_state_changes`
/// returns every tick -- the current session snapshots and the `launch_id ->
/// terminal_target_id` map -- from outside the tick loop, for a caller that
/// needs them without also draining events. Current callers:
/// `send_screenkey_state_immediately` (a physical-key-triggered immediate
/// send) and `actions.rs`'s `resolve_hud_target_slot` (called from
/// `HostActionKind::HudConfirm`), both of which need "what is Claude Code
/// showing right now" without disturbing the tick loop's own event drain.
///
/// Locks `claude_integration` alone and drops the guard before returning --
/// never call this while already holding the `hud` or `ai_display_slots`
/// lock (see this module's and `actions.rs`'s comments on not holding one
/// lock across another).
pub(crate) fn claude_display_inputs(
    integration: &Arc<Mutex<Option<ClaudeIntegration>>>,
) -> (Vec<ClaudeSessionSnapshot>, BTreeMap<String, String>) {
    let guard = integration.lock().unwrap();
    let Some(integration) = guard.as_ref() else {
        return (Vec::new(), BTreeMap::new());
    };
    let terminal_targets = integration
        .launches
        .iter()
        .map(|(launch_id, launch)| (launch_id.clone(), launch.terminal_target_id.clone()))
        .collect();
    (integration.registry.snapshots(), terminal_targets)
}

fn selected_ai_output_with_targets(
    codex_snapshots: Vec<CodexSessionSnapshot>,
    codex_changes: Vec<CodexStateChange>,
    claude_snapshots: Vec<ClaudeSessionSnapshot>,
    claude_changes: Vec<ClaudeStateChange>,
    claude_terminal_targets: &BTreeMap<String, String>,
    selection: &mut AiDisplaySelection,
    last_selection_epoch: &mut u64,
) -> (AiClientStateSnapshot, Vec<AiClientStateChange>) {
    let mut candidates = Vec::new();
    candidates.extend(
        codex_snapshots
            .iter()
            .filter(|snapshot| snapshot.state.session_active && snapshot.is_display_target)
            .map(|snapshot| AiDisplayCandidate {
                target: AiDisplayTarget::Codex {
                    terminal_target_id: snapshot.terminal_target_id.clone(),
                },
                snapshot: snapshot.state,
                registration_order: snapshot.registration_order,
            }),
    );
    candidates.extend(
        claude_snapshots
            .iter()
            .filter(|snapshot| {
                snapshot.session_active
                    && !claude_snapshots.iter().any(|other| {
                        other.session_active
                            && other.launch_id == snapshot.launch_id
                            && other.registration_order > snapshot.registration_order
                    })
            })
            .map(|snapshot| AiDisplayCandidate {
                target: AiDisplayTarget::Claude {
                    terminal_target_id: claude_terminal_targets
                        .get(&snapshot.launch_id)
                        .cloned()
                        .unwrap_or_default(),
                },
                snapshot: claude_as_ai_snapshot(Some(snapshot.clone())),
                registration_order: snapshot.registration_order,
            }),
    );
    selection.update_candidates(candidates);

    let selected_target = selection.selected_target().cloned();
    let snapshot = selection
        .selected_snapshot()
        .unwrap_or(AiClientStateSnapshot {
            client_type: AiClientType::Codex,
            client_variant: AiClientVariant::Cli,
            session_active: false,
            activity_state: AiActivityState::None,
            work_phase: AiWorkPhase::Unspecified,
            revision: 0,
        });
    let selection_changed = selection.epoch() != *last_selection_epoch;
    *last_selection_epoch = selection.epoch();
    if selection_changed {
        return (
            snapshot,
            vec![AiClientStateChange {
                state: snapshot,
                reason: rawhid_host_core::AiClientStateChangeReason::SessionStarted,
            }],
        );
    }

    let changes = match selected_target {
        Some(AiDisplayTarget::Codex { terminal_target_id }) => codex_changes
            .into_iter()
            .filter(|change| {
                // `terminal_target_id` is per-CLI, not per-thread: Codex CLI
                // spawns a second, throwaway thread right after the first
                // user message (to generate the conversation title) that
                // shares the same terminal. Without also checking that the
                // change's thread is still the connection's display target,
                // that throwaway thread's transitions (e.g. its own
                // `TurnCompleted`) would flash onto the slot showing a
                // different, still-active thread. Mirrors the candidate
                // filter above so a non-display thread's changes are simply
                // dropped -- if the display thread itself ends, the
                // candidate set changes, the selection epoch advances, and
                // the branch above resends a full snapshot.
                change.terminal_target_id == terminal_target_id
                    && codex_snapshots.iter().any(|snapshot| {
                        snapshot.state.session_active
                            && snapshot.is_display_target
                            && snapshot.thread_id == change.thread_id
                    })
            })
            .map(|change| AiClientStateChange {
                state: change.state,
                reason: change.reason,
            })
            .collect(),
        Some(AiDisplayTarget::Claude { terminal_target_id }) => claude_changes
            .into_iter()
            .filter(|change| {
                claude_terminal_targets.get(&change.state.launch_id) == Some(&terminal_target_id)
                    && claude_snapshots.iter().any(|snapshot| {
                        snapshot.session_active
                            && snapshot.launch_id == change.state.launch_id
                            && snapshot.session_id == change.state.session_id
                            && !claude_snapshots.iter().any(|other| {
                                other.session_active
                                    && other.launch_id == snapshot.launch_id
                                    && other.registration_order > snapshot.registration_order
                            })
                    })
            })
            .map(|change| AiClientStateChange {
                state: claude_as_ai_snapshot(Some(change.state)),
                // Claude hook changes are full state transitions. Work-phase-only
                // targeting is reserved for Codex's detailed activity stream.
                reason: rawhid_host_core::AiClientStateChangeReason::SessionStarted,
            })
            .collect(),
        None => Vec::new(),
    };
    (snapshot, changes)
}

fn selected_ai_output(
    codex_snapshots: Vec<CodexSessionSnapshot>,
    codex_changes: Vec<CodexStateChange>,
    claude_snapshots: Vec<ClaudeSessionSnapshot>,
    claude_changes: Vec<ClaudeStateChange>,
    selection: &mut AiDisplaySelection,
    last_selection_epoch: &mut u64,
) -> (AiClientStateSnapshot, Vec<AiClientStateChange>) {
    let terminal_targets = claude_snapshots
        .iter()
        .map(|snapshot| (snapshot.launch_id.clone(), snapshot.launch_id.clone()))
        .collect();
    selected_ai_output_with_targets(
        codex_snapshots,
        codex_changes,
        claude_snapshots,
        claude_changes,
        &terminal_targets,
        selection,
        last_selection_epoch,
    )
}

fn sync_ai_client_state<T>(
    transport: &mut T,
    snapshot: AiClientStateSnapshot,
    changes: impl IntoIterator<Item = AiClientStateChange>,
    tracker: &mut AiClientStateSendTracker,
    now: Instant,
    screenkey_state: AiScreenKeyState,
) -> Result<Vec<AiWireSend>, String>
where
    T: AiClientStateTransport,
{
    sync_ai_client_state_slot(
        transport,
        0,
        snapshot,
        changes,
        tracker,
        now,
        screenkey_state,
    )
}

/// Why a wire send happened. Diagnostic-only, kept to enum names and
/// numbers -- never AI-generated content (docs/spec.md).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AiWireSendCause {
    Change(rawhid_host_core::AiClientStateChangeReason),
    Heartbeat,
    DeviceGeneration,
    /// An immediate, out-of-tick resend right after `HostActionKind::HudConfirm`
    /// dispatched an answer, so the `Responded` ScreenKey state reaches the
    /// device without waiting for the next polling tick (see
    /// `send_screenkey_state_immediately`).
    HudResponded,
    /// The computed `AiScreenKeyState` for this slot changed since the last
    /// send even though nothing else did -- a HUD target switch or a
    /// `HUD_RESPONDED_HOLD` expiry between two sessions that stay
    /// `WaitingApproval` throughout never trips `Change`/`DeviceGeneration`,
    /// so this forces a send within one tick instead of waiting up to
    /// `AI_CLIENT_STATE_RESEND_INTERVAL`. See `sync_ai_client_state_slot`.
    ScreenkeyStateChanged,
    /// An immediate, out-of-tick resend right after
    /// `HostActionKind::SelectHudTarget` moved the HUD's explicit target to a
    /// new slot, covering both the newly targeted slot and the
    /// previously-targeted one (see `send_screenkey_state_immediately` and
    /// its caller in `handle_uplink_events`).
    HudTargetSelected,
}

impl AiWireSendCause {
    fn log_label(self) -> String {
        match self {
            Self::Change(reason) => format!("change/{reason:?}"),
            Self::Heartbeat => "heartbeat".to_string(),
            Self::DeviceGeneration => "device-generation".to_string(),
            Self::HudResponded => "hud-responded".to_string(),
            Self::ScreenkeyStateChanged => "screenkey-state-changed".to_string(),
            Self::HudTargetSelected => "hud-target-selected".to_string(),
        }
    }
}

/// One `transport.send_ai_client_state` call, recorded so the monitor loop
/// can log it. Diagnostic for the reported "waiting-approval flashes green
/// first" / "yellow blink stalls then resumes" bugs: enum names, numbers,
/// and the slot only -- never request bodies (docs/spec.md).
#[derive(Debug, Clone)]
struct AiWireSend {
    slot: u8,
    client_type: AiClientType,
    activity_state: AiActivityState,
    work_phase: AiWorkPhase,
    revision: u16,
    cause: AiWireSendCause,
    work_phase_only: bool,
    /// The ScreenKey display-state byte carried in this send's packet (§8 of
    /// `docs/ai-approval-hud-design.md`). Enum name and number only, same as
    /// every other field here.
    screenkey_state: AiScreenKeyState,
    /// `transport.send_ai_client_state`'s return value: how many devices the
    /// packet actually reached. This was previously discarded; zero here
    /// while the caller believed a device was capable is one of the things
    /// this diagnostic exists to surface.
    devices: usize,
}

fn format_ai_wire_send(send: &AiWireSend) -> String {
    format!(
        "ai wire slot={} client={:?} state={:?} phase={:?} rev={} cause={} partial={} screenkey={:?} devices={}",
        send.slot,
        send.client_type,
        send.activity_state,
        send.work_phase,
        send.revision,
        send.cause.log_label(),
        send.work_phase_only,
        send.screenkey_state,
        send.devices
    )
}

fn sync_ai_client_state_slot<T>(
    transport: &mut T,
    display_slot: u8,
    snapshot: AiClientStateSnapshot,
    changes: impl IntoIterator<Item = AiClientStateChange>,
    tracker: &mut AiClientStateSendTracker,
    now: Instant,
    screenkey_state: AiScreenKeyState,
) -> Result<Vec<AiWireSend>, String>
where
    T: AiClientStateTransport,
{
    let device_generation = transport.device_generation();
    let mut sent_change = false;
    let mut sends = Vec::new();

    for change in changes {
        let packet = ai_client_state_packet_for_slot(change.state, display_slot, screenkey_state)?;
        if !transport.has_ai_client_state_device(packet.client_type) {
            tracker.last_device_generation = None;
            tracker.last_sent_at = None;
            tracker.last_screenkey_state = None;
            continue;
        }
        let work_phase_only =
            change.reason == rawhid_host_core::AiClientStateChangeReason::WorkPhaseChanged;
        let devices = transport.send_ai_client_state(packet, work_phase_only)?;
        tracker.push_wire_send(
            &mut sends,
            AiWireSend {
                slot: display_slot,
                client_type: packet.client_type,
                activity_state: packet.activity_state,
                work_phase: packet.work_phase,
                revision: packet.revision,
                cause: AiWireSendCause::Change(change.reason),
                work_phase_only,
                screenkey_state: packet.screenkey_state,
                devices,
            },
        );
        tracker.last_screenkey_state = Some(packet.screenkey_state);
        sent_change = true;
    }

    let periodic_due = tracker.last_sent_at.is_none_or(|last_sent| {
        now.saturating_duration_since(last_sent) >= AI_CLIENT_STATE_RESEND_INTERVAL
    });
    let device_generation_changed = tracker.last_device_generation != Some(device_generation);
    // A slot's ScreenKey display state can change (HUD target switched, or a
    // `HUD_RESPONDED_HOLD` window just expired) without `activity_state`
    // changing at all -- two sessions both stuck at `WaitingApproval` never
    // produce a `Change`. Without this, such a switch would sit unsent until
    // the next `AI_CLIENT_STATE_RESEND_INTERVAL` heartbeat (5s), which is far
    // too slow for a HUD-target indicator meant to disambiguate which of
    // several blinking slots the physical Confirm key currently answers.
    let screenkey_state_changed = tracker.last_screenkey_state != Some(screenkey_state);
    if !sent_change
        && (device_generation_changed || periodic_due || screenkey_state_changed)
        && transport.has_ai_client_state_device(snapshot.client_type)
    {
        let packet = ai_client_state_packet_for_slot(snapshot, display_slot, screenkey_state)?;
        let devices = transport.send_ai_client_state(packet, false)?;
        let cause = if device_generation_changed {
            AiWireSendCause::DeviceGeneration
        } else if screenkey_state_changed {
            AiWireSendCause::ScreenkeyStateChanged
        } else {
            AiWireSendCause::Heartbeat
        };
        tracker.push_wire_send(
            &mut sends,
            AiWireSend {
                slot: display_slot,
                client_type: packet.client_type,
                activity_state: packet.activity_state,
                work_phase: packet.work_phase,
                revision: packet.revision,
                cause,
                work_phase_only: false,
                screenkey_state: packet.screenkey_state,
                devices,
            },
        );
        tracker.last_screenkey_state = Some(packet.screenkey_state);
        sent_change = true;
    }

    if transport.has_ai_client_state_device(snapshot.client_type) {
        tracker.last_device_generation = Some(device_generation);
        if sent_change {
            tracker.last_sent_at = Some(now);
        }
    } else {
        tracker.last_device_generation = None;
        tracker.last_sent_at = None;
        tracker.last_screenkey_state = None;
    }
    Ok(sends)
}

fn ai_client_state_packet_for_slot(
    state: AiClientStateSnapshot,
    display_slot: u8,
    screenkey_state: AiScreenKeyState,
) -> Result<AiClientStatePacket, String> {
    AiClientStatePacket::new_for_slot(
        state.client_type,
        state.client_variant as u8,
        state.session_active,
        state.activity_state,
        state.work_phase,
        state.revision,
        display_slot,
    )
    .map(|packet| packet.with_screenkey_state(screenkey_state))
    .map_err(|error| error.to_string())
}

/// How long a ScreenKey slot's display state is held at
/// `AiScreenKeyState::Responded` after `HostActionKind::HudConfirm`
/// dispatches an answer for it, per `docs/ai-approval-hud-design.md` §8's
/// firmware confirmation animation. Without this hold, the very next state
/// change (the resolved approval moving from `WaitingApproval` to `Working`)
/// can arrive within one polling tick and overwrite byte nine before the
/// animation finishes.
///
/// Firmware's playback is 700 ms, but the Host's hold and firmware's
/// playback do not start at the same instant: the Host starts holding the
/// moment it processes the Confirm `HOST_ACTION`, while firmware only starts
/// animating once the following HID write actually arrives. A hold too close
/// to firmware's playback time races that gap and was observed on real
/// hardware (when firmware's playback was still 600 ms and this hold was 600
/// ms) to let the hold expire -- and byte nine revert to the live computed
/// state -- just before playback finished, truncating the checkmark
/// animation's final frame. 850 ms gives 150 ms of margin over the HID write
/// and firmware draw-start latency, comfortably longer than firmware's own
/// 700 ms so the hold always outlives playback. This only widens the window
/// byte nine is pinned to `Responded`; bytes 0..8 keep carrying the live
/// state throughout, so no other display state is delayed by the extra
/// margin.
const HUD_RESPONDED_HOLD: Duration = Duration::from_millis(850);

/// The `AiScreenKeyState` a slot should currently show, computed from its
/// live waiting/target status (§8/§11: waiting-and-target, waiting-only, or
/// normal) but overridden to `Responded` while that slot is within
/// `HUD_RESPONDED_HOLD` of its last dispatched HUD answer. An expired hold
/// entry is cleared here so it does not linger for a later, unrelated slot
/// reusing the same index.
fn screenkey_state_for_slot(
    slot: u8,
    computed: AiScreenKeyState,
    hud_responded: &Mutex<Option<(u8, Instant)>>,
    now: Instant,
) -> AiScreenKeyState {
    let mut responded = hud_responded.lock().unwrap();
    match *responded {
        Some((responded_slot, at)) if now.saturating_duration_since(at) < HUD_RESPONDED_HOLD => {
            if responded_slot == slot {
                return AiScreenKeyState::Responded;
            }
        }
        Some(_) => {
            *responded = None;
        }
        None => {}
    }
    computed
}

/// Whether the AI session a ScreenKey slot resolved to (via
/// `actions::hud_target_for_slot`) is exactly the HUD's current target
/// (`HudCoordinator::target_session`), client-independent. `None` on either
/// side is never a match -- a slot with no pending-approval target, or no
/// HUD target at all, is not "the" target. A Codex slot can only ever equal
/// a Codex HUD target and a Claude Code slot only a Claude Code one, since
/// `HudTargetSession`'s variants compare unequal across clients.
fn is_hud_target(
    slot_target: &Option<HudTargetSession>,
    hud_target: &Option<HudTargetSession>,
) -> bool {
    matches!((slot_target, hud_target), (Some(a), Some(b)) if a == b)
}

/// The `AiScreenKeyState` a slot's live waiting/target status computes to,
/// before the `HUD_RESPONDED_HOLD` override in `screenkey_state_for_slot`.
/// Per `docs/ai-approval-hud-design.md` §8/§11: waiting on approval/input and
/// the current HUD target, waiting but not the target, or neither.
fn compute_screenkey_state(activity_state: AiActivityState, is_target: bool) -> AiScreenKeyState {
    let waiting = matches!(
        activity_state,
        AiActivityState::WaitingApproval | AiActivityState::WaitingInput
    );
    match (waiting, is_target) {
        (true, true) => AiScreenKeyState::WaitingTarget,
        (true, false) => AiScreenKeyState::WaitingOther,
        (false, _) => AiScreenKeyState::Normal,
    }
}

/// Finds the one other slot (if any) whose last actually-sent ScreenKey
/// state was `AiScreenKeyState::WaitingTarget`, excluding `new_slot`. Used
/// by `ActionOutcome::HudTargetSelected` handling in `handle_uplink_events`
/// so the previously-targeted slot's white frame is cleared in the same
/// immediate send as the newly-targeted slot's, instead of surviving until
/// the next tick. Excluding `new_slot` itself means a target switch that
/// lands back on the slot already recorded as the target (a no-op switch,
/// or the only slot with a live tracker) never produces a duplicate in the
/// caller's send list.
fn previously_targeted_slot(
    ai_client_state_send: &BTreeMap<u8, AiClientStateSendTracker>,
    new_slot: u8,
) -> Option<u8> {
    ai_client_state_send
        .iter()
        .find(|&(&slot, tracker)| {
            slot != new_slot
                && tracker.last_screenkey_state == Some(AiScreenKeyState::WaitingTarget)
        })
        .map(|(&slot, _)| slot)
}

fn apply_monitor_config(
    app: &AppHandle,
    next_config: AppConfig,
    runner: &mut MonitorRunner,
    interval: &mut Duration,
    uplink_interval: &mut Duration,
    status: &Arc<std::sync::Mutex<MonitorStatus>>,
    ai_usage_shared: Option<AiUsageShared>,
    extras: &MonitorExtras,
    actions_cfg: &mut ActionsConfig,
) -> Result<(), String> {
    *actions_cfg = next_config.actions.clone();
    extras
        .ai_display_slots
        .lock()
        .expect("AI display slots lock poisoned")
        .set_slot_count(next_config.ai_client.display.slot_count);
    if let Ok(mut store) = extras.key_stats.lock() {
        store.set_flush_interval(Duration::from_secs(
            next_config.stats.flush_interval_sec.max(1),
        ));
    }
    runner.update_config(next_config.clone(), ai_usage_shared);
    runner.set_key_stats_store(Arc::clone(&extras.key_stats));
    *interval = Duration::from_millis(next_config.polling.interval_ms.max(1));
    *uplink_interval = Duration::from_millis(next_config.polling.uplink_interval_ms.max(5));

    let ai_usage = runner.ai_usage_statuses();
    update_status(app, status, |s| {
        s.current_layer = None;
        s.current_rule = None;
        s.connected_devices = 0;
        s.connected_device_names = Vec::new();
        s.host_link_devices = Vec::new();
        s.device_battery = Vec::new();
        s.device_layers = Vec::new();
        s.ai_usage = ai_usage;
        s.last_error = None;
    });

    Ok(())
}

/// Apply a single monitor command. Returns `true` when the loop should stop.
fn process_command(
    command: MonitorCommand,
    app: &AppHandle,
    runner: &mut MonitorRunner,
    interval: &mut Duration,
    uplink_interval: &mut Duration,
    status: &Arc<std::sync::Mutex<MonitorStatus>>,
    log_entries: &Arc<std::sync::Mutex<std::collections::VecDeque<LogEntry>>>,
    log_counter: &Arc<std::sync::Mutex<u64>>,
    extras: &MonitorExtras,
    actions_cfg: &mut ActionsConfig,
    automation_enabled: &mut bool,
    ai_client_state_send: &mut BTreeMap<u8, AiClientStateSendTracker>,
) -> bool {
    match command {
        MonitorCommand::Shutdown => true,
        MonitorCommand::SetAutomationEnabled(enabled, reply) => {
            *automation_enabled = enabled;
            update_status(app, status, |s| {
                s.running = enabled;
                if !enabled {
                    s.current_layer = None;
                    s.current_rule = None;
                }
            });
            let _ = reply.send(Ok(()));
            false
        }
        MonitorCommand::Probe(reply) => {
            let result = runner.probe_host_link_devices().map_err(hid_error_code);
            let _ = reply.send(result);
            false
        }
        MonitorCommand::Config(call) => {
            let result = if Instant::now() >= call.deadline {
                Err("hostlink_result_unknown".to_string())
            } else {
                execute_host_link_request(runner, call.uid, call.request)
            };
            let _ = call.reply.send(result);
            if *automation_enabled {
                runner.drain_uplink_only();
                if handle_uplink_events(
                    app,
                    runner,
                    actions_cfg,
                    extras,
                    status,
                    log_entries,
                    log_counter,
                    ai_client_state_send,
                ) {
                    *automation_enabled = false;
                    update_status(app, status, |s| {
                        s.running = false;
                        s.current_layer = None;
                        s.current_rule = None;
                    });
                }
            } else {
                runner.discard_uplink_only();
            }
            false
        }
        MonitorCommand::ForegroundChanged => false,
        MonitorCommand::InjectUplink(device, packet) => {
            runner.inject_uplink(device, packet);
            false
        }
        MonitorCommand::UpdateConfig(next_config, ai_usage_shared) => {
            if let Err(e) = apply_monitor_config(
                app,
                next_config,
                runner,
                interval,
                uplink_interval,
                status,
                ai_usage_shared,
                extras,
                actions_cfg,
            ) {
                let msg = format!("HID init error: {}", e);
                update_status(app, status, |s| s.last_error = Some(msg.clone()));
                let entry = add_log(log_entries, log_counter, "error", &msg);
                let _ = app.emit("log-added", entry);
            }
            false
        }
    }
}

fn execute_host_link_request(
    runner: &mut MonitorRunner,
    uid: u64,
    request: HostLinkRequest,
) -> Result<HostLinkResponse, String> {
    match request {
        HostLinkRequest::EncoderGetInfo => runner
            .config_get_encoder_info(uid)
            .map(HostLinkResponse::EncoderInfo),
        HostLinkRequest::EncoderGetBindings {
            layer_id,
            encoder_id,
        } => runner
            .config_get_encoder_bindings(uid, layer_id, encoder_id)
            .map(HostLinkResponse::EncoderBindings),
        HostLinkRequest::EncoderSetBindings {
            layer_id,
            encoder_id,
            cw,
            ccw,
        } => runner
            .config_set_encoder_bindings(uid, layer_id, encoder_id, cw, ccw)
            .map(|()| HostLinkResponse::Done),
        HostLinkRequest::EncoderGetDirty => runner
            .config_get_encoder_dirty(uid)
            .map(HostLinkResponse::Dirty),
        HostLinkRequest::EncoderSave => runner
            .config_save_encoder(uid)
            .map(|()| HostLinkResponse::Done),
        HostLinkRequest::EncoderDiscard => runner
            .config_discard_encoder(uid)
            .map(|()| HostLinkResponse::Done),
        HostLinkRequest::EncoderClearOverride {
            layer_id,
            encoder_id,
        } => runner
            .config_clear_encoder_override(uid, layer_id, encoder_id)
            .map(|()| HostLinkResponse::Done),
        HostLinkRequest::ComboGetInfo => runner
            .config_get_combo_info(uid)
            .map(HostLinkResponse::ComboInfo),
        HostLinkRequest::ComboGet { slot } => runner
            .config_get_combo(uid, slot)
            .map(HostLinkResponse::ComboItem),
        HostLinkRequest::ComboSet { item } => runner
            .config_set_combo(uid, item)
            .map(|()| HostLinkResponse::Done),
        HostLinkRequest::ComboGetDirty => runner
            .config_get_combo_dirty(uid)
            .map(HostLinkResponse::Dirty),
        HostLinkRequest::ComboSave => runner
            .config_save_combos(uid)
            .map(|()| HostLinkResponse::Done),
        HostLinkRequest::ComboDiscard => runner
            .config_discard_combos(uid)
            .map(|()| HostLinkResponse::Done),
        HostLinkRequest::ComboDelete { slot } => runner
            .config_delete_combo(uid, slot)
            .map(|()| HostLinkResponse::Done),
        HostLinkRequest::ComboResetToKeymap => runner
            .config_reset_combos_to_keymap(uid)
            .map(|()| HostLinkResponse::Done),
    }
    .map_err(hid_error_code)
}

#[derive(Debug, Clone, serde::Serialize)]
struct KeyPressPayload {
    device_uid: String,
    position: u8,
    pressed: bool,
}

/// Sends each of `slots`' freshly computed `AiScreenKeyState` right away,
/// rather than waiting for the next polling tick (up to
/// `polling.interval_ms`, 500 ms on real hardware). This is the single
/// immediate-send path shared by every `ActionOutcome` that changes a slot's
/// ScreenKey display state outside the regular monitor-loop tick --
/// `HudResponded` (one slot: the one that was just answered) and
/// `HudTargetSelected` (up to two slots: the newly targeted one and the
/// previously targeted one) -- rather than two parallel send paths.
///
/// Each slot's state is computed exactly like the monitor loop's regular
/// per-tick pass: `compute_screenkey_state` from its live waiting/target
/// status against the HUD's current target, then `screenkey_state_for_slot`'s
/// `HUD_RESPONDED_HOLD` override. Routing every immediate send through the
/// same hold check means a slot currently holding `Responded` after a
/// just-dispatched HUD answer keeps showing `Responded` here too, even when
/// the same tick also fires an unrelated `HudTargetSelected` touching other
/// slots -- the hold is never dropped as a side effect of another slot's
/// immediate send.
///
/// The HUD target is read once, before `ai_display_slots` is locked per
/// slot below -- never hold both locks at once (see the `hud`/
/// `ai_display_slots` boundary in this module's monitor loop and
/// `actions.rs`'s `HudConfirm` arm). Claude Code's display inputs
/// (`claude_display_inputs`) are likewise read up front, before either the
/// `hud` or `ai_display_slots` lock is taken -- see that function's own
/// doc comment on why.
///
/// A missing slot snapshot falls back to `inactive_ai_snapshot()`; any
/// encode or transport failure is logged via `tracing::warn!` and otherwise
/// ignored -- this is a best-effort immediate nudge, and the regular
/// monitor-loop sync a moment later still carries the authoritative state.
///
/// On success, also records the sent `AiScreenKeyState` as that slot's
/// `AiClientStateSendTracker::last_screenkey_state`. Without this, the very
/// next regular tick's `sync_ai_client_state_slot` would see a tracker that
/// still thinks nothing was sent yet, treat the value as a fresh
/// `screenkey_state_changed`, and immediately re-send the same packet.
fn send_screenkey_state_immediately(
    runner: &mut MonitorRunner,
    extras: &MonitorExtras,
    slots: &[u8],
    cause: AiWireSendCause,
    ai_client_state_send: &mut BTreeMap<u8, AiClientStateSendTracker>,
) {
    let (claude_snapshots, claude_terminal_targets) =
        claude_display_inputs(&extras.claude_integration);
    let hud_target_session = extras
        .hud
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|hud| hud.target_session());
    let codex_snapshots = extras.codex_activity.snapshots();
    let now = Instant::now();

    for &slot in slots {
        let (snapshot, target) = {
            let selection = extras.ai_display_slots.lock().unwrap();
            let snapshot = selection
                .slot_snapshot(slot)
                .unwrap_or_else(inactive_ai_snapshot);
            let assigned = selection
                .slots()
                .get(usize::from(slot))
                .and_then(|entry| entry.assigned.clone());
            (
                snapshot,
                actions::hud_target_for_slot(
                    assigned,
                    &codex_snapshots,
                    &claude_snapshots,
                    &claude_terminal_targets,
                ),
            )
        };
        let computed = compute_screenkey_state(
            snapshot.activity_state,
            is_hud_target(&target, &hud_target_session),
        );
        let screenkey_state = screenkey_state_for_slot(slot, computed, &extras.hud_responded, now);
        let packet = match ai_client_state_packet_for_slot(snapshot, slot, screenkey_state) {
            Ok(packet) => packet,
            Err(error) => {
                tracing::warn!(slot, "AI display slot immediate send encode error: {error}");
                continue;
            }
        };
        match runner.send_ai_client_state(packet, false) {
            Ok(devices) => {
                tracing::debug!(
                    target: AI_DISPLAY_LOG_TARGET,
                    "{}",
                    format_ai_wire_send(&AiWireSend {
                        slot,
                        client_type: packet.client_type,
                        activity_state: packet.activity_state,
                        work_phase: packet.work_phase,
                        revision: packet.revision,
                        cause,
                        work_phase_only: false,
                        screenkey_state: packet.screenkey_state,
                        devices,
                    })
                );
                ai_client_state_send
                    .entry(slot)
                    .or_default()
                    .last_screenkey_state = Some(packet.screenkey_state);
            }
            Err(error) => {
                tracing::warn!(slot, "AI display slot immediate send failed: {error}");
            }
        }
    }
}

/// Handle uplink events drained from the runner. Returns `true` when a
/// HOST_ACTION requested the monitor loop to stop.
fn handle_uplink_events(
    app: &AppHandle,
    runner: &mut MonitorRunner,
    actions_cfg: &ActionsConfig,
    extras: &MonitorExtras,
    status: &Arc<std::sync::Mutex<MonitorStatus>>,
    log_entries: &Arc<std::sync::Mutex<std::collections::VecDeque<LogEntry>>>,
    log_counter: &Arc<std::sync::Mutex<u64>>,
    ai_client_state_send: &mut BTreeMap<u8, AiClientStateSendTracker>,
) -> bool {
    let mut should_stop = false;
    for event in runner.take_uplink_events() {
        match &event.packet {
            UplinkPacket::HostAction(action) => {
                // Bindings are per device; a device without an (enabled)
                // config section only has its actions logged.
                let device_key = uplink_device_key(&event.device);
                let binding = actions_cfg
                    .enabled
                    .then(|| {
                        actions_cfg
                            .devices
                            .get(&device_key)
                            .filter(|device_cfg| device_cfg.enabled)
                            .and_then(|device_cfg| {
                                device_cfg
                                    .bindings
                                    .iter()
                                    .find(|b| b.action_id == action.action_id)
                            })
                    })
                    .flatten();
                let (level, message) = match binding {
                    Some(binding) => {
                        match actions::execute(app, binding, action.value, extras, status) {
                            Ok(actions::ActionOutcome::Continue) => (
                                "info",
                                format!(
                                    "host action {} executed ({:?})",
                                    action.action_id, binding.action
                                ),
                            ),
                            Ok(actions::ActionOutcome::HudNoop { reason }) => (
                                "info",
                                format!(
                                    "host action {} ignored safely ({})",
                                    action.action_id, reason
                                ),
                            ),
                            Ok(actions::ActionOutcome::AiSessionSelected { label }) => (
                                "info",
                                format!(
                                    "host action {} selected AI session {}",
                                    action.action_id, label
                                ),
                            ),
                            Ok(actions::ActionOutcome::StopRequested) => {
                                should_stop = true;
                                (
                                    "info",
                                    format!("host action {}: stop monitoring", action.action_id),
                                )
                            }
                            Ok(actions::ActionOutcome::HudResponded { slot }) => {
                                *extras.hud_responded.lock().unwrap() =
                                    Some((slot, Instant::now()));
                                send_screenkey_state_immediately(
                                    runner,
                                    extras,
                                    &[slot],
                                    AiWireSendCause::HudResponded,
                                    ai_client_state_send,
                                );
                                (
                                    "info",
                                    format!(
                                        "host action {} executed ({:?}) hud_responded slot={}",
                                        action.action_id, binding.action, slot
                                    ),
                                )
                            }
                            Ok(actions::ActionOutcome::HudTargetSelected { slot }) => {
                                // Also resend whichever slot was `WaitingTarget`
                                // before this switch (if any and if different),
                                // so its white frame disappears in the same
                                // immediate send instead of surviving until the
                                // next tick -- otherwise two slots would
                                // briefly appear to be the HUD target at once.
                                let mut slots = vec![slot];
                                if let Some(previous_slot) =
                                    previously_targeted_slot(ai_client_state_send, slot)
                                {
                                    slots.push(previous_slot);
                                }
                                send_screenkey_state_immediately(
                                    runner,
                                    extras,
                                    &slots,
                                    AiWireSendCause::HudTargetSelected,
                                    ai_client_state_send,
                                );
                                (
                                    "info",
                                    format!(
                                        "host action {} executed ({:?}) hud_target_selected slot={}",
                                        action.action_id, binding.action, slot
                                    ),
                                )
                            }
                            Err(error) => (
                                "error",
                                format!("host action {} failed: {}", action.action_id, error),
                            ),
                        }
                    }
                    None if actions_cfg.enabled => (
                        "warn",
                        format!(
                            "unbound host action id={} value={} ({})",
                            action.action_id, action.value, device_key
                        ),
                    ),
                    None => (
                        "warn",
                        format!(
                            "host action id={} ignored (actions disabled)",
                            action.action_id
                        ),
                    ),
                };
                let entry = add_log(log_entries, log_counter, level, &message);
                let _ = app.emit("log-added", entry);
            }
            UplinkPacket::KeyStats(_) => {
                let _ = app.emit("key-stats-updated", uplink_device_key(&event.device));
            }
            UplinkPacket::KeyPress(p) => {
                let _ = app.emit(
                    "key-press-event",
                    KeyPressPayload {
                        device_uid: uplink_device_key(&event.device),
                        position: p.position,
                        pressed: p.pressed,
                    },
                );
            }
            // Battery / layer state are part of MonitorStatus and flow
            // through the regular status-update emission.
            UplinkPacket::Battery(_) | UplinkPacket::LayerState(_) => {}
        }
    }
    should_stop
}

#[allow(clippy::too_many_arguments)]
fn run_monitor_loop(
    app: AppHandle,
    mut config: AppConfig,
    status: Arc<std::sync::Mutex<MonitorStatus>>,
    log_entries: Arc<std::sync::Mutex<std::collections::VecDeque<LogEntry>>>,
    log_counter: Arc<std::sync::Mutex<u64>>,
    rx: mpsc::Receiver<MonitorCommand>,
    mut ai_usage_shared: Option<AiUsageShared>,
    extras: MonitorExtras,
    mut automation_enabled: bool,
) {
    let mut init_error_logged = false;
    let mut runner = loop {
        match rebuild_runner(&config, ai_usage_shared.clone()) {
            Ok(runner) => break runner,
            Err(error) => {
                if !init_error_logged {
                    let msg = format!("HID init error: {error}");
                    let entry = add_log(&log_entries, &log_counter, "error", &msg);
                    update_status(&app, &status, |s| {
                        s.running = automation_enabled;
                        s.last_error = Some(msg);
                    });
                    let _ = app.emit("log-added", entry);
                    init_error_logged = true;
                }
                match rx.recv_timeout(Duration::from_secs(1)) {
                    Ok(MonitorCommand::Shutdown) | Err(mpsc::RecvTimeoutError::Disconnected) => {
                        return;
                    }
                    Ok(MonitorCommand::SetAutomationEnabled(enabled, reply)) => {
                        automation_enabled = enabled;
                        status.lock().unwrap().running = enabled;
                        let _ = reply.send(Ok(()));
                    }
                    Ok(MonitorCommand::UpdateConfig(next, shared)) => {
                        config = next;
                        ai_usage_shared = shared;
                        init_error_logged = false;
                    }
                    Ok(MonitorCommand::Probe(reply)) => {
                        let _ = reply.send(Err("hostlink_worker_unavailable".to_string()));
                    }
                    Ok(MonitorCommand::Config(call)) => {
                        let _ = call
                            .reply
                            .send(Err("hostlink_worker_unavailable".to_string()));
                    }
                    Ok(MonitorCommand::ForegroundChanged)
                    | Ok(MonitorCommand::InjectUplink(_, _))
                    | Err(mpsc::RecvTimeoutError::Timeout) => {}
                }
            }
        }
    };
    let mut actions_cfg = config.actions.clone();
    runner.set_key_stats_store(Arc::clone(&extras.key_stats));

    update_status(&app, &status, |s| {
        s.running = automation_enabled;
        s.last_error = None;
    });

    let entry = add_log(
        &log_entries,
        &log_counter,
        "info",
        "Host Link worker started",
    );
    let _ = app.emit("log-added", entry);

    let mut interval = Duration::from_millis(config.polling.interval_ms.max(1));
    let mut uplink_interval = Duration::from_millis(config.polling.uplink_interval_ms.max(5));
    let mut ai_client_state_send = BTreeMap::<u8, AiClientStateSendTracker>::new();
    let mut last_ai_selection_epoch = 0;
    // Diagnostic for the reported ScreenKey display glitches: records the
    // label/epoch this loop last saw for each slot (slot 0 included) so a
    // reassignment can be logged once instead of every tick. IDs come from
    // `AiDisplayTarget::label`, which already truncates to a masked
    // fragment (docs/spec.md: no raw identifiers).
    let mut last_ai_slot_assignment: BTreeMap<u8, (Option<String>, u64)> = BTreeMap::new();

    loop {
        let mut should_stop = false;
        loop {
            match rx.try_recv() {
                Ok(command) => {
                    if process_command(
                        command,
                        &app,
                        &mut runner,
                        &mut interval,
                        &mut uplink_interval,
                        &status,
                        &log_entries,
                        &log_counter,
                        &extras,
                        &mut actions_cfg,
                        &mut automation_enabled,
                        &mut ai_client_state_send,
                    ) {
                        should_stop = true;
                        break;
                    }
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    should_stop = true;
                    break;
                }
                Err(mpsc::TryRecvError::Empty) => break,
            }
        }
        if should_stop {
            break;
        }

        if automation_enabled {
            match runner.tick() {
                Ok(RunEvent::SetLayer { layer, rule_name }) => {
                    update_status(&app, &status, |s| {
                        s.current_layer = Some(layer);
                        s.current_rule = Some(rule_name.clone());
                        apply_runner_view(s, &runner);
                        s.last_error = None;
                    });
                    let msg = format!("Switched to layer {} (rule: {})", layer, rule_name);
                    let entry = add_log(&log_entries, &log_counter, "info", &msg);
                    let _ = app.emit("log-added", entry);
                }
                Ok(RunEvent::Clear) => {
                    update_status(&app, &status, |s| {
                        s.current_layer = None;
                        s.current_rule = None;
                        apply_runner_view(s, &runner);
                    });
                }
                Ok(RunEvent::Unchanged) => {
                    let devices = runner.verified_device_count();
                    let host_link_devices = runner.verified_devices();
                    let ai_usage = runner.ai_usage_statuses();
                    let device_battery = runner.battery_statuses();
                    let device_layers = runner.layer_states();
                    update_status_if(&app, &status, |s| {
                        if s.connected_devices == devices
                            && s.ai_usage == ai_usage
                            && s.host_link_devices == host_link_devices
                            && s.device_battery == device_battery
                            && s.device_layers == device_layers
                        {
                            return false;
                        }
                        apply_runner_view(s, &runner);
                        true
                    });
                }
                Err(e) => {
                    let msg = format!("Error: {}", e);
                    update_status(&app, &status, |s| s.last_error = Some(msg.clone()));
                    let entry = add_log(&log_entries, &log_counter, "error", &msg);
                    let _ = app.emit("log-added", entry);
                }
            }

            if handle_uplink_events(
                &app,
                &mut runner,
                &actions_cfg,
                &extras,
                &status,
                &log_entries,
                &log_counter,
                &mut ai_client_state_send,
            ) {
                automation_enabled = false;
                update_status(&app, &status, |s| {
                    s.running = false;
                    s.current_layer = None;
                    s.current_rule = None;
                });
            }
        } else {
            match runner.tick_transport_only() {
                Ok(()) => {
                    update_status(&app, &status, |s| {
                        apply_runner_view(s, &runner);
                        s.running = false;
                        s.device_battery.clear();
                        s.device_layers.clear();
                        s.last_error = None;
                    });
                }
                Err(error) => {
                    update_status(&app, &status, |s| {
                        s.last_error = Some(format!("Error: {error}"))
                    });
                }
            }
        }

        let now = Instant::now();
        let codex_snapshots = extras.codex_activity.snapshots();
        let codex_changes = drain_codex_state_changes(&extras.codex_activity);
        // Diagnostic for the reported "first approval flashes yellow then
        // turns green while the request is still unanswered": records every
        // Codex activity transition alongside how many approvals the store
        // still holds, without needing to reproduce under a debugger. An
        // earlier version escalated this to `warn` when a Completed state
        // arrived with a non-zero `pending_approvals` count, on the
        // hypothesis that it meant an unresolved approval; that hypothesis
        // was wrong (the real cause -- another thread's state change leaking
        // into this display slot -- was already fixed elsewhere), and
        // `pending_approvals` counts the whole store, so it is routinely
        // non-zero for threads unrelated to this one even in normal
        // operation. Always `debug!` now, under the same
        // `AI_DISPLAY_LOG_TARGET` as the AI wire-send / slot-assignment
        // diagnostics, so this lives in the `[debug_log]` file sink instead
        // of the in-memory UI log. IDs and counts only -- no request bodies
        // (docs/spec.md's sanitized-output rule for the UI log applies here
        // too).
        for change in &codex_changes {
            let pending = extras.codex_activity.pending_approvals().len();
            tracing::debug!(
                target: AI_DISPLAY_LOG_TARGET,
                "codex activity thread={} state={:?} reason={:?} pending_approvals={}",
                thread_tag(&change.thread_id),
                change.state.activity_state,
                change.reason,
                pending
            );
        }
        // Both clients' pending-approval bodies live in the one store Codex
        // already owns (`CodexActivityRuntime`), so a future HUD reads a
        // single source regardless of which AI it is showing.
        let (claude_snapshots, claude_changes, claude_terminal_targets) =
            drain_claude_state_changes(
                &extras.claude_integration,
                &extras.codex_activity.pending_approvals(),
                &extras.claude_permission_gate,
                now,
            );
        // The HUD's current target, read before the `ai_display_slots` lock
        // is taken below -- never hold both locks at once (see the
        // `hud`/`ai_display_slots` boundary in `actions.rs`'s `HudConfirm`
        // arm). Compared against each slot's own resolved target
        // (`actions::hud_target_for_slot`, client-independent) to decide
        // `AiScreenKeyState::WaitingTarget` vs `WaitingOther`; see
        // `compute_screenkey_state`.
        let hud_target_session = extras
            .hud
            .lock()
            .unwrap()
            .as_ref()
            .and_then(|hud| hud.target_session());
        // `claude_snapshots` is moved into `selected_ai_output_with_targets`
        // below, so a clone is kept here for the per-slot target resolution
        // further down -- same reason `codex_snapshots.clone()` exists.
        let claude_snapshots_for_targets = claude_snapshots.clone();
        let (
            ai_snapshot,
            ai_changes,
            selected_codex_thread,
            retired_slots,
            extra_slots,
            slot_assignment_logs,
            slot0_screenkey_state,
        ) = {
            let mut selection = extras.ai_display_slots.lock().unwrap();
            let (snapshot, changes) = selected_ai_output_with_targets(
                codex_snapshots.clone(),
                codex_changes.clone(),
                claude_snapshots,
                claude_changes.clone(),
                &claude_terminal_targets,
                &mut selection,
                &mut last_ai_selection_epoch,
            );
            let selected_codex_thread = match selection.selected_target() {
                Some(AiDisplayTarget::Codex { terminal_target_id }) => codex_snapshots
                    .iter()
                    .find(|snapshot| {
                        snapshot.is_display_target
                            && snapshot.terminal_target_id == *terminal_target_id
                    })
                    .map(|snapshot| snapshot.thread_id.clone()),
                _ => None,
            };
            let slot0_target = actions::hud_target_for_slot(
                selection.selected_target().cloned(),
                &codex_snapshots,
                &claude_snapshots_for_targets,
                &claude_terminal_targets,
            );
            let slot0_screenkey_state = compute_screenkey_state(
                snapshot.activity_state,
                is_hud_target(&slot0_target, &hud_target_session),
            );
            let retired_slots = selection.take_retired_slots();
            let extra_slots: Vec<(
                u8,
                AiClientStateSnapshot,
                Vec<AiClientStateChange>,
                u64,
                AiScreenKeyState,
            )> = selection
                .slots()
                .iter()
                .filter(|entry| entry.slot != 0)
                .map(|entry| {
                    let snapshot = selection
                        .slot_snapshot(entry.slot)
                        .unwrap_or_else(inactive_ai_snapshot);
                    let changes = match entry.assigned.as_ref() {
                        Some(AiDisplayTarget::Codex { terminal_target_id }) => codex_changes
                            .iter()
                            .filter(|change| {
                                // Same fix as `selected_ai_output_with_targets`:
                                // a `terminal_target_id` match alone also
                                // admits the throwaway title-generation
                                // thread Codex CLI spawns after the first
                                // user message, so also require the change's
                                // thread to still be this connection's
                                // display target.
                                change.terminal_target_id == *terminal_target_id
                                    && codex_snapshots.iter().any(|snapshot| {
                                        snapshot.state.session_active
                                            && snapshot.is_display_target
                                            && snapshot.thread_id == change.thread_id
                                    })
                            })
                            .map(|change| AiClientStateChange {
                                state: change.state,
                                reason: change.reason,
                            })
                            .collect(),
                        Some(AiDisplayTarget::Claude { terminal_target_id }) => claude_changes
                            .iter()
                            .filter(|change| {
                                claude_terminal_targets.get(&change.state.launch_id)
                                    == Some(terminal_target_id)
                            })
                            .map(|change| AiClientStateChange {
                                state: claude_as_ai_snapshot(Some(change.state.clone())),
                                reason: rawhid_host_core::AiClientStateChangeReason::SessionStarted,
                            })
                            .collect(),
                        None => Vec::new(),
                    };
                    let target = actions::hud_target_for_slot(
                        entry.assigned.clone(),
                        &codex_snapshots,
                        &claude_snapshots_for_targets,
                        &claude_terminal_targets,
                    );
                    let screenkey_state = compute_screenkey_state(
                        snapshot.activity_state,
                        is_hud_target(&target, &hud_target_session),
                    );
                    (entry.slot, snapshot, changes, entry.epoch, screenkey_state)
                })
                .collect();
            // Diagnostic for the reported ScreenKey display glitches: log a
            // slot's assignment only when its target or epoch actually
            // changed since the loop last saw it, covering slot 0 too
            // (`extra_slots` above only tracks slot != 0). Built while the
            // lock is held, logged after it is released below.
            let slot_assignment_logs: Vec<String> = selection
                .slots()
                .iter()
                .filter_map(|entry| {
                    let label = entry.assigned.as_ref().map(AiDisplayTarget::label);
                    let key = (label.clone(), entry.epoch);
                    let unchanged = last_ai_slot_assignment.get(&entry.slot) == Some(&key);
                    last_ai_slot_assignment.insert(entry.slot, key);
                    if unchanged {
                        None
                    } else {
                        // `label()` returns a display string with a space
                        // (e.g. "Codex 1A2B"), so quote it -- otherwise the
                        // key=value shape of the line breaks when read next
                        // to `format_ai_wire_send`'s output.
                        let assign = match &label {
                            Some(label) => format!("\"{label}\""),
                            None => "none".to_string(),
                        };
                        Some(format!(
                            "ai slot={} assign={} epoch={}",
                            entry.slot, assign, entry.epoch
                        ))
                    }
                })
                .collect();
            (
                snapshot,
                changes,
                selected_codex_thread,
                retired_slots,
                extra_slots,
                slot_assignment_logs,
                slot0_screenkey_state,
            )
        };
        for message in slot_assignment_logs {
            tracing::debug!(target: AI_DISPLAY_LOG_TARGET, "{message}");
        }
        extras
            .codex_activity
            .set_selected_thread(selected_codex_thread);
        // Stage 1 of `docs/ai-approval-hud-design.md`: display only, no
        // response path. Both clients' unresolved-approval bodies were just
        // fed into `pending_approvals` by `drain_codex_state_changes` /
        // `drain_claude_state_changes` above.
        if let Some(hud) = extras.hud.lock().unwrap().as_ref() {
            hud.update(&app, &extras.codex_activity.pending_approvals());
        }
        match sync_ai_client_state(
            &mut runner,
            ai_snapshot,
            ai_changes,
            ai_client_state_send.entry(0).or_default(),
            now,
            screenkey_state_for_slot(0, slot0_screenkey_state, &extras.hud_responded, now),
        ) {
            Ok(sends) => {
                for send in sends {
                    tracing::debug!(target: AI_DISPLAY_LOG_TARGET, "{}", format_ai_wire_send(&send));
                }
            }
            Err(error) => {
                let msg = format!("AI client state send error: {error}");
                let entry = add_log(&log_entries, &log_counter, "error", &msg);
                update_status(&app, &status, |s| s.last_error = Some(msg));
                let _ = app.emit("log-added", entry);
            }
        }
        for slot in retired_slots {
            match sync_ai_client_state_slot(
                &mut runner,
                slot,
                inactive_ai_snapshot(),
                [AiClientStateChange {
                    state: inactive_ai_snapshot(),
                    reason: rawhid_host_core::AiClientStateChangeReason::SessionEnded,
                }],
                ai_client_state_send.entry(slot).or_default(),
                now,
                AiScreenKeyState::Normal,
            ) {
                Ok(sends) => {
                    for send in sends {
                        tracing::debug!(target: AI_DISPLAY_LOG_TARGET, "{}", format_ai_wire_send(&send));
                    }
                }
                Err(error) => {
                    tracing::warn!(slot, "AI display slot clear error: {error}");
                }
            }
            ai_client_state_send.remove(&slot);
        }
        for (slot, snapshot, changes, epoch, screenkey_state) in extra_slots {
            let screenkey_state =
                screenkey_state_for_slot(slot, screenkey_state, &extras.hud_responded, now);
            let tracker = ai_client_state_send.entry(slot).or_default();
            let assignment_changed = tracker.last_slot_epoch != Some(epoch);
            let result = if assignment_changed && changes.is_empty() {
                // A slot assignment may have happened without a state transition.
                // Sending its full snapshot immediately avoids waiting for heartbeat.
                sync_ai_client_state_slot(
                    &mut runner,
                    slot,
                    snapshot,
                    [AiClientStateChange {
                        state: snapshot,
                        reason: rawhid_host_core::AiClientStateChangeReason::SessionStarted,
                    }],
                    tracker,
                    now,
                    screenkey_state,
                )
            } else {
                sync_ai_client_state_slot(
                    &mut runner,
                    slot,
                    snapshot,
                    changes,
                    tracker,
                    now,
                    screenkey_state,
                )
            };
            match result {
                Ok(sends) => {
                    for send in sends {
                        tracing::debug!(target: AI_DISPLAY_LOG_TARGET, "{}", format_ai_wire_send(&send));
                    }
                    tracker.last_slot_epoch = Some(epoch);
                }
                Err(error) => {
                    tracing::warn!(slot, "AI display slot send error: {error}");
                }
            }
        }

        // Wait for the next control-loop tick, draining uplink every uplink_interval_ms.
        let mut should_stop = false;
        let deadline = Instant::now() + interval;
        'wait: loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break 'wait;
            }
            let wait = uplink_interval.min(remaining);
            match rx.recv_timeout(wait) {
                Ok(command) => {
                    if process_command(
                        command,
                        &app,
                        &mut runner,
                        &mut interval,
                        &mut uplink_interval,
                        &status,
                        &log_entries,
                        &log_counter,
                        &extras,
                        &mut actions_cfg,
                        &mut automation_enabled,
                        &mut ai_client_state_send,
                    ) {
                        should_stop = true;
                    }
                    break 'wait;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    should_stop = true;
                    break 'wait;
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }
            if automation_enabled {
                runner.drain_uplink_only();
                if handle_uplink_events(
                    &app,
                    &mut runner,
                    &actions_cfg,
                    &extras,
                    &status,
                    &log_entries,
                    &log_counter,
                    &mut ai_client_state_send,
                ) {
                    automation_enabled = false;
                    update_status(&app, &status, |s| {
                        s.running = false;
                        s.current_layer = None;
                        s.current_rule = None;
                    });
                    break 'wait;
                }
            } else {
                runner.discard_uplink_only();
            }
        }
        if should_stop {
            break;
        }
    }

    if let Ok(mut store) = extras.key_stats.lock() {
        store.flush_all();
    }

    update_status(&app, &status, |s| {
        s.running = false;
        s.connected_devices = 0;
        s.connected_device_names = Vec::new();
        s.host_link_devices = Vec::new();
        s.current_layer = None;
        s.current_rule = None;
        s.device_battery = Vec::new();
        s.device_layers = Vec::new();
    });

    let entry = add_log(
        &log_entries,
        &log_counter,
        "info",
        "Host Link worker stopped",
    );
    let _ = app.emit("log-added", entry);
}

#[tauri::command]
pub fn get_key_stats(
    device_uid: String,
    period: String,
    state: State<AppState>,
) -> Result<KeyStatsSummary, String> {
    let period = StatsPeriod::parse(&period).ok_or_else(|| "invalid_period".to_string())?;
    let today = chrono::Local::now().format("%Y-%m-%d").to_string();
    let mut store = state.key_stats.lock().map_err(|_| "stats_unavailable")?;
    Ok(store.summary(&device_uid, period, &today))
}

#[tauri::command]
pub fn list_key_stats_devices(state: State<AppState>) -> Vec<String> {
    state
        .key_stats
        .lock()
        .map(|store| store.device_keys())
        .unwrap_or_default()
}

/// Debug-only helper to exercise the uplink path without firmware. Accepts a
/// 64-byte packet payload as hex and feeds it through the monitor loop.
#[tauri::command]
pub fn debug_inject_uplink(
    device_uid: String,
    payload_hex: String,
    state: State<AppState>,
) -> Result<(), String> {
    #[cfg(not(debug_assertions))]
    {
        let _ = (device_uid, payload_hex, state);
        Err("debug_only".to_string())
    }
    #[cfg(debug_assertions)]
    {
        let bytes = decode_hex(&payload_hex)?;
        if bytes.len() != rawhid_host_core::PACKET_SIZE {
            return Err(format!(
                "payload must be {} bytes, got {}",
                rawhid_host_core::PACKET_SIZE,
                bytes.len()
            ));
        }
        let packet = UplinkPacket::decode_payload(&bytes).map_err(|e| e.to_string())?;
        let uid = device_uid
            .strip_prefix("uid:")
            .and_then(|hex| u64::from_str_radix(hex, 16).ok());
        let device = DeviceInfo {
            path: format!("debug:{device_uid}"),
            vendor_id: 0,
            product_id: 0,
            usage_page: 0,
            usage: 0,
            connection_type: rawhid_host_core::hid::DeviceConnectionType::Unknown,
            manufacturer: None,
            product: Some("debug".to_string()),
            serial_number: None,
            capabilities: packet.required_capability(),
            device_uid_hash: uid,
        };
        let tx = state.monitor_tx.lock().unwrap();
        match tx.as_ref() {
            Some(tx) => tx
                .send(MonitorCommand::InjectUplink(device, packet))
                .map_err(|_| "monitor_loop_gone".to_string()),
            None => Err("not_running".to_string()),
        }
    }
}

#[cfg(debug_assertions)]
fn decode_hex(input: &str) -> Result<Vec<u8>, String> {
    let cleaned: String = input.chars().filter(|c| !c.is_whitespace()).collect();
    if cleaned.len() % 2 != 0 {
        return Err("hex string must have even length".to_string());
    }
    (0..cleaned.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&cleaned[i..i + 2], 16).map_err(|e| e.to_string()))
        .collect()
}

#[tauri::command]
pub fn get_app_icons(paths: Vec<String>) -> std::collections::HashMap<String, String> {
    let mut icons = std::collections::HashMap::new();
    for path in paths {
        if let Some(data_url) = icon::app_icon_data_url(&path) {
            icons.insert(path, data_url);
        }
    }
    icons
}

#[tauri::command]
pub fn get_launch_at_login() -> bool {
    startup::is_launch_at_login()
}

#[tauri::command]
pub fn set_launch_at_login(enabled: bool) -> Result<(), String> {
    startup::set_launch_at_login(enabled)
}

fn refresh_error_code(error: AiUsageRefreshError) -> String {
    match error {
        AiUsageRefreshError::InProgress => "refresh_in_progress",
        AiUsageRefreshError::Stopped => "not_running",
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rawhid_host_core::claude_activity::ClaudeStateChangeReason;
    use rawhid_host_core::packet::{AiActivityState, AiClientType, AiClientVariant, AiWorkPhase};
    use rawhid_host_core::runner::{DeviceBatterySource, DeviceBatteryStatus};
    use rawhid_host_core::studio::{StudioLayer, StudioLayoutSource};

    #[test]
    fn launcher_update_preserves_every_other_config_value() {
        let original = AppConfig::default();
        let launcher = CodexLauncherConfig {
            environment: CodexLaunchEnvironment::Windows,
            windows_project_directory: Some(r"C:\Work\project".to_string()),
            ..CodexLauncherConfig::default()
        };

        let updated = config_with_codex_launcher(original.clone(), launcher.clone());
        assert_eq!(updated.ai_client.codex_launcher, launcher);

        let mut without_launcher = updated;
        without_launcher.ai_client.codex_launcher = original.ai_client.codex_launcher.clone();
        assert_eq!(without_launcher, original);
    }

    #[test]
    fn codex_broker_config_carries_version_check_setting() {
        let mut config = AppConfig::default();
        config.ai_client.codex.version_check_enabled = true;

        let broker = codex_broker_config(&config).unwrap();

        assert!(broker.version_check_enabled);
    }

    #[derive(Default)]
    struct FakeAiClientTransport {
        device_generation: u64,
        capable: bool,
        claude_capable: bool,
        sent: Vec<AiClientStatePacket>,
        work_phase_only: Vec<bool>,
    }

    impl AiClientStateTransport for FakeAiClientTransport {
        fn device_generation(&self) -> u64 {
            self.device_generation
        }

        fn has_ai_client_state_device(&self, client_type: AiClientType) -> bool {
            self.capable && (client_type != AiClientType::ClaudeCode || self.claude_capable)
        }

        fn send_ai_client_state(
            &mut self,
            packet: AiClientStatePacket,
            work_phase_only: bool,
        ) -> Result<usize, String> {
            self.sent.push(packet);
            self.work_phase_only.push(work_phase_only);
            Ok(1)
        }
    }

    fn ai_state(activity_state: AiActivityState, revision: u16) -> AiClientStateSnapshot {
        AiClientStateSnapshot {
            client_type: AiClientType::Codex,
            client_variant: AiClientVariant::Cli,
            session_active: activity_state != AiActivityState::None,
            activity_state,
            work_phase: AiWorkPhase::Unspecified,
            revision,
        }
    }

    fn claude_state(activity_state: AiActivityState, revision: u16) -> AiClientStateSnapshot {
        AiClientStateSnapshot {
            client_type: AiClientType::ClaudeCode,
            client_variant: AiClientVariant::Cli,
            session_active: activity_state != AiActivityState::None,
            activity_state,
            work_phase: AiWorkPhase::Unspecified,
            revision,
        }
    }

    fn claude_session(
        session_id: &str,
        activity_state: AiActivityState,
        revision: u16,
    ) -> ClaudeSessionSnapshot {
        ClaudeSessionSnapshot {
            launch_id: "launch-1".to_string(),
            session_id: session_id.to_string(),
            session_active: true,
            activity_state,
            work_phase: AiWorkPhase::Unspecified,
            desynchronized: false,
            revision,
            registration_order: u64::from(revision),
        }
    }

    fn codex_session(
        thread_id: &str,
        activity_state: AiActivityState,
        revision: u16,
    ) -> CodexSessionSnapshot {
        CodexSessionSnapshot {
            thread_id: thread_id.to_string(),
            owner_connection_id: "connection-1".to_string(),
            terminal_target_id: format!("codex-{thread_id}"),
            state: ai_state(activity_state, revision),
            registration_order: u64::from(revision),
            is_display_target: true,
        }
    }

    #[test]
    fn ai_selection_cycles_from_codex_to_claude_and_forces_snapshot() {
        let codex = codex_session("thread-1", AiActivityState::Working, 5);
        let claude = claude_session("session-1", AiActivityState::WaitingApproval, 10);
        let mut selection = AiDisplaySelection::default();
        let mut epoch = 0;

        let (snapshot, changes) = selected_ai_output(
            vec![codex.clone()],
            Vec::new(),
            vec![claude.clone()],
            Vec::new(),
            &mut selection,
            &mut epoch,
        );
        assert_eq!(snapshot.client_type, AiClientType::Codex);
        assert_eq!(changes.len(), 1);

        selection.cycle();
        let (snapshot, changes) = selected_ai_output(
            vec![codex],
            Vec::new(),
            vec![claude],
            Vec::new(),
            &mut selection,
            &mut epoch,
        );
        assert_eq!(snapshot.client_type, AiClientType::ClaudeCode);
        assert_eq!(snapshot.activity_state, AiActivityState::WaitingApproval);
        assert_eq!(changes.len(), 1);
    }

    #[test]
    fn ai_output_ignores_changes_from_a_non_selected_session() {
        let codex = codex_session("thread-1", AiActivityState::Working, 5);
        let first = claude_session("session-1", AiActivityState::Available, 10);
        let second = claude_session("session-2", AiActivityState::Working, 20);
        let other_change = ClaudeStateChange {
            state: claude_session("session-1", AiActivityState::Working, 20),
            reason: ClaudeStateChangeReason::TurnStarted,
        };
        let mut selection = AiDisplaySelection::default();
        let mut epoch = 0;
        selected_ai_output(
            vec![codex.clone()],
            Vec::new(),
            vec![first.clone(), second.clone()],
            Vec::new(),
            &mut selection,
            &mut epoch,
        );
        selection.cycle();
        selected_ai_output(
            vec![codex.clone()],
            Vec::new(),
            vec![first.clone(), second.clone()],
            Vec::new(),
            &mut selection,
            &mut epoch,
        );
        let (_, changes) = selected_ai_output(
            vec![codex],
            Vec::new(),
            vec![first, second],
            vec![other_change],
            &mut selection,
            &mut epoch,
        );
        assert!(changes.is_empty());
    }

    #[test]
    fn ai_output_cycles_multiple_codex_and_claude_sessions() {
        let first = codex_session("thread-1", AiActivityState::Working, 1);
        let second = codex_session("thread-2", AiActivityState::Completed, 2);
        let claude = claude_session("session-1", AiActivityState::WaitingInput, 3);
        let mut selection = AiDisplaySelection::default();
        let mut epoch = 0;

        selected_ai_output(
            vec![first.clone(), second.clone()],
            Vec::new(),
            vec![claude.clone()],
            Vec::new(),
            &mut selection,
            &mut epoch,
        );
        selection.cycle();
        let (snapshot, _) = selected_ai_output(
            vec![first.clone(), second.clone()],
            Vec::new(),
            vec![claude.clone()],
            Vec::new(),
            &mut selection,
            &mut epoch,
        );
        assert_eq!(snapshot.activity_state, AiActivityState::Completed);

        selection.cycle();
        let (snapshot, _) = selected_ai_output(
            vec![first, second],
            Vec::new(),
            vec![claude],
            Vec::new(),
            &mut selection,
            &mut epoch,
        );
        assert_eq!(snapshot.client_type, AiClientType::ClaudeCode);
        assert_eq!(snapshot.activity_state, AiActivityState::WaitingInput);
    }

    #[test]
    fn ai_output_ignores_non_selected_codex_changes() {
        let first = codex_session("thread-1", AiActivityState::Working, 1);
        let second = codex_session("thread-2", AiActivityState::Available, 2);
        let mut selection = AiDisplaySelection::default();
        let mut epoch = 0;
        selected_ai_output(
            vec![first.clone(), second.clone()],
            Vec::new(),
            Vec::new(),
            Vec::new(),
            &mut selection,
            &mut epoch,
        );

        let (_, changes) = selected_ai_output(
            vec![first, second.clone()],
            vec![CodexStateChange {
                thread_id: second.thread_id,
                terminal_target_id: second.terminal_target_id,
                state: ai_state(AiActivityState::Working, 3),
                reason: rawhid_host_core::AiClientStateChangeReason::TurnStarted,
            }],
            Vec::new(),
            Vec::new(),
            &mut selection,
            &mut epoch,
        );
        assert!(changes.is_empty());
    }

    #[test]
    fn codex_changes_from_a_non_display_thread_never_reach_the_slot() {
        // Regression test: Codex CLI spawns a second, throwaway thread right
        // after the first user message (to auto-generate the conversation
        // title). That thread shares the same `terminal_target_id` (same
        // CLI/terminal) as the real session but is never the display
        // target. Before this fix, filtering changes by
        // `terminal_target_id` alone let that throwaway thread's
        // transitions (e.g. its own `TurnCompleted`) flash onto the slot
        // that should keep showing the real, still-active session.
        let display = codex_session("thread-1", AiActivityState::Working, 1);
        let mut hidden = codex_session("thread-2", AiActivityState::Working, 2);
        hidden.terminal_target_id = display.terminal_target_id.clone();
        hidden.is_display_target = false;

        let mut selection = AiDisplaySelection::default();
        let mut epoch = 0;
        selected_ai_output(
            vec![display.clone(), hidden.clone()],
            Vec::new(),
            Vec::new(),
            Vec::new(),
            &mut selection,
            &mut epoch,
        );

        let (_, changes) = selected_ai_output(
            vec![display, hidden.clone()],
            vec![CodexStateChange {
                thread_id: hidden.thread_id,
                terminal_target_id: hidden.terminal_target_id,
                state: ai_state(AiActivityState::Completed, 3),
                reason: rawhid_host_core::AiClientStateChangeReason::TurnCompleted,
            }],
            Vec::new(),
            Vec::new(),
            &mut selection,
            &mut epoch,
        );
        assert!(changes.is_empty());
    }

    #[test]
    fn codex_changes_from_the_display_thread_still_reach_the_slot() {
        // Companion to the regression test above: make sure the fix doesn't
        // over-filter -- changes from the thread actually being displayed
        // must still come through even while a hidden, same-terminal thread
        // exists alongside it.
        let display = codex_session("thread-1", AiActivityState::Working, 1);
        let mut hidden = codex_session("thread-2", AiActivityState::Working, 2);
        hidden.terminal_target_id = display.terminal_target_id.clone();
        hidden.is_display_target = false;

        let mut selection = AiDisplaySelection::default();
        let mut epoch = 0;
        selected_ai_output(
            vec![display.clone(), hidden.clone()],
            Vec::new(),
            Vec::new(),
            Vec::new(),
            &mut selection,
            &mut epoch,
        );

        let (_, changes) = selected_ai_output(
            vec![display.clone(), hidden],
            vec![CodexStateChange {
                thread_id: display.thread_id,
                terminal_target_id: display.terminal_target_id,
                state: ai_state(AiActivityState::Completed, 3),
                reason: rawhid_host_core::AiClientStateChangeReason::TurnCompleted,
            }],
            Vec::new(),
            Vec::new(),
            &mut selection,
            &mut epoch,
        );
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].state.activity_state, AiActivityState::Completed);
    }

    fn ai_change(state: AiClientStateSnapshot) -> AiClientStateChange {
        AiClientStateChange {
            state,
            reason: rawhid_host_core::AiClientStateChangeReason::TurnStarted,
        }
    }

    fn work_phase_change(state: AiClientStateSnapshot) -> AiClientStateChange {
        AiClientStateChange {
            state,
            reason: rawhid_host_core::AiClientStateChangeReason::WorkPhaseChanged,
        }
    }

    #[test]
    fn ai_client_state_sends_each_pending_change_to_capable_devices() {
        let mut transport = FakeAiClientTransport {
            device_generation: 4,
            capable: true,
            ..Default::default()
        };
        let mut tracker = AiClientStateSendTracker::default();
        let available = ai_state(AiActivityState::Available, 50);
        let working = ai_state(AiActivityState::Working, 51);
        let now = Instant::now();

        sync_ai_client_state(
            &mut transport,
            working,
            vec![ai_change(available), ai_change(working)],
            &mut tracker,
            now,
            AiScreenKeyState::Normal,
        )
        .unwrap();

        assert_eq!(transport.sent.len(), 2);
        assert_eq!(transport.sent[0].activity_state, AiActivityState::Available);
        assert_eq!(transport.sent[0].revision, 50);
        assert_eq!(transport.sent[1].activity_state, AiActivityState::Working);
        assert_eq!(transport.sent[1].revision, 51);
        assert_eq!(tracker.last_device_generation, Some(4));
    }

    #[test]
    fn ai_client_work_phase_change_is_marked_detail_only() {
        let mut transport = FakeAiClientTransport {
            device_generation: 5,
            capable: true,
            ..Default::default()
        };
        let mut tracker = AiClientStateSendTracker::default();
        let mut working = ai_state(AiActivityState::Working, 60);
        working.work_phase = AiWorkPhase::Executing;

        sync_ai_client_state(
            &mut transport,
            working,
            [work_phase_change(working)],
            &mut tracker,
            Instant::now(),
            AiScreenKeyState::Normal,
        )
        .unwrap();

        assert_eq!(transport.work_phase_only, vec![true]);
        assert_eq!(transport.sent[0].revision, 60);
        assert_eq!(transport.sent[0].work_phase, AiWorkPhase::Executing);
    }

    #[test]
    fn ai_client_state_resends_current_snapshot_after_device_generation_changes() {
        let mut transport = FakeAiClientTransport {
            device_generation: 7,
            capable: true,
            ..Default::default()
        };
        let mut tracker = AiClientStateSendTracker::default();
        let snapshot = ai_state(AiActivityState::WaitingApproval, 90);
        let now = Instant::now();

        sync_ai_client_state(
            &mut transport,
            snapshot,
            [],
            &mut tracker,
            now,
            AiScreenKeyState::Normal,
        )
        .unwrap();
        sync_ai_client_state(
            &mut transport,
            snapshot,
            [],
            &mut tracker,
            now,
            AiScreenKeyState::Normal,
        )
        .unwrap();
        transport.device_generation = 8;
        sync_ai_client_state(
            &mut transport,
            snapshot,
            [],
            &mut tracker,
            now,
            AiScreenKeyState::Normal,
        )
        .unwrap();

        assert_eq!(transport.sent.len(), 2);
        assert!(transport
            .sent
            .iter()
            .all(|packet| packet.activity_state == AiActivityState::WaitingApproval));
        assert_eq!(tracker.last_device_generation, Some(8));
    }

    #[test]
    fn ai_client_state_waits_for_a_capable_device_then_sends_latest_snapshot() {
        let mut transport = FakeAiClientTransport {
            device_generation: 3,
            capable: false,
            ..Default::default()
        };
        let mut tracker = AiClientStateSendTracker::default();
        let snapshot = ai_state(AiActivityState::Completed, 12);
        let now = Instant::now();

        sync_ai_client_state(
            &mut transport,
            snapshot,
            vec![ai_change(snapshot)],
            &mut tracker,
            now,
            AiScreenKeyState::Normal,
        )
        .unwrap();
        assert!(transport.sent.is_empty());
        assert_eq!(tracker.last_device_generation, None);

        transport.capable = true;
        sync_ai_client_state(
            &mut transport,
            snapshot,
            [],
            &mut tracker,
            now,
            AiScreenKeyState::Normal,
        )
        .unwrap();
        assert_eq!(transport.sent.len(), 1);
        assert_eq!(transport.sent[0].activity_state, AiActivityState::Completed);
        assert_eq!(transport.sent[0].revision, 12);
    }

    #[test]
    fn claude_code_state_waits_for_a_device_with_renderer_capability() {
        let mut transport = FakeAiClientTransport {
            device_generation: 10,
            capable: true,
            claude_capable: false,
            ..Default::default()
        };
        let mut tracker = AiClientStateSendTracker::default();
        let snapshot = claude_state(AiActivityState::Available, 20);
        let now = Instant::now();

        sync_ai_client_state(
            &mut transport,
            snapshot,
            [ai_change(snapshot)],
            &mut tracker,
            now,
            AiScreenKeyState::Normal,
        )
        .unwrap();
        assert!(transport.sent.is_empty());
        assert_eq!(tracker.last_device_generation, None);

        transport.claude_capable = true;
        sync_ai_client_state(
            &mut transport,
            snapshot,
            [],
            &mut tracker,
            now,
            AiScreenKeyState::Normal,
        )
        .unwrap();
        assert_eq!(transport.sent.len(), 1);
        assert_eq!(transport.sent[0].client_type, AiClientType::ClaudeCode);
    }

    #[test]
    fn ai_client_state_heartbeat_reuses_revision_every_five_seconds() {
        let mut transport = FakeAiClientTransport {
            device_generation: 9,
            capable: true,
            ..Default::default()
        };
        let mut tracker = AiClientStateSendTracker::default();
        let snapshot = ai_state(AiActivityState::Completed, 1234);
        let now = Instant::now();

        sync_ai_client_state(
            &mut transport,
            snapshot,
            [],
            &mut tracker,
            now,
            AiScreenKeyState::Normal,
        )
        .unwrap();
        sync_ai_client_state(
            &mut transport,
            snapshot,
            [],
            &mut tracker,
            now + Duration::from_secs(4),
            AiScreenKeyState::Normal,
        )
        .unwrap();
        sync_ai_client_state(
            &mut transport,
            snapshot,
            [],
            &mut tracker,
            now + Duration::from_secs(5),
            AiScreenKeyState::Normal,
        )
        .unwrap();

        assert_eq!(transport.sent.len(), 2);
        assert!(transport.sent.iter().all(|packet| packet.revision == 1234));
    }

    #[test]
    fn sync_ai_client_state_carries_the_screenkey_state_into_every_sent_packet() {
        let mut transport = FakeAiClientTransport {
            device_generation: 1,
            capable: true,
            ..Default::default()
        };
        let mut tracker = AiClientStateSendTracker::default();
        let available = ai_state(AiActivityState::Available, 1);
        let working = ai_state(AiActivityState::WaitingApproval, 2);
        let now = Instant::now();

        sync_ai_client_state(
            &mut transport,
            working,
            vec![ai_change(available), ai_change(working)],
            &mut tracker,
            now,
            AiScreenKeyState::WaitingTarget,
        )
        .unwrap();

        assert_eq!(transport.sent.len(), 2);
        assert!(transport
            .sent
            .iter()
            .all(|packet| packet.screenkey_state == AiScreenKeyState::WaitingTarget));
    }

    #[test]
    fn sync_ai_client_state_sends_immediately_when_only_screenkey_state_changes() {
        // Two sessions both stuck at `WaitingApproval` never produce an
        // `AiClientStateChange` when the HUD target switches between them --
        // `activity_state` never moves. Without forcing a send on a
        // `screenkey_state` mismatch, this slot's display would sit stale
        // until the next `AI_CLIENT_STATE_RESEND_INTERVAL` heartbeat (5s).
        let mut transport = FakeAiClientTransport {
            device_generation: 1,
            capable: true,
            ..Default::default()
        };
        let mut tracker = AiClientStateSendTracker::default();
        let snapshot = ai_state(AiActivityState::WaitingApproval, 42);
        let now = Instant::now();

        sync_ai_client_state(
            &mut transport,
            snapshot,
            [],
            &mut tracker,
            now,
            AiScreenKeyState::WaitingOther,
        )
        .unwrap();
        assert_eq!(transport.sent.len(), 1);
        assert_eq!(
            transport.sent[0].screenkey_state,
            AiScreenKeyState::WaitingOther
        );

        // No activity change, no device-generation change, and well within
        // the 5s heartbeat window -- only `screenkey_state` differs.
        sync_ai_client_state(
            &mut transport,
            snapshot,
            [],
            &mut tracker,
            now + Duration::from_millis(50),
            AiScreenKeyState::WaitingTarget,
        )
        .unwrap();

        assert_eq!(transport.sent.len(), 2);
        assert_eq!(
            transport.sent[1].screenkey_state,
            AiScreenKeyState::WaitingTarget
        );
    }

    #[test]
    fn sync_ai_client_state_suppresses_resend_when_nothing_changed() {
        // Existing behavior must survive: identical activity state AND
        // identical screenkey_state, well before the heartbeat interval,
        // sends nothing on the second call.
        let mut transport = FakeAiClientTransport {
            device_generation: 1,
            capable: true,
            ..Default::default()
        };
        let mut tracker = AiClientStateSendTracker::default();
        let snapshot = ai_state(AiActivityState::WaitingApproval, 7);
        let now = Instant::now();

        sync_ai_client_state(
            &mut transport,
            snapshot,
            [],
            &mut tracker,
            now,
            AiScreenKeyState::WaitingOther,
        )
        .unwrap();
        assert_eq!(transport.sent.len(), 1);

        sync_ai_client_state(
            &mut transport,
            snapshot,
            [],
            &mut tracker,
            now + Duration::from_millis(50),
            AiScreenKeyState::WaitingOther,
        )
        .unwrap();

        assert_eq!(transport.sent.len(), 1, "unchanged state must not resend");
    }

    #[test]
    fn compute_screenkey_state_covers_target_other_and_normal() {
        assert_eq!(
            compute_screenkey_state(AiActivityState::WaitingApproval, true),
            AiScreenKeyState::WaitingTarget
        );
        assert_eq!(
            compute_screenkey_state(AiActivityState::WaitingInput, true),
            AiScreenKeyState::WaitingTarget
        );
        assert_eq!(
            compute_screenkey_state(AiActivityState::WaitingApproval, false),
            AiScreenKeyState::WaitingOther
        );
        assert_eq!(
            compute_screenkey_state(AiActivityState::Working, true),
            AiScreenKeyState::Normal
        );
        assert_eq!(
            compute_screenkey_state(AiActivityState::None, false),
            AiScreenKeyState::Normal
        );
    }

    #[test]
    fn is_hud_target_requires_both_sides_present_and_equal() {
        let a = Some(HudTargetSession::Codex {
            connection_id: "connection-a".to_string(),
            thread_id: "thread-a".to_string(),
        });
        let b = Some(HudTargetSession::Codex {
            connection_id: "connection-b".to_string(),
            thread_id: "thread-b".to_string(),
        });
        assert!(is_hud_target(&a, &a));
        assert!(!is_hud_target(&a, &b));
        assert!(!is_hud_target(&a, &None));
        assert!(!is_hud_target(&None, &a));
        assert!(!is_hud_target(&None, &None));
    }

    /// Client-independent: `is_hud_target` must compare Claude Code variants
    /// the same way it compares Codex ones, and a Codex slot must never be
    /// treated as a match for a Claude Code HUD target (or vice versa) even
    /// though both are `Some` -- `HudTargetSession`'s derived `PartialEq`
    /// already guarantees this across enum variants, but this pins the
    /// behavior at the call site that actually matters for ScreenKey display.
    #[test]
    fn is_hud_target_matches_claude_variants_and_rejects_cross_client_pairs() {
        let claude_a = Some(HudTargetSession::Claude {
            launch_id: "launch-1".to_string(),
            session_id: "session-1".to_string(),
        });
        let claude_b = Some(HudTargetSession::Claude {
            launch_id: "launch-1".to_string(),
            session_id: "session-2".to_string(),
        });
        let codex = Some(HudTargetSession::Codex {
            connection_id: "connection-a".to_string(),
            thread_id: "thread-a".to_string(),
        });
        assert!(is_hud_target(&claude_a, &claude_a));
        assert!(!is_hud_target(&claude_a, &claude_b));
        assert!(!is_hud_target(&claude_a, &codex));
        assert!(!is_hud_target(&codex, &claude_a));
    }

    /// End-to-end through the two pieces the monitor loop actually composes
    /// for a Claude Code slot: `actions::hud_target_for_slot` resolves the
    /// slot's live session, and `compute_screenkey_state` turns that
    /// waiting/target status into `WaitingTarget` when it matches the HUD's
    /// current target, or `WaitingOther` when it does not. Before
    /// `hud_target_for_slot` existed, ScreenKey's target display was wired up
    /// for Codex only -- a Claude Code slot never reached a target comparison
    /// at all, by stage 3's deliberate scope limit, so it could never show
    /// `WaitingTarget`. This test pins the fix: a Claude Code slot now
    /// reaches `WaitingTarget` exactly when it is the HUD's target, the same
    /// as a Codex slot always could.
    #[test]
    fn claude_slot_reaches_waiting_target_only_when_it_is_the_hud_target() {
        let claude_terminal_targets =
            BTreeMap::from([("launch-1".to_string(), "terminal-a".to_string())]);
        let claude_snapshots = vec![claude_session(
            "session-1",
            AiActivityState::WaitingApproval,
            1,
        )];
        let assigned = Some(AiDisplayTarget::Claude {
            terminal_target_id: "terminal-a".to_string(),
        });

        let slot_target = actions::hud_target_for_slot(
            assigned,
            &[],
            &claude_snapshots,
            &claude_terminal_targets,
        );
        assert_eq!(
            slot_target,
            Some(HudTargetSession::Claude {
                launch_id: "launch-1".to_string(),
                session_id: "session-1".to_string(),
            })
        );

        // The slot's own resolved session is the HUD's current target.
        assert_eq!(
            compute_screenkey_state(
                AiActivityState::WaitingApproval,
                is_hud_target(&slot_target, &slot_target)
            ),
            AiScreenKeyState::WaitingTarget
        );

        // A different session is currently the HUD's target: the slot still
        // shows as waiting, but not as the target.
        let other_hud_target = Some(HudTargetSession::Claude {
            launch_id: "launch-other".to_string(),
            session_id: "session-other".to_string(),
        });
        assert_eq!(
            compute_screenkey_state(
                AiActivityState::WaitingApproval,
                is_hud_target(&slot_target, &other_hud_target)
            ),
            AiScreenKeyState::WaitingOther
        );
    }

    #[test]
    fn screenkey_state_for_slot_holds_responded_then_reverts_to_computed() {
        let hud_responded = Mutex::new(None);
        let now = Instant::now();

        // No hold recorded yet: the computed value passes through unchanged.
        assert_eq!(
            screenkey_state_for_slot(0, AiScreenKeyState::WaitingOther, &hud_responded, now),
            AiScreenKeyState::WaitingOther
        );

        *hud_responded.lock().unwrap() = Some((2, now));

        // A different slot is unaffected by another slot's hold.
        assert_eq!(
            screenkey_state_for_slot(0, AiScreenKeyState::Normal, &hud_responded, now),
            AiScreenKeyState::Normal
        );
        // The held slot is forced to `Responded` regardless of its computed
        // value, right up to the hold boundary.
        assert_eq!(
            screenkey_state_for_slot(
                2,
                AiScreenKeyState::WaitingTarget,
                &hud_responded,
                now + HUD_RESPONDED_HOLD - Duration::from_millis(1),
            ),
            AiScreenKeyState::Responded
        );
        // Once the hold has fully elapsed, the computed value takes over
        // again and the stale entry is cleared.
        assert_eq!(
            screenkey_state_for_slot(
                2,
                AiScreenKeyState::WaitingTarget,
                &hud_responded,
                now + HUD_RESPONDED_HOLD,
            ),
            AiScreenKeyState::WaitingTarget
        );
        assert!(hud_responded.lock().unwrap().is_none());
    }

    #[test]
    fn hud_responded_hold_expiry_forces_a_resend_even_without_an_activity_change() {
        // End-to-end check that `screenkey_state_for_slot`'s hold expiry and
        // `sync_ai_client_state_slot`'s `screenkey_state_changed` forcing
        // compose correctly: the slot's `activity_state` never changes
        // (`Working` throughout, no `AiClientStateChange` at all), yet the
        // tick right after `HUD_RESPONDED_HOLD` elapses must still send,
        // carrying the reverted-to-computed state instead of a stale
        // `Responded`.
        let hud_responded = Mutex::new(Some((0u8, Instant::now())));
        let mut transport = FakeAiClientTransport {
            device_generation: 1,
            capable: true,
            ..Default::default()
        };
        let mut tracker = AiClientStateSendTracker::default();
        let snapshot = ai_state(AiActivityState::Working, 9);
        let now = Instant::now();

        let held = screenkey_state_for_slot(0, AiScreenKeyState::Normal, &hud_responded, now);
        assert_eq!(held, AiScreenKeyState::Responded);
        sync_ai_client_state(&mut transport, snapshot, [], &mut tracker, now, held).unwrap();
        assert_eq!(transport.sent.len(), 1);
        assert_eq!(
            transport.sent[0].screenkey_state,
            AiScreenKeyState::Responded
        );

        let after_hold = screenkey_state_for_slot(
            0,
            AiScreenKeyState::Normal,
            &hud_responded,
            now + HUD_RESPONDED_HOLD,
        );
        assert_eq!(after_hold, AiScreenKeyState::Normal);
        sync_ai_client_state(
            &mut transport,
            snapshot,
            [],
            &mut tracker,
            now + HUD_RESPONDED_HOLD + Duration::from_millis(1),
            after_hold,
        )
        .unwrap();

        assert_eq!(transport.sent.len(), 2, "hold expiry must force a resend");
        assert_eq!(transport.sent[1].screenkey_state, AiScreenKeyState::Normal);
    }

    #[test]
    fn target_switch_computes_waiting_target_for_the_new_slot_and_waiting_other_for_the_old_one() {
        // Both sessions sit at `WaitingApproval` throughout a
        // `HostActionKind::SelectHudTarget` switch -- exactly the scenario
        // that motivated `send_screenkey_state_immediately`, since neither
        // slot's `activity_state` moves. The new slot's own resolved Codex
        // target now matches the HUD's target; the old slot's does not.
        let old_target = Some(HudTargetSession::Codex {
            connection_id: "connection-a".to_string(),
            thread_id: "thread-a".to_string(),
        });
        let new_target = Some(HudTargetSession::Codex {
            connection_id: "connection-b".to_string(),
            thread_id: "thread-b".to_string(),
        });

        // Before the switch: HUD target is the old slot's session.
        assert_eq!(
            compute_screenkey_state(
                AiActivityState::WaitingApproval,
                is_hud_target(&old_target, &old_target),
            ),
            AiScreenKeyState::WaitingTarget
        );
        assert_eq!(
            compute_screenkey_state(
                AiActivityState::WaitingApproval,
                is_hud_target(&new_target, &old_target),
            ),
            AiScreenKeyState::WaitingOther
        );

        // After the switch: HUD target is the new slot's session. The old
        // slot, still `WaitingApproval`, now reads `WaitingOther`; the new
        // slot reads `WaitingTarget`.
        assert_eq!(
            compute_screenkey_state(
                AiActivityState::WaitingApproval,
                is_hud_target(&new_target, &new_target),
            ),
            AiScreenKeyState::WaitingTarget
        );
        assert_eq!(
            compute_screenkey_state(
                AiActivityState::WaitingApproval,
                is_hud_target(&old_target, &new_target),
            ),
            AiScreenKeyState::WaitingOther
        );
    }

    #[test]
    fn a_target_switch_immediate_send_never_overrides_an_active_responded_hold() {
        // A slot within its `HUD_RESPONDED_HOLD` window must keep showing
        // `Responded` even when the *reason* for this tick's immediate send
        // is an unrelated `HudTargetSelected` touching a different slot.
        // `send_screenkey_state_immediately` computes each slot independently
        // and always routes through `screenkey_state_for_slot`, so the cause
        // of the send never enters this decision.
        let hud_responded = Mutex::new(Some((3u8, Instant::now())));
        let now = Instant::now();

        // Slot 3 just answered and is mid-hold; left to its own computed
        // value it would already be back to `WaitingOther` (session resolved,
        // no longer the HUD target) -- but the hold must still win.
        assert_eq!(
            screenkey_state_for_slot(3, AiScreenKeyState::WaitingOther, &hud_responded, now),
            AiScreenKeyState::Responded
        );
        // A different slot in the same immediate send is unaffected and gets
        // its own freshly computed value.
        assert_eq!(
            screenkey_state_for_slot(5, AiScreenKeyState::WaitingTarget, &hud_responded, now),
            AiScreenKeyState::WaitingTarget
        );
    }

    #[test]
    fn previously_targeted_slot_finds_the_other_waiting_target_slot() {
        let mut trackers = BTreeMap::<u8, AiClientStateSendTracker>::new();
        trackers.insert(2, AiClientStateSendTracker::default());
        trackers.entry(2).or_default().last_screenkey_state = Some(AiScreenKeyState::WaitingTarget);

        assert_eq!(previously_targeted_slot(&trackers, 5), Some(2));
    }

    #[test]
    fn previously_targeted_slot_is_none_when_no_other_slot_was_the_target() {
        let mut trackers = BTreeMap::<u8, AiClientStateSendTracker>::new();
        trackers.entry(1).or_default().last_screenkey_state = Some(AiScreenKeyState::WaitingOther);
        trackers.entry(2).or_default().last_screenkey_state = Some(AiScreenKeyState::Normal);

        assert_eq!(previously_targeted_slot(&trackers, 9), None);
    }

    #[test]
    fn previously_targeted_slot_excludes_the_new_slot_itself_to_avoid_a_duplicate() {
        // If the only tracker recorded as `WaitingTarget` is the slot that
        // was *just* selected (a same-slot "switch", or a stale leftover from
        // before this exact slot became the target), it must not be returned
        // -- the caller would otherwise resend the same slot twice.
        let mut trackers = BTreeMap::<u8, AiClientStateSendTracker>::new();
        trackers.entry(4).or_default().last_screenkey_state = Some(AiScreenKeyState::WaitingTarget);

        assert_eq!(previously_targeted_slot(&trackers, 4), None);
    }

    #[test]
    fn studio_probe_releases_only_lost_connections() {
        assert!(studio_session_connection_lost(&StudioError::Disconnected));
        assert!(studio_session_connection_lost(&StudioError::Timeout));
        assert!(!studio_session_connection_lost(&StudioError::RpcFailed));
        assert!(!studio_session_connection_lost(
            &StudioError::UnsavedChangesExist
        ));
    }

    fn encoder_get_bindings(
        source: EncoderBindingSource,
        cw_param1: u32,
        ccw_param1: u32,
    ) -> EncoderGetBindings {
        EncoderGetBindings {
            layer_id: 10,
            encoder_id: 0,
            source,
            flags: EncoderBindingFlags::default(),
            cw_binding: EncoderBinding {
                behavior_id: 1,
                param1: cw_param1,
                param2: 0,
            },
            ccw_binding: EncoderBinding {
                behavior_id: 1,
                param1: ccw_param1,
                param2: 0,
            },
        }
    }

    fn battery_status(sources: Vec<DeviceBatterySource>) -> MonitorStatus {
        let mut status = MonitorStatus::default();
        status.device_battery = vec![DeviceBatteryStatus {
            device_key: "uid:00000000000000aa".to_string(),
            serial_number: Some("serial-a".to_string()),
            product: Some("Test Keyboard".to_string()),
            sources,
            updated_unix: 0,
        }];
        status
    }

    fn reset_snapshot(layer_ids: &[u32]) -> StudioKeymapSnapshot {
        StudioKeymapSnapshot {
            device_id: "serial:test".to_string(),
            device_name: "Test Keyboard".to_string(),
            connection_type: "usb".to_string(),
            lock_state: rawhid_host_core::studio::StudioLockState::Unlocked,
            physical_layouts: Vec::new(),
            selected_physical_layout_index: None,
            selected_physical_layout_name: None,
            layout_source: StudioLayoutSource::GridFallback,
            selected_layout_keys: Vec::new(),
            layers: layer_ids
                .iter()
                .enumerate()
                .map(|(index, id)| StudioLayer {
                    index,
                    id: *id,
                    name: format!("Layer {id}"),
                    bindings: Vec::new(),
                })
                .collect(),
            updated_ms: 0,
        }
    }

    #[test]
    fn tray_tooltip_hides_unknown_battery_sources() {
        let status = battery_status(vec![
            DeviceBatterySource {
                source: 0,
                level: None,
            },
            DeviceBatterySource {
                source: 1,
                level: Some(78),
            },
        ]);

        assert_eq!(
            build_tray_tooltip(&status),
            "Keylink Studio\nTest Keyboard: P1:78%"
        );
    }

    #[test]
    fn tray_tooltip_shows_unknown_when_no_source_is_available() {
        let status = battery_status(vec![
            DeviceBatterySource {
                source: 0,
                level: None,
            },
            DeviceBatterySource {
                source: 1,
                level: None,
            },
        ]);

        assert_eq!(
            build_tray_tooltip(&status),
            "Keylink Studio\nTest Keyboard: ?"
        );
    }

    #[test]
    fn tray_tooltip_uses_central_and_peripheral_labels() {
        let status = battery_status(vec![
            DeviceBatterySource {
                source: 0,
                level: Some(92),
            },
            DeviceBatterySource {
                source: 1,
                level: Some(78),
            },
            DeviceBatterySource {
                source: 2,
                level: Some(76),
            },
        ]);

        assert_eq!(
            build_tray_tooltip(&status),
            "Keylink Studio\nTest Keyboard: C:92% P1:78% P2:76%"
        );
    }

    #[test]
    fn failed_export_preparation_does_not_replace_destination() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "keylink-studio-export-{}-{}.json",
            std::process::id(),
            nonce
        ));
        std::fs::write(&path, "existing backup").unwrap();

        let result = write_completed_keymap_export(
            &path,
            Err("config_rpc_status_storage_error".to_string()),
        );

        assert_eq!(result, Err("config_rpc_status_storage_error".to_string()));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "existing backup");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn config_errors_are_exposed_as_stable_codes() {
        assert_eq!(
            hid_error_code(HidError::ConfigRpcStatus(ConfigStatus::StorageError)),
            "config_rpc_status_storage_error"
        );
        assert_eq!(
            hid_error_code(HidError::ConfigRpcStatus(ConfigStatus::InvalidArgument)),
            "config_rpc_status_invalid_argument"
        );
    }

    #[test]
    fn encoder_transport_setup_failure_is_a_structured_feature_result() {
        let config = config_save_or_discard_from_feature(failed_encoder_feature(
            "hostlink_worker_unavailable",
        ));

        assert!(config.attempted);
        assert!(!config.skipped);
        assert!(!config.success);
        assert_eq!(config.results.len(), 1);
        assert_eq!(
            config.results[0].error.as_deref(),
            Some("hostlink_worker_unavailable")
        );
    }

    #[test]
    fn combined_config_result_keeps_encoder_and_combo_outcomes_separate() {
        let config = config_save_or_discard_from_features(vec![
            skipped_config_feature("ENCODER"),
            failed_config_feature("COMBO", "config_rpc_status_storage_error"),
        ]);

        assert!(config.attempted);
        assert!(!config.skipped);
        assert!(!config.success);
        assert_eq!(config.results.len(), 2);
        assert_eq!(config.results[0].feature, "ENCODER");
        assert!(config.results[0].skipped);
        assert_eq!(config.results[1].feature, "COMBO");
        assert_eq!(
            config.results[1].error.as_deref(),
            Some("config_rpc_status_storage_error")
        );
    }

    #[test]
    fn encoder_import_discard_clears_override_and_verifies_keymap_baseline() {
        let baseline = encoder_get_bindings(EncoderBindingSource::Keymap, 1, 2);
        let imported = encoder_get_bindings(EncoderBindingSource::Override, 3, 4);
        let (tx, rx) = mpsc::channel();
        let worker = thread::spawn(move || {
            let mut step = 0;
            while let Ok(MonitorCommand::Config(call)) = rx.recv() {
                let response = match (&call.request, step) {
                    (HostLinkRequest::EncoderGetDirty, 0) => HostLinkResponse::Dirty(true),
                    (HostLinkRequest::EncoderDiscard, 1) => HostLinkResponse::Done,
                    (HostLinkRequest::EncoderGetBindings { .. }, 2) => {
                        HostLinkResponse::EncoderBindings(imported)
                    }
                    (
                        HostLinkRequest::EncoderClearOverride {
                            layer_id,
                            encoder_id,
                        },
                        3,
                    ) => {
                        assert_eq!((*layer_id, *encoder_id), (10, 0));
                        HostLinkResponse::Done
                    }
                    (HostLinkRequest::EncoderSave, 4) => HostLinkResponse::Done,
                    (HostLinkRequest::EncoderGetBindings { .. }, 5) => {
                        HostLinkResponse::EncoderBindings(baseline)
                    }
                    _ => panic!("unexpected encoder rollback request at step {step}"),
                };
                call.reply.send(Ok(response)).unwrap();
                step += 1;
                if step == 6 {
                    break;
                }
            }
            assert_eq!(step, 6);
        });

        let result = run_encoder_discard_with_rollback(
            tx,
            Duration::from_secs(1),
            "uid:00000000000000aa".to_string(),
            BTreeMap::from([((10, 0), baseline)]),
        );

        assert!(result.success);
        assert!(result.attempted);
        assert!(!result.skipped);
        worker.join().unwrap();
    }

    #[test]
    fn encoder_reset_uses_stable_layer_ids_then_saves_once() {
        let (tx, rx) = mpsc::channel();
        let worker = thread::spawn(move || {
            let expected = [(3, 0), (3, 1), (7, 0), (7, 1)];
            for (step, expected_target) in expected.into_iter().enumerate() {
                let MonitorCommand::Config(call) = rx.recv().unwrap() else {
                    panic!("expected config call");
                };
                match call.request {
                    HostLinkRequest::EncoderClearOverride {
                        layer_id,
                        encoder_id,
                    } => {
                        assert_eq!((layer_id, encoder_id), expected_target, "step {step}");
                    }
                    request => panic!("unexpected reset request: {request:?}"),
                }
                call.reply.send(Ok(HostLinkResponse::Done)).unwrap();
            }
            let MonitorCommand::Config(call) = rx.recv().unwrap() else {
                panic!("expected encoder save");
            };
            assert!(matches!(call.request, HostLinkRequest::EncoderSave));
            call.reply.send(Ok(HostLinkResponse::Done)).unwrap();
        });

        let result = run_encoder_reset_to_keymap(
            tx,
            Duration::from_secs(1),
            "uid:00000000000000aa".to_string(),
            EncoderGetInfo {
                layer_count: 2,
                encoder_count: 2,
                capabilities: 0,
                scroll_value: None,
                encoder_tap_ms: None,
            },
            &reset_snapshot(&[7, 3]),
        );

        assert!(result.success);
        assert!(result.attempted);
        assert!(!result.skipped);
        worker.join().unwrap();
    }
}

/// Short, *distinguishing* label for a Codex thread id in the app log.
///
/// Codex thread ids are UUIDv7, which are time-ordered: two threads created
/// seconds apart share a long common prefix. Truncating to the front alone
/// collapsed distinct threads into one label during the "approval flashes
/// then reverts" investigation and made two threads look like one, so keep
/// the tail (the random part) as well.
fn thread_tag(thread_id: &str) -> String {
    let head: String = thread_id.chars().take(8).collect();
    let tail: String = {
        let chars: Vec<char> = thread_id.chars().collect();
        chars[chars.len().saturating_sub(6)..].iter().collect()
    };
    format!("{head}..{tail}")
}
