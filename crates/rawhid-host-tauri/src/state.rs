use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    path::PathBuf,
    sync::{atomic::AtomicBool, Arc, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use rawhid_host_core::{
    ai_usage::{AiUsageProviderStatus, AiUsageRuntime, AiUsageShared},
    codex_activity::{AiClientStateSnapshot, CodexActivityRuntime},
    codex_broker::CodexBrokerManager,
    config::AppConfig,
    hid::{DeviceInfo, ProbeResult},
    packet::{
        ComboInfo, ComboItem, EncoderBinding, EncoderGetBindings, EncoderGetInfo, UplinkPacket,
    },
    runner::{DeviceBatteryStatus, DeviceLayerState},
    stats::{default_stats_dir, KeyStatsStore, SharedKeyStatsStore},
    studio::StudioEditSession,
};

use rawhid_host_core::{
    ClaudeObserverCounters, ClaudeObserverEvents, ClaudeObserverReceiver, ClaudePermissionGate,
    ClaudeSessionRegistry,
};

use crate::debug_log::DebugLogHandle;
use crate::hud_coordinator::HudCoordinator;

pub const MAX_LOG_ENTRIES: usize = 200;
pub const MAX_AI_DISPLAY_SLOTS: u8 = 8;

#[derive(Debug, Clone, serde::Serialize)]
pub struct LogEntry {
    pub id: u64,
    pub timestamp_ms: u64,
    pub level: String,
    pub message: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct MonitorStatus {
    pub running: bool,
    pub connected_devices: usize,
    pub connected_device_names: Vec<String>,
    pub host_link_devices: Vec<DeviceInfo>,
    pub current_layer: Option<u8>,
    pub current_rule: Option<String>,
    pub last_error: Option<String>,
    pub ai_usage: Vec<AiUsageProviderStatus>,
    pub device_battery: Vec<DeviceBatteryStatus>,
    pub device_layers: Vec<DeviceLayerState>,
}

impl Default for MonitorStatus {
    fn default() -> Self {
        Self {
            running: false,
            connected_devices: 0,
            connected_device_names: Vec::new(),
            host_link_devices: Vec::new(),
            current_layer: None,
            current_rule: None,
            last_error: None,
            ai_usage: Vec::new(),
            device_battery: Vec::new(),
            device_layers: Vec::new(),
        }
    }
}

pub struct AppState {
    pub config: Arc<Mutex<AppConfig>>,
    pub config_path: Arc<Mutex<Option<PathBuf>>>,
    pub status: Arc<Mutex<MonitorStatus>>,
    pub log_entries: Arc<Mutex<VecDeque<LogEntry>>>,
    pub log_counter: Arc<Mutex<u64>>,
    pub monitor_tx: Arc<Mutex<Option<std::sync::mpsc::Sender<MonitorCommand>>>>,
    pub ai_usage_refreshing: Arc<AtomicBool>,
    pub ai_terminal_focusing: Arc<AtomicBool>,
    pub ai_usage_runtime: Arc<Mutex<Option<AiUsageRuntime>>>,
    pub codex_activity: Arc<CodexActivityRuntime>,
    pub claude_integration: Arc<Mutex<Option<ClaudeIntegration>>>,
    /// Shared between every Claude Code launch's `ClaudeObserverReceiver`
    /// (`commands.rs`'s `launch_claude_code`, which passes a clone into
    /// `ClaudeObserverReceiverOptions`) and the Tauri command / physical
    /// HUD-action paths that answer a `PermissionRequest` hook
    /// (`respond_to_claude_approval_internal`, `actions.rs`'s
    /// `dispatch_hud_response_selection`). One instance for the whole
    /// process, not one per launch, exactly like `codex_activity`'s single
    /// `PendingApprovalStore` -- see that field's own reasoning for sharing
    /// one store/gate rather than scoping it per launch.
    pub claude_permission_gate: Arc<ClaudePermissionGate>,
    pub ai_display_slots: Arc<Mutex<AiDisplaySlots>>,
    pub codex_broker: CodexBrokerManager,
    pub key_stats: SharedKeyStatsStore,
    pub studio_edit: Arc<Mutex<Option<StudioEditSession>>>,
    pub encoder_restore_rollbacks:
        Arc<Mutex<HashMap<(String, u64), BTreeMap<(u32, u8), EncoderGetBindings>>>>,
    /// Populated once at startup by `lib.rs`'s `setup_hud`, which is the
    /// only place an `AppHandle` capable of building the HUD window is
    /// available. `None` only for the brief window between `AppState::new`
    /// and that `setup()` callback running.
    pub hud: Arc<Mutex<Option<HudCoordinator>>>,
    /// `[debug_log]` file sink. Installed once at startup (`lib.rs::run`);
    /// `commands::persist_config` flips `enabled` on it when the Settings
    /// toggle changes. See `debug_log`'s module doc for why this is a
    /// toggleable handle rather than a subscriber re-initialized per save.
    pub debug_log: DebugLogHandle,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AiDisplayTarget {
    Codex { terminal_target_id: String },
    Claude { terminal_target_id: String },
}

impl AiDisplayTarget {
    pub fn label(&self) -> String {
        match self {
            Self::Codex { terminal_target_id } => {
                format!(
                    "Codex {}",
                    terminal_target_id
                        .rsplit('-')
                        .next()
                        .unwrap_or_default()
                        .chars()
                        .take(8)
                        .collect::<String>()
                        .to_ascii_uppercase()
                )
            }
            Self::Claude { terminal_target_id } => format!(
                "Claude Code {}",
                terminal_target_id
                    .rsplit('-')
                    .next()
                    .unwrap_or_default()
                    .chars()
                    .take(8)
                    .collect::<String>()
                    .to_ascii_uppercase()
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AiDisplayCandidate {
    pub target: AiDisplayTarget,
    pub snapshot: AiClientStateSnapshot,
    /// First successful registration order across all AI client types.
    pub registration_order: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "mode", content = "target", rename_all = "snake_case")]
pub enum AiDisplaySlotMode {
    Auto,
    Pinned(AiDisplayTarget),
}

#[derive(Debug, Clone)]
pub struct AiDisplaySlot {
    pub slot: u8,
    pub mode: AiDisplaySlotMode,
    pub assigned: Option<AiDisplayTarget>,
    pub epoch: u64,
    /// Retained after an Auto target disappears so its replacement advances in
    /// the common registration order rather than jumping back to the first.
    last_registration_order: Option<u64>,
}

#[derive(Debug)]
pub struct AiDisplaySlots {
    candidates: Vec<AiDisplayCandidate>,
    slots: Vec<AiDisplaySlot>,
    /// Slots removed by a configuration reduction.  The monitor consumes this
    /// list and emits a single inactive state so capable renderers clear them.
    retired_slots: Vec<u8>,
}

impl Default for AiDisplaySlots {
    fn default() -> Self {
        Self::new(1)
    }
}

impl AiDisplaySlots {
    pub fn new(slot_count: u8) -> Self {
        let count = slot_count.clamp(1, MAX_AI_DISPLAY_SLOTS);
        Self {
            candidates: Vec::new(),
            slots: (0..count)
                .map(|slot| AiDisplaySlot {
                    slot,
                    mode: AiDisplaySlotMode::Auto,
                    assigned: None,
                    epoch: 0,
                    last_registration_order: None,
                })
                .collect(),
            retired_slots: Vec::new(),
        }
    }

    pub fn set_slot_count(&mut self, slot_count: u8) {
        let count = slot_count.clamp(1, MAX_AI_DISPLAY_SLOTS);
        if self.slots.len() > usize::from(count) {
            self.retired_slots.extend(
                self.slots[usize::from(count)..]
                    .iter()
                    .map(|slot| slot.slot),
            );
        }
        self.slots.truncate(usize::from(count));
        while self.slots.len() < usize::from(count) {
            let slot = self.slots.len() as u8;
            self.slots.push(AiDisplaySlot {
                slot,
                mode: AiDisplaySlotMode::Auto,
                assigned: None,
                epoch: 0,
                last_registration_order: None,
            });
        }
        self.refresh_assignments();
    }

    pub fn update_candidates(&mut self, candidates: Vec<AiDisplayCandidate>) {
        let mut candidates = candidates;
        candidates.sort_by_key(|candidate| candidate.registration_order);
        self.candidates = candidates;
        self.refresh_assignments();
    }

    #[cfg(test)]
    pub fn cycle(&mut self) -> Option<AiDisplayTarget> {
        self.cycle_slot(0)
    }

    pub fn cycle_slot(&mut self, slot: u8) -> Option<AiDisplayTarget> {
        let slot_index = usize::from(slot);
        let current = self.slots.get(slot_index)?.assigned.clone();
        let current_index = current.as_ref().and_then(|target| {
            self.candidates
                .iter()
                .position(|candidate| &candidate.target == target)
        });
        for offset in 1..=self.candidates.len() {
            let index = current_index
                .map(|index| (index + offset) % self.candidates.len())
                .unwrap_or(offset - 1);
            let target = self.candidates[index].target.clone();
            if self.slots.iter().enumerate().all(|(index, other)| {
                index == slot_index || other.assigned.as_ref() != Some(&target)
            }) {
                self.set_assigned(slot_index, Some(target.clone()));
                return Some(target);
            }
        }
        current
    }

    pub fn pin(&mut self, slot: u8, target: AiDisplayTarget) -> Result<(), String> {
        if !self
            .candidates
            .iter()
            .any(|candidate| candidate.target == target)
        {
            return Err("ai_display_target_not_active".to_string());
        }
        let index = usize::from(slot);
        let Some(destination) = self.slots.get(index) else {
            return Err("invalid_ai_display_slot".to_string());
        };
        let previous = destination.assigned.clone();
        let source = self
            .slots
            .iter()
            .position(|other| other.assigned.as_ref() == Some(&target));
        if let Some(source) = source.filter(|source| *source != index) {
            // The target is now pinned at the destination. The source must no
            // longer remain pinned to the same target, otherwise refresh would
            // immediately recreate a duplicate assignment.
            self.slots[source].mode = AiDisplaySlotMode::Auto;
            self.set_assigned(source, previous);
        }
        self.slots[index].mode = AiDisplaySlotMode::Pinned(target.clone());
        self.set_assigned(index, Some(target));
        self.refresh_assignments();
        Ok(())
    }

    pub fn set_auto(&mut self, slot: u8) -> Result<(), String> {
        let index = usize::from(slot);
        let Some(entry) = self.slots.get_mut(index) else {
            return Err("invalid_ai_display_slot".to_string());
        };
        entry.mode = AiDisplaySlotMode::Auto;
        self.refresh_assignments();
        Ok(())
    }

    pub fn slots(&self) -> &[AiDisplaySlot] {
        &self.slots
    }

    pub fn all_candidates(&self) -> &[AiDisplayCandidate] {
        &self.candidates
    }

    pub fn candidate_snapshot(&self, target: &AiDisplayTarget) -> Option<AiClientStateSnapshot> {
        self.candidates
            .iter()
            .find(|candidate| &candidate.target == target)
            .map(|candidate| candidate.snapshot)
    }

    pub fn slot_snapshot(&self, slot: u8) -> Option<AiClientStateSnapshot> {
        self.slots
            .get(usize::from(slot))
            .and_then(|entry| entry.assigned.as_ref())
            .and_then(|target| self.candidate_snapshot(target))
    }

    /// Compatibility accessors for the original single-ScreenKey API.
    pub fn selected_target(&self) -> Option<&AiDisplayTarget> {
        self.slots.first().and_then(|slot| slot.assigned.as_ref())
    }

    pub fn selected_snapshot(&self) -> Option<AiClientStateSnapshot> {
        self.slot_snapshot(0)
    }

    pub fn epoch(&self) -> u64 {
        self.slots.first().map(|slot| slot.epoch).unwrap_or(0)
    }

    pub fn take_retired_slots(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.retired_slots)
    }

    fn refresh_assignments(&mut self) {
        for index in 0..self.slots.len() {
            match self.slots[index].mode.clone() {
                AiDisplaySlotMode::Auto => {
                    if self.slots[index]
                        .assigned
                        .as_ref()
                        .is_some_and(|target| self.candidate_snapshot(target).is_none())
                    {
                        self.set_assigned(index, None);
                    }
                }
                AiDisplaySlotMode::Pinned(target) => {
                    let next = self.candidate_snapshot(&target).is_some().then_some(target);
                    self.set_assigned(index, next);
                }
            }
        }
        for index in 0..self.slots.len() {
            if !matches!(self.slots[index].mode, AiDisplaySlotMode::Auto)
                || self.slots[index].assigned.is_some()
            {
                continue;
            }
            let last_registration_order = self.slots[index].last_registration_order;
            let is_available = |candidate: &AiDisplayCandidate| {
                !self
                    .slots
                    .iter()
                    .any(|slot| slot.assigned.as_ref() == Some(&candidate.target))
            };
            let next = self
                .candidates
                .iter()
                .filter(|candidate| is_available(candidate))
                .find(|candidate| {
                    last_registration_order
                        .is_some_and(|order| candidate.registration_order > order)
                })
                .or_else(|| {
                    self.candidates
                        .iter()
                        .find(|candidate| is_available(candidate))
                })
                .map(|candidate| candidate.target.clone());
            self.set_assigned(index, next);
        }
    }

    fn set_assigned(&mut self, index: usize, assigned: Option<AiDisplayTarget>) {
        if self.slots[index].assigned != assigned {
            self.slots[index].assigned = assigned;
            self.slots[index].epoch = self.slots[index].epoch.wrapping_add(1);
        }
        if let Some(target) = self.slots[index].assigned.as_ref() {
            self.slots[index].last_registration_order = self
                .candidates
                .iter()
                .find(|candidate| &candidate.target == target)
                .map(|candidate| candidate.registration_order);
        }
    }
}

/// Compatibility name retained while callers migrate to slot-aware display
/// handling. It always exposes slot zero for the legacy methods.
pub type AiDisplaySelection = AiDisplaySlots;

pub struct ClaudeIntegration {
    pub launches: BTreeMap<String, ClaudeLaunchIntegration>,
    pub registry: ClaudeSessionRegistry,
}

pub struct ClaudeLaunchIntegration {
    pub receiver: ClaudeObserverReceiver,
    pub events: ClaudeObserverEvents,
    pub last_counters: ClaudeObserverCounters,
    pub plugin_root: PathBuf,
    pub terminal_target_id: String,
    pub display_name: String,
    pub timed_out_at: Option<Instant>,
    pub remove_at: Option<Instant>,
}

#[derive(Debug)]
pub enum MonitorCommand {
    SetAutomationEnabled(bool, std::sync::mpsc::Sender<Result<(), String>>),
    Probe(std::sync::mpsc::Sender<Result<Vec<ProbeResult>, String>>),
    Config(HostLinkCall),
    Shutdown,
    UpdateConfig(AppConfig, Option<AiUsageShared>),
    /// The OS foreground window changed; wake the loop to re-evaluate immediately.
    ForegroundChanged,
    /// Debug-only: feed a synthetic uplink packet through the normal path.
    InjectUplink(DeviceInfo, UplinkPacket),
}

#[derive(Debug)]
pub struct HostLinkCall {
    pub uid: u64,
    pub request: HostLinkRequest,
    pub deadline: Instant,
    pub reply: std::sync::mpsc::Sender<Result<HostLinkResponse, String>>,
}

#[derive(Debug, Clone, Copy)]
pub enum HostLinkRequest {
    EncoderGetInfo,
    EncoderGetBindings {
        layer_id: u32,
        encoder_id: u8,
    },
    EncoderSetBindings {
        layer_id: u32,
        encoder_id: u8,
        cw: EncoderBinding,
        ccw: EncoderBinding,
    },
    EncoderGetDirty,
    EncoderSave,
    EncoderDiscard,
    EncoderClearOverride {
        layer_id: u32,
        encoder_id: u8,
    },
    ComboGetInfo,
    ComboGet {
        slot: u8,
    },
    ComboSet {
        item: ComboItem,
    },
    ComboGetDirty,
    ComboSave,
    ComboDiscard,
    ComboDelete {
        slot: u8,
    },
    ComboResetToKeymap,
}

#[derive(Debug)]
pub enum HostLinkResponse {
    EncoderInfo(EncoderGetInfo),
    EncoderBindings(EncoderGetBindings),
    ComboInfo(ComboInfo),
    ComboItem(ComboItem),
    Dirty(bool),
    Done,
}

impl AppState {
    pub fn new(config: AppConfig, config_path: Option<PathBuf>, debug_log: DebugLogHandle) -> Self {
        let display_slot_count = config.ai_client.display.slot_count;
        let codex_broker = CodexBrokerManager::new();
        let codex_activity = Arc::new(CodexActivityRuntime::start(codex_broker.clone()));
        let ai_usage_runtime = AiUsageRuntime::start(config.ai_usage.clone());
        let ai_usage_statuses = ai_usage_runtime
            .as_ref()
            .map(|runtime| runtime.statuses(config.ai_usage.stale_after_sec))
            .unwrap_or_default();
        let mut status = MonitorStatus::default();
        status.ai_usage = ai_usage_statuses;
        let stats_dir = default_stats_dir()
            .unwrap_or_else(|| std::env::temp_dir().join("keylink-studio").join("stats"));
        let key_stats = Arc::new(Mutex::new(KeyStatsStore::new(
            stats_dir,
            Duration::from_secs(config.stats.flush_interval_sec.max(1)),
        )));
        Self {
            config: Arc::new(Mutex::new(config)),
            config_path: Arc::new(Mutex::new(config_path)),
            status: Arc::new(Mutex::new(status)),
            log_entries: Arc::new(Mutex::new(VecDeque::new())),
            log_counter: Arc::new(Mutex::new(0)),
            monitor_tx: Arc::new(Mutex::new(None)),
            ai_usage_refreshing: Arc::new(AtomicBool::new(false)),
            ai_terminal_focusing: Arc::new(AtomicBool::new(false)),
            ai_usage_runtime: Arc::new(Mutex::new(ai_usage_runtime)),
            codex_activity,
            claude_integration: Arc::new(Mutex::new(None)),
            claude_permission_gate: Arc::new(ClaudePermissionGate::default()),
            ai_display_slots: Arc::new(Mutex::new(AiDisplaySlots::new(display_slot_count))),
            codex_broker,
            key_stats,
            studio_edit: Arc::new(Mutex::new(None)),
            encoder_restore_rollbacks: Arc::new(Mutex::new(HashMap::new())),
            hud: Arc::new(Mutex::new(None)),
            debug_log,
        }
    }
}

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

pub fn add_log(
    log_entries: &Arc<Mutex<VecDeque<LogEntry>>>,
    log_counter: &Arc<Mutex<u64>>,
    level: &str,
    message: &str,
) -> LogEntry {
    let id = {
        let mut counter = log_counter.lock().unwrap();
        *counter += 1;
        *counter
    };
    let entry = LogEntry {
        id,
        timestamp_ms: now_ms(),
        level: level.to_string(),
        message: message.to_string(),
    };
    let mut entries = log_entries.lock().unwrap();
    entries.push_back(entry.clone());
    while entries.len() > MAX_LOG_ENTRIES {
        entries.pop_front();
    }
    entry
}

#[cfg(test)]
mod tests {
    use super::*;
    use rawhid_host_core::packet::{AiActivityState, AiClientType, AiClientVariant, AiWorkPhase};

    fn candidate(target: AiDisplayTarget, revision: u16) -> AiDisplayCandidate {
        AiDisplayCandidate {
            target,
            snapshot: AiClientStateSnapshot {
                client_type: AiClientType::Codex,
                client_variant: AiClientVariant::Cli,
                session_active: true,
                activity_state: AiActivityState::Available,
                work_phase: AiWorkPhase::Unspecified,
                revision,
            },
            registration_order: u64::from(revision),
        }
    }

    fn claude(session_id: &str) -> AiDisplayTarget {
        AiDisplayTarget::Claude {
            terminal_target_id: format!("claude-{session_id}"),
        }
    }

    fn codex(thread_id: &str) -> AiDisplayTarget {
        AiDisplayTarget::Codex {
            terminal_target_id: format!("codex-{thread_id}"),
        }
    }

    #[test]
    fn adding_a_candidate_does_not_steal_the_current_selection() {
        let mut selection = AiDisplaySelection::default();
        selection.update_candidates(vec![candidate(claude("one"), 1)]);
        selection.update_candidates(vec![
            candidate(codex("codex-one"), 2),
            candidate(claude("one"), 1),
        ]);

        assert_eq!(selection.selected_target(), Some(&claude("one")));
    }

    #[test]
    fn removing_the_selected_candidate_advances_from_its_old_position() {
        let mut selection = AiDisplaySelection::default();
        selection.update_candidates(vec![
            candidate(codex("codex-one"), 1),
            candidate(claude("one"), 2),
            candidate(claude("two"), 3),
        ]);
        selection.cycle();
        assert_eq!(selection.selected_target(), Some(&claude("one")));

        selection.update_candidates(vec![
            candidate(codex("codex-one"), 1),
            candidate(claude("two"), 3),
        ]);
        assert_eq!(selection.selected_target(), Some(&claude("two")));
    }

    #[test]
    fn candidates_cycle_in_cross_client_registration_order() {
        let mut selection = AiDisplaySelection::default();
        selection.update_candidates(vec![candidate(claude("one"), 1)]);
        selection.update_candidates(vec![
            candidate(codex("codex-one"), 2),
            candidate(claude("one"), 1),
        ]);
        selection.update_candidates(vec![
            candidate(codex("codex-one"), 2),
            candidate(codex("codex-two"), 3),
            candidate(claude("one"), 1),
            candidate(claude("two"), 4),
        ]);

        assert_eq!(selection.selected_target(), Some(&claude("one")));
        assert_eq!(selection.cycle(), Some(codex("codex-one")));
        assert_eq!(selection.cycle(), Some(codex("codex-two")));
        assert_eq!(selection.cycle(), Some(claude("two")));
        assert_eq!(selection.cycle(), Some(claude("one")));
    }

    #[test]
    fn first_batch_uses_shared_registration_order_not_client_type_order() {
        let mut selection = AiDisplaySelection::default();
        selection.update_candidates(vec![
            candidate(codex("codex-first-in_input"), 2),
            candidate(claude("claude-registered-first"), 1),
        ]);

        assert_eq!(
            selection.selected_target(),
            Some(&claude("claude-registered-first"))
        );
        assert_eq!(selection.cycle(), Some(codex("codex-first-in_input")));
    }

    #[test]
    fn reactivated_candidate_returns_to_its_original_registration_position() {
        let mut selection = AiDisplaySelection::default();
        selection.update_candidates(vec![candidate(claude("a"), 1), candidate(claude("b"), 2)]);
        // B temporarily leaves the active candidates, while later sessions
        // become active. A resumed B must retain its original position.
        selection.update_candidates(vec![candidate(claude("a"), 1)]);
        selection.update_candidates(vec![
            candidate(claude("a"), 1),
            candidate(codex("c"), 3),
            candidate(codex("d"), 4),
        ]);
        selection.update_candidates(vec![
            candidate(claude("a"), 1),
            candidate(claude("b"), 2),
            candidate(codex("c"), 3),
            candidate(codex("d"), 4),
        ]);

        assert_eq!(selection.selected_target(), Some(&claude("a")));
        assert_eq!(selection.cycle(), Some(claude("b")));
        assert_eq!(selection.cycle(), Some(codex("c")));
        assert_eq!(selection.cycle(), Some(codex("d")));
        assert_eq!(selection.cycle(), Some(claude("a")));
    }

    #[test]
    fn updating_a_non_selected_candidate_does_not_change_selection() {
        let mut selection = AiDisplaySelection::default();
        selection.update_candidates(vec![
            candidate(codex("codex-one"), 1),
            candidate(codex("codex-two"), 2),
        ]);
        selection.update_candidates(vec![
            candidate(codex("codex-one"), 1),
            candidate(codex("codex-two"), 99),
        ]);

        assert_eq!(selection.selected_target(), Some(&codex("codex-one")));
        assert_eq!(selection.selected_snapshot().unwrap().revision, 1);
    }

    #[test]
    fn zero_and_one_candidate_are_safe_to_cycle() {
        let mut selection = AiDisplaySelection::default();
        assert_eq!(selection.cycle(), None);
        assert_eq!(selection.selected_target(), None);

        let only = codex("codex-one");
        selection.update_candidates(vec![candidate(only.clone(), 1)]);
        assert_eq!(selection.cycle(), Some(only.clone()));
        assert_eq!(selection.cycle(), Some(only));

        selection.update_candidates(Vec::new());
        assert_eq!(selection.selected_target(), None);
    }

    #[test]
    fn auto_slots_fill_in_registration_order_without_duplicates() {
        let mut slots = AiDisplaySlots::new(3);
        slots.update_candidates(vec![
            candidate(codex("one"), 1),
            candidate(claude("two"), 2),
            candidate(codex("three"), 3),
        ]);

        assert_eq!(slots.slots()[0].assigned, Some(codex("one")));
        assert_eq!(slots.slots()[1].assigned, Some(claude("two")));
        assert_eq!(slots.slots()[2].assigned, Some(codex("three")));
    }

    #[test]
    fn pinning_an_assigned_target_swaps_slots_and_pinned_target_waits_for_return() {
        let mut slots = AiDisplaySlots::new(2);
        slots.update_candidates(vec![
            candidate(codex("one"), 1),
            candidate(claude("two"), 2),
        ]);

        slots.pin(1, codex("one")).unwrap();
        assert_eq!(slots.slots()[0].assigned, Some(claude("two")));
        assert_eq!(slots.slots()[1].assigned, Some(codex("one")));
        assert!(matches!(
            slots.slots()[1].mode,
            AiDisplaySlotMode::Pinned(_)
        ));

        slots.update_candidates(vec![candidate(claude("two"), 2)]);
        assert_eq!(slots.slots()[1].assigned, None);
        slots.update_candidates(vec![
            candidate(codex("one"), 1),
            candidate(claude("two"), 2),
        ]);
        assert_eq!(slots.slots()[1].assigned, Some(codex("one")));
    }

    #[test]
    fn reducing_slot_count_reports_each_retired_slot_once() {
        let mut slots = AiDisplaySlots::new(4);
        slots.set_slot_count(2);
        assert_eq!(slots.take_retired_slots(), vec![2, 3]);
        assert!(slots.take_retired_slots().is_empty());
    }
}
