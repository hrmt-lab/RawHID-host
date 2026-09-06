// 笏笏笏 Config Types 笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏

export interface PollingConfig {
  interval_ms: number;
  uplink_interval_ms: number;
}

export interface HidConfig {
  usage_page: number;
  usage: number;
  hello_timeout_ms: number;
  rescan_interval_sec: number;
}

export interface RuleConfig {
  name: string;
  layer: number;
  path: string | null;
  exe: string | null;
  title: string | null;
}

export type TimeFormatHint =
  | "time_hm"
  | "time_hms"
  | "date_ymd"
  | "date_md"
  | "datetime_hm"
  | "weekday_hm";

export type ClockMode = "24h" | "12h";

export interface TimeConfig {
  enabled: boolean;
  format_hint: TimeFormatHint;
  clock_mode: ClockMode;
  periodic_sync_sec: number;
  tz_offset_min: number | null;
}

export interface StudioConfig {
  probe_timeout_ms: number;
  keymap_read_timeout_ms: number;
}

export interface AiUsageConfig {
  enabled: boolean;
  poll_interval_sec: number;
  stale_after_sec: number;
  codex: CodexAiUsageConfig;
  claude_code: ClaudeCodeAiUsageConfig;
}

export interface CodexAiUsageConfig {
  enabled: boolean;
  sessions_dir: string | null;
  sessions_auto_detect: boolean;
  include_wsl_sessions: boolean;
  extra_sessions_paths: string[];
  history_fallback_enabled: boolean;
  allow_activity_baseline: boolean;
  activity_five_hour_token_baseline: number;
  activity_seven_day_token_baseline: number;
}

export interface ClaudeCodeAiUsageConfig {
  enabled: boolean;
  credentials_path: string | null;
  credentials_auto_detect: boolean;
  include_wsl_credentials: boolean;
  extra_credentials_paths: string[];
  api_timeout_sec: number;
}

export type UnmatchedAction = "clear_managed" | "keep";

export interface DeviceLayerSwitchConfig {
  display_name: string | null;
  enabled: boolean;
  rules: RuleConfig[];
  unmatched_action: UnmatchedAction | null;
}

export interface LayerSwitchConfig {
  enabled: boolean;
  unmatched_action: UnmatchedAction;
  devices: Record<string, DeviceLayerSwitchConfig>;
}

export interface AppBehaviorConfig {
  start_monitoring_on_launch: boolean;
}

export interface CodexClientConfig {
  executable_path: string | null;
  version_check_enabled: boolean;
  app_server_port: number;
  broker_port: number;
}

export type CodexLaunchEnvironment = "windows" | "wsl";

export interface CodexLauncherConfig {
  environment: CodexLaunchEnvironment;
  windows_project_directory: string | null;
  wsl_project_directory: string | null;
  wsl_distribution: string | null;
  wsl_executable: string;
}

export interface ClaudeLauncherConfig {
  executable_path: string | null;
  project_directory: string | null;
}

export interface AiClientConfig {
  codex: CodexClientConfig;
  codex_launcher: CodexLauncherConfig;
  claude_launcher: ClaudeLauncherConfig;
  display: AiClientDisplayConfig;
}

export interface AiClientDisplayConfig {
  slot_count: number;
}

export type CodexBrokerPhase =
  | "stopped"
  | "starting"
  | "waiting_for_client"
  | "connected"
  | "reconnecting"
  | "stopping"
  | "error";

export interface CodexBrokerStatus {
  phase: CodexBrokerPhase;
  app_server_port: number | null;
  broker_port: number | null;
  codex_version: string | null;
  client_connected: boolean;
  connected_client_count: number;
  max_client_count: number;
  managed_launches: ManagedLaunchStatus[];
  last_error: string | null;
}

export interface CodexLaunchResult {
    environment: CodexLaunchEnvironment;
    project_directory: string;
    terminal_target_id: string;
    display_name: string;
  config: AppConfig;
}

export interface ClaudeLaunchResult {
  project_directory: string;
    plugin_directory: string;
    terminal_target_id: string;
    display_name: string;
}

export type ManagedLaunchState = "waiting_for_connection" | "connected" | "timed_out" | "ended";

export interface ManagedLaunchStatus {
  terminal_target_id: string;
  display_name: string;
  state: ManagedLaunchState;
}

export interface WslDistribution {
  name: string;
  version: number;
}

export type AiActivityState =
  | "none"
  | "available"
  | "working"
  | "waiting_approval"
  | "waiting_input"
  | "completed"
  | "error";

export type AiWorkPhase =
  | "unspecified"
  | "thinking"
  | "executing"
  | "searching";

export interface AiClientStateSnapshot {
  client_type: "codex" | "claude_code";
  client_variant: "cli" | "vs_code_extension" | "desktop_app";
  session_active: boolean;
  activity_state: AiActivityState;
  work_phase: AiWorkPhase;
  revision: number;
}

export type AiDisplayTarget =
  | { kind: "codex"; terminal_target_id: string }
  | { kind: "claude"; terminal_target_id: string };

export type AiDisplaySlotMode =
  | { mode: "auto" }
  | { mode: "pinned"; target: AiDisplayTarget };

export interface AiDisplaySlot {
  slot: number;
  mode: AiDisplaySlotMode;
  target: AiDisplayTarget | null;
  snapshot: AiClientStateSnapshot;
}

export interface AiDisplaySlots {
  slots: AiDisplaySlot[];
  candidates: AiDisplayCandidate[];
  slot_capable_device_count: number;
}

export interface AiDisplayCandidate {
  target: AiDisplayTarget;
  label: string;
  snapshot: AiClientStateSnapshot;
}

// ─── HUD approval ─────────────────────────────────────────────────────────────

/** Sanitized view of one pending approval request, pushed from
 * `HudApprovalPayload` (rawhid-host-tauri's hud_coordinator.rs) via the
 * "hud-approval-update" event. `null` means no approval is pending. */
export interface HudApprovalPayload {
  request_key: string;
  client: "codex" | "claude_code";
  oversized: boolean;
  kind: string | null;
  primary_text: string | null;
  full_command: string | null;
  reason: string | null;
  cwd: string | null;
  available_decisions: unknown[] | null;
  /** Host-managed cursor into `available_decisions`; null when no decision
   * can safely be selected. */
  selected_decision_index: number | null;
  /** Same length/order as `available_decisions`, Claude Code entries only
   * (rawhid-host-tauri's `HudApprovalPayload::decision_labels`). `null` for
   * a Codex entry -- fall back to `decisionLabel()` below for those, same as
   * before this field existed. Always English, matching this HUD's existing
   * English-only decision list (no i18n). */
  decision_labels: string[] | null;
}

export type HostActionKind =
  | "show_window"
  | "start_monitoring"
  | "stop_monitoring"
  | "refresh_ai_usage"
  | "cycle_ai_session"
  | "launch"
  | "open_folder"
  | "focus_ai_terminal"
  | "hud_previous"
  | "hud_next"
  | "hud_confirm"
  | "hud_reject"
  | "select_hud_target";

export interface ActionBinding {
  action_id: number;
  action: HostActionKind;
  /** Filesystem path: executable for "launch", folder for "open_folder"; null otherwise. */
  path: string | null;
  /** "open_folder" only: prefer reusing an existing Explorer window's tab (best-effort). */
  prefer_tab: boolean;
  /** "launch" only: override the exe file name used to detect an already-running instance. */
  match_exe: string | null;
}

export interface DeviceActionsConfig {
  display_name: string | null;
  enabled: boolean;
  bindings: ActionBinding[];
}

export interface ActionsConfig {
  enabled: boolean;
  devices: Record<string, DeviceActionsConfig>;
}

export interface DebugLogConfig {
  enabled: boolean;
}

export interface AppConfig {
  app: AppBehaviorConfig;
  ai_client: AiClientConfig;
  polling: PollingConfig;
  hid: HidConfig;
  layer_switch: LayerSwitchConfig;
  time: TimeConfig;
  ai_usage: AiUsageConfig;
  studio: StudioConfig;
  actions: ActionsConfig;
  debug_log: DebugLogConfig;
}

// 笏笏笏 Runtime Types 笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏

export interface DeviceBatterySource {
  /** 0 = central/self, 1..3 = peripheral 1..3. */
  source: number;
  /** 0..100, or null when unknown / disconnected. */
  level: number | null;
}

export interface DeviceBatteryStatus {
  device_key: string;
  serial_number: string | null;
  product: string | null;
  sources: DeviceBatterySource[];
  updated_unix: number;
}

export interface DeviceLayerState {
  device_key: string;
  serial_number: string | null;
  product: string | null;
  active_layer: number;
  layer_mask: number;
}

export interface PositionCount {
  position: number;
  count: number;
}

export interface KeyStatsSummary {
  device_key: string;
  total: number;
  per_position: PositionCount[];
  days_covered: number;
}

export type StatsPeriod = "today" | "last7days" | "all";

export interface KeyPressEvent {
  device_uid: string;
  position: number;
  pressed: boolean;
}

export interface MonitorStatus {
  running: boolean;
  connected_devices: number;
  connected_device_names: string[];
  host_link_devices: DeviceInfo[];
  current_layer: number | null;
  current_rule: string | null;
  last_error: string | null;
  ai_usage: AiUsageProviderStatus[];
  device_battery: DeviceBatteryStatus[];
  device_layers: DeviceLayerState[];
}

export type AiUsageStatusKind =
  | "disabled"
  | "ok"
  | "stale"
  | "no_data"
  | "missing_credentials"
  | "expired_credentials"
  | "auth_failed"
  | "rate_limited"
  | "fetch_failed"
  | "parse_failed"
  | "missing_limit";

export type AiUsageSourceKind = "none" | "quota" | "local_history";

export type AiUsageCredentialSourceKind =
  | "explicit_path"
  | "windows_default"
  | "wsl"
  | "extra_path";

export interface AiUsageProviderStatus {
  provider: string;
  status: AiUsageStatusKind;
  source: AiUsageSourceKind;
  updated_unix: number | null;
  stale: boolean;
  last_error_code: number | null;
  five_hour_used_bp: number | null;
  seven_day_used_bp: number | null;
  five_hour_reset_unix: number | null;
  seven_day_reset_unix: number | null;
  five_hour_valid: boolean;
  seven_day_valid: boolean;
  estimated: boolean;
  quota_source: boolean;
  local_history_source: boolean;
  fallback_limit: boolean;
  error_present: boolean;
  credential_source: AiUsageCredentialSourceKind | null;
}

export interface LogEntry {
  id: number;
  timestamp_ms: number;
  level: "info" | "warn" | "error";
  message: string;
}

export interface DeviceInfo {
  path: string;
  vendor_id: number;
  product_id: number;
  usage_page: number;
  usage: number;
  connection_type: "usb" | "bluetooth" | "unknown";
  manufacturer: string | null;
  product: string | null;
  serial_number: string | null;
  capabilities: number;
  device_uid_hash: string | null;
}


export type StudioRpcStatus = "ok" | "failed" | "timeout" | "unavailable";
export type StudioLockState = "locked" | "unlocked" | "unknown";
export type KeymapViewerStatus = "available" | "locked" | "unsupported" | "failed";
export type StudioErrorCode =
  | "none"
  | "no_serial_ports"
  | "open_failed"
  | "rpc_timeout"
  | "rpc_failed"
  | "protocol_mismatch"
  | "locked"
  | "device_not_found"
  | "keymap_read_failed";

export type StudioConnectionType = "usb_serial" | "ble_studio" | string;

export interface StudioDeviceStatus {
  id: string;
  connection_type: StudioConnectionType;
  port_name: string;
  display_name: string;
  vid: number | null;
  pid: number | null;
  serial_number: string | null;
  manufacturer: string | null;
  product: string | null;
  transport_detected: boolean;
  rpc_status: StudioRpcStatus;
  lock_state: StudioLockState;
  keymap_viewer_status: KeymapViewerStatus;
  error_code: StudioErrorCode;
}

export type StudioLayoutSource = "studio_physical_layout" | "grid_fallback";

export interface StudioKeymapSnapshot {
  device_id: string;
  device_name: string;
  connection_type: StudioConnectionType;
  lock_state: StudioLockState;
  physical_layouts: StudioPhysicalLayout[];
  selected_physical_layout_index: number | null;
  selected_physical_layout_name: string | null;
  layout_source: StudioLayoutSource;
  selected_layout_keys: StudioPhysicalKey[];
  layers: StudioLayer[];
  updated_ms: number;
}

export interface StudioPhysicalLayout {
  index: number;
  name: string;
  keys: StudioPhysicalKey[];
}

export interface StudioPhysicalKey {
  position: number;
  x: number;
  y: number;
  width: number;
  height: number;
  r: number;
  rx: number;
  ry: number;
}

export interface StudioLayer {
  index: number;
  id: number;
  name: string;
  bindings: StudioBinding[];
}

export interface StudioBinding {
  position: number;
  binding_label: string;
  primary_label: string;
  secondary_label: string;
  full_label: string;
  behavior: string;
  params: number[];
  raw: StudioRawBinding;
}

export interface StudioBindingLabelPatch {
  behavior_id: number;
  param1: number;
  param2: number;
  behavior: string;
  binding_label: string;
  primary_label: string;
  secondary_label: string;
  full_label: string;
}

export interface StudioRawBinding {
  behavior_id: number;
  param1: number;
  param2: number;
}

export type BehaviorVerification = "done" | "skipped";

export interface RestoreIssue {
  code: string;
  layer_index: number | null;
  position: number | null;
  message: string;
}

export interface RestoreChangedKey {
  layer_index: number;
  position: number;
}

export interface RestoreChangedEncoder {
  layer_index: number;
  encoder_id: number;
}

export interface RestoreChangedCombo {
  name: string;
  action: "add" | "update";
}

export interface KeymapExportReport {
  warnings: RestoreIssue[];
}

export interface RestoreReport {
  can_apply: boolean;
  behavior_verification: BehaviorVerification;
  source_device_name: string;
  exported_at_ms: number;
  will_write: number;
  unchanged_skipped: number;
  blocked: number;
  changed_keys: RestoreChangedKey[];
  warnings: RestoreIssue[];
  errors: RestoreIssue[];
  encoder_will_write: number;
  encoder_unchanged_skipped: number;
  encoder_blocked: number;
  changed_encoders: RestoreChangedEncoder[];
  combo_added: number;
  combo_updated: number;
  combo_unchanged_skipped: number;
  combo_blocked: number;
  changed_combos: RestoreChangedCombo[];
  apply_status: "preview" | "complete" | "partial";
  applied_keys: RestoreChangedKey[];
  applied_encoders: RestoreChangedEncoder[];
  applied_combos: RestoreChangedCombo[];
}

export type KeyCatalogCategory =
  | "letters"
  | "numbers"
  | "symbols"
  | "control"
  | "navigation"
  | "locks"
  | "function"
  | "international"
  | "language"
  | "miscellaneous"
  | "modifiers"
  | "keypad"
  | "editing"
  | "media"
  | "applications"
  | "input_assist"
  | "power_lock"
  | "other";

export interface KeyCatalogEntry {
  display: string;
  canonical: string;
  hid_usage: number;
  category: KeyCatalogCategory;
  aliases: string[];
  names: string[];
}

export type EditBehavior =
  | { kind: "key_press"; hid_usage: number }
  | { kind: "transparent" }
  | { kind: "none" }
  | { kind: "momentary_layer"; target_layer_index: number }
  | { kind: "toggle_layer"; target_layer_index: number }
  | { kind: "to_layer"; target_layer_index: number }
  | { kind: "mod_tap"; hold_hid_usage: number; tap_hid_usage: number }
  | { kind: "layer_tap"; target_layer_index: number; tap_hid_usage: number }
  | { kind: "sticky_key"; hid_usage: number }
  | { kind: "sticky_layer"; target_layer_index: number }
  | { kind: "bluetooth"; command: number; value: number }
  | { kind: "output_selection"; value: number }
  | { kind: "mouse_key_press"; value: number }
  | { kind: "mouse_move"; value: number }
  | { kind: "mouse_scroll"; value: number }
  | { kind: "caps_word" }
  | { kind: "key_repeat" }
  | { kind: "reset" }
  | { kind: "bootloader" }
  | { kind: "studio_unlock" }
  | { kind: "grave_escape" }
  | { kind: "host_action"; action_id: number; value: number };

export interface EncoderInfoDto {
  layer_count: number;
  encoder_count: number;
  capabilities: number;
  scroll_value: number | null;
  encoder_tap_ms: number | null;
}

export interface EncoderBindingDto {
  behavior_id: number;
  param1: number;
  param2: number;
  // Only populated when the parent EncoderBindingsDto.source is "override".
  label: StudioBindingLabelPatch | null;
}

export interface EncoderBindingsDto {
  layer_id: number;
  encoder_id: number;
  source: "keymap" | "override";
  stale_saved_exists: boolean;
  saved_exists: boolean;
  runtime_dirty: boolean;
  invalid_saved_exists: boolean;
  cw: EncoderBindingDto;
  ccw: EncoderBindingDto;
}

export interface ComboInfoDto {
  max_combos: number;
  max_keys_per_combo: number;
  combo_count: number;
  flags: number;
  occupied_slots: number;
  stale_slots: number;
  invalid_slots: number;
}

export interface ComboBindingDto {
  behavior_id: number;
  param1: number;
  param2: number;
  label: StudioBindingLabelPatch | null;
}

export interface ComboItemDto {
  slot: number;
  name: string;
  key_positions: number[];
  slow_release: boolean;
  binding: ComboBindingDto;
  layer_mask: number;
  timeout_ms: number;
  require_prior_idle_ms: number | null;
}

export interface ComboItemInputDto extends ComboItemDto {
  /** A picker result is resolved against the connected firmware on apply. */
  behavior: EditBehavior | null;
}

// Outcome of attempting a save-or-discard operation on a single target
// (Studio RPC keys, or a Config RPC feature). `skipped` implies
// `success: true`: there was nothing to do or no encoder target was part of
// this edit session. A known dirty target that disconnected is a failure.
export interface SaveOrDiscardTargetDto {
  attempted: boolean;
  skipped: boolean;
  success: boolean;
  error: string | null;
}

// Per-feature Config RPC result (currently only "ENCODER").
export interface ConfigFeatureResultDto {
  feature: string;
  attempted: boolean;
  skipped: boolean;
  success: boolean;
  error: string | null;
}

export interface ConfigSaveOrDiscardDto {
  attempted: boolean;
  skipped: boolean;
  success: boolean;
  results: ConfigFeatureResultDto[];
}

export interface SaveOrDiscardResultDto {
  overall_success: boolean;
  studio: SaveOrDiscardTargetDto;
  config: ConfigSaveOrDiscardDto;
}

export interface DiscardChangesDto {
  result: SaveOrDiscardResultDto;
  // Present when the studio-side discard ran successfully (the re-read snapshot).
  snapshot: StudioKeymapSnapshot | null;
}

export interface ResetToKeymapDto {
  overall_success: boolean;
  studio: SaveOrDiscardTargetDto;
  config: ConfigSaveOrDiscardDto;
  snapshot: StudioKeymapSnapshot | null;
  refresh_error: string | null;
}

export interface StudioResyncEditStateDto {
  snapshot: StudioKeymapSnapshot;
  has_unsaved: boolean;
}

export interface EditState {
  mode: "viewing" | "editing";
  dirty: boolean;
  operation: "idle" | "setting" | "saving" | "discarding" | "resetting" | "ending";
  problem: null | "save_failed" | "save_unknown" | "locked_again" | "disconnected";
}
export interface ProbeResult {
  device: DeviceInfo;
  verified: boolean;
  error: string | null;
}

// 笏笏笏 Page Types 笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏笏

export type Page = "devices" | "rules" | "actions" | "timesync" | "ai_usage" | "keymap_viewer" | "settings";
