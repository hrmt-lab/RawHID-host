use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_AI_SESSION_REGISTRATION_ORDER: AtomicU64 = AtomicU64::new(1);

pub(crate) fn next_ai_session_registration_order() -> u64 {
    NEXT_AI_SESSION_REGISTRATION_ORDER.fetch_add(1, Ordering::Relaxed)
}

pub mod active_app;
pub mod ai_usage;
pub mod app_match;
pub mod claude_activity;
pub mod claude_decision;
pub mod claude_hook_event;
pub mod claude_hook_helper;
pub mod claude_hooks;
pub mod claude_observer;
pub mod codex_activity;
pub mod codex_broker;
pub mod config;
pub mod hid;
pub mod packet;
pub mod pending_approval;
pub mod runner;
pub mod stats;
pub mod studio;
pub mod time;

pub use active_app::{ActiveApp, ActiveAppProvider, SystemActiveAppProvider};
pub use ai_usage::{
    AiUsageProviderStatus, AiUsageRuntime, AiUsageSendState, AiUsageShared, AiUsageStatusKind,
};
pub use app_match::{LayerAction, RuleMatch};
pub use claude_activity::{
    ClaudeAdapterDiagnostic, ClaudeApprovalBodyConsumer, ClaudeEventAdapter, ClaudeSessionReducer,
    ClaudeSessionRegistry, ClaudeSessionSnapshot, ClaudeStateChange, ClaudeStateChangeReason,
    CLAUDE_DETAIL_STALE_TIMEOUT,
};
pub use claude_decision::{
    ClaudeDecision, ClaudePermissionGate, CLAUDE_PERMISSION_DECISION_TIMEOUT,
};
pub use claude_hook_event::{ClaudeHookEvent, ClaudeObserverEvent, ClaudeWrapperExited};
pub use claude_hook_helper::run_claude_hook_helper;
pub use claude_hooks::{
    write_claude_observer_plugin, ClaudePluginArtifacts, ClaudePluginError, ClaudePluginOptions,
    CLAUDE_PERMISSION_HOOK_TIMEOUT_SECONDS, CLAUDE_PROMPT_HOOK_TIMEOUT_SECONDS,
    CLAUDE_SESSION_START_TIMEOUT_SECONDS, CLAUDE_STOP_HOOK_TIMEOUT_SECONDS,
    CLAUDE_TOOL_HOOK_TIMEOUT_SECONDS,
};
pub use claude_observer::{
    ClaudeObserverConfig, ClaudeObserverCounters, ClaudeObserverError, ClaudeObserverEvents,
    ClaudeObserverReceiver, ClaudeObserverReceiverOptions,
};
pub use codex_activity::{
    AiClientStateChange, AiClientStateChangeReason, AiClientStateReducer, AiClientStateSnapshot,
    CodexActivityRuntime, CodexEventAdapter, CodexSessionRegistry, CodexSessionSnapshot,
    CodexStateChange, MAX_CODEX_SESSIONS,
};
pub use codex_broker::{
    extract_command_approval_body, BrokerDirection, CodexAppServerRuntime,
    CodexApprovalRequestBody, CodexApprovalResponseOutcome, CodexBrokerConfig, CodexBrokerError,
    CodexBrokerEvent, CodexBrokerManager, CodexBrokerPhase, CodexBrokerStatus,
    CodexClientLaunchInfo, JsonRpcKind, JsonRpcMetadata, MAX_CODEX_CLIENTS,
    SUPPORTED_CODEX_VERSION, SUPPORTED_SCHEMA_SHA256,
};
pub use config::{
    AiClientConfig, AiClientDisplayConfig, AiUsageConfig, AppConfig, ClaudeCodeAiUsageConfig,
    ClaudeLauncherConfig, ClockMode, CodexAiUsageConfig, CodexClientConfig, CodexLaunchEnvironment,
    CodexLauncherConfig, ConfigPaths, DeviceLayerSwitchConfig, HidConfig, LayerSwitchConfig,
    PollingConfig, RuleConfig, StudioConfig, TimeConfig, TimeFormatHint, UnmatchedAction,
};
pub use hid::{DeviceConnectionType, DeviceInfo, HidDeviceManager, HidTransport, ProbeResult};
pub use packet::{
    AiActivityState, AiClientStatePacket, AiClientType, AiClientVariant, AiUsageErrorCode,
    AiUsageFlags, AiUsagePacket, AiUsageProvider, AiWorkPhase, AppLayerAction, BatteryEntry,
    BatteryStatusPacket, ComboBinding, ComboConfigOp, ComboFlags, ComboInfo, ComboInfoFlags,
    ComboItem, ComboName, ConfigFeature, ConfigOp, ConfigRequest, ConfigResponse, ConfigStatus,
    DeviceHello, EncoderBinding, EncoderBindingFlags, EncoderBindingSource, EncoderGetBindings,
    EncoderGetInfo, HostActionPacket, KeyStatsEntry, KeyStatsPacket, LayerStatePacket, Packet,
    PacketType, TimeSyncPacket, UplinkPacket, CAPABILITY_AI_CLIENT_CLAUDE_CODE,
    CAPABILITY_AI_CLIENT_DISPLAY_SLOT, CAPABILITY_AI_CLIENT_STATE, CAPABILITY_AI_CLIENT_WORK_PHASE,
    CAPABILITY_AI_USAGE, CAPABILITY_APP_LAYER, CAPABILITY_BATTERY, CAPABILITY_CONFIG_RPC,
    CAPABILITY_HOST_ACTION, CAPABILITY_KEY_STATS, CAPABILITY_LAYER_STATE, CAPABILITY_THEME,
    CAPABILITY_TIME_SYNC, COMBO_ITEM_LEN, COMBO_MAX_KEYS, COMBO_MAX_SLOTS, COMBO_NAME_LEN,
    FEATURE_AI_CLIENT, FEATURE_SYSTEM, PACKET_SIZE, REPORT_SIZE,
};
pub use pending_approval::{
    claude_key, claude_launch_token_prefix, codex_key, ApprovalClient, ApprovalKey, ApprovalOwner,
    ClaudePendingResponse, CodexPendingResponse, PendingApprovalBody, PendingApprovalContent,
    PendingApprovalSnapshot, PendingApprovalStore, CLAUDE_DECISION_ALLOW, CLAUDE_DECISION_DENY,
    MAX_ENTRIES, MAX_PENDING_APPROVAL_BODY_BYTES,
};
pub use runner::{
    uplink_device_key, DeviceBatterySource, DeviceBatteryStatus, DeviceLayerState, RunEvent,
    Runner, UplinkEvent,
};
pub use stats::{KeyStatsStore, KeyStatsSummary, SharedKeyStatsStore, StatsPeriod};
pub use studio::{
    KeymapViewerStatus, StudioBinding, StudioDeviceStatus, StudioError, StudioErrorCode,
    StudioKeymapSnapshot, StudioLayer, StudioLayoutSource, StudioLockState, StudioPhysicalKey,
    StudioPhysicalLayout, StudioRpcStatus,
};
pub use time::{Clock, SystemClock, TimeSnapshot};
