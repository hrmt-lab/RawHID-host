use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::{mpsc, Arc, Mutex, RwLock},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use serde::Serialize;
use serde_json::Value;

use crate::{
    codex_broker::{
        BrokerDirection, CodexApprovalRequestBody, CodexBrokerEvent, CodexBrokerManager,
        JsonRpcKind, JsonRpcMetadata,
    },
    next_ai_session_registration_order,
    packet::{AiActivityState, AiClientType, AiClientVariant, AiWorkPhase},
    pending_approval::{
        codex_key, ApprovalClient, ApprovalKey, ApprovalOwner, PendingApprovalBody,
        PendingApprovalStore,
    },
};

const RECONNECT_GRACE: Duration = Duration::from_secs(3);
const COMPLETED_DISPLAY_DURATION: Duration = Duration::from_secs(15);
const THINKING_STABILITY: Duration = Duration::from_millis(150);
const EXECUTION_RETURN_STABILITY: Duration = Duration::from_millis(250);
const MAX_PENDING_CHANGES: usize = 64;
pub const MAX_CODEX_SESSIONS: usize = 32;

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AiClientStateChangeReason {
    SessionStarted,
    SessionForked,
    SessionReplaced,
    SessionEnded,
    TurnStarted,
    TurnCompleted,
    CompletedExpired,
    TurnFailed,
    TurnInterrupted,
    RequestStarted,
    RequestResolved,
    WorkPhaseChanged,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct AiClientStateSnapshot {
    pub client_type: AiClientType,
    pub client_variant: AiClientVariant,
    pub session_active: bool,
    pub activity_state: AiActivityState,
    pub work_phase: AiWorkPhase,
    pub revision: u16,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct AiClientStateChange {
    pub state: AiClientStateSnapshot,
    pub reason: AiClientStateChangeReason,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CodexSessionSnapshot {
    pub thread_id: String,
    pub owner_connection_id: String,
    pub terminal_target_id: String,
    pub registration_order: u64,
    pub state: AiClientStateSnapshot,
    /// Whether this thread is the one ScreenKey/HUD should show for
    /// `owner_connection_id`. This is the connection's *display target*,
    /// not necessarily the thread Codex itself last brought to the
    /// foreground (see `CodexSessionRegistry::recompute_display_target`):
    /// a thread waiting on the user (`WaitingApproval`/`WaitingInput`) wins
    /// this over a more recently focused thread that isn't waiting, so an
    /// approval prompt can never be hidden behind a short-lived side thread.
    /// Absent that, a thread that is already `Working` keeps this over a
    /// different thread that only just grabbed Codex's own focus, so a
    /// short-lived side thread (e.g. conversation-title generation) can't
    /// steal the display away from a session actively working for the user.
    /// At most one thread per `owner_connection_id` has this set.
    pub is_display_target: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CodexStateChange {
    pub thread_id: String,
    pub terminal_target_id: String,
    pub state: AiClientStateSnapshot,
    pub reason: AiClientStateChangeReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestKind {
    Approval,
    Input,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TurnOutcome {
    Completed,
    Failed,
    Interrupted,
    InterruptedWithError,
}

#[derive(Debug)]
enum AiClientEvent {
    SessionRequested {
        requested_thread_id: Option<String>,
    },
    SessionStarted {
        thread_id: String,
    },
    SessionForked {
        thread_id: String,
    },
    TurnStarted {
        thread_id: String,
        turn_id: String,
    },
    RequestStarted {
        key: String,
        kind: RequestKind,
        thread_id: String,
        turn_id: Option<String>,
    },
    RequestResolved {
        key: String,
        thread_id: String,
    },
    ItemStarted {
        thread_id: String,
        turn_id: String,
        item_id: String,
        work_phase: AiWorkPhase,
    },
    ItemCompleted {
        thread_id: String,
        turn_id: String,
        item_id: String,
    },
    TurnFinished {
        thread_id: String,
        turn_id: String,
        outcome: TurnOutcome,
    },
    ClientDisconnected,
    SessionEnded,
}

pub struct AiClientStateReducer {
    snapshot: AiClientStateSnapshot,
    has_emitted: bool,
    tracked_thread_id: Option<String>,
    tracked_turn_id: Option<String>,
    requests: HashMap<String, RequestKind>,
    active_items: HashMap<String, AiWorkPhase>,
    observed_work_phase: AiWorkPhase,
    pending_work_phase: Option<(AiWorkPhase, Instant)>,
    completed_deadline: Option<Instant>,
    reconnect_deadline: Option<Instant>,
}

impl AiClientStateReducer {
    pub fn new_codex_cli() -> Self {
        Self::with_initial_revision(initial_revision())
    }

    pub fn with_initial_revision(revision: u16) -> Self {
        Self {
            snapshot: AiClientStateSnapshot {
                client_type: AiClientType::Codex,
                client_variant: AiClientVariant::Cli,
                session_active: false,
                activity_state: AiActivityState::None,
                work_phase: AiWorkPhase::Unspecified,
                revision,
            },
            has_emitted: false,
            tracked_thread_id: None,
            tracked_turn_id: None,
            requests: HashMap::new(),
            active_items: HashMap::new(),
            observed_work_phase: AiWorkPhase::Unspecified,
            pending_work_phase: None,
            completed_deadline: None,
            reconnect_deadline: None,
        }
    }

    pub fn snapshot(&self) -> AiClientStateSnapshot {
        self.snapshot
    }

    fn apply(&mut self, event: AiClientEvent, now: Instant) -> Vec<AiClientStateChange> {
        match event {
            AiClientEvent::SessionRequested {
                requested_thread_id,
            } => {
                if !self.snapshot.session_active {
                    return Vec::new();
                }
                if requested_thread_id.as_deref() == self.tracked_thread_id.as_deref() {
                    return Vec::new();
                }
                self.clear_session();
                vec![self.emit(
                    false,
                    AiActivityState::None,
                    AiClientStateChangeReason::SessionReplaced,
                )]
            }
            AiClientEvent::SessionStarted { thread_id } => {
                if self.tracked_thread_id.as_deref() == Some(thread_id.as_str()) {
                    self.reconnect_deadline = None;
                    return Vec::new();
                }
                let mut changes = Vec::new();
                if self.snapshot.session_active {
                    self.clear_session();
                    changes.push(self.emit(
                        false,
                        AiActivityState::None,
                        AiClientStateChangeReason::SessionReplaced,
                    ));
                }
                self.tracked_thread_id = Some(thread_id);
                self.reconnect_deadline = None;
                changes.push(self.emit(
                    true,
                    AiActivityState::Available,
                    AiClientStateChangeReason::SessionStarted,
                ));
                changes
            }
            AiClientEvent::SessionForked { thread_id } => {
                if self.tracked_thread_id.as_deref() == Some(thread_id.as_str()) {
                    return Vec::new();
                }
                // A fork is a new active display thread, but it does not end the
                // parent CLI thread. Do not emit NONE between the two display
                // states; otherwise ScreenKey visibly blacks out before the forked
                // turn starts.
                self.clear_session();
                self.tracked_thread_id = Some(thread_id);
                vec![self.emit(
                    true,
                    AiActivityState::Available,
                    AiClientStateChangeReason::SessionForked,
                )]
            }
            AiClientEvent::TurnStarted { thread_id, turn_id } => {
                if self.tracked_thread_id.as_deref() != Some(thread_id.as_str()) {
                    return Vec::new();
                }
                self.tracked_turn_id = Some(turn_id);
                self.requests.clear();
                self.clear_items();
                self.completed_deadline = None;
                vec![self.emit(
                    true,
                    AiActivityState::Working,
                    AiClientStateChangeReason::TurnStarted,
                )]
            }
            AiClientEvent::RequestStarted {
                key,
                kind,
                thread_id,
                turn_id,
            } => {
                if self.tracked_thread_id.as_deref() != Some(thread_id.as_str()) {
                    return Vec::new();
                }
                if let Some(turn_id) = turn_id {
                    if self.tracked_turn_id.as_deref() != Some(turn_id.as_str()) {
                        return Vec::new();
                    }
                } else if kind == RequestKind::Approval {
                    return Vec::new();
                }
                if self.requests.insert(key, kind).is_some() {
                    return Vec::new();
                }
                vec![self.emit(
                    true,
                    self.waiting_state(),
                    AiClientStateChangeReason::RequestStarted,
                )]
            }
            AiClientEvent::RequestResolved { key, .. } => {
                if self.requests.remove(&key).is_none() {
                    return Vec::new();
                }
                vec![self.emit(
                    true,
                    self.waiting_state(),
                    AiClientStateChangeReason::RequestResolved,
                )]
            }
            AiClientEvent::ItemStarted {
                thread_id,
                turn_id,
                item_id,
                work_phase,
            } => {
                if !self.matches_turn(&thread_id, &turn_id) {
                    return Vec::new();
                }
                if self.active_items.get(&item_id) == Some(&work_phase) {
                    return Vec::new();
                }
                self.active_items.insert(item_id, work_phase);
                self.update_observed_work_phase(now)
            }
            AiClientEvent::ItemCompleted {
                thread_id,
                turn_id,
                item_id,
            } => {
                if !self.matches_turn(&thread_id, &turn_id)
                    || self.active_items.remove(&item_id).is_none()
                {
                    return Vec::new();
                }
                self.update_observed_work_phase(now)
            }
            AiClientEvent::TurnFinished {
                thread_id,
                turn_id,
                outcome,
            } => {
                if self.tracked_thread_id.as_deref() != Some(thread_id.as_str())
                    || self.tracked_turn_id.as_deref() != Some(turn_id.as_str())
                {
                    return Vec::new();
                }
                self.tracked_turn_id = None;
                self.requests.clear();
                self.clear_items();
                self.completed_deadline = if outcome == TurnOutcome::Completed {
                    Some(now + COMPLETED_DISPLAY_DURATION)
                } else {
                    None
                };
                let (activity, reason) = match outcome {
                    TurnOutcome::Completed => (
                        AiActivityState::Completed,
                        AiClientStateChangeReason::TurnCompleted,
                    ),
                    TurnOutcome::Failed | TurnOutcome::InterruptedWithError => (
                        AiActivityState::Error,
                        AiClientStateChangeReason::TurnFailed,
                    ),
                    TurnOutcome::Interrupted => (
                        AiActivityState::Available,
                        AiClientStateChangeReason::TurnInterrupted,
                    ),
                };
                vec![self.emit(true, activity, reason)]
            }
            AiClientEvent::ClientDisconnected => {
                if self.snapshot.session_active {
                    self.reconnect_deadline = Some(now + RECONNECT_GRACE);
                }
                Vec::new()
            }
            AiClientEvent::SessionEnded => self.end_session(),
        }
    }

    fn tick(&mut self, now: Instant) -> Vec<AiClientStateChange> {
        if self
            .reconnect_deadline
            .is_some_and(|deadline| now >= deadline)
        {
            return self.end_session();
        }
        if self
            .completed_deadline
            .is_some_and(|deadline| now >= deadline)
        {
            self.completed_deadline = None;
            if self.snapshot.session_active
                && self.snapshot.activity_state == AiActivityState::Completed
            {
                return vec![self.emit(
                    true,
                    AiActivityState::Available,
                    AiClientStateChangeReason::CompletedExpired,
                )];
            }
        }
        if let Some((phase, deadline)) = self.pending_work_phase {
            if now >= deadline {
                self.pending_work_phase = None;
                if self.snapshot.activity_state == AiActivityState::Working
                    && self.observed_work_phase == phase
                    && self.snapshot.work_phase != phase
                {
                    return vec![self.emit_work_phase(phase)];
                }
            }
        }
        Vec::new()
    }

    fn end_session(&mut self) -> Vec<AiClientStateChange> {
        if !self.snapshot.session_active {
            self.clear_session();
            return Vec::new();
        }
        self.clear_session();
        vec![self.emit(
            false,
            AiActivityState::None,
            AiClientStateChangeReason::SessionEnded,
        )]
    }

    fn clear_session(&mut self) {
        self.tracked_thread_id = None;
        self.tracked_turn_id = None;
        self.requests.clear();
        self.clear_items();
        self.completed_deadline = None;
        self.reconnect_deadline = None;
    }

    fn waiting_state(&self) -> AiActivityState {
        if self
            .requests
            .values()
            .any(|kind| *kind == RequestKind::Approval)
        {
            AiActivityState::WaitingApproval
        } else if self
            .requests
            .values()
            .any(|kind| *kind == RequestKind::Input)
        {
            AiActivityState::WaitingInput
        } else if self.tracked_turn_id.is_some() {
            AiActivityState::Working
        } else {
            AiActivityState::Available
        }
    }

    fn matches_turn(&self, thread_id: &str, turn_id: &str) -> bool {
        self.tracked_thread_id.as_deref() == Some(thread_id)
            && self.tracked_turn_id.as_deref() == Some(turn_id)
    }

    fn clear_items(&mut self) {
        self.active_items.clear();
        self.observed_work_phase = AiWorkPhase::Unspecified;
        self.pending_work_phase = None;
    }

    fn aggregate_work_phase(&self) -> AiWorkPhase {
        self.active_items
            .values()
            .copied()
            .max_by_key(|phase| match phase {
                AiWorkPhase::Unspecified => 0,
                AiWorkPhase::Thinking => 1,
                AiWorkPhase::Executing => 2,
                AiWorkPhase::Searching => 3,
            })
            .unwrap_or(AiWorkPhase::Unspecified)
    }

    fn update_observed_work_phase(&mut self, now: Instant) -> Vec<AiClientStateChange> {
        let next = self.aggregate_work_phase();
        if next == self.observed_work_phase {
            return Vec::new();
        }
        self.observed_work_phase = next;
        if self.snapshot.activity_state != AiActivityState::Working {
            self.pending_work_phase = None;
            return Vec::new();
        }
        if next == self.snapshot.work_phase {
            self.pending_work_phase = None;
            return Vec::new();
        }
        if matches!(next, AiWorkPhase::Executing | AiWorkPhase::Searching) {
            self.pending_work_phase = None;
            return vec![self.emit_work_phase(next)];
        }
        let delay = if matches!(
            self.snapshot.work_phase,
            AiWorkPhase::Executing | AiWorkPhase::Searching
        ) {
            EXECUTION_RETURN_STABILITY
        } else {
            THINKING_STABILITY
        };
        self.pending_work_phase = Some((next, now + delay));
        Vec::new()
    }

    fn emit_work_phase(&mut self, work_phase: AiWorkPhase) -> AiClientStateChange {
        self.snapshot.work_phase = work_phase;
        AiClientStateChange {
            state: self.snapshot,
            reason: AiClientStateChangeReason::WorkPhaseChanged,
        }
    }

    fn emit(
        &mut self,
        session_active: bool,
        activity_state: AiActivityState,
        reason: AiClientStateChangeReason,
    ) -> AiClientStateChange {
        if self.has_emitted {
            self.snapshot.revision = self.snapshot.revision.wrapping_add(1);
        } else {
            self.has_emitted = true;
        }
        self.snapshot.session_active = session_active;
        self.snapshot.activity_state = activity_state;
        self.pending_work_phase = None;
        self.snapshot.work_phase = if activity_state == AiActivityState::Working {
            self.observed_work_phase
        } else {
            AiWorkPhase::Unspecified
        };
        AiClientStateChange {
            state: self.snapshot,
            reason,
        }
    }
}

impl Default for AiClientStateReducer {
    fn default() -> Self {
        Self::new_codex_cli()
    }
}

#[derive(Debug)]
enum PendingClientRequest {
    ThreadStart,
    ThreadResume { requested_thread_id: String },
    ThreadFork,
}

#[derive(Debug)]
struct PendingServerRequest {
    thread_id: String,
    turn_id: Option<String>,
}

#[derive(Debug, Default)]
pub struct CodexEventAdapter {
    connection_id: Option<String>,
    confirmed_thread_id: Option<String>,
    announced_thread_id: Option<String>,
    owned_thread_ids: HashSet<String>,
    client_requests: HashMap<String, PendingClientRequest>,
    pending_turn_starts: HashMap<String, String>,
    authorized_unknown_turn_threads: HashSet<String>,
    server_requests: HashMap<String, PendingServerRequest>,
}

impl CodexEventAdapter {
    fn adapt(&mut self, event: CodexBrokerEvent) -> Vec<AiClientEvent> {
        match event {
            CodexBrokerEvent::ClientConnected { connection_id, .. } => {
                self.connection_id = Some(connection_id);
                self.announced_thread_id = None;
                self.owned_thread_ids.clear();
                self.client_requests.clear();
                self.pending_turn_starts.clear();
                self.authorized_unknown_turn_threads.clear();
                self.server_requests.clear();
                Vec::new()
            }
            CodexBrokerEvent::ClientDisconnected {
                connection_id,
                origin,
                ..
            } if self.connection_id.as_deref() == Some(connection_id.as_str()) => {
                self.connection_id = None;
                self.announced_thread_id = None;
                self.owned_thread_ids.clear();
                self.client_requests.clear();
                self.pending_turn_starts.clear();
                self.authorized_unknown_turn_threads.clear();
                self.server_requests.clear();
                if origin == "cli" {
                    vec![AiClientEvent::ClientDisconnected]
                } else {
                    vec![AiClientEvent::SessionEnded]
                }
            }
            CodexBrokerEvent::Message {
                connection_id,
                direction,
                metadata,
            } if self.connection_id.as_deref() == Some(connection_id.as_str()) => {
                self.adapt_message(direction, *metadata)
            }
            CodexBrokerEvent::Stopped
            | CodexBrokerEvent::Error {
                component: "lifecycle",
                ..
            } => {
                self.connection_id = None;
                self.confirmed_thread_id = None;
                self.announced_thread_id = None;
                self.owned_thread_ids.clear();
                self.client_requests.clear();
                self.pending_turn_starts.clear();
                self.authorized_unknown_turn_threads.clear();
                self.server_requests.clear();
                vec![AiClientEvent::SessionEnded]
            }
            _ => Vec::new(),
        }
    }

    fn adapt_message(
        &mut self,
        direction: BrokerDirection,
        metadata: JsonRpcMetadata,
    ) -> Vec<AiClientEvent> {
        match direction {
            BrokerDirection::CliToAppServer => self.adapt_cli_message(metadata),
            BrokerDirection::AppServerToCli => self.adapt_server_message(metadata),
        }
    }

    fn adapt_cli_message(&mut self, metadata: JsonRpcMetadata) -> Vec<AiClientEvent> {
        if metadata.kind == JsonRpcKind::Request {
            let Some(key) = metadata.id.as_ref().and_then(rpc_key) else {
                return Vec::new();
            };
            match metadata.method.as_deref() {
                Some("thread/start") => {
                    self.announced_thread_id = None;
                    self.client_requests
                        .insert(key, PendingClientRequest::ThreadStart);
                    return vec![AiClientEvent::SessionRequested {
                        requested_thread_id: None,
                    }];
                }
                Some("thread/resume") => {
                    if let Some(thread_id) = metadata.thread_id {
                        self.announced_thread_id = None;
                        self.client_requests.insert(
                            key,
                            PendingClientRequest::ThreadResume {
                                requested_thread_id: thread_id.clone(),
                            },
                        );
                        return vec![AiClientEvent::SessionRequested {
                            requested_thread_id: Some(thread_id),
                        }];
                    }
                }
                Some("thread/fork") => {
                    self.client_requests
                        .insert(key, PendingClientRequest::ThreadFork);
                }
                Some("turn/start") => {
                    if let Some(thread_id) = metadata.thread_id {
                        self.pending_turn_starts.insert(key, thread_id);
                    }
                }
                _ => {}
            }
            return Vec::new();
        }
        if metadata.kind != JsonRpcKind::Response {
            return Vec::new();
        }
        let Some(key) = metadata.id.as_ref().and_then(rpc_key) else {
            return Vec::new();
        };
        if let Some(request) = self.server_requests.remove(&key) {
            return vec![AiClientEvent::RequestResolved {
                key,
                thread_id: request.thread_id,
            }];
        }
        Vec::new()
    }

    fn handle_thread_response(
        &mut self,
        key: String,
        metadata: JsonRpcMetadata,
    ) -> Vec<AiClientEvent> {
        let Some(request) = self.client_requests.remove(&key) else {
            return Vec::new();
        };
        // This is an App Server -> CLI response, so it is the terminal result
        // for the matching client request even if it is an error response
        // without `result.thread.id`.
        let Some(result_thread_id) = metadata.result_thread_id else {
            return Vec::new();
        };
        match request {
            PendingClientRequest::ThreadStart => {
                self.confirmed_thread_id = Some(result_thread_id.clone());
                self.owned_thread_ids.insert(result_thread_id.clone());
                vec![AiClientEvent::SessionStarted {
                    thread_id: result_thread_id,
                }]
            }
            PendingClientRequest::ThreadResume {
                requested_thread_id,
            } if requested_thread_id == result_thread_id => {
                self.confirmed_thread_id = Some(result_thread_id.clone());
                self.owned_thread_ids.insert(result_thread_id.clone());
                vec![AiClientEvent::SessionStarted {
                    thread_id: result_thread_id,
                }]
            }
            PendingClientRequest::ThreadResume { .. } => {
                self.confirmed_thread_id = None;
                vec![AiClientEvent::SessionEnded]
            }
            PendingClientRequest::ThreadFork => {
                self.confirmed_thread_id = Some(result_thread_id.clone());
                self.announced_thread_id = None;
                self.owned_thread_ids.insert(result_thread_id.clone());
                vec![AiClientEvent::SessionForked {
                    thread_id: result_thread_id,
                }]
            }
        }
    }

    fn adapt_server_message(&mut self, metadata: JsonRpcMetadata) -> Vec<AiClientEvent> {
        if metadata.kind == JsonRpcKind::Response {
            let Some(key) = metadata.id.as_ref().and_then(rpc_key) else {
                return Vec::new();
            };
            if let Some(thread_id) = self.pending_turn_starts.remove(&key) {
                if !metadata.response_is_error {
                    self.authorized_unknown_turn_threads.insert(thread_id);
                }
                return Vec::new();
            }
            return self.handle_thread_response(key, metadata);
        }
        match (metadata.kind, metadata.method.as_deref()) {
            (JsonRpcKind::Notification, Some("thread/started")) => {
                let Some(thread_id) = metadata.thread_id else {
                    return Vec::new();
                };
                // App Server notifications are broadcast to all WebSocket clients.
                // A request/response pair on this connection is the ownership proof.
                let _ = thread_id;
                Vec::new()
            }
            (JsonRpcKind::Notification, Some("turn/started"))
                if metadata.turn_status.as_deref() == Some("inProgress") =>
            {
                match (metadata.thread_id, metadata.turn_id) {
                    (Some(thread_id), Some(turn_id)) => {
                        let is_known_owner = self.owned_thread_ids.contains(&thread_id);
                        let is_authorized_unknown =
                            self.authorized_unknown_turn_threads.remove(&thread_id);
                        if !is_known_owner && !is_authorized_unknown {
                            return Vec::new();
                        }
                        self.server_requests.retain(|_, request| {
                            request.thread_id != thread_id
                                || request.turn_id.as_deref() == Some(turn_id.as_str())
                        });
                        // `/side` can return to its parent without issuing a
                        // `thread/resume` request. The next turn on either the
                        // parent or fork is therefore the authoritative display
                        // focus. Switch without emitting NONE, then apply the turn.
                        let mut events = Vec::new();
                        if !is_known_owner {
                            self.owned_thread_ids.insert(thread_id.clone());
                            events.push(AiClientEvent::SessionStarted {
                                thread_id: thread_id.clone(),
                            });
                        } else if self.confirmed_thread_id.as_deref() != Some(thread_id.as_str()) {
                            self.confirmed_thread_id = Some(thread_id.clone());
                            events.push(AiClientEvent::SessionForked {
                                thread_id: thread_id.clone(),
                            });
                        }
                        events.push(AiClientEvent::TurnStarted { thread_id, turn_id });
                        events
                    }
                    _ => Vec::new(),
                }
            }
            (JsonRpcKind::Notification, Some("turn/completed")) => {
                let outcome = match metadata.turn_status.as_deref() {
                    Some("completed") => Some(TurnOutcome::Completed),
                    Some("failed") => Some(TurnOutcome::Failed),
                    Some("interrupted") if metadata.turn_has_error => {
                        Some(TurnOutcome::InterruptedWithError)
                    }
                    Some("interrupted") => Some(TurnOutcome::Interrupted),
                    _ => None,
                };
                match (metadata.thread_id, metadata.turn_id, outcome) {
                    (Some(thread_id), Some(turn_id), Some(outcome)) => {
                        if !self.owned_thread_ids.contains(&thread_id) {
                            return Vec::new();
                        }
                        self.server_requests.retain(|_, request| {
                            request.thread_id != thread_id
                                || request.turn_id.as_deref() != Some(turn_id.as_str())
                        });
                        vec![AiClientEvent::TurnFinished {
                            thread_id,
                            turn_id,
                            outcome,
                        }]
                    }
                    _ => Vec::new(),
                }
            }
            (JsonRpcKind::Notification, Some("item/started")) => {
                let (Some(thread_id), Some(turn_id), Some(item_id), Some(work_phase)) = (
                    metadata.thread_id,
                    metadata.turn_id,
                    metadata.item_id,
                    metadata.item_type.as_deref().and_then(item_work_phase),
                ) else {
                    return Vec::new();
                };
                if !self.owned_thread_ids.contains(&thread_id) {
                    return Vec::new();
                }
                vec![AiClientEvent::ItemStarted {
                    thread_id,
                    turn_id,
                    item_id,
                    work_phase,
                }]
            }
            (JsonRpcKind::Notification, Some("item/completed")) => {
                let (Some(thread_id), Some(turn_id), Some(item_id)) =
                    (metadata.thread_id, metadata.turn_id, metadata.item_id)
                else {
                    return Vec::new();
                };
                if !self.owned_thread_ids.contains(&thread_id) {
                    return Vec::new();
                }
                vec![AiClientEvent::ItemCompleted {
                    thread_id,
                    turn_id,
                    item_id,
                }]
            }
            (JsonRpcKind::Request, Some(method)) => {
                let kind = if is_approval_method(method) {
                    Some(RequestKind::Approval)
                } else if is_input_method(method) {
                    Some(RequestKind::Input)
                } else {
                    None
                };
                let (Some(kind), Some(key), Some(thread_id)) = (
                    kind,
                    metadata.id.as_ref().and_then(rpc_key),
                    metadata.thread_id,
                ) else {
                    return Vec::new();
                };
                if !self.owned_thread_ids.contains(&thread_id) {
                    return Vec::new();
                }
                if method == "item/tool/requestUserInput" && metadata.turn_id.is_none() {
                    return Vec::new();
                }
                if method.starts_with("item/")
                    && (metadata.turn_id.is_none() || metadata.item_id.is_none())
                {
                    return Vec::new();
                }
                self.server_requests.insert(
                    key.clone(),
                    PendingServerRequest {
                        thread_id: thread_id.clone(),
                        turn_id: metadata.turn_id.clone(),
                    },
                );
                vec![AiClientEvent::RequestStarted {
                    key,
                    kind,
                    thread_id,
                    turn_id: metadata.turn_id,
                }]
            }
            (JsonRpcKind::Notification, Some("serverRequest/resolved")) => {
                let Some(key) = metadata.request_id.as_ref().and_then(rpc_key) else {
                    return Vec::new();
                };
                let Some(request) = self.server_requests.remove(&key) else {
                    return Vec::new();
                };
                vec![AiClientEvent::RequestResolved {
                    key,
                    thread_id: request.thread_id,
                }]
            }
            // `error` notifications remain diagnostic metadata. They do not end a Turn.
            _ => Vec::new(),
        }
    }
}

struct CodexSessionEntry {
    owner_connection_id: String,
    terminal_target_id: String,
    registration_order: u64,
    reducer: AiClientStateReducer,
}

pub struct CodexSessionRegistry {
    sessions: HashMap<String, CodexSessionEntry>,
    order: Vec<String>,
    next_revision: u16,
    selected_thread_id: Option<String>,
    /// Per-connection Codex focus: the one Codex thread that connection last
    /// started/forked/turned on, even though older threads on the same
    /// connection keep their reducers running in `sessions`. This tracks
    /// what Codex itself has brought forward; it is *not* the display
    /// selection ScreenKey/HUD use (see `effective_focus_by_connection`,
    /// which layers waiting-thread priority on top of this).
    focused: HashMap<String, String>,
    connection_targets: HashMap<String, String>,
    /// Monotonic stamp recording, for each thread currently
    /// `WaitingApproval`/`WaitingInput`, when it most recently entered that
    /// state. Used only to break ties in `effective_focus_by_connection`
    /// when a connection has more than one thread waiting at once: the
    /// thread that started waiting first keeps the display slot. Entries
    /// are removed once the thread stops waiting.
    waiting_order: HashMap<String, u64>,
    next_waiting_order: u64,
    /// Per-connection display target, as last computed by
    /// `recompute_display_target` at the end of `apply()`. This is the
    /// single source of truth `effective_focus_by_connection` reads; it
    /// exists (rather than recomputing fresh every time) so that rule 2
    /// there — "stay on a thread that is still `Working` rather than
    /// jumping to whatever `self.focused` says" — has somewhere to
    /// remember what was previously on screen. Cleaned up wherever
    /// `waiting_order` is: non-graceful disconnect, `end_all`, and thread
    /// retirement in `tick`.
    display_target: HashMap<String, String>,
}

impl Default for CodexSessionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl CodexSessionRegistry {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            order: Vec::new(),
            next_revision: initial_revision(),
            selected_thread_id: None,
            focused: HashMap::new(),
            connection_targets: HashMap::new(),
            waiting_order: HashMap::new(),
            next_waiting_order: 0,
            display_target: HashMap::new(),
        }
    }

    pub fn snapshots(&self) -> Vec<CodexSessionSnapshot> {
        let display_targets = self.effective_focus_by_connection();
        self.order
            .iter()
            .filter_map(|thread_id| {
                self.sessions.get(thread_id).map(|entry| {
                    // A thread is the display target only if it is still the
                    // thread its *current* owner resolves to; if ownership
                    // moved on, the previous owner's selection is stale and
                    // simply no longer matches here.
                    let is_display_target =
                        display_targets.get(&entry.owner_connection_id) == Some(thread_id);
                    CodexSessionSnapshot {
                        thread_id: thread_id.clone(),
                        owner_connection_id: entry.owner_connection_id.clone(),
                        terminal_target_id: entry.terminal_target_id.clone(),
                        registration_order: entry.registration_order,
                        state: entry.reducer.snapshot(),
                        is_display_target,
                    }
                })
            })
            .collect()
    }

    /// Returns, for each connection, the single thread `snapshots()` should
    /// mark as the display target. Purely a read of `self.display_target`,
    /// which `recompute_display_target` keeps up to date at the end of every
    /// `apply()` call — see that method for the preference order this
    /// reflects.
    fn effective_focus_by_connection(&self) -> HashMap<String, String> {
        self.display_target.clone()
    }

    /// Recomputes and stores `connection_id`'s display target, reflecting
    /// the event `apply()` just finished processing (thread state,
    /// `self.focused`, and `self.waiting_order` are all already up to date
    /// for it by the time this runs).
    ///
    /// Preference order:
    /// 1. A thread currently `WaitingApproval`/`WaitingInput` — the AI is
    ///    waiting on the user there, and that must never be hidden behind a
    ///    thread that merely happens to be more recently focused. If several
    ///    threads on the same connection are waiting at once, the
    ///    connection's Codex-focused thread (`self.focused`) wins when it is
    ///    among them; otherwise the thread that started waiting first
    ///    (`self.waiting_order`) wins, a stable tiebreak independent of
    ///    wall-clock races.
    /// 2. Otherwise, if the connection's *current* display target was
    ///    already `Working` before this event (`display_target_was_working`,
    ///    sampled by the caller before anything was mutated) and is still
    ///    `Working` now, keep it. A thread actively working for the user
    ///    must not lose the display slot just because some other, possibly
    ///    short-lived thread grabs `self.focused` — or even completes —
    ///    while it runs. Requiring it to have *already* been `Working`
    ///    (not just `Working` now) matters: a thread that only just
    ///    transitioned out of `WaitingApproval`/`WaitingInput` into
    ///    `Working` on this very event hasn't earned this stickiness yet,
    ///    and falls through to rule 3 instead — the display target returns
    ///    to `self.focused` as soon as a wait resolves, same as before this
    ///    rule existed.
    /// 3. Otherwise, the connection's Codex-focused thread (`self.focused`).
    ///
    /// Still exactly one candidate thread per connection either way.
    fn recompute_display_target(&mut self, connection_id: &str, display_target_was_working: bool) {
        let waiting_threads: Vec<String> = self
            .order
            .iter()
            .filter(|thread_id| {
                self.sessions.get(*thread_id).is_some_and(|entry| {
                    entry.owner_connection_id == connection_id
                        && matches!(
                            entry.reducer.snapshot().activity_state,
                            AiActivityState::WaitingApproval | AiActivityState::WaitingInput
                        )
                })
            })
            .cloned()
            .collect();

        let chosen: Option<String> = if !waiting_threads.is_empty() {
            let codex_focused = self.focused.get(connection_id);
            Some(
                codex_focused
                    .filter(|focused_thread| waiting_threads.iter().any(|t| t == *focused_thread))
                    .cloned()
                    .unwrap_or_else(|| {
                        waiting_threads
                            .into_iter()
                            .min_by_key(|thread_id| {
                                self.waiting_order
                                    .get(thread_id)
                                    .copied()
                                    .unwrap_or(u64::MAX)
                            })
                            .expect("checked non-empty above")
                    }),
            )
        } else {
            let sticky = if display_target_was_working {
                self.display_target
                    .get(connection_id)
                    .cloned()
                    .filter(|thread_id| {
                        self.sessions.get(thread_id).is_some_and(|entry| {
                            entry.owner_connection_id == connection_id
                                && entry.reducer.snapshot().activity_state
                                    == AiActivityState::Working
                        })
                    })
            } else {
                None
            };
            sticky.or_else(|| self.focused.get(connection_id).cloned())
        };

        match chosen {
            Some(thread_id) => {
                self.display_target
                    .insert(connection_id.to_string(), thread_id);
            }
            None => {
                self.display_target.remove(connection_id);
            }
        }
    }

    pub fn set_selected_thread(&mut self, thread_id: Option<String>) {
        self.selected_thread_id = thread_id;
    }

    fn register_connection(&mut self, connection_id: String, terminal_target_id: String) {
        self.connection_targets
            .insert(connection_id, terminal_target_id);
    }

    fn apply(
        &mut self,
        connection_id: &str,
        event: AiClientEvent,
        now: Instant,
    ) -> Vec<CodexStateChange> {
        // Sampled before anything below mutates thread state, so this
        // reflects whether the *current* display target was already
        // `Working` going into this event — see `recompute_display_target`
        // rule 2.
        let display_target_was_working = self
            .display_target
            .get(connection_id)
            .and_then(|thread_id| self.sessions.get(thread_id))
            .is_some_and(|entry| {
                entry.reducer.snapshot().activity_state == AiActivityState::Working
            });

        if let AiClientEvent::SessionStarted { thread_id }
        | AiClientEvent::SessionForked { thread_id } = &event
        {
            if !self.sessions.contains_key(thread_id) && !self.make_room() {
                tracing::warn!(thread_id, "Codex session registry limit reached");
                return Vec::new();
            }
            if !self.sessions.contains_key(thread_id) {
                let revision = self.allocate_revision();
                self.sessions.insert(
                    thread_id.clone(),
                    CodexSessionEntry {
                        owner_connection_id: connection_id.to_string(),
                        terminal_target_id: self
                            .connection_targets
                            .get(connection_id)
                            .cloned()
                            .unwrap_or_default(),
                        registration_order: next_ai_session_registration_order(),
                        reducer: AiClientStateReducer::with_initial_revision(revision),
                    },
                );
                self.order.push(thread_id.clone());
            } else {
                let entry = self.sessions.get_mut(thread_id).expect("checked above");
                if entry.owner_connection_id != connection_id {
                    tracing::info!(
                        thread_id,
                        old_connection_id = %entry.owner_connection_id,
                        new_connection_id = connection_id,
                        "Codex thread ownership transferred"
                    );
                    entry.owner_connection_id = connection_id.to_string();
                    entry.terminal_target_id = self
                        .connection_targets
                        .get(connection_id)
                        .cloned()
                        .unwrap_or_default();
                }
            }
        }

        let Some(thread_id) = semantic_thread_id(&event).map(str::to_string) else {
            return Vec::new();
        };
        let Some(entry) = self.sessions.get_mut(&thread_id) else {
            return Vec::new();
        };
        if entry.owner_connection_id != connection_id {
            tracing::warn!(thread_id, connection_id, "ignored non-owner Codex event");
            return Vec::new();
        }
        match &event {
            // A fresh start/fork is unambiguously the connection's active
            // thread. `TurnStarted` also refocuses: Codex's `/side` can
            // return control to the parent thread without ever issuing a
            // `thread/resume`, so a turn on an owned thread is the only
            // signal we get that focus moved back.
            AiClientEvent::SessionStarted { .. }
            | AiClientEvent::SessionForked { .. }
            | AiClientEvent::TurnStarted { .. } => {
                self.focused
                    .insert(connection_id.to_string(), thread_id.clone());
            }
            _ => {}
        }
        let changes = entry.reducer.apply(event, now);
        let activity_state = entry.reducer.snapshot().activity_state;
        let terminal_target_id = entry.terminal_target_id.clone();
        let result: Vec<CodexStateChange> = changes
            .into_iter()
            .map(|change| CodexStateChange {
                thread_id: thread_id.clone(),
                terminal_target_id: terminal_target_id.clone(),
                state: change.state,
                reason: change.reason,
            })
            .collect();

        // Stamp (or clear) this thread's waiting-order entry so
        // `effective_focus_by_connection` can break ties between
        // simultaneously waiting threads deterministically.
        if matches!(
            activity_state,
            AiActivityState::WaitingApproval | AiActivityState::WaitingInput
        ) {
            self.waiting_order
                .entry(thread_id.clone())
                .or_insert_with(|| {
                    let order = self.next_waiting_order;
                    self.next_waiting_order = self.next_waiting_order.wrapping_add(1);
                    order
                });
        } else {
            self.waiting_order.remove(&thread_id);
        }

        self.recompute_display_target(connection_id, display_target_was_working);

        result
    }

    fn disconnect(
        &mut self,
        connection_id: &str,
        graceful: bool,
        now: Instant,
    ) -> Vec<CodexStateChange> {
        let owned: Vec<String> = self
            .order
            .iter()
            .filter(|thread_id| {
                self.sessions
                    .get(*thread_id)
                    .is_some_and(|entry| entry.owner_connection_id == connection_id)
            })
            .cloned()
            .collect();
        let changes = owned
            .iter()
            .cloned()
            .into_iter()
            .flat_map(|thread_id| {
                let entry = self.sessions.get_mut(&thread_id).expect("collected above");
                let event = if graceful {
                    AiClientEvent::ClientDisconnected
                } else {
                    AiClientEvent::SessionEnded
                };
                let terminal_target_id = entry.terminal_target_id.clone();
                entry
                    .reducer
                    .apply(event, now)
                    .into_iter()
                    .map(move |change| CodexStateChange {
                        thread_id: thread_id.clone(),
                        terminal_target_id: terminal_target_id.clone(),
                        state: change.state,
                        reason: change.reason,
                    })
            })
            .collect();
        if !graceful {
            for thread_id in &owned {
                self.sessions.remove(thread_id);
                self.waiting_order.remove(thread_id);
            }
            self.order.retain(|thread_id| !owned.contains(thread_id));
            // A non-graceful disconnect ends this connection outright; leaving
            // its focus behind would let a later reused connection_id inherit
            // a stale focus pointer.
            self.focused.remove(connection_id);
            self.display_target.remove(connection_id);
            self.connection_targets.remove(connection_id);
        }
        changes
    }

    fn end_all(&mut self, now: Instant) -> Vec<CodexStateChange> {
        self.focused.clear();
        self.waiting_order.clear();
        self.display_target.clear();
        let keys = self.order.clone();
        let mut changes = Vec::new();
        for thread_id in keys {
            let Some(entry) = self.sessions.get_mut(&thread_id) else {
                continue;
            };
            changes.extend(
                entry
                    .reducer
                    .apply(AiClientEvent::SessionEnded, now)
                    .into_iter()
                    .map(|change| CodexStateChange {
                        thread_id: thread_id.clone(),
                        terminal_target_id: entry.terminal_target_id.clone(),
                        state: change.state,
                        reason: change.reason,
                    }),
            );
        }
        self.sessions.clear();
        self.order.clear();
        changes
    }

    fn tick(&mut self, now: Instant) -> Vec<CodexStateChange> {
        let mut changes = Vec::new();
        let mut retired = Vec::new();
        for thread_id in &self.order {
            let Some(entry) = self.sessions.get_mut(thread_id) else {
                continue;
            };
            for change in entry.reducer.tick(now) {
                if change.reason == AiClientStateChangeReason::SessionEnded {
                    retired.push(thread_id.clone());
                }
                changes.push(CodexStateChange {
                    thread_id: thread_id.clone(),
                    terminal_target_id: entry.terminal_target_id.clone(),
                    state: change.state,
                    reason: change.reason,
                });
            }
        }
        for thread_id in retired {
            self.sessions.remove(&thread_id);
            self.order.retain(|candidate| candidate != &thread_id);
            self.waiting_order.remove(&thread_id);
            self.display_target.retain(|_, target| target != &thread_id);
        }
        changes
    }

    fn make_room(&mut self) -> bool {
        if self.sessions.len() < MAX_CODEX_SESSIONS {
            return true;
        }
        let candidate = self.order.iter().find(|thread_id| {
            self.selected_thread_id.as_ref() != Some(*thread_id)
                && self.sessions.get(*thread_id).is_some_and(|entry| {
                    !matches!(
                        entry.reducer.snapshot().activity_state,
                        AiActivityState::Working
                            | AiActivityState::WaitingApproval
                            | AiActivityState::WaitingInput
                    )
                })
        });
        let Some(candidate) = candidate.cloned() else {
            return false;
        };
        self.sessions.remove(&candidate);
        self.order.retain(|thread_id| thread_id != &candidate);
        true
    }

    fn allocate_revision(&mut self) -> u16 {
        let revision = self.next_revision;
        self.next_revision = self.next_revision.wrapping_add(1);
        revision
    }
}

fn semantic_thread_id(event: &AiClientEvent) -> Option<&str> {
    match event {
        AiClientEvent::SessionStarted { thread_id }
        | AiClientEvent::SessionForked { thread_id }
        | AiClientEvent::TurnStarted { thread_id, .. }
        | AiClientEvent::RequestStarted { thread_id, .. }
        | AiClientEvent::RequestResolved { thread_id, .. }
        | AiClientEvent::ItemStarted { thread_id, .. }
        | AiClientEvent::ItemCompleted { thread_id, .. }
        | AiClientEvent::TurnFinished { thread_id, .. } => Some(thread_id),
        AiClientEvent::SessionRequested { .. }
        | AiClientEvent::ClientDisconnected
        | AiClientEvent::SessionEnded => None,
    }
}

pub struct CodexActivityRuntime {
    snapshots: Arc<RwLock<Vec<CodexSessionSnapshot>>>,
    changes: Arc<Mutex<VecDeque<CodexStateChange>>>,
    selected_thread_id: Arc<RwLock<Option<String>>>,
    pending_approvals: Arc<PendingApprovalStore>,
    stop_tx: mpsc::Sender<()>,
    worker: Option<thread::JoinHandle<()>>,
}

impl CodexActivityRuntime {
    pub fn start(broker: CodexBrokerManager) -> Self {
        let snapshots = Arc::new(RwLock::new(Vec::new()));
        let worker_snapshots = snapshots.clone();
        let changes = Arc::new(Mutex::new(VecDeque::new()));
        let worker_changes = changes.clone();
        let selected_thread_id = Arc::new(RwLock::new(None));
        let worker_selected_thread_id = selected_thread_id.clone();
        let pending_approvals = Arc::new(PendingApprovalStore::new());
        let worker_pending_approvals = pending_approvals.clone();
        let (stop_tx, stop_rx) = mpsc::channel();
        let worker = thread::Builder::new()
            .name("codex-activity-reducer".to_string())
            .spawn(move || {
                run_activity_loop(
                    broker,
                    worker_snapshots,
                    worker_changes,
                    worker_selected_thread_id,
                    worker_pending_approvals,
                    stop_rx,
                )
            })
            .expect("failed to create Codex Activity reducer thread");
        Self {
            snapshots,
            changes,
            selected_thread_id,
            pending_approvals,
            stop_tx,
            worker: Some(worker),
        }
    }

    /// Shared handle to the store of unresolved Codex approval-request
    /// bodies. Populated by this runtime's own event loop (see
    /// `resolve_codex_approval` below) regardless of whether anything
    /// reads from it yet -- there is no HUD consumer in this phase (see
    /// `docs/ai-approval-hud-design.md` §13, stage 1).
    pub fn pending_approvals(&self) -> Arc<PendingApprovalStore> {
        self.pending_approvals.clone()
    }

    pub fn snapshot(&self) -> AiClientStateSnapshot {
        self.snapshots()
            .into_iter()
            .find(|snapshot| snapshot.state.session_active)
            .map(|snapshot| snapshot.state)
            .unwrap_or(AiClientStateSnapshot {
                client_type: AiClientType::Codex,
                client_variant: AiClientVariant::Cli,
                session_active: false,
                activity_state: AiActivityState::Available,
                work_phase: AiWorkPhase::Unspecified,
                revision: 0,
            })
    }

    pub fn try_recv_change(&self) -> Option<AiClientStateChange> {
        self.try_recv_session_change()
            .map(|change| AiClientStateChange {
                state: change.state,
                reason: change.reason,
            })
    }

    pub fn snapshots(&self) -> Vec<CodexSessionSnapshot> {
        self.snapshots.read().unwrap().clone()
    }

    pub fn try_recv_session_change(&self) -> Option<CodexStateChange> {
        self.changes.lock().unwrap().pop_front()
    }

    pub fn set_selected_thread(&self, thread_id: Option<String>) {
        *self.selected_thread_id.write().unwrap() = thread_id;
    }
}

impl Drop for CodexActivityRuntime {
    fn drop(&mut self) {
        let _ = self.stop_tx.send(());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn run_activity_loop(
    broker: CodexBrokerManager,
    snapshots: Arc<RwLock<Vec<CodexSessionSnapshot>>>,
    changes: Arc<Mutex<VecDeque<CodexStateChange>>>,
    selected_thread_id: Arc<RwLock<Option<String>>>,
    pending_approvals: Arc<PendingApprovalStore>,
    stop_rx: mpsc::Receiver<()>,
) {
    let mut adapters: HashMap<String, CodexEventAdapter> = HashMap::new();
    let mut registry = CodexSessionRegistry::new();
    // Tracks the approval key of every still-open requestApproval by the
    // turn it belongs to, purely so `turn/completed` can discard a
    // leftover entry as a safety net (see `resolve_codex_approval`). The
    // primary resolution paths -- the CLI's own response, and
    // `serverRequest/resolved` -- key off the request id directly and
    // don't need this map. Keyed by (connection_id, thread_id, turn_id)
    // because more than one command can be approved within one turn.
    let mut approval_turns: HashMap<(String, String, String), Vec<ApprovalKey>> = HashMap::new();
    loop {
        if stop_rx.try_recv().is_ok() {
            break;
        }
        let now = Instant::now();
        registry.set_selected_thread(selected_thread_id.read().unwrap().clone());
        publish_registry_changes(
            registry.tick(now),
            &registry,
            &snapshots,
            &changes,
            &selected_thread_id,
        );
        match broker.recv_event_timeout(Duration::from_millis(100)) {
            Ok(event) => {
                let now = Instant::now();
                publish_registry_changes(
                    registry.tick(now),
                    &registry,
                    &snapshots,
                    &changes,
                    &selected_thread_id,
                );
                match event {
                    CodexBrokerEvent::ManagedClientConnected {
                        connection_id,
                        terminal_target_id,
                    } => {
                        registry
                            .register_connection(connection_id.clone(), terminal_target_id.clone());
                        let mut adapter = CodexEventAdapter::default();
                        adapter.adapt(CodexBrokerEvent::ClientConnected {
                            connection_id: connection_id.clone(),
                        });
                        adapters.insert(connection_id, adapter);
                    }
                    CodexBrokerEvent::ClientDisconnected {
                        connection_id,
                        origin,
                        ..
                    } => {
                        adapters.remove(&connection_id);
                        pending_approvals.clear_owner(&ApprovalOwner::Codex {
                            connection_id: connection_id.clone(),
                        });
                        approval_turns.retain(|(owner, _, _), _| owner != &connection_id);
                        let changes_for_connection =
                            registry.disconnect(&connection_id, origin == "cli", now);
                        publish_registry_changes(
                            changes_for_connection,
                            &registry,
                            &snapshots,
                            &changes,
                            &selected_thread_id,
                        );
                    }
                    CodexBrokerEvent::ApprovalRequestBody {
                        connection_id,
                        request_id,
                        body,
                    } => {
                        ingest_codex_approval(
                            &pending_approvals,
                            &mut approval_turns,
                            &connection_id,
                            &request_id,
                            *body,
                        );
                    }
                    CodexBrokerEvent::Message {
                        connection_id,
                        direction,
                        metadata,
                    } => {
                        resolve_codex_approval(
                            &pending_approvals,
                            &mut approval_turns,
                            &connection_id,
                            direction,
                            &metadata,
                        );
                        let Some(adapter) = adapters.get_mut(&connection_id) else {
                            continue;
                        };
                        let semantic = adapter.adapt(CodexBrokerEvent::Message {
                            connection_id: connection_id.clone(),
                            direction,
                            metadata,
                        });
                        for event in semantic {
                            let applied = registry.apply(&connection_id, event, now);
                            publish_registry_changes(
                                applied,
                                &registry,
                                &snapshots,
                                &changes,
                                &selected_thread_id,
                            );
                        }
                    }
                    CodexBrokerEvent::Stopped
                    | CodexBrokerEvent::Error {
                        component: "lifecycle",
                        ..
                    } => {
                        adapters.clear();
                        pending_approvals.clear_client(ApprovalClient::Codex);
                        approval_turns.clear();
                        let ended = registry.end_all(now);
                        publish_registry_changes(
                            ended,
                            &registry,
                            &snapshots,
                            &changes,
                            &selected_thread_id,
                        );
                    }
                    CodexBrokerEvent::Error { component, detail } => {
                        tracing::warn!(%component, %detail, "Codex Broker connection-local error");
                    }
                    _ => {}
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                pending_approvals.clear_client(ApprovalClient::Codex);
                approval_turns.clear();
                let ended = registry.end_all(Instant::now());
                publish_registry_changes(
                    ended,
                    &registry,
                    &snapshots,
                    &changes,
                    &selected_thread_id,
                );
                break;
            }
        }
    }
}

/// Records the body of one Codex `requestApproval` into `pending_approvals`.
///
/// This is a sibling of `CodexEventAdapter`/`CodexSessionRegistry`, not a
/// part of them: it never reads or writes `AiClientStateReducer`'s state,
/// and its failure or absence has no effect on the activity state machine
/// those drive. It exists purely to keep the store in sync with what the
/// Broker observed.
fn ingest_codex_approval(
    pending_approvals: &PendingApprovalStore,
    approval_turns: &mut HashMap<(String, String, String), Vec<ApprovalKey>>,
    connection_id: &str,
    request_id: &Value,
    body: CodexApprovalRequestBody,
) {
    let key = crate::pending_approval::codex_key_for_thread(
        connection_id,
        request_id,
        body.thread_id.as_deref(),
    );
    if let (Some(thread_id), Some(turn_id)) = (body.thread_id.clone(), body.turn_id.clone()) {
        approval_turns
            .entry((connection_id.to_string(), thread_id, turn_id))
            .or_default()
            .push(key.clone());
    }
    let owner = ApprovalOwner::Codex {
        connection_id: connection_id.to_string(),
    };
    let normalized = PendingApprovalBody {
        primary_text: body.command_actions.first().cloned(),
        full_command: body.command,
        reason: body.reason,
        cwd: body.cwd,
        kind: body.kind,
        available_decisions: Some(body.available_decisions),
        // Codex's own request id lives in `key`/`approval_turns` above, not
        // in the body -- these three fields are Claude-only. Codex has no
        // `permission_suggestions` counterpart at all (see that field's own
        // doc comment on `PendingApprovalBody`).
        tool_use_id: None,
        prompt_id: None,
        permission_suggestions: None,
    };
    pending_approvals.insert(key, ApprovalClient::Codex, owner, normalized);
}

/// Resolution triggers for `pending_approvals`, read from the same
/// `JsonRpcMetadata` the activity reducer already receives -- never from
/// the request body itself (that only ever flows once, via
/// `ApprovalRequestBody`/`ingest_codex_approval`). Three signals resolve an
/// entry:
/// - the CLI's own response to a `requestApproval` it saw normally
///   (`CliToAppServer`, matching JSON-RPC id);
/// - `serverRequest/resolved`, observed ~2ms after a Broker-held request is
///   answered (KO-2 §4);
/// - `turn/completed`, a ~2.8s-later safety net for any request the first
///   two signals missed (KO-2 §4).
fn resolve_codex_approval(
    pending_approvals: &PendingApprovalStore,
    approval_turns: &mut HashMap<(String, String, String), Vec<ApprovalKey>>,
    connection_id: &str,
    direction: BrokerDirection,
    metadata: &JsonRpcMetadata,
) {
    match direction {
        BrokerDirection::CliToAppServer => {
            if metadata.kind == JsonRpcKind::Response {
                if let Some(id) = metadata.id.as_ref() {
                    pending_approvals.resolve(&codex_key(connection_id, id));
                }
            }
        }
        BrokerDirection::AppServerToCli => match metadata.method.as_deref() {
            Some("serverRequest/resolved") => {
                if let Some(request_id) = metadata.request_id.as_ref() {
                    pending_approvals.resolve(&codex_key(connection_id, request_id));
                }
            }
            Some("turn/completed") => {
                if let (Some(thread_id), Some(turn_id)) =
                    (metadata.thread_id.as_deref(), metadata.turn_id.as_deref())
                {
                    let map_key = (
                        connection_id.to_string(),
                        thread_id.to_string(),
                        turn_id.to_string(),
                    );
                    if let Some(keys) = approval_turns.remove(&map_key) {
                        for key in keys {
                            pending_approvals.resolve(&key);
                        }
                    }
                }
            }
            _ => {}
        },
    }
}

fn publish_registry_changes(
    changes: Vec<CodexStateChange>,
    registry: &CodexSessionRegistry,
    snapshots: &Arc<RwLock<Vec<CodexSessionSnapshot>>>,
    pending: &Arc<Mutex<VecDeque<CodexStateChange>>>,
    selected_thread_id: &Arc<RwLock<Option<String>>>,
) {
    *snapshots.write().unwrap() = registry.snapshots();
    let selected_thread_id = selected_thread_id.read().unwrap().clone();
    for change in changes {
        let mut pending = pending.lock().unwrap();
        if pending.len() == MAX_PENDING_CHANGES {
            let eviction_index = pending
                .iter()
                .position(|queued| Some(&queued.thread_id) != selected_thread_id.as_ref())
                .unwrap_or(0);
            pending.remove(eviction_index);
        }
        pending.push_back(change);
    }
}

fn is_approval_method(method: &str) -> bool {
    matches!(
        method,
        "item/commandExecution/requestApproval"
            | "item/fileChange/requestApproval"
            | "item/permissions/requestApproval"
    )
}

fn is_input_method(method: &str) -> bool {
    matches!(
        method,
        "item/tool/requestUserInput" | "mcpServer/elicitation/request"
    )
}

fn item_work_phase(item_type: &str) -> Option<AiWorkPhase> {
    match item_type {
        "reasoning" | "agentMessage" | "plan" => Some(AiWorkPhase::Thinking),
        "commandExecution"
        | "fileChange"
        | "mcpToolCall"
        | "dynamicToolCall"
        | "collabAgentToolCall"
        | "subAgentActivity"
        | "imageView"
        | "imageGeneration"
        | "sleep" => Some(AiWorkPhase::Executing),
        "webSearch" => Some(AiWorkPhase::Searching),
        _ => None,
    }
}

fn rpc_key(value: &Value) -> Option<String> {
    match value {
        Value::Null => Some("null".to_string()),
        Value::String(value) => Some(format!("s:{value}")),
        Value::Number(value) => Some(format!("n:{value}")),
        _ => None,
    }
}

fn initial_revision() -> u16 {
    let mut bytes = [0_u8; 2];
    if getrandom::fill(&mut bytes).is_ok() {
        return u16::from_le_bytes(bytes);
    }
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or_default();
    (nanos ^ u64::from(std::process::id())) as u16
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pending_approval::PendingApprovalContent;

    const THREAD_A: &str = "thread-a";
    const THREAD_B: &str = "thread-b";
    const TURN_A: &str = "turn-a";
    const TURN_B: &str = "turn-b";

    fn apply_one(
        reducer: &mut AiClientStateReducer,
        event: AiClientEvent,
        now: Instant,
    ) -> AiClientStateChange {
        let changes = reducer.apply(event, now);
        assert_eq!(changes.len(), 1);
        changes[0]
    }

    fn start_session(reducer: &mut AiClientStateReducer, now: Instant) -> AiClientStateChange {
        apply_one(
            reducer,
            AiClientEvent::SessionStarted {
                thread_id: THREAD_A.to_string(),
            },
            now,
        )
    }

    fn start_turn(reducer: &mut AiClientStateReducer, now: Instant) -> AiClientStateChange {
        apply_one(
            reducer,
            AiClientEvent::TurnStarted {
                thread_id: THREAD_A.to_string(),
                turn_id: TURN_A.to_string(),
            },
            now,
        )
    }

    fn message(direction: BrokerDirection, json: &str) -> CodexBrokerEvent {
        CodexBrokerEvent::Message {
            connection_id: "connection-1".to_string(),
            direction,
            metadata: Box::new(crate::codex_broker::classify_json_rpc(json)),
        }
    }

    #[test]
    fn first_state_uses_initial_revision_and_subsequent_states_wrap() {
        let now = Instant::now();
        let mut reducer = AiClientStateReducer::with_initial_revision(u16::MAX);

        let session = start_session(&mut reducer, now);
        assert_eq!(session.reason, AiClientStateChangeReason::SessionStarted);
        assert_eq!(session.state.revision, u16::MAX);
        assert_eq!(session.state.activity_state, AiActivityState::Available);

        let turn = start_turn(&mut reducer, now);
        assert_eq!(turn.state.revision, 0);
        assert_eq!(turn.state.activity_state, AiActivityState::Working);
    }

    #[test]
    fn approval_has_priority_over_input_and_resolution_restores_prior_state() {
        let now = Instant::now();
        let mut reducer = AiClientStateReducer::with_initial_revision(10);
        start_session(&mut reducer, now);
        start_turn(&mut reducer, now);

        let input = apply_one(
            &mut reducer,
            AiClientEvent::RequestStarted {
                key: "input".to_string(),
                kind: RequestKind::Input,
                thread_id: THREAD_A.to_string(),
                turn_id: Some(TURN_A.to_string()),
            },
            now,
        );
        assert_eq!(input.state.activity_state, AiActivityState::WaitingInput);

        let approval = apply_one(
            &mut reducer,
            AiClientEvent::RequestStarted {
                key: "approval".to_string(),
                kind: RequestKind::Approval,
                thread_id: THREAD_A.to_string(),
                turn_id: Some(TURN_A.to_string()),
            },
            now,
        );
        assert_eq!(
            approval.state.activity_state,
            AiActivityState::WaitingApproval
        );

        let after_approval = apply_one(
            &mut reducer,
            AiClientEvent::RequestResolved {
                key: "approval".to_string(),
                thread_id: THREAD_A.to_string(),
            },
            now,
        );
        assert_eq!(
            after_approval.state.activity_state,
            AiActivityState::WaitingInput
        );

        let after_input = apply_one(
            &mut reducer,
            AiClientEvent::RequestResolved {
                key: "input".to_string(),
                thread_id: THREAD_A.to_string(),
            },
            now,
        );
        assert_eq!(after_input.state.activity_state, AiActivityState::Working);
    }

    #[test]
    fn completed_failed_and_interrupted_turns_follow_the_contract() {
        let now = Instant::now();
        let outcomes = [
            (TurnOutcome::Completed, AiActivityState::Completed),
            (TurnOutcome::Failed, AiActivityState::Error),
            (TurnOutcome::InterruptedWithError, AiActivityState::Error),
            (TurnOutcome::Interrupted, AiActivityState::Available),
        ];

        for (outcome, expected) in outcomes {
            let mut reducer = AiClientStateReducer::with_initial_revision(1);
            start_session(&mut reducer, now);
            start_turn(&mut reducer, now);
            let change = apply_one(
                &mut reducer,
                AiClientEvent::TurnFinished {
                    thread_id: THREAD_A.to_string(),
                    turn_id: TURN_A.to_string(),
                    outcome,
                },
                now,
            );
            assert_eq!(change.state.activity_state, expected);
            assert!(change.state.session_active);
        }
    }

    #[test]
    fn completed_state_expires_to_available_after_fifteen_seconds() {
        assert_eq!(COMPLETED_DISPLAY_DURATION, Duration::from_secs(15));
        let now = Instant::now();
        let mut reducer = AiClientStateReducer::with_initial_revision(70);
        start_session(&mut reducer, now);
        start_turn(&mut reducer, now);
        let completed = apply_one(
            &mut reducer,
            AiClientEvent::TurnFinished {
                thread_id: THREAD_A.to_string(),
                turn_id: TURN_A.to_string(),
                outcome: TurnOutcome::Completed,
            },
            now,
        );

        assert_eq!(completed.state.activity_state, AiActivityState::Completed);
        assert!(reducer
            .tick(now + COMPLETED_DISPLAY_DURATION - Duration::from_millis(1))
            .is_empty());

        let expired = reducer.tick(now + COMPLETED_DISPLAY_DURATION);
        assert_eq!(expired.len(), 1);
        assert_eq!(
            expired[0].reason,
            AiClientStateChangeReason::CompletedExpired
        );
        assert_eq!(expired[0].state.activity_state, AiActivityState::Available);
        assert_eq!(expired[0].state.work_phase, AiWorkPhase::Unspecified);
        assert_eq!(
            expired[0].state.revision,
            completed.state.revision.wrapping_add(1)
        );
        assert!(reducer
            .tick(now + COMPLETED_DISPLAY_DURATION + Duration::from_secs(1))
            .is_empty());
    }

    #[test]
    fn starting_a_new_turn_cancels_completed_expiration() {
        let now = Instant::now();
        let mut reducer = AiClientStateReducer::with_initial_revision(80);
        start_session(&mut reducer, now);
        start_turn(&mut reducer, now);
        apply_one(
            &mut reducer,
            AiClientEvent::TurnFinished {
                thread_id: THREAD_A.to_string(),
                turn_id: TURN_A.to_string(),
                outcome: TurnOutcome::Completed,
            },
            now,
        );
        apply_one(
            &mut reducer,
            AiClientEvent::TurnStarted {
                thread_id: THREAD_A.to_string(),
                turn_id: "turn-b".to_string(),
            },
            now + Duration::from_secs(1),
        );

        assert!(reducer.tick(now + COMPLETED_DISPLAY_DURATION).is_empty());
        assert_eq!(reducer.snapshot().activity_state, AiActivityState::Working);
    }

    #[test]
    fn reconnecting_same_thread_preserves_state_and_revision_until_grace_expires() {
        let now = Instant::now();
        let mut reducer = AiClientStateReducer::with_initial_revision(100);
        start_session(&mut reducer, now);
        let working = start_turn(&mut reducer, now);

        assert!(reducer
            .apply(AiClientEvent::ClientDisconnected, now)
            .is_empty());
        assert!(reducer
            .apply(
                AiClientEvent::SessionStarted {
                    thread_id: THREAD_A.to_string(),
                },
                now + Duration::from_secs(1),
            )
            .is_empty());
        assert_eq!(reducer.snapshot(), working.state);

        assert!(reducer
            .apply(
                AiClientEvent::ClientDisconnected,
                now + Duration::from_secs(2)
            )
            .is_empty());
        let changes = reducer.tick(now + Duration::from_secs(6));
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].state.activity_state, AiActivityState::None);
        assert!(!changes[0].state.session_active);
        assert_eq!(
            changes[0].state.revision,
            working.state.revision.wrapping_add(1)
        );
    }

    #[test]
    fn replacing_a_session_emits_none_before_new_thread_becomes_available() {
        let now = Instant::now();
        let mut reducer = AiClientStateReducer::with_initial_revision(1);
        start_session(&mut reducer, now);

        let ended = apply_one(
            &mut reducer,
            AiClientEvent::SessionRequested {
                requested_thread_id: Some(THREAD_B.to_string()),
            },
            now,
        );
        assert_eq!(ended.reason, AiClientStateChangeReason::SessionReplaced);
        assert_eq!(ended.state.activity_state, AiActivityState::None);

        let new_session = apply_one(
            &mut reducer,
            AiClientEvent::SessionStarted {
                thread_id: THREAD_B.to_string(),
            },
            now,
        );
        assert_eq!(new_session.state.activity_state, AiActivityState::Available);
        assert!(new_session.state.session_active);
    }

    #[test]
    fn item_types_map_without_inspecting_item_content() {
        for item_type in ["reasoning", "agentMessage", "plan"] {
            assert_eq!(item_work_phase(item_type), Some(AiWorkPhase::Thinking));
        }
        for item_type in [
            "commandExecution",
            "fileChange",
            "mcpToolCall",
            "dynamicToolCall",
            "collabAgentToolCall",
            "subAgentActivity",
            "imageView",
            "imageGeneration",
            "sleep",
        ] {
            assert_eq!(item_work_phase(item_type), Some(AiWorkPhase::Executing));
        }
        assert_eq!(item_work_phase("webSearch"), Some(AiWorkPhase::Searching));
        assert_eq!(item_work_phase("userMessage"), None);
        assert_eq!(item_work_phase("futureItemType"), None);
    }

    #[test]
    fn adapter_emits_structured_item_lifecycle_events() {
        let mut adapter = CodexEventAdapter::default();
        adapter.adapt(CodexBrokerEvent::ClientConnected {
            connection_id: "connection-1".to_string(),
        });
        adapter.adapt(message(
            BrokerDirection::CliToAppServer,
            r#"{"jsonrpc":"2.0","id":"start","method":"thread/start","params":{}}"#,
        ));
        adapter.adapt(message(
            BrokerDirection::AppServerToCli,
            r#"{"jsonrpc":"2.0","id":"start","result":{"thread":{"id":"thread-a"}}}"#,
        ));

        let started = adapter.adapt(message(
            BrokerDirection::AppServerToCli,
            r#"{"jsonrpc":"2.0","method":"item/started","params":{"threadId":"thread-a","turnId":"turn-a","item":{"id":"item-a","type":"webSearch","query":"must-not-be-inspected"}}}"#,
        ));
        assert!(matches!(
            started.as_slice(),
            [AiClientEvent::ItemStarted {
                thread_id,
                turn_id,
                item_id,
                work_phase: AiWorkPhase::Searching,
            }] if thread_id == THREAD_A && turn_id == TURN_A && item_id == "item-a"
        ));

        let completed = adapter.adapt(message(
            BrokerDirection::AppServerToCli,
            r#"{"jsonrpc":"2.0","method":"item/completed","params":{"threadId":"thread-a","turnId":"turn-a","item":{"id":"item-a","type":"webSearch"}}}"#,
        ));
        assert!(matches!(
            completed.as_slice(),
            [AiClientEvent::ItemCompleted {
                thread_id,
                turn_id,
                item_id,
            }] if thread_id == THREAD_A && turn_id == TURN_A && item_id == "item-a"
        ));
    }

    #[test]
    fn work_phase_precedence_and_debounce_do_not_change_base_revision() {
        let now = Instant::now();
        let mut reducer = AiClientStateReducer::with_initial_revision(20);
        start_session(&mut reducer, now);
        let working = start_turn(&mut reducer, now);

        assert!(reducer
            .apply(
                AiClientEvent::ItemStarted {
                    thread_id: THREAD_A.to_string(),
                    turn_id: TURN_A.to_string(),
                    item_id: "thinking".to_string(),
                    work_phase: AiWorkPhase::Thinking,
                },
                now,
            )
            .is_empty());
        let thinking = reducer.tick(now + THINKING_STABILITY);
        assert_eq!(thinking.len(), 1);
        assert_eq!(thinking[0].state.work_phase, AiWorkPhase::Thinking);
        assert_eq!(thinking[0].state.revision, working.state.revision);

        let executing = apply_one(
            &mut reducer,
            AiClientEvent::ItemStarted {
                thread_id: THREAD_A.to_string(),
                turn_id: TURN_A.to_string(),
                item_id: "tool".to_string(),
                work_phase: AiWorkPhase::Executing,
            },
            now + THINKING_STABILITY,
        );
        assert_eq!(executing.state.work_phase, AiWorkPhase::Executing);
        assert_eq!(executing.state.revision, working.state.revision);

        let searching = apply_one(
            &mut reducer,
            AiClientEvent::ItemStarted {
                thread_id: THREAD_A.to_string(),
                turn_id: TURN_A.to_string(),
                item_id: "search".to_string(),
                work_phase: AiWorkPhase::Searching,
            },
            now + THINKING_STABILITY,
        );
        assert_eq!(searching.state.work_phase, AiWorkPhase::Searching);

        let back_to_executing = apply_one(
            &mut reducer,
            AiClientEvent::ItemCompleted {
                thread_id: THREAD_A.to_string(),
                turn_id: TURN_A.to_string(),
                item_id: "search".to_string(),
            },
            now + THINKING_STABILITY,
        );
        assert_eq!(back_to_executing.state.work_phase, AiWorkPhase::Executing);
        assert!(reducer
            .apply(
                AiClientEvent::ItemCompleted {
                    thread_id: THREAD_A.to_string(),
                    turn_id: TURN_A.to_string(),
                    item_id: "tool".to_string(),
                },
                now + THINKING_STABILITY,
            )
            .is_empty());
        assert!(reducer
            .tick(now + THINKING_STABILITY + Duration::from_millis(249))
            .is_empty());
        let returned = reducer.tick(now + THINKING_STABILITY + EXECUTION_RETURN_STABILITY);
        assert_eq!(returned.len(), 1);
        assert_eq!(returned[0].state.work_phase, AiWorkPhase::Thinking);
        assert_eq!(returned[0].state.revision, working.state.revision);
    }

    #[test]
    fn waiting_state_hides_phase_and_resolution_restores_active_phase_immediately() {
        let now = Instant::now();
        let mut reducer = AiClientStateReducer::with_initial_revision(30);
        start_session(&mut reducer, now);
        start_turn(&mut reducer, now);
        apply_one(
            &mut reducer,
            AiClientEvent::ItemStarted {
                thread_id: THREAD_A.to_string(),
                turn_id: TURN_A.to_string(),
                item_id: "tool".to_string(),
                work_phase: AiWorkPhase::Executing,
            },
            now,
        );

        let waiting = apply_one(
            &mut reducer,
            AiClientEvent::RequestStarted {
                key: "approval".to_string(),
                kind: RequestKind::Approval,
                thread_id: THREAD_A.to_string(),
                turn_id: Some(TURN_A.to_string()),
            },
            now,
        );
        assert_eq!(
            waiting.state.activity_state,
            AiActivityState::WaitingApproval
        );
        assert_eq!(waiting.state.work_phase, AiWorkPhase::Unspecified);

        let restored = apply_one(
            &mut reducer,
            AiClientEvent::RequestResolved {
                key: "approval".to_string(),
                thread_id: THREAD_A.to_string(),
            },
            now,
        );
        assert_eq!(restored.state.activity_state, AiActivityState::Working);
        assert_eq!(restored.state.work_phase, AiWorkPhase::Executing);
    }

    #[test]
    fn item_events_for_other_turns_and_unknown_completions_are_ignored() {
        let now = Instant::now();
        let mut reducer = AiClientStateReducer::with_initial_revision(40);
        start_session(&mut reducer, now);
        start_turn(&mut reducer, now);

        assert!(reducer
            .apply(
                AiClientEvent::ItemStarted {
                    thread_id: THREAD_A.to_string(),
                    turn_id: "other-turn".to_string(),
                    item_id: "tool".to_string(),
                    work_phase: AiWorkPhase::Executing,
                },
                now,
            )
            .is_empty());
        assert!(reducer
            .apply(
                AiClientEvent::ItemCompleted {
                    thread_id: THREAD_A.to_string(),
                    turn_id: TURN_A.to_string(),
                    item_id: "missing".to_string(),
                },
                now,
            )
            .is_empty());
        assert_eq!(reducer.snapshot().work_phase, AiWorkPhase::Unspecified);
    }

    #[test]
    fn adapter_correlates_thread_turn_and_response_required_requests() {
        let mut adapter = CodexEventAdapter::default();
        assert!(adapter
            .adapt(CodexBrokerEvent::ClientConnected {
                connection_id: "connection-1".to_string(),
            })
            .is_empty());

        let events = adapter.adapt(message(
            BrokerDirection::CliToAppServer,
            r#"{"jsonrpc":"2.0","id":1,"method":"thread/start","params":{}}"#,
        ));
        assert!(matches!(
            events.as_slice(),
            [AiClientEvent::SessionRequested {
                requested_thread_id: None
            }]
        ));

        let events = adapter.adapt(message(
            BrokerDirection::AppServerToCli,
            r#"{"jsonrpc":"2.0","id":1,"result":{"thread":{"id":"thread-a"}}}"#,
        ));
        assert!(matches!(
            events.as_slice(),
            [AiClientEvent::SessionStarted { thread_id }] if thread_id == THREAD_A
        ));

        assert!(adapter
            .adapt(message(
                BrokerDirection::AppServerToCli,
                r#"{"jsonrpc":"2.0","method":"thread/started","params":{"thread":{"id":"thread-a"}}}"#,
            ))
            .is_empty());

        let events = adapter.adapt(message(
            BrokerDirection::AppServerToCli,
            r#"{"jsonrpc":"2.0","method":"turn/started","params":{"threadId":"thread-a","turn":{"id":"turn-a","status":"inProgress"}}}"#,
        ));
        assert!(matches!(
            events.as_slice(),
            [AiClientEvent::TurnStarted { thread_id, turn_id }]
                if thread_id == THREAD_A && turn_id == TURN_A
        ));

        let events = adapter.adapt(message(
            BrokerDirection::AppServerToCli,
            r#"{"jsonrpc":"2.0","id":99,"method":"item/commandExecution/requestApproval","params":{"threadId":"thread-a","turnId":"turn-a","item":{"id":"item-a"}}}"#,
        ));
        assert!(matches!(
            events.as_slice(),
            [AiClientEvent::RequestStarted {
                key,
                kind: RequestKind::Approval,
                thread_id,
                turn_id,
            }] if key == "n:99"
                && thread_id == THREAD_A
                && turn_id.as_deref() == Some(TURN_A)
        ));

        let events = adapter.adapt(message(
            BrokerDirection::CliToAppServer,
            r#"{"jsonrpc":"2.0","id":99,"result":{}}"#,
        ));
        assert!(matches!(
            events.as_slice(),
            [AiClientEvent::RequestResolved { key, thread_id }]
                if key == "n:99" && thread_id == THREAD_A
        ));
    }

    #[test]
    fn adapter_ignores_a_side_thread_started_notification() {
        let mut adapter = CodexEventAdapter::default();
        adapter.adapt(CodexBrokerEvent::ClientConnected {
            connection_id: "connection-1".to_string(),
        });
        adapter.adapt(message(
            BrokerDirection::CliToAppServer,
            r#"{"jsonrpc":"2.0","id":"start","method":"thread/start","params":{}}"#,
        ));
        adapter.adapt(message(
            BrokerDirection::AppServerToCli,
            r#"{"jsonrpc":"2.0","id":"start","result":{"thread":{"id":"thread-a"}}}"#,
        ));

        let events = adapter.adapt(message(
            BrokerDirection::AppServerToCli,
            r#"{"jsonrpc":"2.0","method":"thread/started","params":{"thread":{"id":"thread-b"}}}"#,
        ));
        assert!(events.is_empty());
        assert_eq!(adapter.confirmed_thread_id.as_deref(), Some(THREAD_A));

        let events = adapter.adapt(message(
            BrokerDirection::AppServerToCli,
            r#"{"jsonrpc":"2.0","method":"turn/started","params":{"threadId":"thread-a","turn":{"id":"turn-a","status":"inProgress"}}}"#,
        ));
        assert!(matches!(
            events.as_slice(),
            [AiClientEvent::TurnStarted {
                thread_id,
                turn_id,
            }] if thread_id == THREAD_A && turn_id == TURN_A
        ));
    }

    #[test]
    fn adapter_keeps_approval_correlation_for_another_thread_on_the_same_connection() {
        let mut adapter = CodexEventAdapter::default();
        adapter.adapt(CodexBrokerEvent::ClientConnected {
            connection_id: "connection-1".to_string(),
        });
        for (request_id, thread_id) in [("start-a", THREAD_A), ("start-b", THREAD_B)] {
            adapter.adapt(message(
                BrokerDirection::CliToAppServer,
                &format!(r#"{{"jsonrpc":"2.0","id":"{request_id}","method":"thread/start","params":{{}}}}"#),
            ));
            adapter.adapt(message(
                BrokerDirection::AppServerToCli,
                &format!(r#"{{"jsonrpc":"2.0","id":"{request_id}","result":{{"thread":{{"id":"{thread_id}"}}}}}}"#),
            ));
        }
        adapter.adapt(message(
            BrokerDirection::AppServerToCli,
            r#"{"jsonrpc":"2.0","method":"turn/started","params":{"threadId":"thread-a","turn":{"id":"turn-a","status":"inProgress"}}}"#,
        ));
        adapter.adapt(message(
            BrokerDirection::AppServerToCli,
            r#"{"jsonrpc":"2.0","id":11,"method":"item/commandExecution/requestApproval","params":{"threadId":"thread-a","turnId":"turn-a","item":{"id":"item-a"}}}"#,
        ));
        adapter.adapt(message(
            BrokerDirection::AppServerToCli,
            r#"{"jsonrpc":"2.0","method":"turn/started","params":{"threadId":"thread-b","turn":{"id":"turn-b","status":"inProgress"}}}"#,
        ));

        let resolved = adapter.adapt(message(
            BrokerDirection::CliToAppServer,
            r#"{"jsonrpc":"2.0","id":11,"result":{}}"#,
        ));
        assert!(matches!(
            resolved.as_slice(),
            [AiClientEvent::RequestResolved { thread_id, .. }] if thread_id == THREAD_A
        ));
    }

    #[test]
    fn adapter_keeps_client_request_when_a_server_response_reuses_its_json_rpc_id() {
        let mut adapter = CodexEventAdapter::default();
        adapter.adapt(CodexBrokerEvent::ClientConnected {
            connection_id: "connection-1".to_string(),
        });
        adapter.adapt(message(
            BrokerDirection::CliToAppServer,
            r#"{"jsonrpc":"2.0","id":"seed","method":"thread/start","params":{}}"#,
        ));
        adapter.adapt(message(
            BrokerDirection::AppServerToCli,
            r#"{"jsonrpc":"2.0","id":"seed","result":{"thread":{"id":"thread-a"}}}"#,
        ));
        adapter.adapt(message(
            BrokerDirection::CliToAppServer,
            r#"{"jsonrpc":"2.0","id":1,"method":"thread/start","params":{}}"#,
        ));
        adapter.adapt(message(
            BrokerDirection::AppServerToCli,
            r#"{"jsonrpc":"2.0","id":1,"method":"item/commandExecution/requestApproval","params":{"threadId":"thread-a","turnId":"turn-a","item":{"id":"item-a"}}}"#,
        ));
        let resolved = adapter.adapt(message(
            BrokerDirection::CliToAppServer,
            r#"{"jsonrpc":"2.0","id":1,"result":{}}"#,
        ));
        assert!(matches!(
            resolved.as_slice(),
            [AiClientEvent::RequestResolved { .. }]
        ));

        let started = adapter.adapt(message(
            BrokerDirection::AppServerToCli,
            r#"{"jsonrpc":"2.0","id":1,"result":{"thread":{"id":"thread-b"}}}"#,
        ));
        assert!(matches!(
            started.as_slice(),
            [AiClientEvent::SessionStarted { thread_id }] if thread_id == THREAD_B
        ));
    }

    #[test]
    fn adapter_registers_an_unknown_thread_only_after_its_local_turn_start_succeeds() {
        let mut adapter = CodexEventAdapter::default();
        adapter.adapt(CodexBrokerEvent::ClientConnected {
            connection_id: "connection-1".to_string(),
        });
        adapter.adapt(message(
            BrokerDirection::CliToAppServer,
            r#"{"jsonrpc":"2.0","id":"turn","method":"turn/start","params":{"threadId":"thread-b"}}"#,
        ));
        adapter.adapt(message(
            BrokerDirection::AppServerToCli,
            r#"{"jsonrpc":"2.0","id":"turn","result":{}}"#,
        ));
        let events = adapter.adapt(message(
            BrokerDirection::AppServerToCli,
            r#"{"jsonrpc":"2.0","method":"turn/started","params":{"threadId":"thread-b","turn":{"id":"turn-b","status":"inProgress"}}}"#,
        ));
        assert!(matches!(
            events.as_slice(),
            [
                AiClientEvent::SessionStarted { thread_id },
                AiClientEvent::TurnStarted { thread_id: turn_thread_id, turn_id }
            ] if thread_id == THREAD_B && turn_thread_id == THREAD_B && turn_id == TURN_B
        ));
    }

    #[test]
    fn adapter_ignores_connection_local_broker_errors() {
        let mut adapter = CodexEventAdapter::default();
        adapter.adapt(CodexBrokerEvent::ClientConnected {
            connection_id: "connection-1".to_string(),
        });
        assert!(adapter
            .adapt(CodexBrokerEvent::Error {
                component: "connection",
                detail: "reset".to_string(),
            })
            .is_empty());
        assert_eq!(adapter.connection_id.as_deref(), Some("connection-1"));
    }

    #[test]
    fn adapter_releases_client_request_correlations_after_error_responses() {
        let mut adapter = CodexEventAdapter::default();
        adapter.adapt(CodexBrokerEvent::ClientConnected {
            connection_id: "connection-1".to_string(),
        });
        adapter.adapt(message(
            BrokerDirection::CliToAppServer,
            r#"{"jsonrpc":"2.0","id":"failed-thread","method":"thread/start","params":{}}"#,
        ));
        adapter.adapt(message(
            BrokerDirection::CliToAppServer,
            r#"{"jsonrpc":"2.0","id":"failed-turn","method":"turn/start","params":{"threadId":"thread-a"}}"#,
        ));

        assert!(adapter
            .adapt(message(
                BrokerDirection::AppServerToCli,
                r#"{"jsonrpc":"2.0","id":"failed-thread","error":{"code":-1,"message":"failed"}}"#,
            ))
            .is_empty());
        assert!(adapter
            .adapt(message(
                BrokerDirection::AppServerToCli,
                r#"{"jsonrpc":"2.0","id":"failed-turn","error":{"code":-1,"message":"failed"}}"#,
            ))
            .is_empty());
        assert!(adapter.client_requests.is_empty());
        assert!(adapter.pending_turn_starts.is_empty());
    }

    #[test]
    fn adapter_promotes_a_forked_thread_without_blacking_out_screenkey() {
        let now = Instant::now();
        let mut adapter = CodexEventAdapter::default();
        let mut reducer = AiClientStateReducer::with_initial_revision(90);
        start_session(&mut reducer, now);
        adapter.adapt(CodexBrokerEvent::ClientConnected {
            connection_id: "connection-1".to_string(),
        });
        adapter.adapt(message(
            BrokerDirection::CliToAppServer,
            r#"{"jsonrpc":"2.0","id":"start","method":"thread/start","params":{}}"#,
        ));
        adapter.adapt(message(
            BrokerDirection::AppServerToCli,
            r#"{"jsonrpc":"2.0","id":"start","result":{"thread":{"id":"thread-a"}}}"#,
        ));

        assert!(adapter
            .adapt(message(
                BrokerDirection::CliToAppServer,
                r#"{"jsonrpc":"2.0","id":"fork","method":"thread/fork","params":{"threadId":"thread-a","ephemeral":true}}"#,
            ))
            .is_empty());
        let forked = adapter.adapt(message(
            BrokerDirection::AppServerToCli,
            r#"{"jsonrpc":"2.0","id":"fork","result":{"thread":{"id":"thread-b","forkedFromId":"thread-a"}}}"#,
        ));
        assert!(matches!(
            forked.as_slice(),
            [AiClientEvent::SessionForked { thread_id }] if thread_id == THREAD_B
        ));
        let switched = apply_one(&mut reducer, forked.into_iter().next().unwrap(), now);
        assert_eq!(switched.reason, AiClientStateChangeReason::SessionForked);
        assert!(switched.state.session_active);
        assert_eq!(switched.state.activity_state, AiActivityState::Available);

        assert!(adapter
            .adapt(message(
                BrokerDirection::AppServerToCli,
                r#"{"jsonrpc":"2.0","method":"thread/started","params":{"thread":{"id":"thread-b"}}}"#,
            ))
            .is_empty());
        let started = adapter.adapt(message(
            BrokerDirection::AppServerToCli,
            r#"{"jsonrpc":"2.0","method":"turn/started","params":{"threadId":"thread-b","turn":{"id":"turn-b","status":"inProgress"}}}"#,
        ));
        let working = apply_one(&mut reducer, started.into_iter().next().unwrap(), now);
        assert_eq!(working.state.activity_state, AiActivityState::Working);

        let completed = apply_one(
            &mut reducer,
            AiClientEvent::TurnFinished {
                thread_id: THREAD_B.to_string(),
                turn_id: "turn-b".to_string(),
                outcome: TurnOutcome::Completed,
            },
            now,
        );
        assert_eq!(completed.state.activity_state, AiActivityState::Completed);

        let returned_to_parent = adapter.adapt(message(
            BrokerDirection::AppServerToCli,
            r#"{"jsonrpc":"2.0","method":"turn/started","params":{"threadId":"thread-a","turn":{"id":"turn-c","status":"inProgress"}}}"#,
        ));
        assert!(matches!(
            returned_to_parent.as_slice(),
            [
                AiClientEvent::SessionForked { thread_id },
                AiClientEvent::TurnStarted {
                    thread_id: turn_thread_id,
                    turn_id,
                }
            ] if thread_id == THREAD_A && turn_thread_id == THREAD_A && turn_id == "turn-c"
        ));
        let returned_to_parent = returned_to_parent
            .into_iter()
            .flat_map(|event| reducer.apply(event, now))
            .collect::<Vec<_>>();
        assert_eq!(returned_to_parent.len(), 2);
        assert_eq!(
            returned_to_parent[0].reason,
            AiClientStateChangeReason::SessionForked
        );
        assert_eq!(
            returned_to_parent[1].state.activity_state,
            AiActivityState::Working
        );
    }

    #[test]
    fn registry_keeps_threads_and_runtime_state_independent() {
        let now = Instant::now();
        let mut registry = CodexSessionRegistry::new();
        registry.apply(
            "connection-a",
            AiClientEvent::SessionStarted {
                thread_id: THREAD_A.to_string(),
            },
            now,
        );
        registry.apply(
            "connection-b",
            AiClientEvent::SessionStarted {
                thread_id: THREAD_B.to_string(),
            },
            now,
        );
        registry.apply(
            "connection-a",
            AiClientEvent::TurnStarted {
                thread_id: THREAD_A.to_string(),
                turn_id: TURN_A.to_string(),
            },
            now,
        );
        registry.apply(
            "connection-a",
            AiClientEvent::RequestStarted {
                key: "input-a".to_string(),
                kind: RequestKind::Input,
                thread_id: THREAD_A.to_string(),
                turn_id: Some(TURN_A.to_string()),
            },
            now,
        );
        registry.apply(
            "connection-b",
            AiClientEvent::TurnStarted {
                thread_id: THREAD_B.to_string(),
                turn_id: TURN_B.to_string(),
            },
            now,
        );
        registry.apply(
            "connection-b",
            AiClientEvent::ItemStarted {
                thread_id: THREAD_B.to_string(),
                turn_id: TURN_B.to_string(),
                item_id: "command-b".to_string(),
                work_phase: AiWorkPhase::Executing,
            },
            now,
        );
        registry.apply(
            "connection-b",
            AiClientEvent::RequestStarted {
                key: "approval-b".to_string(),
                kind: RequestKind::Approval,
                thread_id: THREAD_B.to_string(),
                turn_id: Some(TURN_B.to_string()),
            },
            now,
        );

        let snapshots = registry.snapshots();
        assert_eq!(snapshots.len(), 2);
        assert_eq!(
            snapshots[0].state.activity_state,
            AiActivityState::WaitingInput
        );
        assert_eq!(
            snapshots[1].state.activity_state,
            AiActivityState::WaitingApproval
        );
        assert_ne!(snapshots[0].state.revision, snapshots[1].state.revision);

        registry.apply(
            "connection-a",
            AiClientEvent::RequestResolved {
                key: "input-a".to_string(),
                thread_id: THREAD_A.to_string(),
            },
            now,
        );
        let snapshots = registry.snapshots();
        assert_eq!(snapshots[0].state.activity_state, AiActivityState::Working);
        assert_eq!(
            snapshots[1].state.activity_state,
            AiActivityState::WaitingApproval
        );
    }

    #[test]
    fn registry_retains_old_thread_when_one_connection_starts_another() {
        let now = Instant::now();
        let mut registry = CodexSessionRegistry::new();
        for thread_id in [THREAD_A, THREAD_B] {
            registry.apply(
                "connection-a",
                AiClientEvent::SessionStarted {
                    thread_id: thread_id.to_string(),
                },
                now,
            );
        }

        let snapshots = registry.snapshots();
        assert_eq!(snapshots.len(), 2);
        assert_eq!(snapshots[0].thread_id, THREAD_A);
        assert_eq!(snapshots[1].thread_id, THREAD_B);
        // Both entries survive for state tracking, but only the connection's
        // most recently started thread should be a ScreenKey display candidate.
        assert!(!snapshots[0].is_display_target);
        assert!(snapshots[1].is_display_target);
    }

    #[test]
    fn turn_started_on_owned_thread_refocuses_it() {
        let now = Instant::now();
        let mut registry = CodexSessionRegistry::new();
        for thread_id in [THREAD_A, THREAD_B] {
            registry.apply(
                "connection-a",
                AiClientEvent::SessionStarted {
                    thread_id: thread_id.to_string(),
                },
                now,
            );
        }

        // `/side` can return to the parent thread without a `thread/resume`,
        // so a turn on the older, owned thread must move focus back to it.
        registry.apply(
            "connection-a",
            AiClientEvent::TurnStarted {
                thread_id: THREAD_A.to_string(),
                turn_id: TURN_A.to_string(),
            },
            now,
        );

        let snapshots = registry.snapshots();
        assert_eq!(snapshots.len(), 2);
        assert!(
            snapshots
                .iter()
                .find(|snapshot| snapshot.thread_id == THREAD_A)
                .unwrap()
                .is_display_target
        );
        assert!(
            !snapshots
                .iter()
                .find(|snapshot| snapshot.thread_id == THREAD_B)
                .unwrap()
                .is_display_target
        );
    }

    #[test]
    fn separate_connections_each_keep_their_own_focused_thread() {
        let now = Instant::now();
        let mut registry = CodexSessionRegistry::new();
        registry.apply(
            "connection-a",
            AiClientEvent::SessionStarted {
                thread_id: THREAD_A.to_string(),
            },
            now,
        );
        registry.apply(
            "connection-b",
            AiClientEvent::SessionStarted {
                thread_id: THREAD_B.to_string(),
            },
            now,
        );

        // Two independent CLI connections should both remain visible.
        let snapshots = registry.snapshots();
        assert_eq!(snapshots.len(), 2);
        assert!(snapshots.iter().all(|snapshot| snapshot.is_display_target));
    }

    #[test]
    fn ownership_transfer_moves_focus_to_the_new_owner() {
        let now = Instant::now();
        let mut registry = CodexSessionRegistry::new();
        registry.apply(
            "connection-a",
            AiClientEvent::SessionStarted {
                thread_id: THREAD_A.to_string(),
            },
            now,
        );
        // connection-b adopts thread A (e.g. after a resume from a new process).
        registry.apply(
            "connection-b",
            AiClientEvent::SessionStarted {
                thread_id: THREAD_A.to_string(),
            },
            now,
        );

        let snapshots = registry.snapshots();
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].owner_connection_id, "connection-b");
        assert!(snapshots[0].is_display_target);

        // connection-a no longer owns anything, so it has no valid focus left.
        registry.apply(
            "connection-a",
            AiClientEvent::TurnStarted {
                thread_id: THREAD_A.to_string(),
                turn_id: TURN_A.to_string(),
            },
            now,
        );
        let snapshots = registry.snapshots();
        assert_eq!(snapshots[0].owner_connection_id, "connection-b");
        assert!(snapshots[0].is_display_target);
    }

    #[test]
    fn non_graceful_disconnect_clears_the_connections_focus() {
        let now = Instant::now();
        let mut registry = CodexSessionRegistry::new();
        registry.apply(
            "connection-a",
            AiClientEvent::SessionStarted {
                thread_id: THREAD_A.to_string(),
            },
            now,
        );
        assert!(registry.focused.contains_key("connection-a"));

        registry.disconnect("connection-a", false, now);

        assert!(!registry.focused.contains_key("connection-a"));
    }

    #[test]
    fn resume_transfers_ownership_without_resetting_revision() {
        let now = Instant::now();
        let mut registry = CodexSessionRegistry::new();
        registry.apply(
            "connection-a",
            AiClientEvent::SessionStarted {
                thread_id: THREAD_A.to_string(),
            },
            now,
        );
        let revision = registry.snapshots()[0].state.revision;
        registry.disconnect("connection-a", true, now);
        // The production loop ticks every 100 ms while the reconnect grace is
        // active; it must not retire this thread before the new owner resumes.
        registry.tick(now + Duration::from_secs(1));
        registry.apply(
            "connection-b",
            AiClientEvent::SessionStarted {
                thread_id: THREAD_A.to_string(),
            },
            now + Duration::from_secs(2),
        );
        let ignored = registry.apply(
            "connection-a",
            AiClientEvent::TurnStarted {
                thread_id: THREAD_A.to_string(),
                turn_id: TURN_A.to_string(),
            },
            now + Duration::from_secs(2),
        );

        let snapshots = registry.snapshots();
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].owner_connection_id, "connection-b");
        assert_eq!(snapshots[0].state.revision, revision);
        assert!(ignored.is_empty());
        assert_eq!(
            snapshots[0].state.activity_state,
            AiActivityState::Available
        );
    }

    #[test]
    fn disconnect_expiry_retires_only_threads_owned_by_that_connection() {
        let now = Instant::now();
        let mut registry = CodexSessionRegistry::new();
        registry.apply(
            "connection-a",
            AiClientEvent::SessionStarted {
                thread_id: THREAD_A.to_string(),
            },
            now,
        );
        registry.apply(
            "connection-b",
            AiClientEvent::SessionStarted {
                thread_id: THREAD_B.to_string(),
            },
            now,
        );
        registry.disconnect("connection-a", true, now);
        registry.tick(now + RECONNECT_GRACE + Duration::from_millis(1));

        let snapshots = registry.snapshots();
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].thread_id, THREAD_B);
        assert!(snapshots[0].state.session_active);
    }

    #[test]
    fn waiting_thread_outranks_a_more_recently_focused_completed_thread() {
        // Reproduces the field bug: Codex opens a short-lived side thread
        // (observed in practice around conversation-title generation) that
        // steals `self.focused` right as the original thread starts waiting
        // on an approval. ScreenKey must still show the thread the user
        // actually needs to answer, not the side thread that just finished.
        let now = Instant::now();
        let mut registry = CodexSessionRegistry::new();

        registry.apply(
            "connection-a",
            AiClientEvent::SessionStarted {
                thread_id: THREAD_A.to_string(),
            },
            now,
        );
        registry.apply(
            "connection-a",
            AiClientEvent::TurnStarted {
                thread_id: THREAD_A.to_string(),
                turn_id: TURN_A.to_string(),
            },
            now,
        );

        // The short-lived side thread starts and finishes, moving
        // `self.focused` onto it and leaving it there.
        registry.apply(
            "connection-a",
            AiClientEvent::SessionStarted {
                thread_id: THREAD_B.to_string(),
            },
            now,
        );
        registry.apply(
            "connection-a",
            AiClientEvent::TurnStarted {
                thread_id: THREAD_B.to_string(),
                turn_id: TURN_B.to_string(),
            },
            now,
        );
        registry.apply(
            "connection-a",
            AiClientEvent::TurnFinished {
                thread_id: THREAD_B.to_string(),
                turn_id: TURN_B.to_string(),
                outcome: TurnOutcome::Completed,
            },
            now,
        );

        // Thread A now asks for approval.
        registry.apply(
            "connection-a",
            AiClientEvent::RequestStarted {
                key: "approval-a".to_string(),
                kind: RequestKind::Approval,
                thread_id: THREAD_A.to_string(),
                turn_id: Some(TURN_A.to_string()),
            },
            now,
        );

        let snapshots = registry.snapshots();
        let a = snapshots
            .iter()
            .find(|snapshot| snapshot.thread_id == THREAD_A)
            .unwrap();
        let b = snapshots
            .iter()
            .find(|snapshot| snapshot.thread_id == THREAD_B)
            .unwrap();
        assert_eq!(a.state.activity_state, AiActivityState::WaitingApproval);
        assert_eq!(b.state.activity_state, AiActivityState::Completed);
        // `self.focused` (Codex's own notion of "current thread") is still
        // B, but the display target must be A: it is the thread actually
        // waiting on the user.
        assert!(a.is_display_target);
        assert!(!b.is_display_target);
    }

    #[test]
    fn display_target_returns_to_the_focused_thread_once_the_wait_resolves() {
        let now = Instant::now();
        let mut registry = CodexSessionRegistry::new();

        registry.apply(
            "connection-a",
            AiClientEvent::SessionStarted {
                thread_id: THREAD_A.to_string(),
            },
            now,
        );
        registry.apply(
            "connection-a",
            AiClientEvent::TurnStarted {
                thread_id: THREAD_A.to_string(),
                turn_id: TURN_A.to_string(),
            },
            now,
        );
        registry.apply(
            "connection-a",
            AiClientEvent::SessionStarted {
                thread_id: THREAD_B.to_string(),
            },
            now,
        );
        // B is now `self.focused` (started last); A is waiting on approval.
        registry.apply(
            "connection-a",
            AiClientEvent::RequestStarted {
                key: "approval-a".to_string(),
                kind: RequestKind::Approval,
                thread_id: THREAD_A.to_string(),
                turn_id: Some(TURN_A.to_string()),
            },
            now,
        );

        let waiting = registry.snapshots();
        assert!(
            waiting
                .iter()
                .find(|snapshot| snapshot.thread_id == THREAD_A)
                .unwrap()
                .is_display_target
        );
        assert!(
            !waiting
                .iter()
                .find(|snapshot| snapshot.thread_id == THREAD_B)
                .unwrap()
                .is_display_target
        );

        registry.apply(
            "connection-a",
            AiClientEvent::RequestResolved {
                key: "approval-a".to_string(),
                thread_id: THREAD_A.to_string(),
            },
            now,
        );

        let resolved = registry.snapshots();
        let a = resolved
            .iter()
            .find(|snapshot| snapshot.thread_id == THREAD_A)
            .unwrap();
        let b = resolved
            .iter()
            .find(|snapshot| snapshot.thread_id == THREAD_B)
            .unwrap();
        assert_eq!(a.state.activity_state, AiActivityState::Working);
        // Nothing is waiting any more, so display falls back to
        // `self.focused` exactly as before this fix.
        assert!(!a.is_display_target);
        assert!(b.is_display_target);
    }

    #[test]
    fn focused_thread_wins_when_multiple_threads_are_waiting() {
        let now = Instant::now();
        let mut registry = CodexSessionRegistry::new();

        registry.apply(
            "connection-a",
            AiClientEvent::SessionStarted {
                thread_id: THREAD_A.to_string(),
            },
            now,
        );
        registry.apply(
            "connection-a",
            AiClientEvent::TurnStarted {
                thread_id: THREAD_A.to_string(),
                turn_id: TURN_A.to_string(),
            },
            now,
        );
        registry.apply(
            "connection-a",
            AiClientEvent::RequestStarted {
                key: "approval-a".to_string(),
                kind: RequestKind::Approval,
                thread_id: THREAD_A.to_string(),
                turn_id: Some(TURN_A.to_string()),
            },
            now,
        );

        registry.apply(
            "connection-a",
            AiClientEvent::SessionStarted {
                thread_id: THREAD_B.to_string(),
            },
            now,
        );
        registry.apply(
            "connection-a",
            AiClientEvent::TurnStarted {
                thread_id: THREAD_B.to_string(),
                turn_id: TURN_B.to_string(),
            },
            now,
        );
        // B becomes `self.focused` via `TurnStarted`, and now also starts
        // waiting, so both threads are waiting at once.
        registry.apply(
            "connection-a",
            AiClientEvent::RequestStarted {
                key: "approval-b".to_string(),
                kind: RequestKind::Approval,
                thread_id: THREAD_B.to_string(),
                turn_id: Some(TURN_B.to_string()),
            },
            now,
        );

        let snapshots = registry.snapshots();
        let a = snapshots
            .iter()
            .find(|snapshot| snapshot.thread_id == THREAD_A)
            .unwrap();
        let b = snapshots
            .iter()
            .find(|snapshot| snapshot.thread_id == THREAD_B)
            .unwrap();
        assert_eq!(a.state.activity_state, AiActivityState::WaitingApproval);
        assert_eq!(b.state.activity_state, AiActivityState::WaitingApproval);
        // A started waiting first, but B is Codex's own focused thread, so
        // it wins the single display slot per the tiebreak rule.
        assert!(!a.is_display_target);
        assert!(b.is_display_target);
    }

    #[test]
    fn earliest_waiting_thread_wins_when_the_focused_thread_is_not_waiting() {
        const THREAD_C: &str = "thread-c";
        let now = Instant::now();
        let mut registry = CodexSessionRegistry::new();

        registry.apply(
            "connection-a",
            AiClientEvent::SessionStarted {
                thread_id: THREAD_A.to_string(),
            },
            now,
        );
        registry.apply(
            "connection-a",
            AiClientEvent::TurnStarted {
                thread_id: THREAD_A.to_string(),
                turn_id: TURN_A.to_string(),
            },
            now,
        );
        registry.apply(
            "connection-a",
            AiClientEvent::RequestStarted {
                key: "approval-a".to_string(),
                kind: RequestKind::Approval,
                thread_id: THREAD_A.to_string(),
                turn_id: Some(TURN_A.to_string()),
            },
            now,
        );

        registry.apply(
            "connection-a",
            AiClientEvent::SessionStarted {
                thread_id: THREAD_B.to_string(),
            },
            now,
        );
        registry.apply(
            "connection-a",
            AiClientEvent::TurnStarted {
                thread_id: THREAD_B.to_string(),
                turn_id: TURN_B.to_string(),
            },
            now,
        );
        registry.apply(
            "connection-a",
            AiClientEvent::RequestStarted {
                key: "approval-b".to_string(),
                kind: RequestKind::Approval,
                thread_id: THREAD_B.to_string(),
                turn_id: Some(TURN_B.to_string()),
            },
            now,
        );

        // C becomes `self.focused` via `SessionStarted`, but never waits.
        registry.apply(
            "connection-a",
            AiClientEvent::SessionStarted {
                thread_id: THREAD_C.to_string(),
            },
            now,
        );

        let snapshots = registry.snapshots();
        let a = snapshots
            .iter()
            .find(|snapshot| snapshot.thread_id == THREAD_A)
            .unwrap();
        let b = snapshots
            .iter()
            .find(|snapshot| snapshot.thread_id == THREAD_B)
            .unwrap();
        let c = snapshots
            .iter()
            .find(|snapshot| snapshot.thread_id == THREAD_C)
            .unwrap();
        assert_eq!(a.state.activity_state, AiActivityState::WaitingApproval);
        assert_eq!(b.state.activity_state, AiActivityState::WaitingApproval);
        // Neither waiting thread is `self.focused` (that's C), so the one
        // that started waiting first (A) keeps the display slot — a stable
        // choice that does not depend on wall-clock timing.
        assert!(a.is_display_target);
        assert!(!b.is_display_target);
        assert!(!c.is_display_target);
    }

    #[test]
    fn working_thread_keeps_the_display_target_while_a_side_thread_completes() {
        // Reproduces the field bug this test guards against: a short-lived
        // side thread (observed in practice around conversation-title
        // generation) starts and completes *while* the real thread the user
        // is waiting on is still `Working`. Before this fix, the side
        // thread's `SessionStarted`/`TurnStarted` stole `self.focused`, and
        // its `TurnCompleted` briefly showed green on ScreenKey/the keyboard
        // LEDs for a thread the user never asked about. The display target
        // must stay on thread A throughout.
        let now = Instant::now();
        let mut registry = CodexSessionRegistry::new();

        registry.apply(
            "connection-a",
            AiClientEvent::SessionStarted {
                thread_id: THREAD_A.to_string(),
            },
            now,
        );
        registry.apply(
            "connection-a",
            AiClientEvent::TurnStarted {
                thread_id: THREAD_A.to_string(),
                turn_id: TURN_A.to_string(),
            },
            now,
        );

        let is_display_target = |registry: &CodexSessionRegistry, thread_id: &str| {
            registry
                .snapshots()
                .into_iter()
                .find(|snapshot| snapshot.thread_id == thread_id)
                .unwrap()
                .is_display_target
        };
        assert!(is_display_target(&registry, THREAD_A));

        // The side thread starts (this alone used to move `self.focused`,
        // and with it the display target, onto B).
        registry.apply(
            "connection-a",
            AiClientEvent::SessionStarted {
                thread_id: THREAD_B.to_string(),
            },
            now,
        );
        assert!(is_display_target(&registry, THREAD_A));
        assert!(!is_display_target(&registry, THREAD_B));

        registry.apply(
            "connection-a",
            AiClientEvent::TurnStarted {
                thread_id: THREAD_B.to_string(),
                turn_id: TURN_B.to_string(),
            },
            now,
        );
        assert!(is_display_target(&registry, THREAD_A));
        assert!(!is_display_target(&registry, THREAD_B));

        // The side thread completes — this is exactly where the green flash
        // used to happen.
        registry.apply(
            "connection-a",
            AiClientEvent::TurnFinished {
                thread_id: THREAD_B.to_string(),
                turn_id: TURN_B.to_string(),
                outcome: TurnOutcome::Completed,
            },
            now,
        );

        let snapshots = registry.snapshots();
        let a = snapshots
            .iter()
            .find(|snapshot| snapshot.thread_id == THREAD_A)
            .unwrap();
        let b = snapshots
            .iter()
            .find(|snapshot| snapshot.thread_id == THREAD_B)
            .unwrap();
        assert_eq!(a.state.activity_state, AiActivityState::Working);
        assert_eq!(b.state.activity_state, AiActivityState::Completed);
        assert!(a.is_display_target);
        assert!(!b.is_display_target);
    }

    #[test]
    fn display_target_moves_to_the_focused_thread_once_the_working_thread_stops() {
        let now = Instant::now();
        let mut registry = CodexSessionRegistry::new();

        registry.apply(
            "connection-a",
            AiClientEvent::SessionStarted {
                thread_id: THREAD_A.to_string(),
            },
            now,
        );
        registry.apply(
            "connection-a",
            AiClientEvent::TurnStarted {
                thread_id: THREAD_A.to_string(),
                turn_id: TURN_A.to_string(),
            },
            now,
        );
        registry.apply(
            "connection-a",
            AiClientEvent::SessionStarted {
                thread_id: THREAD_B.to_string(),
            },
            now,
        );

        // A is still the display target while it works, even though B is
        // `self.focused` (started more recently).
        let before = registry.snapshots();
        assert!(
            before
                .iter()
                .find(|snapshot| snapshot.thread_id == THREAD_A)
                .unwrap()
                .is_display_target
        );

        // A's turn ends without ever waiting on the user — it stops being
        // `Working`, so it no longer has a claim on the display slot and
        // control returns to Codex's own focus (B).
        registry.apply(
            "connection-a",
            AiClientEvent::TurnFinished {
                thread_id: THREAD_A.to_string(),
                turn_id: TURN_A.to_string(),
                outcome: TurnOutcome::Interrupted,
            },
            now,
        );

        let after = registry.snapshots();
        let a = after
            .iter()
            .find(|snapshot| snapshot.thread_id == THREAD_A)
            .unwrap();
        let b = after
            .iter()
            .find(|snapshot| snapshot.thread_id == THREAD_B)
            .unwrap();
        assert_eq!(a.state.activity_state, AiActivityState::Available);
        assert!(!a.is_display_target);
        assert!(b.is_display_target);
    }

    #[test]
    fn a_newly_waiting_thread_still_outranks_a_working_display_target() {
        // Rule 1 (a thread waiting on the user) must keep outranking rule 2
        // (stickiness to a `Working` thread) — the new stickiness must not
        // resurrect the bug rule 1 was added to fix.
        let now = Instant::now();
        let mut registry = CodexSessionRegistry::new();

        registry.apply(
            "connection-a",
            AiClientEvent::SessionStarted {
                thread_id: THREAD_A.to_string(),
            },
            now,
        );
        registry.apply(
            "connection-a",
            AiClientEvent::TurnStarted {
                thread_id: THREAD_A.to_string(),
                turn_id: TURN_A.to_string(),
            },
            now,
        );
        registry.apply(
            "connection-a",
            AiClientEvent::SessionStarted {
                thread_id: THREAD_B.to_string(),
            },
            now,
        );
        registry.apply(
            "connection-a",
            AiClientEvent::TurnStarted {
                thread_id: THREAD_B.to_string(),
                turn_id: TURN_B.to_string(),
            },
            now,
        );

        // A is still Working and sticky-holding the display slot at this
        // point.
        let before = registry.snapshots();
        assert!(
            before
                .iter()
                .find(|snapshot| snapshot.thread_id == THREAD_A)
                .unwrap()
                .is_display_target
        );

        registry.apply(
            "connection-a",
            AiClientEvent::RequestStarted {
                key: "approval-b".to_string(),
                kind: RequestKind::Approval,
                thread_id: THREAD_B.to_string(),
                turn_id: Some(TURN_B.to_string()),
            },
            now,
        );

        let after = registry.snapshots();
        let a = after
            .iter()
            .find(|snapshot| snapshot.thread_id == THREAD_A)
            .unwrap();
        let b = after
            .iter()
            .find(|snapshot| snapshot.thread_id == THREAD_B)
            .unwrap();
        assert_eq!(a.state.activity_state, AiActivityState::Working);
        assert_eq!(b.state.activity_state, AiActivityState::WaitingApproval);
        assert!(!a.is_display_target);
        assert!(b.is_display_target);
    }

    #[test]
    fn registry_caps_sessions_at_thirty_two_with_safe_oldest_eviction() {
        let now = Instant::now();
        let mut registry = CodexSessionRegistry::new();
        registry.set_selected_thread(Some("thread-0".to_string()));
        for index in 0..=MAX_CODEX_SESSIONS {
            registry.apply(
                "connection-a",
                AiClientEvent::SessionStarted {
                    thread_id: format!("thread-{index}"),
                },
                now,
            );
        }

        let snapshots = registry.snapshots();
        assert_eq!(snapshots.len(), MAX_CODEX_SESSIONS);
        assert!(snapshots
            .iter()
            .any(|snapshot| snapshot.thread_id == "thread-0"));
        assert!(snapshots
            .iter()
            .all(|snapshot| snapshot.thread_id != "thread-1"));
        assert!(snapshots
            .iter()
            .any(|snapshot| snapshot.thread_id == format!("thread-{MAX_CODEX_SESSIONS}")));
    }

    #[test]
    fn registry_refuses_a_new_thread_when_every_session_is_protected() {
        let now = Instant::now();
        let mut registry = CodexSessionRegistry::new();
        for index in 0..MAX_CODEX_SESSIONS {
            let thread_id = format!("thread-{index}");
            registry.apply(
                "connection-a",
                AiClientEvent::SessionStarted {
                    thread_id: thread_id.clone(),
                },
                now,
            );
            registry.apply(
                "connection-a",
                AiClientEvent::TurnStarted {
                    thread_id,
                    turn_id: format!("turn-{index}"),
                },
                now,
            );
        }

        let changes = registry.apply(
            "connection-b",
            AiClientEvent::SessionStarted {
                thread_id: "thread-overflow".to_string(),
            },
            now,
        );
        assert!(changes.is_empty());
        assert_eq!(registry.snapshots().len(), MAX_CODEX_SESSIONS);
        assert!(registry
            .snapshots()
            .iter()
            .all(|snapshot| snapshot.thread_id != "thread-overflow"));
    }

    #[test]
    fn registry_expires_completed_session_while_another_thread_is_selected() {
        let now = Instant::now();
        let mut registry = CodexSessionRegistry::new();
        for (connection_id, thread_id, turn_id) in [
            ("connection-a", THREAD_A, TURN_A),
            ("connection-b", THREAD_B, TURN_B),
        ] {
            registry.apply(
                connection_id,
                AiClientEvent::SessionStarted {
                    thread_id: thread_id.to_string(),
                },
                now,
            );
            registry.apply(
                connection_id,
                AiClientEvent::TurnStarted {
                    thread_id: thread_id.to_string(),
                    turn_id: turn_id.to_string(),
                },
                now,
            );
        }
        registry.apply(
            "connection-b",
            AiClientEvent::TurnFinished {
                thread_id: THREAD_B.to_string(),
                turn_id: TURN_B.to_string(),
                outcome: TurnOutcome::Completed,
            },
            now,
        );
        registry.set_selected_thread(Some(THREAD_A.to_string()));

        registry.tick(now + COMPLETED_DISPLAY_DURATION + Duration::from_millis(1));

        let snapshots = registry.snapshots();
        assert_eq!(snapshots[0].state.activity_state, AiActivityState::Working);
        assert_eq!(
            snapshots[1].state.activity_state,
            AiActivityState::Available
        );
    }

    fn ko2_body() -> CodexApprovalRequestBody {
        CodexApprovalRequestBody {
            command_actions: vec!["mkdir ko2-test".to_string()],
            command: Some(
                "\"C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe\" -Command 'mkdir ko2-test'"
                    .to_string(),
            ),
            reason: Some("ワークスペース内に ko2-test ディレクトリを作成してよいですか？".to_string()),
            cwd: Some("C:\\01.keyboards\\OriginalKeyboards\\02.SW\\Keylink-Studio".to_string()),
            kind: Some("command".to_string()),
            available_decisions: vec![
                Value::String("accept".to_string()),
                serde_json::json!({"acceptWithExecpolicyAmendment": {"execpolicy_amendment": ["mkdir"]}}),
                Value::String("cancel".to_string()),
            ],
            thread_id: Some(THREAD_A.to_string()),
            turn_id: Some(TURN_A.to_string()),
            item_id: Some("exec-882ac982".to_string()),
        }
    }

    #[test]
    fn ingest_codex_approval_normalizes_the_ko2_body_and_keeps_decisions_opaque() {
        let store = PendingApprovalStore::new();
        let mut approval_turns = HashMap::new();
        let request_id = Value::from(0);
        ingest_codex_approval(
            &store,
            &mut approval_turns,
            "connection-1",
            &request_id,
            ko2_body(),
        );

        let key = codex_key("connection-1", &request_id);
        let snapshot = store.get(&key).expect("entry inserted");
        match snapshot.content {
            PendingApprovalContent::Body(body) => {
                assert_eq!(body.primary_text.as_deref(), Some("mkdir ko2-test"));
                assert_eq!(
                    body.full_command.as_deref(),
                    Some(
                        "\"C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe\" -Command 'mkdir ko2-test'"
                    )
                );
                assert_eq!(
                    body.reason.as_deref(),
                    Some("ワークスペース内に ko2-test ディレクトリを作成してよいですか？")
                );
                assert_eq!(body.kind.as_deref(), Some("command"));
                let decisions = body.available_decisions.expect("decisions present");
                assert_eq!(decisions.len(), 3);
                assert_eq!(decisions[0], Value::String("accept".to_string()));
                assert_eq!(
                    decisions[1],
                    serde_json::json!({"acceptWithExecpolicyAmendment": {"execpolicy_amendment": ["mkdir"]}})
                );
                assert_eq!(decisions[2], Value::String("cancel".to_string()));
            }
            PendingApprovalContent::Oversized => panic!("unexpected oversized marker"),
        }
        assert_eq!(
            approval_turns
                .get(&(
                    "connection-1".to_string(),
                    THREAD_A.to_string(),
                    TURN_A.to_string()
                ))
                .map(Vec::len),
            Some(1)
        );
    }

    #[test]
    fn resolve_codex_approval_via_cli_response_discards_the_entry() {
        let store = PendingApprovalStore::new();
        let mut approval_turns = HashMap::new();
        let request_id = Value::from(0);
        ingest_codex_approval(
            &store,
            &mut approval_turns,
            "connection-1",
            &request_id,
            ko2_body(),
        );
        let key = codex_key("connection-1", &request_id);
        assert!(store.get(&key).is_some());

        let response = crate::codex_broker::classify_json_rpc(
            r#"{"jsonrpc":"2.0","id":0,"result":{"decision":"accept"}}"#,
        );
        resolve_codex_approval(
            &store,
            &mut approval_turns,
            "connection-1",
            BrokerDirection::CliToAppServer,
            &response,
        );
        assert!(store.get(&key).is_none());
    }

    #[test]
    fn resolve_codex_approval_via_server_request_resolved_discards_the_entry() {
        let store = PendingApprovalStore::new();
        let mut approval_turns = HashMap::new();
        let request_id = Value::from(0);
        ingest_codex_approval(
            &store,
            &mut approval_turns,
            "connection-1",
            &request_id,
            ko2_body(),
        );
        let key = codex_key("connection-1", &request_id);

        let resolved = crate::codex_broker::classify_json_rpc(
            r#"{"jsonrpc":"2.0","method":"serverRequest/resolved","params":{"requestId":0}}"#,
        );
        resolve_codex_approval(
            &store,
            &mut approval_turns,
            "connection-1",
            BrokerDirection::AppServerToCli,
            &resolved,
        );
        assert!(store.get(&key).is_none());
    }

    #[test]
    fn resolve_codex_approval_via_turn_completed_is_a_safety_net() {
        let store = PendingApprovalStore::new();
        let mut approval_turns = HashMap::new();
        let request_id = Value::from(0);
        ingest_codex_approval(
            &store,
            &mut approval_turns,
            "connection-1",
            &request_id,
            ko2_body(),
        );
        let key = codex_key("connection-1", &request_id);
        // Neither the CLI response nor `serverRequest/resolved` arrived
        // (e.g. a Broker-held request answered without either signal
        // reaching this connection) -- `turn/completed` must still clear
        // it, per docs/ai-approval-hud-design.md §9.1.
        let completed = crate::codex_broker::classify_json_rpc(&format!(
            r#"{{"jsonrpc":"2.0","method":"turn/completed","params":{{"threadId":"{THREAD_A}","turnId":"{TURN_A}","turn":{{"status":"completed"}}}}}}"#
        ));
        resolve_codex_approval(
            &store,
            &mut approval_turns,
            "connection-1",
            BrokerDirection::AppServerToCli,
            &completed,
        );
        assert!(store.get(&key).is_none());
        assert!(approval_turns.is_empty());
    }

    /// KO-2 observed `serverRequest/resolved` arriving before
    /// `turn/completed`, but nothing here should depend on that order:
    /// `PendingApprovalStore::resolve` is idempotent, so whichever signal
    /// arrives first clears the entry (and, for `turn/completed`, the
    /// `approval_turns` tracking) and the second is a harmless no-op.
    /// Covers both orders explicitly rather than relying on that being an
    /// accident of the implementation.
    #[test]
    fn resolution_signals_are_order_independent() {
        for reverse_order in [false, true] {
            let store = PendingApprovalStore::new();
            let mut approval_turns = HashMap::new();
            let request_id = Value::from(0);
            ingest_codex_approval(
                &store,
                &mut approval_turns,
                "connection-1",
                &request_id,
                ko2_body(),
            );
            let key = codex_key("connection-1", &request_id);

            let resolved = crate::codex_broker::classify_json_rpc(
                r#"{"jsonrpc":"2.0","method":"serverRequest/resolved","params":{"requestId":0}}"#,
            );
            let completed = crate::codex_broker::classify_json_rpc(&format!(
                r#"{{"jsonrpc":"2.0","method":"turn/completed","params":{{"threadId":"{THREAD_A}","turnId":"{TURN_A}","turn":{{"status":"completed"}}}}}}"#
            ));
            let mut signals = [
                (BrokerDirection::AppServerToCli, resolved),
                (BrokerDirection::AppServerToCli, completed),
            ];
            if reverse_order {
                signals.reverse();
            }
            for (direction, metadata) in signals {
                resolve_codex_approval(
                    &store,
                    &mut approval_turns,
                    "connection-1",
                    direction,
                    &metadata,
                );
            }

            assert!(
                store.get(&key).is_none(),
                "reverse_order={reverse_order}: entry must be resolved regardless of signal order"
            );
            assert!(
                approval_turns.is_empty(),
                "reverse_order={reverse_order}: turn tracking must be cleaned up either way"
            );
        }
    }

    #[test]
    fn client_disconnected_discards_only_that_connections_pending_approvals() {
        let store = Arc::new(PendingApprovalStore::new());
        let store_a = store.clone();
        let request_id = Value::from(0);
        let mut approval_turns = HashMap::new();
        ingest_codex_approval(
            &store_a,
            &mut approval_turns,
            "connection-a",
            &request_id,
            ko2_body(),
        );
        ingest_codex_approval(
            &store_a,
            &mut approval_turns,
            "connection-b",
            &request_id,
            ko2_body(),
        );
        let key_a = codex_key("connection-a", &request_id);
        let key_b = codex_key("connection-b", &request_id);

        store.clear_owner(&ApprovalOwner::Codex {
            connection_id: "connection-a".to_string(),
        });
        approval_turns.retain(|(owner, _, _), _| owner != "connection-a");

        assert!(store.get(&key_a).is_none());
        assert!(store.get(&key_b).is_some());
    }
}
