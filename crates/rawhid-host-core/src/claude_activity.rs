use std::{
    collections::{HashMap, VecDeque},
    time::{Duration, Instant},
};

use serde::Serialize;
use serde_json::{json, Value};

use crate::{
    claude_decision::ClaudePermissionGate,
    claude_hook_event::{ClaudeHookEvent, ClaudeObserverEvent},
    next_ai_session_registration_order,
    packet::{AiActivityState, AiWorkPhase},
    pending_approval::{
        claude_key, ApprovalClient, ApprovalOwner, PendingApprovalBody, PendingApprovalContent,
        PendingApprovalStore, CLAUDE_DECISION_ALLOW, CLAUDE_DECISION_ALLOW_WITH_PERMISSIONS,
        CLAUDE_DECISION_DENY,
    },
};

pub const CLAUDE_DETAIL_STALE_TIMEOUT: Duration = Duration::from_secs(120);
const CLAUDE_COMPLETED_DISPLAY_DURATION: Duration = Duration::from_secs(15);
const CLAUDE_TOOL_TOMBSTONE_TTL: Duration = Duration::from_secs(120);
const MAX_TOOL_TOMBSTONES: usize = 256;

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ClaudeStateChangeReason {
    SessionStarted,
    TurnStarted,
    ToolStarted,
    ToolCompleted,
    WaitingApproval,
    WaitingInput,
    InputResolved,
    TurnCompleted,
    CompletedExpired,
    TurnFailed,
    DetailStale,
    Desynchronized,
    SessionEnded,
    WrapperExited,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ClaudeSessionSnapshot {
    pub launch_id: String,
    pub session_id: String,
    pub registration_order: u64,
    pub session_active: bool,
    pub activity_state: AiActivityState,
    pub work_phase: AiWorkPhase,
    pub desynchronized: bool,
    pub revision: u16,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ClaudeStateChange {
    pub state: ClaudeSessionSnapshot,
    pub reason: ClaudeStateChangeReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaudeAdapterDiagnostic {
    MissingSessionId,
    MissingToolUseId,
    InvalidPayload,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestKind {
    Approval,
    Input,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ClaudeCanonicalEvent {
    SessionStart,
    TurnStart,
    ToolStart {
        tool_use_id: String,
        phase: AiWorkPhase,
    },
    ToolComplete {
        tool_use_id: String,
    },
    ToolFailure {
        tool_use_id: String,
    },
    RequestStart {
        key: String,
        kind: RequestKind,
    },
    RequestResolve {
        key: String,
    },
    TurnComplete,
    TurnFailure,
    SessionEnd,
    WrapperExit,
}

/// Converts Claude Code hook payloads into events that do not retain answer text,
/// compact summaries, credentials, or other raw payload fields.
#[derive(Debug, Default)]
pub struct ClaudeEventAdapter;

impl ClaudeEventAdapter {
    fn adapt(
        &self,
        event: &ClaudeObserverEvent,
    ) -> Result<Option<ClaudeCanonicalEvent>, ClaudeAdapterDiagnostic> {
        match event {
            ClaudeObserverEvent::WrapperExited(_) => Ok(Some(ClaudeCanonicalEvent::WrapperExit)),
            ClaudeObserverEvent::Hook(hook) => self.adapt_hook(hook),
        }
    }

    fn adapt_hook(
        &self,
        hook: &ClaudeHookEvent,
    ) -> Result<Option<ClaudeCanonicalEvent>, ClaudeAdapterDiagnostic> {
        if hook.session_id.is_none() {
            return Err(ClaudeAdapterDiagnostic::MissingSessionId);
        }
        let tool_use_id = || required_string(&hook.body, "tool_use_id");
        match hook.hook_event_name.as_str() {
            "SessionStart" => Ok(Some(ClaudeCanonicalEvent::SessionStart)),
            "UserPromptSubmit" => Ok(Some(ClaudeCanonicalEvent::TurnStart)),
            "PreToolUse" => {
                let tool_use_id = tool_use_id().ok_or(ClaudeAdapterDiagnostic::MissingToolUseId)?;
                Ok(Some(ClaudeCanonicalEvent::ToolStart {
                    phase: tool_phase(required_string(&hook.body, "tool_name").as_deref()),
                    tool_use_id,
                }))
            }
            "PostToolUse" => Ok(Some(ClaudeCanonicalEvent::ToolComplete {
                tool_use_id: tool_use_id().ok_or(ClaudeAdapterDiagnostic::MissingToolUseId)?,
            })),
            "PostToolUseFailure" => Ok(Some(ClaudeCanonicalEvent::ToolFailure {
                tool_use_id: tool_use_id().ok_or(ClaudeAdapterDiagnostic::MissingToolUseId)?,
            })),
            "PermissionRequest" => Ok(Some(ClaudeCanonicalEvent::RequestStart {
                key: request_key(&hook.body, RequestKind::Approval),
                kind: RequestKind::Approval,
            })),
            "PermissionDenied" => Ok(Some(ClaudeCanonicalEvent::RequestResolve {
                key: request_key(&hook.body, RequestKind::Approval),
            })),
            "Elicitation" => Ok(Some(ClaudeCanonicalEvent::RequestStart {
                key: request_key(&hook.body, RequestKind::Input),
                kind: RequestKind::Input,
            })),
            "ElicitationResult" => Ok(Some(ClaudeCanonicalEvent::RequestResolve {
                key: request_key(&hook.body, RequestKind::Input),
            })),
            // PermissionRequest and Elicitation are the authoritative waiting-state
            // signals. Their related notifications are delayed supplementary events;
            // treating them as new requests can reopen an approval after PostToolUse.
            "Notification" => Ok(None),
            "Stop" => Ok(Some(ClaudeCanonicalEvent::TurnComplete)),
            "StopFailure" => Ok(Some(ClaudeCanonicalEvent::TurnFailure)),
            "SessionEnd" => Ok(Some(ClaudeCanonicalEvent::SessionEnd)),
            _ => Ok(None),
        }
    }
}

/// Reads unresolved-approval bodies out of Claude Code hook events and
/// keeps `PendingApprovalStore` in sync with them.
///
/// This is deliberately a sibling of `ClaudeEventAdapter`/
/// `ClaudeSessionReducer`, not a part of them: the doc comment on
/// `ClaudeEventAdapter` above states that it "does not retain answer text,
/// compact summaries, credentials, or other raw payload fields," and that
/// invariant is exactly what this type must not disturb. This consumer
/// reads `hook.body` -- the one place in the crate allowed to do so for
/// approval content -- and only ever hands it to `PendingApprovalStore`. It
/// holds no `AiActivityState` of its own and never feeds back into
/// `ClaudeSessionRegistry`.
///
/// Call `ingest` alongside (not instead of) `ClaudeSessionRegistry::apply`
/// for the same event; the two are independent observers of the same
/// stream.
///
/// `ingest` also takes a `&ClaudePermissionGate`, which was not true before
/// stage 3 of `docs/ai-approval-hud-design.md`. The reason is first-wins
/// arbitration (§9.4): a `PermissionRequest` hook connection is held open
/// by `claude_observer.rs` waiting on the gate for a decision, in parallel
/// with Claude Code's own terminal prompt. When the *terminal* answers
/// first, the tool actually runs (or the request is explicitly denied),
/// which arrives here as `PostToolUse` / `PermissionDenied` / `Stop` /
/// `SessionEnd` / `WrapperExited` -- exactly the events that already
/// resolved `store` before this stage existed. Each of those must now also
/// cancel the matching gate waiter, or the still-open hook connection would
/// eventually time out and fall back to 204 correctly, but only after
/// needlessly holding the connection (and the user's mental model of "did
/// my keyboard press do anything") for the rest of the hook's timeout.
/// Canceling immediately here is what makes the terminal's first answer
/// visibly final.
#[derive(Debug, Default)]
pub struct ClaudeApprovalBodyConsumer;

impl ClaudeApprovalBodyConsumer {
    pub fn ingest(
        &self,
        store: &PendingApprovalStore,
        gate: &ClaudePermissionGate,
        event: &ClaudeObserverEvent,
    ) {
        match event {
            ClaudeObserverEvent::Hook(hook) => self.ingest_hook(store, gate, hook),
            // A wrapper ending takes every session of that launch down with
            // it, even ones this consumer never saw a SessionStart for. Any
            // hook connection still waiting on a decision for one of that
            // launch's sessions is released the same way.
            ClaudeObserverEvent::WrapperExited(exit) => {
                store.clear_claude_launch(&exit.launch_id);
                gate.cancel_launch(&exit.launch_id);
            }
        }
    }

    fn ingest_hook(
        &self,
        store: &PendingApprovalStore,
        gate: &ClaudePermissionGate,
        hook: &ClaudeHookEvent,
    ) {
        let Some(session_id) = hook.session_id.as_deref() else {
            return;
        };
        match hook.hook_event_name.as_str() {
            // The real captured body
            // (`docs/claude-permission-hook-gate-results.md` §4) has no
            // `tool_use_id`, so the key here is `(launch_id, session_id)`,
            // not a per-tool id (see `claude_key`). One session holds at
            // most one unresolved request: a new `PermissionRequest`
            // overwrites whatever was pending for this session, per
            // `PendingApprovalStore::insert`'s overwrite semantics.
            "PermissionRequest" => {
                let owner = ApprovalOwner::ClaudeSession {
                    launch_id: hook.launch_id.clone(),
                    session_id: session_id.to_string(),
                };
                store.insert(
                    claude_key(&hook.launch_id, session_id),
                    ApprovalClient::ClaudeCode,
                    owner,
                    claude_approval_body(&hook.body),
                );
            }
            // PermissionDenied is Claude's explicit denial; PostToolUse
            // fires once the tool has actually run, which for an approved
            // tool is the resolution (Notification/permission_prompt is
            // deliberately not a trigger here, mirroring
            // ClaudeEventAdapter's own treatment of it above).
            //
            // Since the key no longer names a specific tool, use the
            // stored entry's `tool_use_id` (when both it and this event
            // have one) purely to avoid clearing a still-pending request
            // because an unrelated concurrent tool finished -- see
            // `pending_approval_has_priority_over_later_tool_activity` in
            // `ClaudeSessionReducer`'s own tests for that scenario. When
            // either side lacks a `tool_use_id` (the common case per the
            // real capture above), resolve unconditionally: with at most
            // one pending entry per session there is nothing more precise
            // to check against.
            "PermissionDenied" | "PostToolUse" => {
                let key = claude_key(&hook.launch_id, session_id);
                let event_tool_use_id = required_string(&hook.body, "tool_use_id");
                let should_resolve = match store.get(&key).map(|snapshot| snapshot.content) {
                    Some(PendingApprovalContent::Body(stored)) => {
                        match (stored.tool_use_id, event_tool_use_id) {
                            (Some(stored_id), Some(event_id)) => stored_id == event_id,
                            _ => true,
                        }
                    }
                    Some(PendingApprovalContent::Oversized) => true,
                    None => false,
                };
                if should_resolve {
                    store.resolve(&key);
                    // The terminal (or an already-run tool) resolved this
                    // request first; a hook connection still waiting on the
                    // gate for the same token must not also receive a
                    // decision -- see this consumer's own doc comment on
                    // why `ingest` touches the gate at all.
                    gate.cancel(key.token());
                }
            }
            // The turn (or session) ending leaves nothing left to answer,
            // even if no explicit resolution for a given request arrived.
            "Stop" | "SessionEnd" => {
                store.clear_owner(&ApprovalOwner::ClaudeSession {
                    launch_id: hook.launch_id.clone(),
                    session_id: session_id.to_string(),
                });
                gate.cancel(claude_key(&hook.launch_id, session_id).token());
            }
            _ => {}
        }
    }
}

/// Builds the normalized body for one Claude Code `PermissionRequest`. See
/// the comparison table in `docs/ai-approval-hud-design.md` §7.2: Claude
/// Code has no `reason` and no `availableDecisions` in the hook body --
/// the terminal's own choices aren't even present in it
/// (`docs/claude-permission-hook-gate-results.md` §4). Stage 3 normalizes
/// this on the Host side instead, into `[CLAUDE_DECISION_ALLOW,
/// CLAUDE_DECISION_DENY]` a HUD can offer exactly like Codex's own
/// `availableDecisions` array (`pending_approval.rs`'s
/// `PendingApprovalStore::claude_response` is what turns an index into it
/// back into a `ClaudeDecision`).
///
/// A real hook body can also carry a `permission_suggestions` array -- the
/// terminal's own "always allow" candidates, captured and verified against a
/// real Claude Code instance on 2026-09-06 (see `permission_suggestions`'s
/// own doc comment on `PendingApprovalBody` for the three shapes observed
/// and why this array is retained verbatim rather than interpreted here).
/// When that array is present and non-empty, the normalized
/// `available_decisions` grows a third, middle element,
/// `CLAUDE_DECISION_ALLOW_WITH_PERMISSIONS`, so a HUD can offer "allow, and
/// remember this choice" as a single combined option (per the design's
/// instruction to apply the whole suggestion set at once, never one rule at
/// a time). `allow` stays first and `deny` stays last in both shapes --
/// first so a HUD's default selection is never "always allow" by accident,
/// last because `hud_coordinator.rs`'s `reject_decision_index_from_body`
/// finds Claude Code's reject side via an exact `"deny"` match and must keep
/// finding it regardless of which shape is in play. An absent field, an
/// empty array, or a non-array value all fall back to the plain two-element
/// shape -- none of those are "there is a suggestion to offer."
fn claude_approval_body(body: &Value) -> PendingApprovalBody {
    let tool_input = body.get("tool_input");
    let command = tool_input
        .and_then(|input| input.get("command"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let primary_text = command.clone().or_else(|| {
        tool_input.map(|input| {
            serde_json::to_string(input).unwrap_or_else(|_| "<tool_input>".to_string())
        })
    });
    // Only a non-empty JSON array counts as "there is a suggestion to
    // offer" -- see this function's own doc comment on why an absent field,
    // an empty array, and a non-array value are all treated identically
    // (plain allow/deny, no `permission_suggestions` retained).
    let permission_suggestions = body
        .get("permission_suggestions")
        .and_then(Value::as_array)
        .filter(|suggestions| !suggestions.is_empty())
        .cloned();
    let available_decisions = if permission_suggestions.is_some() {
        vec![
            json!(CLAUDE_DECISION_ALLOW),
            json!(CLAUDE_DECISION_ALLOW_WITH_PERMISSIONS),
            json!(CLAUDE_DECISION_DENY),
        ]
    } else {
        vec![json!(CLAUDE_DECISION_ALLOW), json!(CLAUDE_DECISION_DENY)]
    };
    PendingApprovalBody {
        primary_text,
        full_command: command,
        reason: None,
        cwd: required_string(body, "cwd"),
        kind: required_string(body, "tool_name"),
        available_decisions: Some(available_decisions),
        // Auxiliary only -- see the doc comments on these fields in
        // `pending_approval.rs`. Absent in the real capture (§4), present
        // when Claude Code happens to include it.
        tool_use_id: required_string(body, "tool_use_id"),
        prompt_id: required_string(body, "prompt_id"),
        permission_suggestions,
    }
}

pub struct ClaudeSessionReducer {
    adapter: ClaudeEventAdapter,
    snapshot: ClaudeSessionSnapshot,
    retired: bool,
    launch_ended: bool,
    turn_active: bool,
    requests: HashMap<String, RequestKind>,
    active_items: HashMap<String, AiWorkPhase>,
    active_item_order: VecDeque<String>,
    tool_tombstones: VecDeque<(String, Instant)>,
    last_relevant_event: Option<Instant>,
    completed_deadline: Option<Instant>,
}

/// Owns all Claude Code sessions observed by Keylink Studio in stable
/// registration order. Cross-client display selection belongs to the host app.
pub struct ClaudeSessionRegistry {
    sessions: HashMap<(String, String), ClaudeSessionReducer>,
    order: Vec<(String, String)>,
    next_revision: u16,
}

impl Default for ClaudeSessionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ClaudeSessionRegistry {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            order: Vec::new(),
            next_revision: 1,
        }
    }

    pub fn apply(
        &mut self,
        event: ClaudeObserverEvent,
        now: Instant,
    ) -> Result<Vec<ClaudeStateChange>, ClaudeAdapterDiagnostic> {
        match &event {
            ClaudeObserverEvent::WrapperExited(exit) => {
                let mut changes = Vec::new();
                for ((launch_id, _), reducer) in &mut self.sessions {
                    if launch_id == &exit.launch_id {
                        changes.extend(reducer.apply(event.clone(), now)?);
                    }
                }
                Ok(changes)
            }
            ClaudeObserverEvent::Hook(hook) => {
                let session_id = hook
                    .session_id
                    .as_deref()
                    .ok_or(ClaudeAdapterDiagnostic::MissingSessionId)?;
                let key = (hook.launch_id.clone(), session_id.to_string());
                if !self.sessions.contains_key(&key) {
                    if hook.hook_event_name != "SessionStart" {
                        return Ok(Vec::new());
                    }
                    let revision = self.allocate_revision();
                    self.sessions.insert(
                        key.clone(),
                        ClaudeSessionReducer::new(&hook.launch_id, session_id, revision),
                    );
                    self.order.push(key.clone());
                }
                let reducer = self.sessions.get_mut(&key).expect("inserted above");
                let changes = reducer.apply(event, now)?;
                Ok(changes)
            }
        }
    }

    pub fn tick(&mut self, now: Instant) -> Vec<ClaudeStateChange> {
        self.sessions
            .values_mut()
            .flat_map(|reducer| reducer.tick(now))
            .collect()
    }

    pub fn mark_launch_desynchronized(&mut self, launch_id: &str) -> Vec<ClaudeStateChange> {
        self.sessions
            .iter_mut()
            .filter(|((candidate, _), _)| candidate == launch_id)
            .flat_map(|(_, reducer)| reducer.mark_desynchronized())
            .collect()
    }

    pub fn snapshots(&self) -> Vec<ClaudeSessionSnapshot> {
        self.order
            .iter()
            .filter_map(|key| self.sessions.get(key).map(ClaudeSessionReducer::snapshot))
            .cloned()
            .collect()
    }

    fn allocate_revision(&mut self) -> u16 {
        let revision = self.next_revision;
        self.next_revision = self.next_revision.wrapping_add(1);
        revision
    }
}

impl ClaudeSessionReducer {
    pub fn new(launch_id: impl Into<String>, session_id: impl Into<String>, revision: u16) -> Self {
        Self {
            adapter: ClaudeEventAdapter,
            snapshot: ClaudeSessionSnapshot {
                launch_id: launch_id.into(),
                session_id: session_id.into(),
                registration_order: next_ai_session_registration_order(),
                session_active: false,
                activity_state: AiActivityState::None,
                work_phase: AiWorkPhase::Unspecified,
                desynchronized: false,
                revision,
            },
            retired: false,
            launch_ended: false,
            turn_active: false,
            requests: HashMap::new(),
            active_items: HashMap::new(),
            active_item_order: VecDeque::new(),
            tool_tombstones: VecDeque::new(),
            last_relevant_event: None,
            completed_deadline: None,
        }
    }

    pub fn snapshot(&self) -> &ClaudeSessionSnapshot {
        &self.snapshot
    }

    pub fn apply(
        &mut self,
        event: ClaudeObserverEvent,
        now: Instant,
    ) -> Result<Vec<ClaudeStateChange>, ClaudeAdapterDiagnostic> {
        if !self.matches_event(&event) {
            return Ok(Vec::new());
        }
        self.prune_tombstones(now);
        let Some(event) = self.adapter.adapt(&event)? else {
            return Ok(Vec::new());
        };
        if self.launch_ended && event != ClaudeCanonicalEvent::WrapperExit {
            return Ok(Vec::new());
        }
        self.apply_canonical(event, now)
    }

    pub fn tick(&mut self, now: Instant) -> Vec<ClaudeStateChange> {
        self.prune_tombstones(now);
        if self
            .completed_deadline
            .is_some_and(|deadline| now >= deadline)
        {
            self.completed_deadline = None;
            if self.snapshot.session_active
                && self.snapshot.activity_state == AiActivityState::Completed
            {
                return vec![self.emit(
                    AiActivityState::Available,
                    AiWorkPhase::Unspecified,
                    ClaudeStateChangeReason::CompletedExpired,
                )];
            }
        }
        if !self.snapshot.session_active || !self.turn_active {
            return Vec::new();
        }
        let Some(last_event) = self.last_relevant_event else {
            return Vec::new();
        };
        if now.duration_since(last_event) < CLAUDE_DETAIL_STALE_TIMEOUT
            || (self.snapshot.activity_state == AiActivityState::Working
                && self.snapshot.work_phase == AiWorkPhase::Unspecified)
        {
            return Vec::new();
        }
        vec![self.emit(
            AiActivityState::Working,
            AiWorkPhase::Unspecified,
            ClaudeStateChangeReason::DetailStale,
        )]
    }

    /// Records a receiver-side loss of ordering information (for example, a
    /// bounded queue overflow). This never invents a terminal state: an active
    /// turn remains `WORKING + UNSPECIFIED` until Claude supplies an end event.
    pub fn mark_desynchronized(&mut self) -> Vec<ClaudeStateChange> {
        if self.snapshot.desynchronized {
            return Vec::new();
        }
        self.snapshot.desynchronized = true;
        if !self.snapshot.session_active {
            return Vec::new();
        }
        let activity = if self.turn_active {
            AiActivityState::Working
        } else {
            self.snapshot.activity_state
        };
        vec![self.emit(
            activity,
            AiWorkPhase::Unspecified,
            ClaudeStateChangeReason::Desynchronized,
        )]
    }

    fn matches_event(&self, event: &ClaudeObserverEvent) -> bool {
        match event {
            ClaudeObserverEvent::WrapperExited(event) => event.launch_id == self.snapshot.launch_id,
            ClaudeObserverEvent::Hook(event) => {
                event.launch_id == self.snapshot.launch_id
                    && event.session_id.as_deref() == Some(self.snapshot.session_id.as_str())
            }
        }
    }

    fn apply_canonical(
        &mut self,
        event: ClaudeCanonicalEvent,
        now: Instant,
    ) -> Result<Vec<ClaudeStateChange>, ClaudeAdapterDiagnostic> {
        match event {
            ClaudeCanonicalEvent::SessionStart => {
                if self.launch_ended || self.snapshot.session_active {
                    return Ok(Vec::new());
                }
                self.retired = false;
                self.snapshot.desynchronized = false;
                self.last_relevant_event = Some(now);
                self.completed_deadline = None;
                Ok(vec![self.emit(
                    AiActivityState::Available,
                    AiWorkPhase::Unspecified,
                    ClaudeStateChangeReason::SessionStarted,
                )])
            }
            ClaudeCanonicalEvent::TurnStart => {
                if self.retired || !self.snapshot.session_active {
                    return Ok(Vec::new());
                }
                self.turn_active = true;
                self.requests.clear();
                self.active_items.clear();
                self.active_item_order.clear();
                self.last_relevant_event = Some(now);
                self.completed_deadline = None;
                Ok(vec![self.emit(
                    AiActivityState::Working,
                    AiWorkPhase::Thinking,
                    ClaudeStateChangeReason::TurnStarted,
                )])
            }
            ClaudeCanonicalEvent::ToolStart { tool_use_id, phase } => {
                if self.retired || !self.snapshot.session_active || self.has_tombstone(&tool_use_id)
                {
                    return Ok(Vec::new());
                }
                self.turn_active = true;
                self.last_relevant_event = Some(now);
                let repeated_phase =
                    self.active_items.insert(tool_use_id.clone(), phase) == Some(phase);
                self.active_item_order.retain(|key| key != &tool_use_id);
                self.active_item_order.push_back(tool_use_id);
                if repeated_phase
                    && self.snapshot.activity_state == AiActivityState::Working
                    && self.snapshot.work_phase == phase
                {
                    return Ok(Vec::new());
                }
                Ok(vec![self.emit(
                    self.waiting_or_working(),
                    self.active_phase(),
                    ClaudeStateChangeReason::ToolStarted,
                )])
            }
            ClaudeCanonicalEvent::ToolComplete { tool_use_id }
            | ClaudeCanonicalEvent::ToolFailure { tool_use_id } => {
                if self.retired || !self.snapshot.session_active {
                    return Ok(Vec::new());
                }
                self.last_relevant_event = Some(now);
                if self.has_tombstone(&tool_use_id) {
                    return Ok(Vec::new());
                }
                self.insert_tombstone(tool_use_id.clone(), now);
                let was_active = self.active_items.remove(&tool_use_id).is_some();
                self.active_item_order.retain(|key| key != &tool_use_id);
                self.requests.remove(&tool_use_id);
                if !was_active {
                    return Ok(Vec::new());
                }
                Ok(vec![self.emit(
                    self.waiting_or_working(),
                    self.active_phase(),
                    ClaudeStateChangeReason::ToolCompleted,
                )])
            }
            ClaudeCanonicalEvent::RequestStart { key, kind } => {
                if self.retired || !self.snapshot.session_active {
                    return Ok(Vec::new());
                }
                self.turn_active = true;
                self.last_relevant_event = Some(now);
                let key = self.correlate_anonymous_approval(key);
                if self.requests.insert(key, kind).is_some() {
                    return Ok(Vec::new());
                }
                let reason = match kind {
                    RequestKind::Approval => ClaudeStateChangeReason::WaitingApproval,
                    RequestKind::Input => ClaudeStateChangeReason::WaitingInput,
                };
                Ok(vec![self.emit(
                    self.waiting_or_working(),
                    AiWorkPhase::Unspecified,
                    reason,
                )])
            }
            ClaudeCanonicalEvent::RequestResolve { key } => {
                if self.retired || !self.snapshot.session_active {
                    return Ok(Vec::new());
                }
                self.last_relevant_event = Some(now);
                let key = self.correlate_anonymous_approval(key);
                if self.requests.remove(&key).is_none() {
                    return Ok(Vec::new());
                }
                Ok(vec![self.emit(
                    self.waiting_or_working(),
                    self.active_phase(),
                    ClaudeStateChangeReason::InputResolved,
                )])
            }
            ClaudeCanonicalEvent::TurnComplete | ClaudeCanonicalEvent::TurnFailure => {
                if self.retired || !self.snapshot.session_active || !self.turn_active {
                    return Ok(Vec::new());
                }
                let (activity, reason) = if event == ClaudeCanonicalEvent::TurnComplete {
                    self.completed_deadline = Some(now + CLAUDE_COMPLETED_DISPLAY_DURATION);
                    (
                        AiActivityState::Completed,
                        ClaudeStateChangeReason::TurnCompleted,
                    )
                } else {
                    self.completed_deadline = None;
                    (AiActivityState::Error, ClaudeStateChangeReason::TurnFailed)
                };
                self.finish_turn(now);
                Ok(vec![self.emit(activity, AiWorkPhase::Unspecified, reason)])
            }
            ClaudeCanonicalEvent::SessionEnd => {
                Ok(self.retire(ClaudeStateChangeReason::SessionEnded))
            }
            ClaudeCanonicalEvent::WrapperExit => {
                self.launch_ended = true;
                Ok(self.retire(ClaudeStateChangeReason::WrapperExited))
            }
        }
    }

    fn retire(&mut self, reason: ClaudeStateChangeReason) -> Vec<ClaudeStateChange> {
        if self.retired {
            return Vec::new();
        }
        self.retired = true;
        self.turn_active = false;
        self.requests.clear();
        self.active_items.clear();
        self.active_item_order.clear();
        self.last_relevant_event = None;
        self.completed_deadline = None;
        if !self.snapshot.session_active {
            return Vec::new();
        }
        vec![self.emit(AiActivityState::None, AiWorkPhase::Unspecified, reason)]
    }

    fn finish_turn(&mut self, now: Instant) {
        let active_keys = self.active_items.keys().cloned().collect::<Vec<_>>();
        for key in active_keys {
            self.insert_tombstone(key, now);
        }
        self.turn_active = false;
        self.requests.clear();
        self.active_items.clear();
        self.active_item_order.clear();
        self.last_relevant_event = None;
    }

    fn correlate_anonymous_approval(&self, key: String) -> String {
        if key != "approval:unknown" {
            return key;
        }
        self.active_item_order
            .back()
            .filter(|tool_use_id| self.active_items.contains_key(*tool_use_id))
            .cloned()
            .unwrap_or(key)
    }

    fn waiting_or_working(&self) -> AiActivityState {
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
        } else {
            AiActivityState::Working
        }
    }

    fn active_phase(&self) -> AiWorkPhase {
        self.active_items
            .values()
            .copied()
            .max_by_key(|phase| match phase {
                AiWorkPhase::Unspecified => 0,
                AiWorkPhase::Thinking => 1,
                AiWorkPhase::Executing => 2,
                AiWorkPhase::Searching => 3,
            })
            .unwrap_or(AiWorkPhase::Thinking)
    }

    fn has_tombstone(&self, tool_use_id: &str) -> bool {
        self.tool_tombstones
            .iter()
            .any(|(key, _)| key == tool_use_id)
    }

    fn insert_tombstone(&mut self, tool_use_id: String, now: Instant) {
        self.tool_tombstones.retain(|(key, _)| key != &tool_use_id);
        self.tool_tombstones.push_back((tool_use_id, now));
        while self.tool_tombstones.len() > MAX_TOOL_TOMBSTONES {
            self.tool_tombstones.pop_front();
        }
    }

    fn prune_tombstones(&mut self, now: Instant) {
        while self
            .tool_tombstones
            .front()
            .is_some_and(|(_, created)| now.duration_since(*created) >= CLAUDE_TOOL_TOMBSTONE_TTL)
        {
            self.tool_tombstones.pop_front();
        }
    }

    fn emit(
        &mut self,
        activity_state: AiActivityState,
        work_phase: AiWorkPhase,
        reason: ClaudeStateChangeReason,
    ) -> ClaudeStateChange {
        self.snapshot.revision = self.snapshot.revision.wrapping_add(1);
        self.snapshot.session_active = activity_state != AiActivityState::None;
        self.snapshot.activity_state = activity_state;
        self.snapshot.work_phase = if activity_state == AiActivityState::Working {
            work_phase
        } else {
            AiWorkPhase::Unspecified
        };
        ClaudeStateChange {
            state: self.snapshot.clone(),
            reason,
        }
    }
}

fn required_string(body: &Value, key: &str) -> Option<String> {
    body.get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn request_key(body: &Value, kind: RequestKind) -> String {
    match kind {
        RequestKind::Approval => required_string(body, "tool_use_id")
            .or_else(|| {
                required_string(body, "request_id").map(|value| format!("approval:{value}"))
            })
            .unwrap_or_else(|| "approval:unknown".to_string()),
        RequestKind::Input => required_string(body, "elicitation_id")
            .or_else(|| required_string(body, "request_id"))
            .map(|value| format!("input:{value}"))
            .unwrap_or_else(|| "input:unknown".to_string()),
    }
}

fn tool_phase(tool_name: Option<&str>) -> AiWorkPhase {
    match tool_name {
        Some("WebSearch") | Some("WebFetch") => AiWorkPhase::Searching,
        _ => AiWorkPhase::Executing,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::claude_hook_event::{ClaudeHookEvent, ClaudeObserverEvent, ClaudeWrapperExited};

    fn reducer() -> ClaudeSessionReducer {
        ClaudeSessionReducer::new("launch-1", "session-1", 100)
    }

    fn hook(name: &str, body: Value) -> ClaudeObserverEvent {
        hook_for_session("session-1", name, body)
    }

    fn hook_for_session(session_id: &str, name: &str, body: Value) -> ClaudeObserverEvent {
        hook_for_launch_session("launch-1", session_id, name, body)
    }

    fn hook_for_launch_session(
        launch_id: &str,
        session_id: &str,
        name: &str,
        body: Value,
    ) -> ClaudeObserverEvent {
        ClaudeObserverEvent::Hook(ClaudeHookEvent {
            launch_id: launch_id.to_string(),
            hook_event_name: name.to_string(),
            session_id: Some(session_id.to_string()),
            body,
        })
    }

    fn start_session(reducer: &mut ClaudeSessionReducer, now: Instant) {
        let change = reducer
            .apply(hook("SessionStart", serde_json::json!({})), now)
            .unwrap();
        assert_eq!(change[0].state.activity_state, AiActivityState::Available);
    }

    #[test]
    fn tool_completion_before_start_is_tombstoned() {
        let now = Instant::now();
        let mut reducer = reducer();
        start_session(&mut reducer, now);
        reducer
            .apply(hook("UserPromptSubmit", serde_json::json!({})), now)
            .unwrap();
        assert!(reducer
            .apply(
                hook("PostToolUse", serde_json::json!({"tool_use_id": "tool-a"})),
                now + Duration::from_secs(1),
            )
            .unwrap()
            .is_empty());
        assert!(reducer
            .apply(
                hook(
                    "PreToolUse",
                    serde_json::json!({"tool_use_id": "tool-a", "tool_name": "Bash"}),
                ),
                now + Duration::from_secs(2),
            )
            .unwrap()
            .is_empty());
        assert_eq!(reducer.snapshot().work_phase, AiWorkPhase::Thinking);
    }

    #[test]
    fn stale_only_downgrades_detail() {
        let now = Instant::now();
        let mut reducer = reducer();
        start_session(&mut reducer, now);
        reducer
            .apply(hook("UserPromptSubmit", serde_json::json!({})), now)
            .unwrap();
        reducer
            .apply(
                hook(
                    "PreToolUse",
                    serde_json::json!({"tool_use_id": "tool-a", "tool_name": "Bash"}),
                ),
                now,
            )
            .unwrap();
        assert!(reducer
            .tick(now + CLAUDE_DETAIL_STALE_TIMEOUT - Duration::from_millis(1))
            .is_empty());
        let changes = reducer.tick(now + CLAUDE_DETAIL_STALE_TIMEOUT);
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].reason, ClaudeStateChangeReason::DetailStale);
        assert_eq!(changes[0].state.activity_state, AiActivityState::Working);
        assert_eq!(changes[0].state.work_phase, AiWorkPhase::Unspecified);
        assert!(changes[0].state.session_active);
    }

    #[test]
    fn completed_expires_after_display_duration() {
        assert_eq!(CLAUDE_COMPLETED_DISPLAY_DURATION, Duration::from_secs(15));
        let now = Instant::now();
        let mut reducer = reducer();
        start_session(&mut reducer, now);
        reducer
            .apply(hook("UserPromptSubmit", serde_json::json!({})), now)
            .unwrap();
        let completed = reducer
            .apply(hook("Stop", serde_json::json!({})), now)
            .unwrap();
        assert_eq!(
            completed[0].state.activity_state,
            AiActivityState::Completed
        );
        assert!(reducer
            .tick(now + CLAUDE_COMPLETED_DISPLAY_DURATION - Duration::from_millis(1))
            .is_empty());

        let expired = reducer.tick(now + CLAUDE_COMPLETED_DISPLAY_DURATION);
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].reason, ClaudeStateChangeReason::CompletedExpired);
        assert_eq!(expired[0].state.activity_state, AiActivityState::Available);
        assert_eq!(
            expired[0].state.revision,
            completed[0].state.revision.wrapping_add(1)
        );
        assert!(reducer
            .tick(now + CLAUDE_COMPLETED_DISPLAY_DURATION + Duration::from_secs(1))
            .is_empty());
    }

    #[test]
    fn starting_a_new_turn_cancels_completed_expiration() {
        let now = Instant::now();
        let mut reducer = reducer();
        start_session(&mut reducer, now);
        reducer
            .apply(hook("UserPromptSubmit", serde_json::json!({})), now)
            .unwrap();
        reducer
            .apply(hook("Stop", serde_json::json!({})), now)
            .unwrap();
        let working = reducer
            .apply(
                hook("UserPromptSubmit", serde_json::json!({})),
                now + Duration::from_secs(1),
            )
            .unwrap();
        assert_eq!(working[0].state.activity_state, AiActivityState::Working);
        assert!(reducer
            .tick(now + CLAUDE_COMPLETED_DISPLAY_DURATION)
            .is_empty());
        assert_eq!(reducer.snapshot().activity_state, AiActivityState::Working);
    }

    #[test]
    fn manual_permission_never_guesses_execution() {
        let now = Instant::now();
        let mut reducer = reducer();
        start_session(&mut reducer, now);
        reducer
            .apply(hook("UserPromptSubmit", serde_json::json!({})), now)
            .unwrap();
        reducer
            .apply(
                hook(
                    "PreToolUse",
                    serde_json::json!({"tool_use_id": "tool-a", "tool_name": "Bash"}),
                ),
                now,
            )
            .unwrap();
        let changes = reducer
            .apply(
                hook(
                    "PermissionRequest",
                    serde_json::json!({"tool_use_id": "tool-a"}),
                ),
                now,
            )
            .unwrap();
        assert_eq!(
            changes[0].state.activity_state,
            AiActivityState::WaitingApproval
        );
        assert_eq!(changes[0].state.work_phase, AiWorkPhase::Unspecified);
        let changes = reducer
            .apply(
                hook("PostToolUse", serde_json::json!({"tool_use_id": "tool-a"})),
                now + Duration::from_secs(1),
            )
            .unwrap();
        assert_eq!(changes[0].state.activity_state, AiActivityState::Working);
        assert_eq!(changes[0].state.work_phase, AiWorkPhase::Thinking);
        assert!(
            reducer
                .tick(now + Duration::from_secs(1) + CLAUDE_DETAIL_STALE_TIMEOUT)
                .len()
                == 1
        );
        assert_eq!(reducer.snapshot().activity_state, AiActivityState::Working);
        assert_eq!(reducer.snapshot().work_phase, AiWorkPhase::Unspecified);
    }

    #[test]
    fn permission_without_request_id_is_correlated_with_the_latest_tool() {
        let now = Instant::now();
        let mut reducer = reducer();
        start_session(&mut reducer, now);
        reducer
            .apply(hook("UserPromptSubmit", serde_json::json!({})), now)
            .unwrap();
        reducer
            .apply(
                hook(
                    "PreToolUse",
                    serde_json::json!({"tool_use_id": "tool-a", "tool_name": "Write"}),
                ),
                now,
            )
            .unwrap();

        let waiting = reducer
            .apply(hook("PermissionRequest", serde_json::json!({})), now)
            .unwrap();
        assert_eq!(waiting.len(), 1);
        assert_eq!(
            waiting[0].state.activity_state,
            AiActivityState::WaitingApproval
        );

        let completed = reducer
            .apply(
                hook("PostToolUse", serde_json::json!({"tool_use_id": "tool-a"})),
                now + Duration::from_secs(1),
            )
            .unwrap();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].state.activity_state, AiActivityState::Working);
        assert_eq!(completed[0].state.work_phase, AiWorkPhase::Thinking);
    }

    #[test]
    fn pending_approval_has_priority_over_later_tool_activity() {
        let now = Instant::now();
        let mut reducer = reducer();
        start_session(&mut reducer, now);
        reducer
            .apply(hook("UserPromptSubmit", serde_json::json!({})), now)
            .unwrap();
        reducer
            .apply(
                hook(
                    "PreToolUse",
                    serde_json::json!({"tool_use_id": "approval-tool", "tool_name": "Bash"}),
                ),
                now,
            )
            .unwrap();
        reducer
            .apply(
                hook(
                    "PermissionRequest",
                    serde_json::json!({"tool_use_id": "approval-tool"}),
                ),
                now,
            )
            .unwrap();

        let started = reducer
            .apply(
                hook(
                    "PreToolUse",
                    serde_json::json!({"tool_use_id": "search-tool", "tool_name": "WebSearch"}),
                ),
                now + Duration::from_secs(1),
            )
            .unwrap();
        assert_eq!(started.len(), 1);
        assert_eq!(
            started[0].state.activity_state,
            AiActivityState::WaitingApproval
        );
        assert_eq!(started[0].state.work_phase, AiWorkPhase::Unspecified);

        let search_completed = reducer
            .apply(
                hook(
                    "PostToolUse",
                    serde_json::json!({"tool_use_id": "search-tool"}),
                ),
                now + Duration::from_secs(2),
            )
            .unwrap();
        assert_eq!(
            search_completed[0].state.activity_state,
            AiActivityState::WaitingApproval
        );

        let approval_completed = reducer
            .apply(
                hook(
                    "PostToolUse",
                    serde_json::json!({"tool_use_id": "approval-tool"}),
                ),
                now + Duration::from_secs(3),
            )
            .unwrap();
        assert_eq!(
            approval_completed[0].state.activity_state,
            AiActivityState::Working
        );
        assert_eq!(
            approval_completed[0].state.work_phase,
            AiWorkPhase::Thinking
        );
    }

    #[test]
    fn delayed_permission_notification_does_not_reopen_completed_approval() {
        let now = Instant::now();
        let mut reducer = reducer();
        start_session(&mut reducer, now);
        reducer
            .apply(hook("UserPromptSubmit", serde_json::json!({})), now)
            .unwrap();
        reducer
            .apply(
                hook(
                    "PreToolUse",
                    serde_json::json!({"tool_use_id": "tool-a", "tool_name": "Bash"}),
                ),
                now,
            )
            .unwrap();
        reducer
            .apply(
                hook(
                    "PermissionRequest",
                    serde_json::json!({"tool_use_id": "tool-a"}),
                ),
                now,
            )
            .unwrap();
        reducer
            .apply(
                hook("PostToolUse", serde_json::json!({"tool_use_id": "tool-a"})),
                now + Duration::from_secs(1),
            )
            .unwrap();

        let delayed = reducer
            .apply(
                hook(
                    "Notification",
                    serde_json::json!({
                        "notification_type": "permission_prompt",
                        "tool_use_id": "tool-a"
                    }),
                ),
                now + Duration::from_secs(6),
            )
            .unwrap();
        assert!(delayed.is_empty());
        assert_eq!(reducer.snapshot().activity_state, AiActivityState::Working);
        assert_eq!(reducer.snapshot().work_phase, AiWorkPhase::Thinking);
    }

    #[test]
    fn session_end_and_wrapper_exit_are_idempotent() {
        let now = Instant::now();
        let mut reducer = reducer();
        start_session(&mut reducer, now);
        let ended = reducer
            .apply(hook("SessionEnd", serde_json::json!({})), now)
            .unwrap();
        assert_eq!(ended.len(), 1);
        assert_eq!(ended[0].reason, ClaudeStateChangeReason::SessionEnded);
        let wrapper = ClaudeObserverEvent::WrapperExited(ClaudeWrapperExited {
            launch_id: "launch-1".to_string(),
            exit_code: 0,
        });
        assert!(reducer.apply(wrapper.clone(), now).unwrap().is_empty());
        assert!(reducer.apply(wrapper, now).unwrap().is_empty());
        assert!(reducer
            .apply(hook("SessionStart", serde_json::json!({})), now)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn a_session_can_resume_after_session_end_until_its_wrapper_exits() {
        let now = Instant::now();
        let mut reducer = reducer();
        start_session(&mut reducer, now);
        reducer
            .apply(hook("SessionEnd", serde_json::json!({})), now)
            .unwrap();
        let resumed = reducer
            .apply(
                hook("SessionStart", serde_json::json!({})),
                now + Duration::from_secs(1),
            )
            .unwrap();
        assert_eq!(resumed.len(), 1);
        assert_eq!(resumed[0].reason, ClaudeStateChangeReason::SessionStarted);
        assert_eq!(resumed[0].state.activity_state, AiActivityState::Available);
    }

    #[test]
    fn desynchronization_keeps_an_active_turn_non_terminal_and_is_idempotent() {
        let now = Instant::now();
        let mut reducer = reducer();
        start_session(&mut reducer, now);
        reducer
            .apply(hook("UserPromptSubmit", serde_json::json!({})), now)
            .unwrap();
        let changes = reducer.mark_desynchronized();
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].reason, ClaudeStateChangeReason::Desynchronized);
        assert_eq!(changes[0].state.activity_state, AiActivityState::Working);
        assert_eq!(changes[0].state.work_phase, AiWorkPhase::Unspecified);
        assert!(changes[0].state.desynchronized);
        assert!(reducer.mark_desynchronized().is_empty());
    }

    #[test]
    fn wrapper_exit_retires_active_session_when_session_end_is_missing() {
        let now = Instant::now();
        let mut reducer = reducer();
        start_session(&mut reducer, now);
        let changes = reducer
            .apply(
                ClaudeObserverEvent::WrapperExited(ClaudeWrapperExited {
                    launch_id: "launch-1".to_string(),
                    exit_code: 9,
                }),
                now,
            )
            .unwrap();
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].reason, ClaudeStateChangeReason::WrapperExited);
        assert!(!changes[0].state.session_active);
    }

    #[test]
    fn session_start_for_a_different_session_is_not_applied() {
        let now = Instant::now();
        let mut reducer = reducer();
        let event = ClaudeObserverEvent::Hook(ClaudeHookEvent {
            launch_id: "launch-1".to_string(),
            hook_event_name: "SessionStart".to_string(),
            session_id: Some("session-2".to_string()),
            body: serde_json::json!({}),
        });
        assert!(reducer.apply(event, now).unwrap().is_empty());
        assert!(!reducer.snapshot().session_active);
    }

    #[test]
    fn raw_payload_requires_session_and_tool_identity() {
        let adapter = ClaudeEventAdapter;
        let missing_session = ClaudeObserverEvent::Hook(ClaudeHookEvent {
            launch_id: "launch-1".to_string(),
            hook_event_name: "PreToolUse".to_string(),
            session_id: None,
            body: serde_json::json!({"tool_use_id": "tool-a"}),
        });
        assert_eq!(
            adapter.adapt(&missing_session),
            Err(ClaudeAdapterDiagnostic::MissingSessionId)
        );
        let missing_tool = hook("PreToolUse", serde_json::json!({"tool_name": "Bash"}));
        assert_eq!(
            adapter.adapt(&missing_tool),
            Err(ClaudeAdapterDiagnostic::MissingToolUseId)
        );
    }

    #[test]
    fn tombstone_expires_after_the_detail_stale_window() {
        let now = Instant::now();
        let mut reducer = reducer();
        start_session(&mut reducer, now);
        reducer
            .apply(hook("UserPromptSubmit", serde_json::json!({})), now)
            .unwrap();
        reducer
            .apply(
                hook("PostToolUse", serde_json::json!({"tool_use_id": "tool-a"})),
                now,
            )
            .unwrap();
        assert!(reducer
            .apply(
                hook(
                    "PreToolUse",
                    serde_json::json!({"tool_use_id": "tool-a", "tool_name": "Bash"}),
                ),
                now + Duration::from_secs(1),
            )
            .unwrap()
            .is_empty());
        let changes = reducer
            .apply(
                hook(
                    "PreToolUse",
                    serde_json::json!({"tool_use_id": "tool-a", "tool_name": "Bash"}),
                ),
                now + CLAUDE_TOOL_TOMBSTONE_TTL,
            )
            .unwrap();
        assert_eq!(changes[0].state.work_phase, AiWorkPhase::Executing);
    }

    #[test]
    fn registry_keeps_sessions_in_stable_registration_order() {
        let now = Instant::now();
        let mut registry = ClaudeSessionRegistry::new();
        registry
            .apply(hook("SessionStart", serde_json::json!({})), now)
            .unwrap();
        let second = hook_for_session("session-2", "SessionStart", serde_json::json!({}));
        registry.apply(second, now).unwrap();
        registry
            .apply(
                hook_for_session("session-1", "UserPromptSubmit", serde_json::json!({})),
                now,
            )
            .unwrap();
        let sessions = registry.snapshots();
        assert_eq!(sessions.len(), 2);
        assert_eq!(sessions[0].session_id, "session-1");
        assert_eq!(sessions[1].session_id, "session-2");
    }

    #[test]
    fn registry_expires_completed_sessions_while_they_are_not_displayed() {
        let now = Instant::now();
        let mut registry = ClaudeSessionRegistry::new();
        registry
            .apply(hook("SessionStart", serde_json::json!({})), now)
            .unwrap();
        registry
            .apply(
                hook_for_session("session-2", "SessionStart", serde_json::json!({})),
                now,
            )
            .unwrap();
        registry
            .apply(hook("UserPromptSubmit", serde_json::json!({})), now)
            .unwrap();
        registry
            .apply(hook("Stop", serde_json::json!({})), now)
            .unwrap();

        let changes = registry.tick(now + CLAUDE_COMPLETED_DISPLAY_DURATION);
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].state.session_id, "session-1");
        assert_eq!(changes[0].reason, ClaudeStateChangeReason::CompletedExpired);
        assert_eq!(changes[0].state.activity_state, AiActivityState::Available);
        let snapshots = registry.snapshots();
        assert_eq!(snapshots[0].activity_state, AiActivityState::Available);
        assert_eq!(snapshots[1].activity_state, AiActivityState::Available);
    }

    #[test]
    fn registry_retires_all_sessions_for_a_launch() {
        let now = Instant::now();
        let mut registry = ClaudeSessionRegistry::new();
        registry
            .apply(hook("SessionStart", serde_json::json!({})), now)
            .unwrap();
        registry
            .apply(
                hook_for_session("session-2", "SessionStart", serde_json::json!({})),
                now,
            )
            .unwrap();
        let changes = registry
            .apply(
                ClaudeObserverEvent::WrapperExited(ClaudeWrapperExited {
                    launch_id: "launch-1".to_string(),
                    exit_code: 0,
                }),
                now,
            )
            .unwrap();
        assert_eq!(changes.len(), 2);
        assert!(registry
            .snapshots()
            .iter()
            .all(|snapshot| !snapshot.session_active));
    }

    #[test]
    fn registry_keeps_independent_launches_as_distinct_sessions() {
        let now = Instant::now();
        let mut registry = ClaudeSessionRegistry::new();
        registry
            .apply(
                hook_for_launch_session(
                    "launch-1",
                    "session-1",
                    "SessionStart",
                    serde_json::json!({}),
                ),
                now,
            )
            .unwrap();
        registry
            .apply(
                hook_for_launch_session(
                    "launch-2",
                    "session-2",
                    "SessionStart",
                    serde_json::json!({}),
                ),
                now,
            )
            .unwrap();
        let sessions = registry.snapshots();
        assert_eq!(sessions.len(), 2);
        assert_eq!(sessions[0].launch_id, "launch-1");
        assert_eq!(sessions[1].launch_id, "launch-2");
    }

    #[test]
    fn registry_marks_only_the_overflowed_launch_desynchronized() {
        let now = Instant::now();
        let mut registry = ClaudeSessionRegistry::new();
        registry
            .apply(hook("SessionStart", serde_json::json!({})), now)
            .unwrap();
        registry
            .apply(hook("UserPromptSubmit", serde_json::json!({})), now)
            .unwrap();
        let changes = registry.mark_launch_desynchronized("launch-1");
        assert_eq!(changes.len(), 1);
        assert!(changes[0].state.desynchronized);
        assert!(registry
            .mark_launch_desynchronized("other-launch")
            .is_empty());
    }

    /// The real `PermissionRequest` hook body captured in
    /// `docs/claude-permission-hook-gate-results.md` §4. Note it has no
    /// `tool_use_id` -- see
    /// `permission_request_without_tool_use_id_is_still_stored_and_keyed_by_session`
    /// below for what that means for this consumer.
    fn ko3_hook_body() -> Value {
        serde_json::json!({
            "session_id": "c21f2516-0000-0000-0000-000000000000",
            "transcript_path": "C:\\Users\\example\\session.jsonl",
            "cwd": "C:\\Users\\example\\keylink-claude-permission-probe-8",
            "scratchpad_dir": "C:\\Users\\example\\scratchpad",
            "prompt_id": "ad630989-0000-0000-0000-000000000000",
            "permission_mode": "acceptEdits",
            "effort": {"level": "high"},
            "hook_event_name": "PermissionRequest",
            "tool_name": "PowerShell",
            "tool_input": {
                "command": "New-Item -ItemType Directory ko3-test8",
                "description": "Create ko3-test8 directory"
            },
            "permission_suggestions": [
                {
                    "type": "addRules",
                    "rules": [
                        {"toolName": "PowerShell", "ruleContent": "New-Item -ItemType Directory ko3-test8"}
                    ],
                    "behavior": "allow",
                    "destination": "localSettings"
                }
            ]
        })
    }

    #[test]
    fn claude_approval_body_extracts_the_real_ko3_fields() {
        let extracted = claude_approval_body(&ko3_hook_body());
        assert_eq!(
            extracted.full_command.as_deref(),
            Some("New-Item -ItemType Directory ko3-test8")
        );
        assert_eq!(
            extracted.primary_text.as_deref(),
            Some("New-Item -ItemType Directory ko3-test8")
        );
        assert_eq!(
            extracted.cwd.as_deref(),
            Some("C:\\Users\\example\\keylink-claude-permission-probe-8")
        );
        assert_eq!(extracted.kind.as_deref(), Some("PowerShell"));
        // Claude Code's hook body itself has no `reason` field.
        assert_eq!(extracted.reason, None);
        // The hook body has no `availableDecisions` field at all, but the
        // Host normalizes one anyway (see `claude_approval_body`'s doc
        // comment) -- and the real KO3 capture's `permission_suggestions` is
        // non-empty, so the normalized array grows the middle
        // `allow_with_permissions` element rather than staying two-element.
        assert_eq!(
            extracted.available_decisions,
            Some(vec![
                json!(CLAUDE_DECISION_ALLOW),
                json!(CLAUDE_DECISION_ALLOW_WITH_PERMISSIONS),
                json!(CLAUDE_DECISION_DENY)
            ])
        );
        // permission_suggestions itself is retained verbatim, not just
        // reflected in available_decisions's shape.
        assert_eq!(
            extracted.permission_suggestions,
            ko3_hook_body()
                .get("permission_suggestions")
                .and_then(Value::as_array)
                .cloned()
        );
        // The real capture has no `tool_use_id`, but does carry `prompt_id`.
        assert_eq!(extracted.tool_use_id, None);
        assert_eq!(
            extracted.prompt_id.as_deref(),
            Some("ad630989-0000-0000-0000-000000000000")
        );
    }

    /// Pins the three "no suggestion to offer" cases together: a missing
    /// `permission_suggestions` field, an explicit empty array, and a
    /// present-but-not-an-array value all normalize identically to the
    /// plain two-element `[allow, deny]` -- none of them retain anything in
    /// `permission_suggestions` either. This is test #1 from the design's
    /// required-test list, covering every input the design calls "not a
    /// suggestion to offer" in one place so a future change to one branch
    /// cannot silently diverge from the other two.
    #[test]
    fn claude_approval_body_offers_only_allow_and_deny_when_there_is_no_real_suggestion() {
        let missing = serde_json::json!({"tool_name": "Bash"});
        let empty = serde_json::json!({"tool_name": "Bash", "permission_suggestions": []});
        let not_an_array =
            serde_json::json!({"tool_name": "Bash", "permission_suggestions": "not-an-array"});

        for body in [missing, empty, not_an_array] {
            let extracted = claude_approval_body(&body);
            assert_eq!(
                extracted.available_decisions,
                Some(vec![
                    json!(CLAUDE_DECISION_ALLOW),
                    json!(CLAUDE_DECISION_DENY)
                ]),
                "unexpected available_decisions for input: {body:?}"
            );
            assert_eq!(
                extracted.permission_suggestions, None,
                "unexpected permission_suggestions for input: {body:?}"
            );
        }
    }

    /// The positive counterpart to the test above: test #1's third
    /// requirement, that a non-empty `permission_suggestions` array
    /// produces the three-element `[allow, allow_with_permissions, deny]`
    /// array in exactly that order, with the input array retained
    /// unmodified.
    #[test]
    fn claude_approval_body_offers_allow_with_permissions_when_suggestions_are_present() {
        let suggestions = vec![serde_json::json!({
            "type": "addRules",
            "behavior": "allow",
            "destination": "session",
            "rules": [{"toolName": "Read", "ruleContent": "//c/temp/**"}]
        })];
        let body = serde_json::json!({
            "tool_name": "Read",
            "permission_suggestions": suggestions
        });

        let extracted = claude_approval_body(&body);
        assert_eq!(
            extracted.available_decisions,
            Some(vec![
                json!(CLAUDE_DECISION_ALLOW),
                json!(CLAUDE_DECISION_ALLOW_WITH_PERMISSIONS),
                json!(CLAUDE_DECISION_DENY)
            ])
        );
        assert_eq!(extracted.permission_suggestions, Some(suggestions));
    }

    #[test]
    fn claude_approval_body_falls_back_to_a_tool_input_summary_without_command() {
        let body = serde_json::json!({"tool_input": {"file_path": "notes.md"}});
        let extracted = claude_approval_body(&body);
        assert_eq!(extracted.full_command, None);
        assert!(extracted
            .primary_text
            .as_deref()
            .is_some_and(|text| text.contains("notes.md")));
    }

    #[test]
    fn consumer_inserts_on_permission_request_and_resolves_on_permission_denied() {
        let store = PendingApprovalStore::new();
        let consumer = ClaudeApprovalBodyConsumer;
        let gate = ClaudePermissionGate::default();
        let key = claude_key("launch-1", "session-1");

        consumer.ingest(&store, &gate,
            &hook(
                "PermissionRequest",
                serde_json::json!({"tool_use_id": "tool-a", "cwd": "C:\\work", "tool_name": "Bash"}),
            ),
        );
        assert!(store.get(&key).is_some());
        // A hook connection is registered on the gate the moment
        // `claude_observer.rs` sees the `PermissionRequest` -- simulate
        // that here so this test can observe the consumer canceling it.
        let waiter = gate.register(key.token().to_string());

        consumer.ingest(
            &store,
            &gate,
            &hook(
                "PermissionDenied",
                serde_json::json!({"tool_use_id": "tool-a"}),
            ),
        );
        assert!(store.get(&key).is_none());
        assert!(
            waiter.blocking_recv().is_err(),
            "PermissionDenied must cancel the gate waiter, not leave it open"
        );
    }

    #[test]
    fn consumer_resolves_on_post_tool_use() {
        let store = PendingApprovalStore::new();
        let consumer = ClaudeApprovalBodyConsumer;
        let gate = ClaudePermissionGate::default();
        let key = claude_key("launch-1", "session-1");
        consumer.ingest(
            &store,
            &gate,
            &hook(
                "PermissionRequest",
                serde_json::json!({"tool_use_id": "tool-a"}),
            ),
        );
        let waiter = gate.register(key.token().to_string());
        consumer.ingest(
            &store,
            &gate,
            &hook("PostToolUse", serde_json::json!({"tool_use_id": "tool-a"})),
        );
        assert!(store.get(&key).is_none());
        assert!(
            waiter.blocking_recv().is_err(),
            "PostToolUse must cancel the gate waiter, not leave it open"
        );
    }

    /// The correlation key is `(launch_id, session_id)`, not `tool_use_id`
    /// -- but a stored `tool_use_id` (when present) is still used to avoid
    /// clearing a genuinely still-pending request just because some other,
    /// concurrently running tool in the same session finished. This
    /// mirrors the scenario `ClaudeSessionReducer`'s own
    /// `pending_approval_has_priority_over_later_tool_activity` test
    /// covers for activity state.
    #[test]
    fn consumer_does_not_resolve_when_a_different_tools_post_tool_use_arrives() {
        let store = PendingApprovalStore::new();
        let consumer = ClaudeApprovalBodyConsumer;
        let gate = ClaudePermissionGate::default();
        let key = claude_key("launch-1", "session-1");
        consumer.ingest(
            &store,
            &gate,
            &hook(
                "PermissionRequest",
                serde_json::json!({"tool_use_id": "approval-tool"}),
            ),
        );
        // An unrelated, concurrently running tool completes.
        consumer.ingest(
            &store,
            &gate,
            &hook(
                "PostToolUse",
                serde_json::json!({"tool_use_id": "search-tool"}),
            ),
        );
        assert!(
            store.get(&key).is_some(),
            "an unrelated tool's completion must not clear a still-pending approval"
        );

        // The approval's own tool finishing does resolve it.
        consumer.ingest(
            &store,
            &gate,
            &hook(
                "PostToolUse",
                serde_json::json!({"tool_use_id": "approval-tool"}),
            ),
        );
        assert!(store.get(&key).is_none());
    }

    #[test]
    fn consumer_clears_the_session_on_stop_and_session_end() {
        let store = PendingApprovalStore::new();
        let consumer = ClaudeApprovalBodyConsumer;
        let gate = ClaudePermissionGate::default();
        let key = claude_key("launch-1", "session-1");
        consumer.ingest(
            &store,
            &gate,
            &hook(
                "PermissionRequest",
                serde_json::json!({"tool_use_id": "tool-a"}),
            ),
        );
        let stop_waiter = gate.register(key.token().to_string());
        consumer.ingest(&store, &gate, &hook("Stop", serde_json::json!({})));
        assert!(store.get(&key).is_none());
        assert!(
            stop_waiter.blocking_recv().is_err(),
            "Stop must cancel the gate waiter, not leave it open"
        );

        consumer.ingest(
            &store,
            &gate,
            &hook(
                "PermissionRequest",
                serde_json::json!({"tool_use_id": "tool-b"}),
            ),
        );
        let session_end_waiter = gate.register(key.token().to_string());
        consumer.ingest(&store, &gate, &hook("SessionEnd", serde_json::json!({})));
        assert!(store.get(&key).is_none());
        assert!(
            session_end_waiter.blocking_recv().is_err(),
            "SessionEnd must cancel the gate waiter, not leave it open"
        );
    }

    #[test]
    fn a_new_permission_request_overwrites_the_sessions_previous_pending_request() {
        let store = PendingApprovalStore::new();
        let consumer = ClaudeApprovalBodyConsumer;
        let gate = ClaudePermissionGate::default();
        let key = claude_key("launch-1", "session-1");
        consumer.ingest(
            &store,
            &gate,
            &hook(
                "PermissionRequest",
                serde_json::json!({"tool_use_id": "tool-a", "tool_name": "Bash"}),
            ),
        );
        consumer.ingest(
            &store,
            &gate,
            &hook(
                "PermissionRequest",
                serde_json::json!({"tool_use_id": "tool-b", "tool_name": "Write"}),
            ),
        );
        assert_eq!(store.len(), 1);
        match store.get(&key).expect("entry present").content {
            PendingApprovalContent::Body(body) => {
                assert_eq!(body.tool_use_id.as_deref(), Some("tool-b"));
                assert_eq!(body.kind.as_deref(), Some("Write"));
            }
            PendingApprovalContent::Oversized => panic!("unexpected oversized marker"),
        }
    }

    #[test]
    fn consumer_clears_every_session_of_a_launch_on_wrapper_exit() {
        let store = PendingApprovalStore::new();
        let consumer = ClaudeApprovalBodyConsumer;
        let gate = ClaudePermissionGate::default();
        let key_a = claude_key("launch-1", "session-1");
        let key_b = claude_key("launch-1", "session-2");
        consumer.ingest(
            &store,
            &gate,
            &hook_for_session(
                "session-1",
                "PermissionRequest",
                serde_json::json!({"tool_use_id": "tool-a"}),
            ),
        );
        consumer.ingest(
            &store,
            &gate,
            &hook_for_session(
                "session-2",
                "PermissionRequest",
                serde_json::json!({"tool_use_id": "tool-b"}),
            ),
        );
        let waiter_a = gate.register(key_a.token().to_string());
        let waiter_b = gate.register(key_b.token().to_string());

        consumer.ingest(
            &store,
            &gate,
            &ClaudeObserverEvent::WrapperExited(ClaudeWrapperExited {
                launch_id: "launch-1".to_string(),
                exit_code: 0,
            }),
        );

        assert!(store.get(&key_a).is_none());
        assert!(store.get(&key_b).is_none());
        assert!(
            waiter_a.blocking_recv().is_err(),
            "WrapperExited must cancel every waiter of that launch"
        );
        assert!(
            waiter_b.blocking_recv().is_err(),
            "WrapperExited must cancel every waiter of that launch"
        );
    }

    #[test]
    fn wrapper_exit_only_clears_its_own_launch() {
        let store = PendingApprovalStore::new();
        let consumer = ClaudeApprovalBodyConsumer;
        let gate = ClaudePermissionGate::default();
        let key_other_launch = claude_key("launch-2", "session-2");
        consumer.ingest(
            &store,
            &gate,
            &hook_for_launch_session(
                "launch-2",
                "session-2",
                "PermissionRequest",
                serde_json::json!({"tool_use_id": "tool-b"}),
            ),
        );
        consumer.ingest(
            &store,
            &gate,
            &ClaudeObserverEvent::WrapperExited(ClaudeWrapperExited {
                launch_id: "launch-1".to_string(),
                exit_code: 0,
            }),
        );
        assert!(store.get(&key_other_launch).is_some());
    }

    /// The coordinator's original correlation-key spec (`session_id` /
    /// `tool_use_id`) was based on an older design document, not the real
    /// capture. The real `PermissionRequest` body in
    /// `docs/claude-permission-hook-gate-results.md` §4 has no
    /// `tool_use_id` at all -- confirmed here by feeding that exact body
    /// through the consumer and checking it is still stored, keyed by
    /// `(launch_id, session_id)` alone.
    #[test]
    fn permission_request_without_tool_use_id_is_still_stored_and_keyed_by_session() {
        let store = PendingApprovalStore::new();
        let consumer = ClaudeApprovalBodyConsumer;
        let gate = ClaudePermissionGate::default();
        let key = claude_key("launch-1", "session-1");
        consumer.ingest(&store, &gate, &hook("PermissionRequest", ko3_hook_body()));
        let snapshot = store.get(&key).expect("stored despite missing tool_use_id");
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
    }
}
