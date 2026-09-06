//! In-memory retention layer for unresolved AI approval-request bodies.
//!
//! This is the layer beneath the future HUD described in
//! `docs/ai-approval-hud-design.md` (see its §7.2 for the fields a HUD
//! would show and §9 for the resolution triggers that discard an entry).
//! It exists so a later HUD window can read *what* an AI is asking
//! approval for without either client's activity reducer
//! (`codex_activity.rs`'s `AiClientStateReducer`/`CodexEventAdapter`,
//! `claude_activity.rs`'s `ClaudeSessionReducer`/`ClaudeEventAdapter`)
//! having to carry request content through its state transitions. Those
//! reducers stay metadata-only; this store is the only place request
//! bodies are kept, and only for as long as a request is unresolved.
//!
//! Rules enforced here, not left to callers:
//! - Only *unresolved* requests are kept. Every insertion is expected to be
//!   paired with a later `resolve`/`clear_*` call from the per-client
//!   consumer that populated it (see `codex_activity.rs`'s
//!   `resolve_codex_approval` and `claude_activity.rs`'s
//!   `ClaudeApprovalBodyConsumer`).
//! - Nothing is written to disk. This is a plain in-memory `Mutex`; drop
//!   the process and every entry is gone.
//! - A body larger than [`MAX_PENDING_APPROVAL_BODY_BYTES`] is replaced by
//!   [`PendingApprovalContent::Oversized`] -- the marker is kept, the text
//!   is not.
//! - The store caps its total entry count at [`MAX_ENTRIES`] and evicts the
//!   least-recently-touched *unprotected* entry to make room. An entry
//!   marked protected (i.e. currently shown by a HUD, via
//!   [`PendingApprovalStore::set_protected`]) is never evicted by capacity
//!   pressure; it can still be removed by `resolve`/`clear_*`.

use std::{
    collections::{HashMap, VecDeque},
    sync::Mutex,
};

use serde::Serialize;
use serde_json::Value;

use crate::claude_decision::{ClaudeDecision, CLAUDE_HUD_DENY_MESSAGE};

/// A single retained body may not exceed this many serialized bytes.
pub const MAX_PENDING_APPROVAL_BODY_BYTES: usize = 1024 * 1024;

/// The store keeps at most this many unresolved entries at once, across
/// both clients.
pub const MAX_ENTRIES: usize = 256;

/// The Host-normalized "allow" element of a Claude Code entry's
/// `available_decisions` (see that field's doc comment on
/// [`PendingApprovalBody`] for why Claude Code's array is Host-built rather
/// than protocol-supplied).
pub const CLAUDE_DECISION_ALLOW: &str = "allow";
/// The Host-normalized "deny" element of a Claude Code entry's
/// `available_decisions`. See [`CLAUDE_DECISION_ALLOW`].
pub const CLAUDE_DECISION_DENY: &str = "deny";
/// The Host-normalized "allow, and remember this choice" element of a
/// Claude Code entry's `available_decisions`. Only present when the real
/// hook body carried a non-empty `permission_suggestions` array
/// (`claude_activity.rs`'s `claude_approval_body`) -- unlike
/// [`CLAUDE_DECISION_ALLOW`]/[`CLAUDE_DECISION_DENY`], which are offered
/// unconditionally, this element only exists when there is something for it
/// to apply. Selecting it resolves to `ClaudeDecision::AllowWithPermissions`
/// (`PendingApprovalStore::claude_response`), carrying the exact
/// `permission_suggestions` array back out as `updatedPermissions` -- see
/// that variant's own doc comment for why the array is never reconstructed.
pub const CLAUDE_DECISION_ALLOW_WITH_PERMISSIONS: &str = "allow_with_permissions";

/// Which AI client an approval request originated from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ApprovalClient {
    Codex,
    ClaudeCode,
}

/// Groups every pending entry belonging to one connection/session, so a
/// disconnect, a `Stop`, or a `SessionEnd` can discard them together
/// without the caller needing to know individual request keys.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ApprovalOwner {
    /// One Codex Broker WebSocket connection (`connection_id` from
    /// `codex_broker.rs`).
    Codex { connection_id: String },
    /// One Claude Code session within one wrapper launch.
    ClaudeSession {
        launch_id: String,
        session_id: String,
    },
}

/// Correlates a pending entry with the frames that later resolve it.
///
/// Build this with [`codex_key`] or [`claude_key`]; the store itself
/// treats it as an opaque handle.
#[derive(Debug, Clone)]
pub struct ApprovalKey {
    token: String,
    codex_target: Option<CodexApprovalTarget>,
}

#[derive(Debug, Clone)]
struct CodexApprovalTarget {
    connection_id: String,
    request_id: Value,
    /// Internal only: binds this request to the exact Codex display thread.
    /// It is never included in a HUD payload or Host Link packet.
    thread_id: Option<String>,
}

impl PartialEq for ApprovalKey {
    fn eq(&self, other: &Self) -> bool {
        self.token == other.token
    }
}

impl Eq for ApprovalKey {}

impl std::hash::Hash for ApprovalKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.token.hash(state);
    }
}

impl ApprovalKey {
    pub fn new(value: impl Into<String>) -> Self {
        Self {
            token: value.into(),
            codex_target: None,
        }
    }

    /// Opaque identifier sent to the HUD and returned with an answer. It
    /// contains no request body or credential.
    pub fn token(&self) -> &str {
        &self.token
    }

    fn codex_target(&self) -> Option<&CodexApprovalTarget> {
        self.codex_target.as_ref()
    }

    /// Internal-only: the exact `(connection_id, thread_id)` this key was
    /// built for, when it is a Codex key with a known thread id. `None` for
    /// a Claude Code key, a Codex key built via [`codex_key`] (no thread
    /// id), or any key whose thread id could not be determined.
    ///
    /// This exists purely for Host-side slot/target matching -- comparing a
    /// ScreenKey slot's assigned session against the HUD's current target
    /// (`hud_coordinator.rs`'s `target_codex_thread`, `actions.rs`'s
    /// `codex_target_for_slot` comparison). Like
    /// `CodexApprovalTarget::thread_id`, the returned thread id must never
    /// be placed in a HUD payload or a Host Link packet.
    pub fn codex_thread(&self) -> Option<(&str, &str)> {
        let target = self.codex_target.as_ref()?;
        let thread_id = target.thread_id.as_deref()?;
        Some((target.connection_id.as_str(), thread_id))
    }
}

/// Builds the correlation key for a Codex `requestApproval`, from the
/// Broker connection id and the request's JSON-RPC id.
pub fn codex_key(connection_id: &str, request_id: &Value) -> ApprovalKey {
    codex_key_for_thread(connection_id, request_id, None)
}

/// Builds a Codex correlation key with its request's optional `threadId`.
/// A missing `threadId` remains deliberately unselectable from a ScreenKey:
/// the strict display-slot lookup below never infers a thread from only a
/// connection id.
pub fn codex_key_for_thread(
    connection_id: &str,
    request_id: &Value,
    thread_id: Option<&str>,
) -> ApprovalKey {
    ApprovalKey {
        token: format!("codex:{connection_id}:{}", json_rpc_id_token(request_id)),
        codex_target: Some(CodexApprovalTarget {
            connection_id: connection_id.to_string(),
            request_id: request_id.clone(),
            thread_id: thread_id.map(str::to_string),
        }),
    }
}

/// Builds the correlation key for a Claude Code `PermissionRequest`, from
/// `(launch_id, session_id)` -- the identifier pair already used elsewhere
/// in Studio to name a Claude Code session (see
/// `docs/ai-session-display-switching.md`,
/// `docs/ai-display-slot-multiscreen-host-design.md`).
///
/// This is deliberately *not* keyed on `tool_use_id`: the real captured
/// `PermissionRequest` body
/// (`docs/claude-permission-hook-gate-results.md` §4) does not have one.
/// One session holds at most one unresolved request; a new
/// `PermissionRequest` for the same session overwrites the previous one
/// (see [`PendingApprovalStore::insert`]).
pub fn claude_key(launch_id: &str, session_id: &str) -> ApprovalKey {
    ApprovalKey::new(format!(
        "{}{session_id}",
        claude_launch_token_prefix(launch_id)
    ))
}

/// The token prefix shared by every [`claude_key`] built for one Claude
/// Code wrapper launch. This module is the only place that knows a Claude
/// Code token's format (`"claude:{launch_id}:{session_id}"`); rather than
/// let another module reconstruct that format to find "every token of this
/// launch", it calls this function instead. `claude_decision.rs`'s
/// `ClaudePermissionGate::cancel_launch` is the current caller, for
/// discarding every hook connection still waiting on a decision when the
/// launch's wrapper process exits.
pub fn claude_launch_token_prefix(launch_id: &str) -> String {
    format!("claude:{launch_id}:")
}

fn json_rpc_id_token(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::String(value) => format!("s:{value}"),
        Value::Number(value) => format!("n:{value}"),
        other => format!("?:{other}"),
    }
}

/// The normalized, cross-client body of one approval request. Every field
/// is optional because each client populates a different subset -- see the
/// comparison table in `docs/ai-approval-hud-design.md` §7.2.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PendingApprovalBody {
    /// What to show as the primary line: Codex's
    /// `commandActions[].command`, or Claude's `tool_input.command` (or a
    /// summary of `tool_input` when there is no `command` field).
    pub primary_text: Option<String>,
    /// The command in full, including any shell wrapper. Codex's `command`
    /// / Claude's `tool_input.command`.
    pub full_command: Option<String>,
    /// Codex's `reason`, written by the model in the user's language.
    /// Claude Code has no equivalent field.
    pub reason: Option<String>,
    pub cwd: Option<String>,
    /// Codex's `kind`, or Claude's `tool_name`.
    pub kind: Option<String>,
    /// The offered decisions, kept exactly as received/normalized -- **but
    /// the two clients populate this very differently, and only one of them
    /// may be reconstructed:**
    ///
    /// - **Codex**'s `availableDecisions`, carried through *verbatim*.
    ///   Elements are a mix of strings and objects and the set changes per
    ///   request (see `docs/codex-approval-proxy-gate-results.md` §5.1).
    ///   **Never reconstruct or summarize a Codex entry** -- an opaque
    ///   value must be returned to Codex byte-for-byte, or the App Server
    ///   may reject it.
    ///  - **Claude Code**'s hook body has no `availableDecisions` field at
    ///    all (`docs/claude-permission-hook-gate-results.md` §4) -- the
    ///    terminal's own allow/deny choice is never sent to Studio. The
    ///    Host normalizes this itself (`claude_activity.rs`'s
    ///    `claude_approval_body`), purely so a HUD can offer Claude Code
    ///    requests through the same index-into-this-array shape as Codex's.
    ///    The normalized array is `[CLAUDE_DECISION_ALLOW,
    ///    CLAUDE_DECISION_DENY]`, or, when the hook body also carried a
    ///    non-empty `permission_suggestions` array, the three-element
    ///    `[CLAUDE_DECISION_ALLOW, CLAUDE_DECISION_ALLOW_WITH_PERMISSIONS,
    ///    CLAUDE_DECISION_DENY]` -- always in that order, `allow` first and
    ///    `deny` last, so a HUD's default selection is never "always allow"
    ///    and `reject_decision_index_from_body`'s exact-match `"deny"`
    ///    lookup keeps working regardless of which shape is in play. Unlike
    ///    Codex's elements, a Claude Code element *is* meant to be
    ///    interpreted -- `PendingApprovalStore::claude_response` is the one
    ///    place that does, turning an index back into a `ClaudeDecision`.
    pub available_decisions: Option<Vec<Value>>,
    /// Claude's `tool_use_id`, when the hook body happens to carry one.
    /// Auxiliary only -- the store's correlation key is `(launch_id,
    /// session_id)` via [`claude_key`], never this field. Codex has no
    /// equivalent (its per-request id lives in the key itself).
    pub tool_use_id: Option<String>,
    /// Claude's `prompt_id`, the turn this request belongs to. Kept as a
    /// hint for resolution logic that needs to reason about turns; Codex's
    /// analogous turn tracking lives outside this body, in
    /// `codex_activity.rs`'s `approval_turns` map.
    pub prompt_id: Option<String>,
    /// Claude Code's `permission_suggestions` array, verbatim, when the hook
    /// body carried a non-empty one. Codex has no counterpart at all -- its
    /// App Server never sends anything like this, so this field is always
    /// `None` for a Codex entry.
    ///
    /// This is a separate field from `available_decisions` on purpose, even
    /// though both hold opaque arrays the Host must never reinterpret:
    /// `available_decisions` is the *menu* -- the fixed, Host-built list of
    /// choices a HUD offers (`[allow, allow_with_permissions, deny]` or
    /// `[allow, deny]`, per `claude_activity.rs`'s `claude_approval_body`) --
    /// while this field is the *payload* one specific menu entry
    /// (`CLAUDE_DECISION_ALLOW_WITH_PERMISSIONS`) applies when chosen. The
    /// menu is Host-normalized and stable in shape; this field is whatever
    /// Claude Code happened to send for this one request, cloned exactly as
    /// received, and handed back out byte-for-byte as `updatedPermissions`
    /// (`ClaudeDecision::hook_response_body`) if that entry is picked. Never
    /// interpret, reorder, or rewrite an element of this array -- it was
    /// captured verbatim from a real Claude Code instance on 2026-09-06 in
    /// three shapes that shared no common structure beyond being JSON
    /// objects, so any assumption about its fields beyond "pass it through"
    /// is unverified.
    pub permission_suggestions: Option<Vec<Value>>,
}

impl PendingApprovalBody {
    fn estimated_size(&self) -> usize {
        serde_json::to_vec(self)
            .map(|bytes| bytes.len())
            .unwrap_or(usize::MAX)
    }
}

/// What the store actually holds for one key: either the body, or a marker
/// recording that the body was too large to retain.
#[derive(Debug, Clone, PartialEq)]
pub enum PendingApprovalContent {
    Body(PendingApprovalBody),
    /// The body exceeded [`MAX_PENDING_APPROVAL_BODY_BYTES`] and was not
    /// retained.
    Oversized,
}

/// A read-only view of one stored entry.
#[derive(Debug, Clone, PartialEq)]
pub struct PendingApprovalSnapshot {
    pub client: ApprovalClient,
    pub content: PendingApprovalContent,
    pub protected: bool,
}

/// Validated routing data for one Codex HUD answer. The decision is cloned
/// directly from that request's opaque `availableDecisions` array.
#[derive(Debug, Clone, PartialEq)]
pub struct CodexPendingResponse {
    pub key: ApprovalKey,
    pub connection_id: String,
    pub request_id: Value,
    pub decision: Value,
}

/// Validated routing data for one Claude Code HUD answer. Unlike
/// [`CodexPendingResponse`], `decision` is not the raw array element --
/// [`PendingApprovalStore::claude_response`] resolves the Host-normalized
/// string (`CLAUDE_DECISION_ALLOW`/`CLAUDE_DECISION_DENY`) into an actual
/// [`ClaudeDecision`], because a Claude Code element (unlike a Codex one) is
/// meant to be interpreted, not carried through opaque.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaudePendingResponse {
    pub key: ApprovalKey,
    pub decision: ClaudeDecision,
}

struct Entry {
    client: ApprovalClient,
    owner: ApprovalOwner,
    content: PendingApprovalContent,
    protected: bool,
}

#[derive(Default)]
struct Inner {
    entries: HashMap<ApprovalKey, Entry>,
    /// Least-recently-touched key at the front, most-recently-touched at
    /// the back. `insert` moves a key to the back.
    lru: VecDeque<ApprovalKey>,
}

/// In-memory store of unresolved approval-request bodies. Share it via
/// `Arc`; every method takes `&self` and locks internally.
pub struct PendingApprovalStore {
    inner: Mutex<Inner>,
}

impl Default for PendingApprovalStore {
    fn default() -> Self {
        Self::new()
    }
}

impl PendingApprovalStore {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner::default()),
        }
    }

    /// Records a new unresolved request. Replaces any existing entry under
    /// the same key (a retried/duplicate request overwrites; it does not
    /// stack), preserving that entry's `protected` flag.
    pub fn insert(
        &self,
        key: ApprovalKey,
        client: ApprovalClient,
        owner: ApprovalOwner,
        body: PendingApprovalBody,
    ) {
        let content = if body.estimated_size() > MAX_PENDING_APPROVAL_BODY_BYTES {
            PendingApprovalContent::Oversized
        } else {
            PendingApprovalContent::Body(body)
        };
        let mut inner = self.inner.lock().unwrap();
        inner.lru.retain(|existing| existing != &key);
        let protected = inner
            .entries
            .get(&key)
            .map(|existing| existing.protected)
            .unwrap_or(false);
        if !inner.entries.contains_key(&key) {
            evict_for_capacity(&mut inner);
        }
        inner.entries.insert(
            key.clone(),
            Entry {
                client,
                owner,
                content,
                protected,
            },
        );
        inner.lru.push_back(key);
    }

    /// Marks an entry as currently displayed (or no longer displayed),
    /// exempting it from capacity eviction while `protected` is true. No-op
    /// if the key is not present.
    pub fn set_protected(&self, key: &ApprovalKey, protected: bool) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(entry) = inner.entries.get_mut(key) {
            entry.protected = protected;
        }
    }

    /// Discards one resolved request, regardless of its `protected` flag.
    /// No-op if the key is not present.
    pub fn resolve(&self, key: &ApprovalKey) {
        let mut inner = self.inner.lock().unwrap();
        inner.entries.remove(key);
        inner.lru.retain(|existing| existing != key);
    }

    /// Discards every entry belonging to one connection/session (a
    /// disconnect, a `Stop`, or a `SessionEnd`).
    pub fn clear_owner(&self, owner: &ApprovalOwner) {
        self.retain_dropping(|entry| &entry.owner != owner);
    }

    /// Discards every entry for every session of one Claude Code wrapper
    /// launch (`WrapperExited`, which ends all of a launch's sessions at
    /// once).
    pub fn clear_claude_launch(&self, launch_id: &str) {
        self.retain_dropping(|entry| {
            !matches!(
                &entry.owner,
                ApprovalOwner::ClaudeSession { launch_id: candidate, .. }
                    if candidate == launch_id
            )
        });
    }

    /// Discards every entry for one client (Codex Broker stop / lifecycle
    /// error, which ends every connection at once).
    pub fn clear_client(&self, client: ApprovalClient) {
        self.retain_dropping(|entry| entry.client != client);
    }

    fn retain_dropping(&self, keep: impl Fn(&Entry) -> bool) {
        let mut inner = self.inner.lock().unwrap();
        let doomed: Vec<ApprovalKey> = inner
            .entries
            .iter()
            .filter(|(_, entry)| !keep(entry))
            .map(|(key, _)| key.clone())
            .collect();
        for key in doomed {
            inner.entries.remove(&key);
            inner.lru.retain(|existing| existing != &key);
        }
    }

    pub fn get(&self, key: &ApprovalKey) -> Option<PendingApprovalSnapshot> {
        let inner = self.inner.lock().unwrap();
        inner.entries.get(key).map(|entry| PendingApprovalSnapshot {
            client: entry.client,
            content: entry.content.clone(),
            protected: entry.protected,
        })
    }

    /// Looks up a HUD answer by its opaque request token and decision index.
    /// The caller never supplies a reconstructed decision value.
    pub fn codex_response(
        &self,
        request_token: &str,
        decision_index: usize,
    ) -> Option<CodexPendingResponse> {
        let inner = self.inner.lock().unwrap();
        let (key, entry) = inner
            .entries
            .iter()
            .find(|(key, _)| key.token() == request_token)?;
        if entry.client != ApprovalClient::Codex {
            return None;
        }
        let target = key.codex_target()?;
        let PendingApprovalContent::Body(body) = &entry.content else {
            return None;
        };
        let decision = body
            .available_decisions
            .as_ref()?
            .get(decision_index)?
            .clone();
        Some(CodexPendingResponse {
            key: key.clone(),
            connection_id: target.connection_id.clone(),
            request_id: target.request_id.clone(),
            decision,
        })
    }

    /// Looks up a HUD answer for a Claude Code request by its opaque
    /// request token and decision index, the Claude Code counterpart to
    /// [`Self::codex_response`]. `None` for a Codex entry -- callers must
    /// never fall through to treating a Claude Code answer as a Codex one
    /// or vice versa.
    ///
    /// Unlike `codex_response`, the returned decision is not the raw array
    /// element: only `CLAUDE_DECISION_ALLOW`/
    /// `CLAUDE_DECISION_ALLOW_WITH_PERMISSIONS`/`CLAUDE_DECISION_DENY` are
    /// recognized (an unknown value returns `None` rather than inventing a
    /// decision from it -- see `available_decisions`'s own doc comment on
    /// why a Claude Code element is interpreted at all).
    pub fn claude_response(
        &self,
        request_token: &str,
        decision_index: usize,
    ) -> Option<ClaudePendingResponse> {
        let inner = self.inner.lock().unwrap();
        let (key, entry) = inner
            .entries
            .iter()
            .find(|(key, _)| key.token() == request_token)?;
        if entry.client != ApprovalClient::ClaudeCode {
            return None;
        }
        let PendingApprovalContent::Body(body) = &entry.content else {
            return None;
        };
        let raw = body.available_decisions.as_ref()?.get(decision_index)?;
        let decision = match raw.as_str()? {
            CLAUDE_DECISION_ALLOW => ClaudeDecision::Allow,
            // Defensive, not expected in practice: `available_decisions`
            // only ever contains this element when `claude_approval_body`
            // also populated `permission_suggestions` (see that function's
            // doc comment), so this `?` should never actually fire. But a
            // future bug that decouples the two must not fall through to
            // fabricating an `AllowWithPermissions` with an empty/missing
            // `updates` -- that would apply nothing while looking like it
            // applied something, which is worse than just failing the
            // lookup here.
            CLAUDE_DECISION_ALLOW_WITH_PERMISSIONS => ClaudeDecision::AllowWithPermissions {
                updates: body.permission_suggestions.clone()?,
            },
            CLAUDE_DECISION_DENY => ClaudeDecision::Deny {
                message: CLAUDE_HUD_DENY_MESSAGE.to_string(),
            },
            _ => return None,
        };
        Some(ClaudePendingResponse {
            key: key.clone(),
            decision,
        })
    }

    /// Returns the key and snapshot of the most recently *inserted* entry
    /// (the LRU's back), for a HUD showing "the newest unresolved request"
    /// per `docs/ai-approval-hud-design.md` §10: "複数セッションが同時に承認
    /// 待ちのときは、最新の1件を表示する". `get` does not move a key within
    /// the LRU, so a HUD repeatedly reading the same entry does not disturb
    /// this ordering.
    pub fn latest(&self) -> Option<(ApprovalKey, PendingApprovalSnapshot)> {
        let inner = self.inner.lock().unwrap();
        let key = inner.lru.back()?.clone();
        inner.entries.get(&key).map(|entry| {
            (
                key.clone(),
                PendingApprovalSnapshot {
                    client: entry.client,
                    content: entry.content.clone(),
                    protected: entry.protected,
                },
            )
        })
    }

    /// Returns the newest unresolved Codex request for exactly one display
    /// connection and thread. The caller supplies both internal values from
    /// the Codex display registry; neither a ScreenKey nor the UI can
    /// manufacture this relationship. Entries from requests without a
    /// `threadId` are intentionally not selectable here.
    pub fn latest_codex_for_connection_and_thread(
        &self,
        connection_id: &str,
        thread_id: &str,
    ) -> Option<(ApprovalKey, PendingApprovalSnapshot)> {
        let inner = self.inner.lock().unwrap();
        inner.lru.iter().rev().find_map(|key| {
            let entry = inner.entries.get(key)?;
            (entry.client == ApprovalClient::Codex
                && key.codex_target().is_some_and(|target| {
                    target.connection_id == connection_id
                        && target.thread_id.as_deref() == Some(thread_id)
                }))
            .then(|| {
                (
                    key.clone(),
                    PendingApprovalSnapshot {
                        client: entry.client,
                        content: entry.content.clone(),
                        protected: entry.protected,
                    },
                )
            })
        })
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Evicts the least-recently-touched unprotected entry, if the store is at
/// capacity. If every entry happens to be protected, this does nothing --
/// insertion proceeds and the store is allowed past `MAX_ENTRIES` rather
/// than discarding a request currently shown to the user.
fn evict_for_capacity(inner: &mut Inner) {
    if inner.entries.len() < MAX_ENTRIES {
        return;
    }
    let victim = inner
        .lru
        .iter()
        .find(|key| {
            inner
                .entries
                .get(*key)
                .is_some_and(|entry| !entry.protected)
        })
        .cloned();
    let Some(victim) = victim else {
        return;
    };
    inner.entries.remove(&victim);
    inner.lru.retain(|existing| existing != &victim);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(primary: &str) -> PendingApprovalBody {
        PendingApprovalBody {
            primary_text: Some(primary.to_string()),
            full_command: Some(primary.to_string()),
            reason: None,
            cwd: None,
            kind: None,
            available_decisions: None,
            tool_use_id: None,
            prompt_id: None,
            permission_suggestions: None,
        }
    }

    fn codex_owner(id: &str) -> ApprovalOwner {
        ApprovalOwner::Codex {
            connection_id: id.to_string(),
        }
    }

    #[test]
    fn resolve_discards_the_entry() {
        let store = PendingApprovalStore::new();
        let key = ApprovalKey::new("k1");
        store.insert(
            key.clone(),
            ApprovalClient::Codex,
            codex_owner("c1"),
            body("mkdir foo"),
        );
        assert_eq!(store.len(), 1);
        store.resolve(&key);
        assert!(store.is_empty());
        assert!(store.get(&key).is_none());
        // Resolving an absent key is a harmless no-op.
        store.resolve(&key);
    }

    #[test]
    fn clear_owner_discards_only_that_owners_entries() {
        let store = PendingApprovalStore::new();
        let key_a = ApprovalKey::new("a");
        let key_b = ApprovalKey::new("b");
        store.insert(
            key_a.clone(),
            ApprovalClient::Codex,
            codex_owner("conn-1"),
            body("a"),
        );
        store.insert(
            key_b.clone(),
            ApprovalClient::Codex,
            codex_owner("conn-2"),
            body("b"),
        );
        store.clear_owner(&codex_owner("conn-1"));
        assert!(store.get(&key_a).is_none());
        assert!(store.get(&key_b).is_some());
    }

    #[test]
    fn clear_claude_launch_discards_every_session_of_that_launch() {
        let store = PendingApprovalStore::new();
        let key_a = claude_key("launch-1", "session-1");
        let key_b = claude_key("launch-1", "session-2");
        let key_other = claude_key("launch-2", "session-3");
        store.insert(
            key_a.clone(),
            ApprovalClient::ClaudeCode,
            ApprovalOwner::ClaudeSession {
                launch_id: "launch-1".to_string(),
                session_id: "session-1".to_string(),
            },
            body("a"),
        );
        store.insert(
            key_b.clone(),
            ApprovalClient::ClaudeCode,
            ApprovalOwner::ClaudeSession {
                launch_id: "launch-1".to_string(),
                session_id: "session-2".to_string(),
            },
            body("b"),
        );
        store.insert(
            key_other.clone(),
            ApprovalClient::ClaudeCode,
            ApprovalOwner::ClaudeSession {
                launch_id: "launch-2".to_string(),
                session_id: "session-3".to_string(),
            },
            body("c"),
        );
        store.clear_claude_launch("launch-1");
        assert!(store.get(&key_a).is_none());
        assert!(store.get(&key_b).is_none());
        assert!(store.get(&key_other).is_some());
    }

    #[test]
    fn clear_client_discards_only_that_clients_entries() {
        let store = PendingApprovalStore::new();
        let codex = ApprovalKey::new("codex-1");
        let claude = claude_key("launch-1", "session-1");
        store.insert(
            codex.clone(),
            ApprovalClient::Codex,
            codex_owner("conn-1"),
            body("a"),
        );
        store.insert(
            claude.clone(),
            ApprovalClient::ClaudeCode,
            ApprovalOwner::ClaudeSession {
                launch_id: "launch-1".to_string(),
                session_id: "session-1".to_string(),
            },
            body("b"),
        );
        store.clear_client(ApprovalClient::Codex);
        assert!(store.get(&codex).is_none());
        assert!(store.get(&claude).is_some());
    }

    #[test]
    fn oversized_body_is_replaced_by_a_marker() {
        let store = PendingApprovalStore::new();
        let key = ApprovalKey::new("huge");
        let mut oversized = body("mkdir foo");
        oversized.full_command = Some("x".repeat(MAX_PENDING_APPROVAL_BODY_BYTES + 1));
        store.insert(
            key.clone(),
            ApprovalClient::Codex,
            codex_owner("c1"),
            oversized,
        );
        let snapshot = store.get(&key).expect("entry retained");
        assert_eq!(snapshot.content, PendingApprovalContent::Oversized);
    }

    #[test]
    fn lru_eviction_protects_the_displayed_entry() {
        let store = PendingApprovalStore::new();
        // Fill the store to capacity.
        for i in 0..MAX_ENTRIES {
            let key = ApprovalKey::new(format!("k{i}"));
            store.insert(
                key,
                ApprovalClient::Codex,
                codex_owner(&format!("c{i}")),
                body("cmd"),
            );
        }
        assert_eq!(store.len(), MAX_ENTRIES);

        // Protect the oldest entry -- it must survive capacity eviction
        // even though it would otherwise be the LRU victim.
        let oldest = ApprovalKey::new("k0");
        let second_oldest = ApprovalKey::new("k1");
        store.set_protected(&oldest, true);

        // Insert one more entry: this must evict the least-recently-touched
        // *unprotected* entry (k1), not the protected k0.
        let newcomer = ApprovalKey::new("newcomer");
        store.insert(
            newcomer.clone(),
            ApprovalClient::Codex,
            codex_owner("c-new"),
            body("cmd"),
        );

        assert!(store.get(&oldest).is_some(), "protected entry was evicted");
        assert!(
            store.get(&second_oldest).is_none(),
            "unprotected LRU entry should have been evicted instead"
        );
        assert!(store.get(&newcomer).is_some());
        assert_eq!(store.len(), MAX_ENTRIES);
    }

    #[test]
    fn insert_overwrites_rather_than_stacking_and_preserves_protection() {
        let store = PendingApprovalStore::new();
        let key = ApprovalKey::new("k1");
        store.insert(
            key.clone(),
            ApprovalClient::Codex,
            codex_owner("c1"),
            body("first"),
        );
        store.set_protected(&key, true);
        store.insert(
            key.clone(),
            ApprovalClient::Codex,
            codex_owner("c1"),
            body("second"),
        );
        assert_eq!(store.len(), 1);
        let snapshot = store.get(&key).expect("entry present");
        assert!(snapshot.protected, "protection should survive an overwrite");
        assert_eq!(
            snapshot.content,
            PendingApprovalContent::Body(body("second"))
        );
    }

    #[test]
    fn latest_returns_the_most_recently_inserted_unresolved_entry() {
        let store = PendingApprovalStore::new();
        assert!(store.latest().is_none());

        let key_a = ApprovalKey::new("a");
        let key_b = ApprovalKey::new("b");
        store.insert(
            key_a.clone(),
            ApprovalClient::Codex,
            codex_owner("c1"),
            body("a"),
        );
        store.insert(
            key_b.clone(),
            ApprovalClient::Codex,
            codex_owner("c2"),
            body("b"),
        );

        let (latest_key, _) = store.latest().expect("an entry is pending");
        assert_eq!(latest_key, key_b);

        // Reading via `get` must not disturb insertion order.
        store.get(&key_a);
        let (latest_key, _) = store.latest().expect("an entry is pending");
        assert_eq!(latest_key, key_b);

        // Resolving the newest entry falls back to the next-newest.
        store.resolve(&key_b);
        let (latest_key, _) = store.latest().expect("an entry is pending");
        assert_eq!(latest_key, key_a);

        store.resolve(&key_a);
        assert!(store.latest().is_none());
    }

    #[test]
    fn latest_codex_for_connection_and_thread_requires_an_exact_match() {
        let store = PendingApprovalStore::new();
        let first = codex_key_for_thread("connection-a", &Value::from(1), Some("thread-a"));
        let other = codex_key_for_thread("connection-b", &Value::from(2), Some("thread-a"));
        let latest = codex_key_for_thread("connection-a", &Value::from(3), Some("thread-a"));
        for (key, connection) in [
            (first, "connection-a"),
            (other, "connection-b"),
            (latest.clone(), "connection-a"),
        ] {
            store.insert(
                key,
                ApprovalClient::Codex,
                codex_owner(connection),
                body("command"),
            );
        }

        assert_eq!(
            store
                .latest_codex_for_connection_and_thread("connection-a", "thread-a")
                .map(|(key, _)| key),
            Some(latest)
        );
        assert!(store
            .latest_codex_for_connection_and_thread("connection-a", "missing")
            .is_none());
    }

    #[test]
    fn exact_thread_lookup_never_falls_back_to_another_thread_on_connection() {
        let store = PendingApprovalStore::new();
        let thread_a = codex_key_for_thread("connection-a", &Value::from(1), Some("thread-a"));
        let thread_b = codex_key_for_thread("connection-a", &Value::from(2), Some("thread-b"));
        let missing_thread = codex_key("connection-a", &Value::from(3));
        for key in [thread_a.clone(), thread_b.clone(), missing_thread] {
            store.insert(
                key,
                ApprovalClient::Codex,
                codex_owner("connection-a"),
                body("command"),
            );
        }

        assert_eq!(
            store
                .latest_codex_for_connection_and_thread("connection-a", "thread-a")
                .map(|(key, _)| key),
            Some(thread_a)
        );
        assert_eq!(
            store
                .latest_codex_for_connection_and_thread("connection-a", "thread-b")
                .map(|(key, _)| key),
            Some(thread_b)
        );
        assert!(store
            .latest_codex_for_connection_and_thread("connection-a", "thread-c")
            .is_none());
    }

    #[test]
    fn available_decisions_round_trip_mixed_string_and_object_values() {
        let store = PendingApprovalStore::new();
        let key = ApprovalKey::new("k1");
        let decisions = vec![
            Value::String("accept".to_string()),
            serde_json::json!({"acceptWithExecpolicyAmendment": {"execpolicy_amendment": ["mkdir"]}}),
            Value::String("cancel".to_string()),
        ];
        let mut with_decisions = body("mkdir ko2-test");
        with_decisions.available_decisions = Some(decisions.clone());
        store.insert(
            key.clone(),
            ApprovalClient::Codex,
            codex_owner("c1"),
            with_decisions,
        );
        let snapshot = store.get(&key).expect("entry present");
        match snapshot.content {
            PendingApprovalContent::Body(stored) => {
                assert_eq!(stored.available_decisions, Some(decisions));
            }
            PendingApprovalContent::Oversized => panic!("unexpected oversized marker"),
        }
    }

    #[test]
    fn codex_thread_returns_none_without_a_known_thread_id() {
        let with_thread = codex_key_for_thread("connection-a", &Value::from(1), Some("thread-a"));
        assert_eq!(
            with_thread.codex_thread(),
            Some(("connection-a", "thread-a"))
        );

        let without_thread = codex_key("connection-a", &Value::from(2));
        assert_eq!(without_thread.codex_thread(), None);

        let claude = claude_key("launch-1", "session-1");
        assert_eq!(claude.codex_thread(), None);
    }

    #[test]
    fn codex_response_returns_the_exact_offered_decision_and_route() {
        let store = PendingApprovalStore::new();
        let request_id = serde_json::json!(42);
        let key = codex_key("connection-a", &request_id);
        let decisions = vec![
            serde_json::json!("accept"),
            serde_json::json!({
                "acceptWithExecpolicyAmendment": {
                    "execpolicy_amendment": ["mkdir"]
                }
            }),
            serde_json::json!("cancel"),
        ];
        let mut approval = body("mkdir foo");
        approval.available_decisions = Some(decisions.clone());
        store.insert(
            key.clone(),
            ApprovalClient::Codex,
            codex_owner("connection-a"),
            approval,
        );

        let response = store
            .codex_response(key.token(), 1)
            .expect("valid response target");
        assert_eq!(response.key, key);
        assert_eq!(response.connection_id, "connection-a");
        assert_eq!(response.request_id, request_id);
        assert_eq!(response.decision, decisions[1]);
        assert!(store.codex_response(response.key.token(), 3).is_none());
    }

    fn insert_claude(
        store: &PendingApprovalStore,
        launch_id: &str,
        session_id: &str,
    ) -> ApprovalKey {
        let key = claude_key(launch_id, session_id);
        let mut approval = body("mkdir foo");
        approval.available_decisions = Some(vec![
            Value::from(CLAUDE_DECISION_ALLOW),
            Value::from(CLAUDE_DECISION_DENY),
        ]);
        store.insert(
            key.clone(),
            ApprovalClient::ClaudeCode,
            ApprovalOwner::ClaudeSession {
                launch_id: launch_id.to_string(),
                session_id: session_id.to_string(),
            },
            approval,
        );
        key
    }

    #[test]
    fn claude_response_resolves_index_zero_and_one_to_allow_and_deny() {
        let store = PendingApprovalStore::new();
        let key = insert_claude(&store, "launch-1", "session-1");

        let allow = store
            .claude_response(key.token(), 0)
            .expect("index 0 is CLAUDE_DECISION_ALLOW");
        assert_eq!(allow.key, key);
        assert_eq!(allow.decision, ClaudeDecision::Allow);

        let deny = store
            .claude_response(key.token(), 1)
            .expect("index 1 is CLAUDE_DECISION_DENY");
        assert_eq!(
            deny.decision,
            ClaudeDecision::Deny {
                message: CLAUDE_HUD_DENY_MESSAGE.to_string()
            }
        );

        // An out-of-range index has no interpretation.
        assert!(store.claude_response(key.token(), 2).is_none());
    }

    #[test]
    fn claude_response_returns_none_for_a_codex_entry() {
        let store = PendingApprovalStore::new();
        let key = codex_key("connection-a", &Value::from(1));
        let mut approval = body("mkdir foo");
        approval.available_decisions = Some(vec![Value::from("accept")]);
        store.insert(
            key.clone(),
            ApprovalClient::Codex,
            codex_owner("connection-a"),
            approval,
        );

        assert!(store.claude_response(key.token(), 0).is_none());
    }

    /// Regression for the boundary the design calls out explicitly: a
    /// Claude Code entry's opaque-looking `available_decisions` must never
    /// be handed to Codex's response path, even though both entries share
    /// the same store and lookup-by-token mechanism.
    #[test]
    fn codex_response_returns_none_for_a_claude_entry() {
        let store = PendingApprovalStore::new();
        let key = insert_claude(&store, "launch-1", "session-1");

        assert!(store.codex_response(key.token(), 0).is_none());
    }

    /// Builds a Claude Code entry with the three-element
    /// `[allow, allow_with_permissions, deny]` array, the shape
    /// `claude_approval_body` only produces when the real hook body also
    /// carried a non-empty `permission_suggestions` array.
    fn insert_claude_with_permission_suggestions(
        store: &PendingApprovalStore,
        launch_id: &str,
        session_id: &str,
        suggestions: Vec<Value>,
    ) -> ApprovalKey {
        let key = claude_key(launch_id, session_id);
        let mut approval = body("mkdir foo");
        approval.available_decisions = Some(vec![
            Value::from(CLAUDE_DECISION_ALLOW),
            Value::from(CLAUDE_DECISION_ALLOW_WITH_PERMISSIONS),
            Value::from(CLAUDE_DECISION_DENY),
        ]);
        approval.permission_suggestions = Some(suggestions);
        store.insert(
            key.clone(),
            ApprovalClient::ClaudeCode,
            ApprovalOwner::ClaudeSession {
                launch_id: launch_id.to_string(),
                session_id: session_id.to_string(),
            },
            approval,
        );
        key
    }

    /// Pins the exact contract `commands.rs`'s `respond_to_claude_approval`
    /// depends on: picking the middle index of a three-element array must
    /// resolve to `AllowWithPermissions`, and the `updates` it carries must
    /// be identical -- same elements, same order -- to whatever
    /// `permission_suggestions` the hook body actually sent. Nothing here is
    /// reconstructed; `claude_response` only ever clones what was stored.
    #[test]
    fn claude_response_resolves_the_middle_index_to_allow_with_permissions_with_untouched_updates()
    {
        let store = PendingApprovalStore::new();
        let suggestions = vec![serde_json::json!({
            "type": "addRules",
            "behavior": "allow",
            "destination": "localSettings",
            "rules": [{"toolName": "PowerShell", "ruleContent": "New-Item -ItemType Directory ko3-test8"}]
        })];
        let key = insert_claude_with_permission_suggestions(
            &store,
            "launch-1",
            "session-1",
            suggestions.clone(),
        );

        let response = store
            .claude_response(key.token(), 1)
            .expect("index 1 is CLAUDE_DECISION_ALLOW_WITH_PERMISSIONS");
        assert_eq!(response.key, key);
        assert_eq!(
            response.decision,
            ClaudeDecision::AllowWithPermissions {
                updates: suggestions
            }
        );

        // index 0 and 2 still resolve to plain allow/deny in this shape.
        assert_eq!(
            store.claude_response(key.token(), 0).unwrap().decision,
            ClaudeDecision::Allow
        );
        assert_eq!(
            store.claude_response(key.token(), 2).unwrap().decision,
            ClaudeDecision::Deny {
                message: CLAUDE_HUD_DENY_MESSAGE.to_string()
            }
        );
    }

    /// Defensive regression for the fallback documented on
    /// `claude_response`'s `CLAUDE_DECISION_ALLOW_WITH_PERMISSIONS` arm: if
    /// an entry's `available_decisions` somehow offers this element without
    /// a matching `permission_suggestions` body (should not happen in
    /// practice -- see `claude_approval_body`), the lookup must fail closed
    /// rather than fabricate a decision with empty `updates`.
    #[test]
    fn claude_response_returns_none_for_allow_with_permissions_without_a_suggestions_body() {
        let store = PendingApprovalStore::new();
        let key = claude_key("launch-1", "session-1");
        let mut approval = body("mkdir foo");
        approval.available_decisions = Some(vec![
            Value::from(CLAUDE_DECISION_ALLOW),
            Value::from(CLAUDE_DECISION_ALLOW_WITH_PERMISSIONS),
            Value::from(CLAUDE_DECISION_DENY),
        ]);
        // permission_suggestions deliberately left `None`.
        store.insert(
            key.clone(),
            ApprovalClient::ClaudeCode,
            ApprovalOwner::ClaudeSession {
                launch_id: "launch-1".to_string(),
                session_id: "session-1".to_string(),
            },
            approval,
        );

        assert!(store.claude_response(key.token(), 1).is_none());
    }
}
