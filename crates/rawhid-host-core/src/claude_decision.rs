//! Stage 3 of `docs/ai-approval-hud-design.md` (§9.2): lets Studio answer a
//! Claude Code `PermissionRequest` hook with an actual decision instead of
//! the bare 204 `claude_observer.rs` returns for every other hook.
//!
//! Claude Code has no approval API of its own -- the only way in is the
//! `PermissionRequest` hook's HTTP response body
//! (`docs/claude-permission-hook-gate-results.md` §Q3/§Q4). That response
//! is produced far from where the hook connection is held open
//! (`claude_observer.rs`'s `handle_connection`, running on the Tokio
//! receiver task) -- the actual decision comes from a physical HUD action
//! dispatched on its own thread (`rawhid-host-tauri`'s
//! `actions::dispatch_hud_response`), or from `ClaudeApprovalBodyConsumer`
//! canceling the wait because the terminal answered first (§9.4's
//! first-wins rule). [`ClaudePermissionGate`] is the handoff between the
//! two: one side registers a token and awaits a
//! [`tokio::sync::oneshot::Receiver`], the other looks the same token up
//! and either sends a decision into it or drops it.
//!
//! This module owns the wire *shape* of a decision
//! ([`ClaudeDecision::hook_response_body`]) and the gate itself. It does
//! not know the token format (`pending_approval.rs`'s [`claude_key`] and
//! `claude_launch_token_prefix` own that) and it never touches
//! `PendingApprovalStore` directly -- `claude_activity.rs`'s
//! `ClaudeApprovalBodyConsumer` and `pending_approval.rs`'s
//! `PendingApprovalStore::claude_response` are the only callers that bridge
//! the two.

use std::{collections::HashMap, sync::Mutex, time::Duration};

use serde_json::{json, Value};
use tokio::sync::oneshot;

use crate::pending_approval::claude_launch_token_prefix;

/// How long the Host waits on [`ClaudePermissionGate::register`]'s receiver
/// before giving up and letting the hook connection fall back to 204.
///
/// This is deliberately shorter than
/// `claude_hooks::CLAUDE_PERMISSION_HOOK_TIMEOUT_SECONDS` (600s) -- kept
/// that way on purpose, see this module's tests -- and was raised from an
/// earlier 55s alongside that value going from 60s to 600s. Extending it is
/// safe for the same reason extending the hook's own `timeout` is safe
/// (`docs/claude-permission-hook-gate-results.md` §Q6, confirmed by an
/// actual round trip): Claude Code does not wait for the hook at all -- it
/// shows its own terminal prompt after about three seconds regardless -- so
/// a longer wait here never makes the user wait longer. It only extends how
/// long the *keyboard* answer path stays reachable, which is the entire
/// point: at 55s, a person had to notice and press the HUD within about a
/// minute of the request appearing, which real usage showed was routinely
/// missed.
///
/// What is *not* known is how large a `timeout` Claude Code's own hook
/// config will actually honor -- that upper bound has never been measured.
/// If Claude Code enforces some smaller cap of its own and closes the hook
/// connection early, this Host-side wait simply keeps running past that
/// point for nothing: the hook is already gone, so nothing this gate does
/// can reach Claude Code for that request any more. That failure mode is
/// exactly what `ClaudePermissionGate::note_unanswerable` /
/// `drain_unanswerable` exist to contain -- once this wait's own timeout
/// (595s) elapses with nobody having answered, the token is recorded and
/// the Host drops the matching `PendingApprovalStore` entry, so the HUD
/// stops offering a request it can no longer deliver an answer for, even
/// though Claude Code stopped listening earlier. The real cap, if any, can
/// be found by leaving a request unanswered and then answering it from the
/// HUD anyway, reading the resulting `answered=` diagnostic.
pub const CLAUDE_PERMISSION_DECISION_TIMEOUT: Duration = Duration::from_secs(595);

/// One decision for a Claude Code `PermissionRequest` hook, already
/// resolved to a `behavior` (never the raw opaque string from
/// `PendingApprovalBody::available_decisions` -- see
/// `PendingApprovalStore::claude_response`, the only place that turns an
/// index into one of these).
///
/// `Deny` carries a `message`: the hook response's free-text `message`
/// field reaches the model itself, confirmed by an actual round trip
/// (`docs/claude-permission-hook-gate-results.md` §Q4) -- this is not
/// speculative. But the keyboard ❌ press itself carries no reason beyond
/// "the user pressed the reject key", so [`CLAUDE_HUD_DENY_MESSAGE`] states
/// only *who* denied it (the HUD, not the model or the terminal) and
/// deliberately gives no instruction ("don't retry", "ask the user first",
/// etc.) -- injecting the same directive into every denial regardless of
/// context would take a judgment call away from the model and the user that
/// a bare statement of fact does not.
///
/// `AllowWithPermissions` is the "allow, and remember this choice" decision
/// (`pending_approval.rs`'s `CLAUDE_DECISION_ALLOW_WITH_PERMISSIONS`). Its
/// `updates` field is always an exact clone of the hook body's own
/// `permission_suggestions` array (`PendingApprovalStore::claude_response`
/// is the only place that fills it in, from `PendingApprovalBody`'s field of
/// the same nature) -- this type never builds that array itself, because its
/// shape and meaning vary per request in ways only Claude Code's terminal
/// itself understands (see `permission_suggestions`'s own doc comment for
/// the three real shapes observed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClaudeDecision {
    Allow,
    AllowWithPermissions { updates: Vec<Value> },
    Deny { message: String },
}

/// The fixed `message` attached to every HUD-issued `Deny`. Deliberately a
/// single short factual sentence and nothing else -- see [`ClaudeDecision`]'s
/// own doc comment for why it carries no action instruction. Shortened from
/// an earlier, longer wording that also named the HUD as the source: real
/// Claude Code terminals already print their own
/// `Denied by PermissionRequest hook` line for every hook-driven denial, so
/// restating where the denial came from a second time added nothing the
/// user didn't already see (confirmed against a real terminal on
/// 2026-09-06).
pub const CLAUDE_HUD_DENY_MESSAGE: &str = "ユーザがこの操作を拒否しました。";

impl ClaudeDecision {
    /// The wire `behavior` string for this decision. Shared by
    /// `hook_response_body` and (indirectly, via `diagnostic_label`'s
    /// distinct spelling) `commands.rs`'s decision-dispatch diagnostic log.
    /// `AllowWithPermissions` deliberately returns the same string as
    /// `Allow`: on the wire, both are `"behavior": "allow"` --
    /// `updatedPermissions` is what actually distinguishes them
    /// (`hook_response_body` below). Use [`Self::diagnostic_label`] instead
    /// of this method anywhere the two need to be told apart, such as a log
    /// line.
    pub fn behavior(&self) -> &'static str {
        match self {
            ClaudeDecision::Allow | ClaudeDecision::AllowWithPermissions { .. } => "allow",
            ClaudeDecision::Deny { .. } => "deny",
        }
    }

    /// A diagnostics-only label that, unlike [`Self::behavior`], keeps
    /// `AllowWithPermissions` distinguishable from a plain `Allow`. Exists
    /// solely so `commands.rs`'s decision-dispatch log can show which one
    /// actually fired without printing `updates` itself -- that field must
    /// never reach a log line (it can carry a full command string via
    /// `ruleContent`, which is exactly the content this log already
    /// deliberately excludes for every other decision).
    pub fn diagnostic_label(&self) -> &'static str {
        match self {
            ClaudeDecision::Allow => "allow",
            ClaudeDecision::AllowWithPermissions { .. } => "allow_with_permissions",
            ClaudeDecision::Deny { .. } => "deny",
        }
    }

    /// Builds the exact JSON body `claude_observer.rs` writes back to the
    /// hook connection.
    ///
    /// The `Allow`/`Deny` shapes were confirmed against a real Claude Code
    /// instance (`docs/claude-permission-hook-gate-results.md` §Q3/§Q4) --
    /// do not rename or nest those fields differently. `AllowWithPermissions`
    /// adds `updatedPermissions`, whose key name and placement (a sibling of
    /// `behavior` inside `decision`, not nested under it) were separately
    /// confirmed by an actual round trip against a real Claude Code instance
    /// on 2026-09-06: a response missing this exact key applies nothing --
    /// the terminal shows no error, the press simply has no effect -- so
    /// this is not a "reasonable-looking addition," it is the one verified
    /// shape. `updates` is passed through unmodified; see `ClaudeDecision`'s
    /// own doc comment for why this type never builds that array itself.
    pub fn hook_response_body(&self) -> Value {
        let decision = match self {
            ClaudeDecision::Allow => json!({ "behavior": self.behavior() }),
            ClaudeDecision::AllowWithPermissions { updates } => {
                json!({ "behavior": self.behavior(), "updatedPermissions": updates })
            }
            ClaudeDecision::Deny { message } => {
                json!({ "behavior": self.behavior(), "message": message })
            }
        };
        json!({
            "hookSpecificOutput": {
                "hookEventName": "PermissionRequest",
                "decision": decision
            }
        })
    }
}

/// Connects a held-open `PermissionRequest` hook connection
/// (`claude_observer.rs`) to whichever Host-side path answers it first: a
/// physical HUD action, or the terminal resolving the same request on its
/// own (`claude_activity.rs`'s `ClaudeApprovalBodyConsumer`, which cancels
/// the matching waiter -- see its own doc comment for why).
///
/// Keyed by the same opaque token `PendingApprovalStore` uses
/// (`pending_approval.rs`'s `claude_key(...).token()`), so a HUD selection
/// and the hook connection it answers always agree on which request they
/// mean without either side knowing the other's internals.
#[derive(Default)]
pub struct ClaudePermissionGate {
    waiters: Mutex<HashMap<String, oneshot::Sender<ClaudeDecision>>>,
    /// Entries whose waiter ended with nobody ever answering -- either the
    /// decision wait itself timed out, or the hook connection's read side
    /// closed while the waiter was still live. See [`UnansweredReason`] for
    /// the two cases; `claude_observer.rs`'s `handle_permission_request` is
    /// the only caller of [`Self::note_unanswerable`], recording one of these
    /// exactly when its own wait on [`Self::register`]'s receiver ends that
    /// way (see that call site's comments on why this must never happen for
    /// the other losing branch, a dropped sender).
    ///
    /// A token landing here means the Host can no longer deliver a decision
    /// for it at all -- the hook connection it belonged to already closed
    /// itself (either just now, or once its 204 fallback was written). The
    /// HUD must not keep offering a request as "answerable" once that has
    /// happened, or a press just disappears with no effect (the real-machine
    /// symptom this exists to fix). Host code (`rawhid-host-tauri`'s
    /// `commands.rs`) drains this list every tick and uses it to drop the
    /// matching entry from `PendingApprovalStore`, and (via
    /// `claude_activity::ClaudeSessionRegistry::withdraw_approval_requests`)
    /// to withdraw the matching session's own unresolved approval request --
    /// see that method's doc comment for why the session side must also give
    /// up, not just the HUD offer.
    unanswerable: Mutex<Vec<UnansweredApproval>>,
}

/// Why a [`ClaudePermissionGate`] waiter ended up in
/// [`ClaudePermissionGate::drain_unanswerable`] without ever receiving a
/// decision. `commands.rs` logs this (via [`Self::diagnostic_label`]) but
/// applies the exact same cleanup regardless of which one it was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnansweredReason {
    /// Nobody -- neither a HUD action nor the terminal -- answered before
    /// [`CLAUDE_PERMISSION_DECISION_TIMEOUT`] elapsed. The Host still cannot
    /// tell, at this point, whether the request is genuinely abandoned or was
    /// answered somewhere Studio can no longer observe -- see
    /// `claude_activity::ClaudeStateChangeReason::ApprovalWithdrawn`'s doc
    /// comment for the tradeoff withdrawing the HUD's offer here accepts.
    DecisionTimeout,
    /// The hook connection's read side observed an EOF or error while its
    /// waiter was still live and unanswered. Confirmed against a real Claude
    /// Code instance on 2026-09-06, comparing two otherwise identical
    /// `PermissionRequest`s: one rejected from the terminal at 14:59:22 had
    /// its connection close 10.7s later; one left completely untouched,
    /// received at 15:02:51, still had its connection open three and a half
    /// minutes later with nothing recorded. A close is therefore treated as
    /// "this request was settled somewhere other than Studio" -- almost
    /// always the terminal, and almost always within a second or two of the
    /// person actually answering it there -- not a fluke of some intermediary
    /// closing and reopening the socket.
    ConnectionClosed,
}

impl UnansweredReason {
    /// Diagnostic label for `commands.rs`'s `reason=` log field. Kept as its
    /// own method (rather than relying on `Debug`) so that field's wire
    /// string stays stable even if the variant names above change for
    /// unrelated reasons.
    pub fn diagnostic_label(&self) -> &'static str {
        match self {
            UnansweredReason::DecisionTimeout => "decision_timeout",
            UnansweredReason::ConnectionClosed => "connection_closed",
        }
    }
}

/// One token [`ClaudePermissionGate::note_unanswerable`] recorded, together
/// with the `(launch_id, session_id)` pair the hook connection belonged to
/// and why it ended up here. `commands.rs`'s `drain_claude_state_changes`
/// needs the token to withdraw the matching `PendingApprovalStore` entry, the
/// launch/session pair to withdraw the matching session's own approval
/// request via `claude_activity::ClaudeSessionRegistry::withdraw_approval_requests`,
/// and the reason purely for its diagnostic log line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnansweredApproval {
    pub token: String,
    pub launch_id: String,
    pub session_id: String,
    pub reason: UnansweredReason,
}

impl std::fmt::Debug for ClaudePermissionGate {
    // A manual impl rather than `#[derive(Debug)]`: this never prints a
    // token (opaque, but still request-identifying) or leans on
    // `oneshot::Sender`'s own `Debug` impl, keeping the "never log hook
    // body/token" rule (docs/ai-approval-hud-design.md's implementation
    // notes) true even for incidental `{:?}` logging elsewhere.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ClaudePermissionGate")
            .field("waiting_count", &self.waiting_count())
            .finish()
    }
}

impl ClaudePermissionGate {
    /// Registers a new waiter for `token`, returning the receiver half the
    /// caller awaits. If a waiter already existed for this token (a retried
    /// or duplicate `PermissionRequest` for the same session -- one session
    /// holds at most one unresolved request, per `claude_key`'s own doc
    /// comment), it is replaced: the old `Sender` is dropped, which fails
    /// its still-open `Receiver` and lets that earlier hook connection
    /// degrade to 204. This mirrors `PendingApprovalStore::insert`'s own
    /// overwrite-rather-than-stack semantics for the same reason.
    pub fn register(&self, token: String) -> oneshot::Receiver<ClaudeDecision> {
        let (sender, receiver) = oneshot::channel();
        self.waiters.lock().unwrap().insert(token, sender);
        receiver
    }

    /// Delivers `decision` to the waiter registered for `token`, if any.
    /// Synchronous and non-blocking (a bare `Mutex` lock plus a channel
    /// send) so it can be called directly from the Tauri monitor thread
    /// that dispatches a physical HUD response, without an `async` runtime
    /// in the loop. Returns `true` only when a waiter was found and the
    /// send succeeded; a second call for the same token -- after either an
    /// answer or a `cancel` -- always returns `false`, since the waiter was
    /// removed on the first call.
    pub fn answer(&self, token: &str, decision: ClaudeDecision) -> bool {
        let sender = self.waiters.lock().unwrap().remove(token);
        match sender {
            Some(sender) => sender.send(decision).is_ok(),
            None => false,
        }
    }

    /// Discards the waiter for `token` without answering it, letting its
    /// hook connection fall back to 204 once it notices the sender was
    /// dropped. No-op if there is no such waiter.
    pub fn cancel(&self, token: &str) {
        self.waiters.lock().unwrap().remove(token);
    }

    /// Discards every waiter belonging to one Claude Code wrapper launch
    /// (`WrapperExited`, which ends every session of that launch at once).
    /// Uses `claude_launch_token_prefix` rather than reconstructing the
    /// token format itself -- see `pending_approval.rs`'s doc comment on
    /// that function for why only it owns the format.
    pub fn cancel_launch(&self, launch_id: &str) {
        let prefix = claude_launch_token_prefix(launch_id);
        self.waiters
            .lock()
            .unwrap()
            .retain(|token, _| !token.starts_with(&prefix));
    }

    /// Number of connections currently held open awaiting a decision.
    /// Diagnostics/tests only.
    pub fn waiting_count(&self) -> usize {
        self.waiters.lock().unwrap().len()
    }

    /// Records that `token`'s waiter ended with nobody answering, for
    /// `reason`, so a later [`Self::drain_unanswerable`] call can pick it up.
    /// See the `unanswerable` field's own doc comment for why this exists and
    /// who is expected to call it (only `claude_observer.rs`'s
    /// decision-timeout arm and its read-probe arm's live-waiter case --
    /// never the dropped-sender branch). `launch_id`/`session_id` are the
    /// same pair `token` was built from (`pending_approval.rs`'s
    /// `claude_key`); they are carried alongside the token purely so the
    /// caller can also withdraw the session's own approval request without
    /// having to parse either back out of the opaque token string.
    pub fn note_unanswerable(
        &self,
        token: &str,
        launch_id: &str,
        session_id: &str,
        reason: UnansweredReason,
    ) {
        self.unanswerable.lock().unwrap().push(UnansweredApproval {
            token: token.to_string(),
            launch_id: launch_id.to_string(),
            session_id: session_id.to_string(),
            reason,
        });
    }

    /// Takes every entry recorded by [`Self::note_unanswerable`] since the
    /// last call, leaving the list empty. Host code calls this once per
    /// tick to reconcile `PendingApprovalStore` (and the matching session's
    /// own approval request) with requests the gate can no longer deliver a
    /// decision for.
    pub fn drain_unanswerable(&self) -> Vec<UnansweredApproval> {
        std::mem::take(&mut self.unanswerable.lock().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Runtime::new().unwrap()
    }

    #[test]
    fn answer_delivers_the_decision_to_the_registered_waiter() {
        runtime().block_on(async {
            let gate = ClaudePermissionGate::default();
            let receiver = gate.register("claude:launch-1:session-1".to_string());
            assert!(gate.answer("claude:launch-1:session-1", ClaudeDecision::Allow));
            assert_eq!(receiver.await, Ok(ClaudeDecision::Allow));
        });
    }

    #[test]
    fn answer_without_a_waiter_returns_false() {
        let gate = ClaudePermissionGate::default();
        assert!(!gate.answer("no-such-token", ClaudeDecision::Allow));
    }

    #[test]
    fn a_second_answer_for_the_same_token_returns_false() {
        let gate = ClaudePermissionGate::default();
        let _receiver = gate.register("token-a".to_string());
        assert!(gate.answer("token-a", ClaudeDecision::Allow));
        assert!(!gate.answer(
            "token-a",
            ClaudeDecision::Deny {
                message: CLAUDE_HUD_DENY_MESSAGE.to_string()
            }
        ));
    }

    #[test]
    fn behavior_and_diagnostic_label_agree_on_the_wire_string_but_diverge_on_the_label() {
        // `behavior()` collapses Allow/AllowWithPermissions to the same wire
        // string on purpose -- that is what the real hook protocol expects
        // (both mean "behavior": "allow"). `diagnostic_label()` exists
        // precisely so a log line does not lose that distinction.
        assert_eq!(ClaudeDecision::Allow.behavior(), "allow");
        assert_eq!(
            ClaudeDecision::AllowWithPermissions { updates: vec![] }.behavior(),
            "allow"
        );
        assert_eq!(
            ClaudeDecision::Deny {
                message: CLAUDE_HUD_DENY_MESSAGE.to_string()
            }
            .behavior(),
            "deny"
        );

        assert_eq!(ClaudeDecision::Allow.diagnostic_label(), "allow");
        assert_eq!(
            ClaudeDecision::AllowWithPermissions { updates: vec![] }.diagnostic_label(),
            "allow_with_permissions"
        );
        assert_eq!(
            ClaudeDecision::Deny {
                message: CLAUDE_HUD_DENY_MESSAGE.to_string()
            }
            .diagnostic_label(),
            "deny"
        );
    }

    /// Each of the three real `permission_suggestions` shapes captured
    /// against an actual Claude Code instance on 2026-09-06 (see this
    /// module's doc comment on `ClaudeDecision::hook_response_body` and
    /// `pending_approval.rs`'s doc comment on `permission_suggestions`),
    /// round-tripped through `hook_response_body` untouched: same elements,
    /// same order, same nesting, under the confirmed `updatedPermissions`
    /// key. This is the one place all three are exercised together as
    /// `updates` so a future refactor cannot special-case just one shape.
    #[test]
    fn allow_with_permissions_carries_every_observed_shape_of_updates_untouched() {
        let session_scoped_read = vec![json!({
            "type": "addRules",
            "behavior": "allow",
            "destination": "session",
            "rules": [{"toolName": "Read", "ruleContent": "//c/temp/**"}]
        })];
        let local_settings_command = vec![json!({
            "type": "addRules",
            "behavior": "allow",
            "destination": "localSettings",
            "rules": [{"toolName": "PowerShell", "ruleContent": "New-Item -ItemType Directory ko3-test8"}]
        })];
        let mode_and_directory_pair = vec![
            json!({"type": "setMode", "mode": "acceptEdits", "destination": "session"}),
            json!({"type": "addDirectories", "directories": ["C:\\temp"], "destination": "session"}),
        ];

        for updates in [
            session_scoped_read,
            local_settings_command,
            mode_and_directory_pair,
        ] {
            let body = ClaudeDecision::AllowWithPermissions {
                updates: updates.clone(),
            }
            .hook_response_body();
            assert_eq!(
                body,
                json!({
                    "hookSpecificOutput": {
                        "hookEventName": "PermissionRequest",
                        "decision": {
                            "behavior": "allow",
                            "updatedPermissions": updates
                        }
                    }
                }),
                "updates must reach the wire byte-for-byte: {updates:?}"
            );
        }
    }

    #[test]
    fn deny_hook_response_body_carries_the_fixed_reason() {
        // Pins the wire shape: `message` is always exactly
        // `CLAUDE_HUD_DENY_MESSAGE`, never something assembled per-request
        // (there is nothing per-request to assemble from -- see
        // `ClaudeDecision`'s own doc comment).
        let body = ClaudeDecision::Deny {
            message: CLAUDE_HUD_DENY_MESSAGE.to_string(),
        }
        .hook_response_body();
        assert_eq!(
            body,
            json!({
                "hookSpecificOutput": {
                    "hookEventName": "PermissionRequest",
                    "decision": {
                        "behavior": "deny",
                        "message": CLAUDE_HUD_DENY_MESSAGE
                    }
                }
            })
        );
    }

    #[test]
    fn cancel_makes_answer_return_false_and_drops_the_receiver() {
        runtime().block_on(async {
            let gate = ClaudePermissionGate::default();
            let receiver = gate.register("token-a".to_string());
            gate.cancel("token-a");
            assert!(!gate.answer("token-a", ClaudeDecision::Allow));
            assert!(receiver.await.is_err(), "sender was dropped by cancel");
        });
    }

    #[test]
    fn cancel_launch_drops_every_waiter_of_that_launch_only() {
        let gate = ClaudePermissionGate::default();
        let _a = gate.register(claude_key_token("launch-1", "session-1"));
        let _b = gate.register(claude_key_token("launch-1", "session-2"));
        let _other = gate.register(claude_key_token("launch-2", "session-3"));
        assert_eq!(gate.waiting_count(), 3);

        gate.cancel_launch("launch-1");

        assert_eq!(gate.waiting_count(), 1);
        assert!(!gate.answer(
            &claude_key_token("launch-1", "session-1"),
            ClaudeDecision::Allow
        ));
        assert!(!gate.answer(
            &claude_key_token("launch-1", "session-2"),
            ClaudeDecision::Allow
        ));
        assert!(gate.answer(
            &claude_key_token("launch-2", "session-3"),
            ClaudeDecision::Allow
        ));
    }

    #[test]
    fn registering_the_same_token_again_drops_the_earlier_waiter() {
        runtime().block_on(async {
            let gate = ClaudePermissionGate::default();
            let first = gate.register("token-a".to_string());
            let second = gate.register("token-a".to_string());

            assert!(gate.answer("token-a", ClaudeDecision::Allow));
            assert!(
                first.await.is_err(),
                "the first receiver was replaced, so its sender was dropped"
            );
            assert_eq!(second.await, Ok(ClaudeDecision::Allow));
        });
    }

    fn claude_key_token(launch_id: &str, session_id: &str) -> String {
        crate::pending_approval::claude_key(launch_id, session_id)
            .token()
            .to_string()
    }

    /// Pins the relationship documented on both constants: the Host's own
    /// wait must always give up strictly before the hook's `timeout` would,
    /// so it is the Host -- not Claude Code's HTTP client -- that decides
    /// when a `PermissionRequest` connection falls back to 204. Changing
    /// either constant without keeping this true would silently invert
    /// which side controls that fallback.
    #[test]
    fn decision_timeout_stays_shorter_than_the_hook_timeout() {
        assert!(
            CLAUDE_PERMISSION_DECISION_TIMEOUT
                < Duration::from_secs(crate::claude_hooks::CLAUDE_PERMISSION_HOOK_TIMEOUT_SECONDS),
            "the Host's decision wait must stay shorter than the hook's own timeout"
        );
    }

    #[test]
    fn drain_unanswerable_returns_recorded_entries_and_then_empties() {
        let gate = ClaudePermissionGate::default();
        gate.note_unanswerable(
            "token-a",
            "launch-1",
            "session-1",
            UnansweredReason::DecisionTimeout,
        );
        gate.note_unanswerable(
            "token-b",
            "launch-2",
            "session-2",
            UnansweredReason::ConnectionClosed,
        );

        assert_eq!(
            gate.drain_unanswerable(),
            vec![
                UnansweredApproval {
                    token: "token-a".to_string(),
                    launch_id: "launch-1".to_string(),
                    session_id: "session-1".to_string(),
                    reason: UnansweredReason::DecisionTimeout,
                },
                UnansweredApproval {
                    token: "token-b".to_string(),
                    launch_id: "launch-2".to_string(),
                    session_id: "session-2".to_string(),
                    reason: UnansweredReason::ConnectionClosed,
                },
            ]
        );
        assert!(
            gate.drain_unanswerable().is_empty(),
            "a second drain must find nothing left"
        );
    }

    #[test]
    fn unanswered_reason_diagnostic_label_matches_the_expected_wire_string() {
        assert_eq!(
            UnansweredReason::DecisionTimeout.diagnostic_label(),
            "decision_timeout"
        );
        assert_eq!(
            UnansweredReason::ConnectionClosed.diagnostic_label(),
            "connection_closed"
        );
    }
}
