//! Execution of HOST_ACTION packets. The keyboard only sends an opaque
//! action id; everything an action *does* is defined by the local config
//! allowlist. The HID value byte is never interpreted as a path or command.

use std::{
    sync::{atomic::Ordering, Arc, Mutex},
    time::Instant,
};

use rawhid_host_core::codex_activity::CodexSessionSnapshot;
use rawhid_host_core::config::{ActionBinding, HostActionKind};
use tauri::{AppHandle, Manager};

use crate::state::{AiDisplayTarget, MonitorStatus};
use crate::{
    commands::{respond_to_codex_approval_internal, spawn_ai_refresh_watcher, MonitorExtras},
    hud_coordinator::HudSelectionDirection,
};

pub enum ActionOutcome {
    Continue,
    /// A valid physical HUD action had no safe live target. It is deliberately
    /// non-fatal: stale packets, Claude slots, and double presses must never
    /// turn into a synthetic approval.
    HudNoop {
        reason: &'static str,
    },
    AiSessionSelected {
        label: String,
    },
    /// Automatic monitoring should stop while the Host Link worker remains
    /// available for discovery and keymap Config RPC.
    StopRequested,
    /// A HUD Confirm actually dispatched an answer, and the ScreenKey slot it
    /// answered for was resolved. `commands.rs`'s monitor loop uses this to
    /// hold that slot's ScreenKey state at `AiScreenKeyState::Responded` for
    /// a short window (see its `HUD_RESPONDED_HOLD` hold logic) and to send
    /// that slot's packet immediately rather than waiting for the next tick.
    HudResponded {
        slot: u8,
    },
    /// `HostActionKind::SelectHudTarget` actually moved the HUD's explicit
    /// target to this ScreenKey slot (it did not fall back to
    /// `focus_ai_terminal_for_slot`). `commands.rs`'s `handle_uplink_events`
    /// uses this to immediately resend this slot's (and, if there was one,
    /// the previously-targeted slot's) ScreenKey state, rather than waiting
    /// for the next tick -- a target switch alone never changes
    /// `activity_state`, so `sync_ai_client_state_slot` would otherwise not
    /// force a send until `screenkey_state_changed` catches it on the next
    /// tick.
    HudTargetSelected {
        slot: u8,
    },
}

pub fn execute(
    app: &AppHandle,
    binding: &ActionBinding,
    value: u8,
    extras: &MonitorExtras,
    status: &Arc<Mutex<MonitorStatus>>,
) -> Result<ActionOutcome, String> {
    match binding.action {
        HostActionKind::ShowWindow => {
            if let Some(window) = app.get_webview_window("main") {
                // unminimize() first: show()/set_focus() do not restore a window
                // that is minimized to the taskbar.
                let _ = window.unminimize();
                let _ = window.show();
                let _ = window.set_focus();
            }
            Ok(ActionOutcome::Continue)
        }
        // Triggered via HID, so the monitor loop is already running.
        HostActionKind::StartMonitoring => Ok(ActionOutcome::Continue),
        HostActionKind::StopMonitoring => Ok(ActionOutcome::StopRequested),
        HostActionKind::RefreshAiUsage => {
            if extras
                .ai_usage_refreshing
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                .is_err()
            {
                return Err("refresh_in_progress".to_string());
            }
            let baseline = {
                let runtime = extras.ai_usage_runtime.lock().unwrap();
                let Some(runtime) = runtime.as_ref() else {
                    extras.ai_usage_refreshing.store(false, Ordering::SeqCst);
                    return Err("source_disabled".to_string());
                };
                let generation = runtime.shared().generation();
                if runtime.refresh().is_err() {
                    extras.ai_usage_refreshing.store(false, Ordering::SeqCst);
                    return Err("refresh_failed".to_string());
                }
                generation
            };
            spawn_ai_refresh_watcher(
                app.clone(),
                Arc::clone(&extras.config),
                Arc::clone(&extras.ai_usage_runtime),
                Arc::clone(status),
                Arc::clone(&extras.ai_usage_refreshing),
                baseline,
            );
            Ok(ActionOutcome::Continue)
        }
        HostActionKind::CycleAiSession => {
            let selected = extras
                .ai_display_slots
                .lock()
                .unwrap()
                .cycle_slot(value)
                .ok_or_else(|| "no_active_ai_sessions".to_string())?;
            Ok(ActionOutcome::AiSessionSelected {
                label: selected.label(),
            })
        }
        HostActionKind::Launch => {
            let path = binding
                .path
                .as_deref()
                .ok_or_else(|| "launch path not configured".to_string())?;
            crate::app_launch::focus_or_launch(path, binding.match_exe.as_deref())?;
            Ok(ActionOutcome::Continue)
        }
        HostActionKind::OpenFolder => {
            let path = binding
                .path
                .as_deref()
                .ok_or_else(|| "open_folder path not configured".to_string())?;
            crate::explorer::open_folder(path, binding.prefer_tab)?;
            Ok(ActionOutcome::Continue)
        }
        HostActionKind::FocusAiTerminal => focus_ai_terminal_for_slot(value, extras),
        HostActionKind::HudPrevious => Ok(move_hud_selection_and_render(
            app,
            extras,
            "no_selectable_hud_approval",
            |hud, pending| hud.move_selection(pending, HudSelectionDirection::Previous),
        )),
        HostActionKind::HudNext => Ok(move_hud_selection_and_render(
            app,
            extras,
            "no_selectable_hud_approval",
            |hud, pending| hud.move_selection(pending, HudSelectionDirection::Next),
        )),
        HostActionKind::HudConfirm => {
            let pending = extras.codex_activity.pending_approvals();
            // Read the live target and reserve the response under the same
            // `hud` lock acquisition, then drop it before touching any other
            // lock (`ai_display_slots`) below -- see this module's
            // `move_hud_selection_and_render` doc comment on not holding the
            // lock used for one step across another.
            let (dispatch, target_thread) = {
                let hud_guard = extras.hud.lock().unwrap();
                let Some(hud) = hud_guard.as_ref() else {
                    return Ok(ActionOutcome::HudNoop {
                        reason: "hud_response_in_flight_guard_or_no_selection",
                    });
                };
                let dispatch = hud.begin_response(&pending, Instant::now());
                let target_thread = match &dispatch {
                    Some(_) => hud.target_codex_thread(),
                    None => None,
                };
                (dispatch, target_thread)
            };
            let Some(dispatch) = dispatch else {
                return Ok(ActionOutcome::HudNoop {
                    reason: "hud_response_in_flight_guard_or_no_selection",
                });
            };
            // `respond_to_approval` may wait for the App Server response.
            // Never make the Host Link monitor loop wait for that round trip.
            dispatch_hud_response(pending, extras.codex_broker.clone(), dispatch);
            let slot = target_thread.and_then(|target| resolve_hud_target_slot(extras, &target));
            match slot {
                Some(slot) => Ok(ActionOutcome::HudResponded { slot }),
                None => Ok(ActionOutcome::Continue),
            }
        }
        HostActionKind::HudReject => {
            // Reject only relocates the highlight to the reject-side decision
            // (exact "decline", else exact "cancel"; see
            // `HudInteractionState::move_selection_toward_reject`). It never
            // sends a response itself -- only a later HudConfirm does that,
            // for whatever index ends up selected.
            Ok(move_hud_selection_and_render(
                app,
                extras,
                "no_reject_decision_available",
                |hud, pending| hud.move_selection_toward_reject(pending),
            ))
        }
        HostActionKind::SelectHudTarget => {
            let assigned = extras
                .ai_display_slots
                .lock()
                .unwrap()
                .slots()
                .get(usize::from(value))
                .and_then(|slot| slot.assigned.clone());
            let target = codex_target_for_slot(assigned, extras.codex_activity.snapshots());
            // ScreenKey's uplink means "make this session the one I'm
            // dealing with" (docs/ai-approval-hud-design.md §6.1/§11): a
            // slot with nothing waiting for HUD approval falls back to the
            // same terminal-focus behavior as a standalone FocusAiTerminal
            // press, rather than staying a silent HudNoop.
            let Some((connection_id, thread_id)) = target else {
                return focus_ai_terminal_for_slot(value, extras);
            };
            let pending = extras.codex_activity.pending_approvals();
            let selected = {
                let hud = extras.hud.lock().unwrap();
                hud.as_ref().is_some_and(|hud| {
                    hud.select_codex_thread(&pending, &connection_id, &thread_id)
                })
            };
            if !selected {
                return focus_ai_terminal_for_slot(value, extras);
            }
            // Render immediately, rather than waiting for the next periodic
            // update; `update` preserves this explicit target thereafter.
            if let Some(hud) = extras.hud.lock().unwrap().as_ref() {
                hud.update(app, &pending);
            }
            Ok(ActionOutcome::HudTargetSelected { slot: value })
        }
    }
}

/// Applies a HUD selection move (`HudPrevious`/`HudNext`/`HudReject`) and, only
/// when it actually moved the highlight, renders immediately rather than
/// waiting for the next periodic update (`SelectHudTarget` above does the same
/// for its own move, with the same comment). Without this, a physical
/// encoder turn stayed invisible until the next host-link tick -- up to
/// `polling.interval_ms` (500 ms on real hardware).
///
/// The lock used to move the selection is dropped (it lives only for the
/// `let moved = ...;` statement) before `update` is called, matching
/// `SelectHudTarget`'s own "drop the lock used to move, then re-acquire it to
/// render" structure. The re-acquired guard is held across `update` itself,
/// exactly as `SelectHudTarget` has always done; the point of dropping first
/// is only that the move and the render do not share one guard.
fn move_hud_selection_and_render(
    app: &AppHandle,
    extras: &MonitorExtras,
    no_move_reason: &'static str,
    move_fn: impl FnOnce(
        &crate::hud_coordinator::HudCoordinator,
        &rawhid_host_core::pending_approval::PendingApprovalStore,
    ) -> Option<usize>,
) -> ActionOutcome {
    let pending = extras.codex_activity.pending_approvals();
    let moved = extras
        .hud
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|hud| move_fn(hud, &pending));
    if moved.is_none() {
        return ActionOutcome::HudNoop {
            reason: no_move_reason,
        };
    }
    if let Some(hud) = extras.hud.lock().unwrap().as_ref() {
        hud.update(app, &pending);
    }
    ActionOutcome::Continue
}

/// Resolves the ScreenKey slot to its assigned terminal and brings it to the
/// foreground. Shared by `HostActionKind::FocusAiTerminal` (a standalone key
/// binding) and `HostActionKind::SelectHudTarget`'s fallback when the slot
/// has no pending HUD approval to switch to.
fn focus_ai_terminal_for_slot(value: u8, extras: &MonitorExtras) -> Result<ActionOutcome, String> {
    // `value` is the ScreenKey's physical index, i.e. the display
    // slot to resolve. Anything short of "exactly one session
    // assigned to this slot" is a silent no-op per the design
    // (docs/screenkey-terminal-focus-design.md 3.5): out-of-range
    // slot, unassigned slot, and (unreachable in practice, see the
    // design's F11-F13) an empty terminal_target_id. None of these
    // touch `ai_terminal_focusing`: they never start a focus
    // sequence, so there is nothing to guard against re-entry.
    let assigned = extras
        .ai_display_slots
        .lock()
        .unwrap()
        .slots()
        .get(usize::from(value))
        .and_then(|slot| slot.assigned.clone());
    let terminal_target_id = match assigned {
        Some(AiDisplayTarget::Codex { terminal_target_id })
        | Some(AiDisplayTarget::Claude { terminal_target_id }) => terminal_target_id,
        None => return Ok(ActionOutcome::Continue),
    };
    if terminal_target_id.is_empty() {
        return Ok(ActionOutcome::Continue);
    }
    // Only claim the in-progress flag once we know a focus sequence
    // will actually run, so every early return above needs no
    // matching `store(false)`; `FocusGuard` releases it when the
    // spawned thread finishes.
    if extras
        .ai_terminal_focusing
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return Err("focus_in_progress".to_string());
    }
    // The search-and-focus sequence runs on its own thread: the
    // monitor loop that calls `execute` must not block on it (it can
    // take ~1.1s when SetForegroundWindow is denied, see F2/F8).
    crate::ai_terminal_focus::spawn_focus(
        terminal_target_id,
        Arc::clone(&extras.ai_terminal_focusing),
    );
    Ok(ActionOutcome::Continue)
}

/// Finds which ScreenKey slot is assigned the exact Codex `(connection_id,
/// thread_id)` that just received a HUD answer, by reusing the same
/// predicate (`codex_target_for_slot`) that decides whether a slot is the
/// HUD's current target elsewhere (`commands.rs`'s per-slot ScreenKey state
/// calculation, `HostActionKind::SelectHudTarget` above). Reusing rather than
/// re-deriving this comparison is deliberate -- see the 2026-09-04 handoff
/// note on keeping "which slot counts as the target" as one predicate.
fn resolve_hud_target_slot(extras: &MonitorExtras, target: &(String, String)) -> Option<u8> {
    let snapshots = extras.codex_activity.snapshots();
    let slots = extras.ai_display_slots.lock().unwrap();
    slots
        .slots()
        .iter()
        .find(|entry| {
            codex_target_for_slot(entry.assigned.clone(), snapshots.clone()).as_ref()
                == Some(target)
        })
        .map(|entry| entry.slot)
}

pub(crate) fn codex_target_for_slot(
    assigned: Option<AiDisplayTarget>,
    snapshots: Vec<CodexSessionSnapshot>,
) -> Option<(String, String)> {
    let AiDisplayTarget::Codex { terminal_target_id } = assigned? else {
        return None;
    };
    snapshots
        .into_iter()
        .find(|snapshot| {
            snapshot.state.session_active
                && snapshot.is_display_target
                && snapshot.state.activity_state
                    == rawhid_host_core::packet::AiActivityState::WaitingApproval
                && snapshot.terminal_target_id == terminal_target_id
        })
        .map(|snapshot| (snapshot.owner_connection_id, snapshot.thread_id))
}

fn dispatch_hud_response(
    pending: Arc<rawhid_host_core::pending_approval::PendingApprovalStore>,
    broker: rawhid_host_core::codex_broker::CodexBrokerManager,
    dispatch: crate::hud_coordinator::HudResponseDispatch,
) {
    // `Builder::spawn` returns an error rather than panicking on the usual
    // OS thread-creation failure. In that case `dispatch` drops here and its
    // reservation is released, leaving a still-pending request retryable.
    let _ = std::thread::Builder::new()
        .name("hud-approval-response".to_string())
        .spawn(move || {
            // A duplicate press or a CLI-first response is expected to lose the
            // Broker first-wins race. The monitor already recorded dispatch, and
            // there is no safe recovery action for a stale physical packet.
            let _ = dispatch_hud_response_selection(&pending, &broker, &dispatch.selection);
        });
}

/// The response half of a physical HUD action after the coordinator has
/// atomically chosen one exact pending request. Keeping this separate from
/// the thread spawn lets tests exercise the same Host dispatcher without a
/// Tauri window or a physical HID packet.
pub(crate) fn dispatch_hud_response_selection(
    pending: &rawhid_host_core::pending_approval::PendingApprovalStore,
    broker: &rawhid_host_core::codex_broker::CodexBrokerManager,
    selection: &crate::hud_coordinator::HudApprovalSelection,
) -> Result<bool, String> {
    respond_to_codex_approval_internal(
        pending,
        broker,
        selection.key.token(),
        selection.decision_index,
    )
}

#[cfg(test)]
fn spawn_response_task(task: impl FnOnce() + Send + 'static) {
    std::thread::spawn(task);
}

#[cfg(test)]
mod tests {
    use std::{
        sync::mpsc,
        time::{Duration, Instant},
    };

    use futures_util::{SinkExt, StreamExt};
    use rawhid_host_core::{
        codex_activity::{AiClientStateSnapshot, CodexActivityRuntime, CodexSessionSnapshot},
        codex_broker::{test_support::BrokerHarness, CodexApprovalResponseOutcome},
        packet::{AiActivityState, AiClientType, AiClientVariant, AiWorkPhase},
        pending_approval::{
            codex_key_for_thread, ApprovalClient, ApprovalOwner, PendingApprovalBody,
            PendingApprovalStore,
        },
    };
    use serde_json::{json, Value};
    use tokio::{net::TcpListener, time};
    use tokio_tungstenite::{
        accept_async, connect_async,
        tungstenite::{client::IntoClientRequest, Message},
    };

    use crate::{
        hud_coordinator::{response_selection_from_state, HudInteractionState, HUD_CONFIRM_GUARD},
        state::AiDisplayTarget,
    };

    fn codex_snapshot(
        connection_id: &str,
        thread_id: &str,
        terminal_target_id: &str,
    ) -> CodexSessionSnapshot {
        CodexSessionSnapshot {
            thread_id: thread_id.to_string(),
            owner_connection_id: connection_id.to_string(),
            terminal_target_id: terminal_target_id.to_string(),
            registration_order: 1,
            state: AiClientStateSnapshot {
                client_type: AiClientType::Codex,
                client_variant: AiClientVariant::Cli,
                session_active: true,
                activity_state: AiActivityState::WaitingApproval,
                work_phase: AiWorkPhase::Unspecified,
                revision: 1,
            },
            is_display_target: true,
        }
    }

    #[test]
    fn response_dispatch_returns_before_the_response_work_finishes() {
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let before = std::time::Instant::now();
        super::spawn_response_task(move || {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        });
        assert!(before.elapsed() < Duration::from_millis(50));
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        release_tx.send(()).unwrap();
    }

    #[test]
    fn slot_selection_uses_the_display_targets_exact_connection_and_thread() {
        let target = super::codex_target_for_slot(
            Some(AiDisplayTarget::Codex {
                terminal_target_id: "terminal-b".to_string(),
            }),
            vec![
                codex_snapshot("connection-a", "thread-a", "terminal-a"),
                codex_snapshot("connection-b", "thread-b", "terminal-b"),
            ],
        );
        assert_eq!(
            target,
            Some(("connection-b".to_string(), "thread-b".to_string()))
        );
        assert!(super::codex_target_for_slot(
            Some(AiDisplayTarget::Claude {
                terminal_target_id: "claude".to_string(),
            }),
            Vec::new(),
        )
        .is_none());
    }

    #[test]
    fn slot_selection_keeps_thread_identity_when_one_connection_owns_two_threads() {
        let target = super::codex_target_for_slot(
            Some(AiDisplayTarget::Codex {
                terminal_target_id: "terminal-a".to_string(),
            }),
            vec![
                codex_snapshot("connection-a", "thread-a", "terminal-a"),
                codex_snapshot("connection-a", "thread-b", "terminal-b"),
            ],
        );

        assert_eq!(
            target,
            Some(("connection-a".to_string(), "thread-a".to_string()))
        );
    }

    // `SelectHudTarget`'s routing decision cannot be exercised through
    // `execute()` itself in this crate's tests: that arm needs a live
    // `tauri::AppHandle` to build a `HudCoordinator`
    // (`HudCoordinator::create`), and nothing in this crate's test suite
    // constructs one -- see `hud_coordinator.rs`'s own tests, which only
    // ever exercise `HudInteractionState`/free-function pure state, never
    // the coordinator itself. The two tests below instead pin down the
    // exact precondition `execute()`'s `SelectHudTarget` arm checks before
    // falling back to `focus_ai_terminal_for_slot`:
    // `codex_target_for_slot` finding a target, and (mirroring
    // `HudCoordinator::select_codex_thread`'s own success condition)
    // `PendingApprovalStore::latest_codex_for_connection_and_thread` finding
    // a live entry for it.

    #[test]
    fn select_hud_target_precondition_holds_when_a_pending_approval_exists() {
        let store = PendingApprovalStore::new();
        let key = codex_key_for_thread("connection-a", &json!(7), Some("thread-a"));
        store.insert(
            key,
            ApprovalClient::Codex,
            ApprovalOwner::Codex {
                connection_id: "connection-a".to_string(),
            },
            PendingApprovalBody {
                primary_text: None,
                full_command: None,
                reason: None,
                cwd: None,
                kind: None,
                available_decisions: Some(vec![json!("approve")]),
                tool_use_id: None,
                prompt_id: None,
            },
        );

        let target = super::codex_target_for_slot(
            Some(AiDisplayTarget::Codex {
                terminal_target_id: "terminal-a".to_string(),
            }),
            vec![codex_snapshot("connection-a", "thread-a", "terminal-a")],
        )
        .expect("slot resolves the pending Codex session's connection/thread");
        // This is exactly `HudCoordinator::select_codex_thread`'s success
        // condition: when it holds, `execute()`'s `SelectHudTarget` arm
        // switches the HUD target and returns `Continue` without ever
        // reaching either former `HudNoop` (now fallback) point.
        assert!(store
            .latest_codex_for_connection_and_thread(&target.0, &target.1)
            .is_some());
    }

    #[test]
    fn select_hud_target_falls_back_when_slot_has_no_pending_codex_target() {
        // No Codex session is `WaitingApproval` for this slot's assigned
        // target, so `codex_target_for_slot` finds nothing -- the first of
        // `SelectHudTarget`'s two former `HudNoop` points
        // ("slot_has_no_codex_pending_target"), which `execute()` now
        // routes to `focus_ai_terminal_for_slot` instead.
        let mut idle_snapshot =
            codex_snapshot("connection-a", "thread-a", "codex-deadbeefdeadbeef");
        idle_snapshot.state.activity_state = AiActivityState::Available;
        assert!(super::codex_target_for_slot(
            Some(AiDisplayTarget::Codex {
                terminal_target_id: "codex-deadbeefdeadbeef".to_string(),
            }),
            vec![idle_snapshot],
        )
        .is_none());
    }

    async fn receive_json<Stream>(socket: &mut tokio_tungstenite::WebSocketStream<Stream>) -> Value
    where
        Stream: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
    {
        let message = time::timeout(Duration::from_secs(2), socket.next())
            .await
            .expect("WebSocket frame timed out")
            .expect("WebSocket closed")
            .expect("WebSocket read failed");
        serde_json::from_str(message.to_text().expect("text JSON-RPC frame").as_ref())
            .expect("valid JSON-RPC")
    }

    async fn wait_until(description: &str, predicate: impl Fn() -> bool) {
        time::timeout(Duration::from_secs(2), async {
            while !predicate() {
                time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("timed out waiting for {description}"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn host_approval_lifecycle_e2e_uses_real_broker_and_selected_thread_only() {
        // The local App Server below is the sole mock.  Everything between it
        // and the simulated HOST_ACTION is production code: the WebSocket
        // Broker, event reducer/PendingApprovalStore, HUD pure state, and the
        // same response dispatcher called by `actions::execute`.
        let upstream_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let upstream_addr = upstream_listener.local_addr().unwrap();
        let harness = BrokerHarness::start(
            format!("ws://{upstream_addr}"),
            &[("client-token", "screenkey-terminal")],
        )
        .await
        .unwrap();
        let manager = harness.manager();
        let activity = CodexActivityRuntime::start(manager.clone());
        let upstream = tokio::spawn(async move {
            let (upstream_socket, _) = upstream_listener.accept().await.unwrap();
            accept_async(upstream_socket).await.unwrap()
        });

        let mut request = format!("ws://{}", harness.broker_addr())
            .into_client_request()
            .unwrap();
        request
            .headers_mut()
            .insert("authorization", "Bearer client-token".parse().unwrap());
        let (mut cli, _) = connect_async(request).await.unwrap();
        let mut app_server = upstream.await.unwrap();

        // Establish two owned threads on one Broker connection, then put both
        // in WaitingApproval. Thread B is the display target because it is the
        // current Codex focus; the Host must not answer thread A instead.
        for (rpc_id, thread_id, turn_id) in [(1, "thread-a", "turn-a"), (2, "thread-b", "turn-b")] {
            cli.send(Message::Text(
                json!({
                    "jsonrpc": "2.0", "id": rpc_id, "method": "thread/start", "params": {}
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();
            assert_eq!(receive_json(&mut app_server).await["id"], rpc_id);
            app_server
                .send(Message::Text(
                    json!({
                        "jsonrpc": "2.0", "id": rpc_id,
                        "result": { "thread": { "id": thread_id } }
                    })
                    .to_string()
                    .into(),
                ))
                .await
                .unwrap();
            let _ = receive_json(&mut cli).await;
            app_server.send(Message::Text(json!({
                "jsonrpc": "2.0", "method": "turn/started",
                "params": { "threadId": thread_id, "turn": { "id": turn_id, "status": "inProgress" } }
            }).to_string().into())).await.unwrap();
            let _ = receive_json(&mut cli).await;
        }

        let approval = |id, thread_id, turn_id, decisions: Value| {
            json!({
                "jsonrpc": "2.0", "id": id,
                "method": "item/commandExecution/requestApproval",
                "params": {
                    "threadId": thread_id, "turnId": turn_id,
                    "item": { "id": format!("item-{id}") },
                    "commandActions": [{ "command": format!("echo {thread_id}") }],
                    "availableDecisions": decisions
                }
            })
        };
        for frame in [
            approval(101, "thread-a", "turn-a", json!(["approve-a", "cancel"])),
            approval(
                102,
                "thread-b",
                "turn-b",
                json!(["approve-b", "decline", "cancel"]),
            ),
        ] {
            app_server
                .send(Message::Text(frame.to_string().into()))
                .await
                .unwrap();
            let _ = receive_json(&mut cli).await;
        }

        let pending = activity.pending_approvals();
        wait_until("both pending approvals", || pending.len() == 2).await;
        wait_until("thread-b display target", || {
            activity.snapshots().iter().any(|snapshot| {
                snapshot.thread_id == "thread-b"
                    && snapshot.is_display_target
                    && snapshot.state.activity_state == AiActivityState::WaitingApproval
            })
        })
        .await;
        let (connection_id, display_thread_id) = super::codex_target_for_slot(
            Some(AiDisplayTarget::Codex {
                terminal_target_id: "screenkey-terminal".to_string(),
            }),
            activity.snapshots(),
        )
        .expect("ScreenKey slot resolves its exact Codex display target");
        assert_eq!(display_thread_id, "thread-b");
        let (selected_key, selected_snapshot) = pending
            .latest_codex_for_connection_and_thread(&connection_id, &display_thread_id)
            .expect("display thread has its own pending approval");
        assert_eq!(
            selected_snapshot.client,
            rawhid_host_core::ApprovalClient::Codex
        );

        let shown_at = Instant::now();
        let mut hud_state = HudInteractionState::default();
        hud_state.sync_target(Some(&selected_key), 3, shown_at);
        assert!(response_selection_from_state(
            &hud_state,
            &pending,
            shown_at + HUD_CONFIRM_GUARD - Duration::from_nanos(1)
        )
        .is_none());
        let selection =
            response_selection_from_state(&hud_state, &pending, shown_at + HUD_CONFIRM_GUARD)
                .expect("400 ms guard releases the selected exact decision");
        assert_eq!(selection.key, selected_key);
        assert_eq!(selection.decision_index, 0);
        assert!(super::dispatch_hud_response_selection(&pending, &manager, &selection).unwrap());
        let host_response = receive_json(&mut app_server).await;
        assert_eq!(host_response["id"], 102);
        assert_eq!(host_response["result"]["decision"], "approve-b");
        assert!(pending.get(&selected_key).is_none());

        // HUD/Host won; the later CLI response cannot make a second upstream
        // delivery. Thread A remains pending and was never answered.
        cli.send(Message::Text(
            json!({
                "jsonrpc": "2.0", "id": 102, "result": { "decision": "cancel" }
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
        assert!(time::timeout(Duration::from_millis(100), app_server.next())
            .await
            .is_err());
        assert!(pending
            .latest_codex_for_connection_and_thread(&connection_id, "thread-a")
            .is_some());

        // CLI wins request 103. The real manager's Host route then reports
        // AlreadyResolved and does not emit a second JSON-RPC response.
        let cli_first = approval(103, "thread-b", "turn-b", json!(["approve", "cancel"]));
        app_server
            .send(Message::Text(cli_first.to_string().into()))
            .await
            .unwrap();
        let _ = receive_json(&mut cli).await;
        cli.send(Message::Text(
            json!({
                "jsonrpc": "2.0", "id": 103, "result": { "decision": "cancel" }
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
        let cli_response = receive_json(&mut app_server).await;
        assert_eq!(cli_response["result"]["decision"], "cancel");
        assert_eq!(
            manager
                .respond_to_approval(&connection_id, json!(103), json!("approve"))
                .unwrap(),
            CodexApprovalResponseOutcome::AlreadyResolved
        );
        assert!(time::timeout(Duration::from_millis(100), app_server.next())
            .await
            .is_err());

        // Reject only relocates the highlight; it prefers an exact `decline`
        // option here even though `cancel` is also offered (see the priority
        // order in `HudInteractionState::move_selection_toward_reject`).
        // Sending it still requires a later HudConfirm, guard included.
        let decline = approval(104, "thread-b", "turn-b", json!(["cancel", "decline"]));
        app_server
            .send(Message::Text(decline.to_string().into()))
            .await
            .unwrap();
        let _ = receive_json(&mut cli).await;
        wait_until("decline request in pending store", || pending.len() >= 2).await;
        let (decline_key, _) = pending
            .latest_codex_for_connection_and_thread(&connection_id, "thread-b")
            .unwrap();
        let decline_shown_at = Instant::now();
        hud_state.sync_target(Some(&decline_key), 2, decline_shown_at);
        let decline_snapshot = pending.get(&decline_key).unwrap();
        assert_eq!(
            hud_state.move_selection_toward_reject(&decline_snapshot),
            Some(1)
        );
        let reject = response_selection_from_state(
            &hud_state,
            &pending,
            decline_shown_at + HUD_CONFIRM_GUARD,
        )
        .expect("guard-elapsed confirm sends whatever reject moved the highlight to");
        assert_eq!(reject.decision_index, 1);
        assert!(super::dispatch_hud_response_selection(&pending, &manager, &reject).unwrap());
        let decline_response = receive_json(&mut app_server).await;
        assert_eq!(decline_response["id"], 104);
        assert_eq!(decline_response["result"]["decision"], "decline");

        // With no `decline` present, reject now falls back to an exact
        // `cancel` option instead of being a no-op -- the old Host behavior
        // (only `decline` was ever selectable) has changed.
        let cancel_only = approval(105, "thread-b", "turn-b", json!(["approve", "cancel"]));
        app_server
            .send(Message::Text(cancel_only.to_string().into()))
            .await
            .unwrap();
        let _ = receive_json(&mut cli).await;
        wait_until("cancel-only request in pending store", || {
            pending.len() >= 2
        })
        .await;
        let (cancel_key, _) = pending
            .latest_codex_for_connection_and_thread(&connection_id, "thread-b")
            .unwrap();
        let cancel_shown_at = Instant::now();
        hud_state.sync_target(Some(&cancel_key), 2, cancel_shown_at);
        let cancel_snapshot = pending.get(&cancel_key).unwrap();
        assert_eq!(
            hud_state.move_selection_toward_reject(&cancel_snapshot),
            Some(1)
        );
        let cancel_selection = response_selection_from_state(
            &hud_state,
            &pending,
            cancel_shown_at + HUD_CONFIRM_GUARD,
        )
        .expect("guard-elapsed confirm sends the reject-moved-to cancel index");
        assert_eq!(cancel_selection.decision_index, 1);
        assert!(
            super::dispatch_hud_response_selection(&pending, &manager, &cancel_selection).unwrap()
        );
        let cancel_response = receive_json(&mut app_server).await;
        assert_eq!(cancel_response["id"], 105);
        assert_eq!(cancel_response["result"]["decision"], "cancel");

        // HUD/Host won this race too; the later CLI response cannot make a
        // second upstream delivery.
        cli.send(Message::Text(
            json!({
                "jsonrpc": "2.0", "id": 105, "result": { "decision": "cancel" }
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
        assert!(time::timeout(Duration::from_millis(100), app_server.next())
            .await
            .is_err());

        // A stale pure HUD target is harmless after disconnect: the runtime
        // clears every pending request owned by that Broker connection.
        cli.close(None).await.unwrap();
        wait_until("disconnect clears pending approvals", || pending.is_empty()).await;
        assert!(response_selection_from_state(&hud_state, &pending, Instant::now()).is_none());
        drop(activity);
        harness.shutdown().await.unwrap();
    }
}
