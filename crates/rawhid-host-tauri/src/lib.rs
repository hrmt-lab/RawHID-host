mod actions;
mod ai_terminal_focus;
mod app_launch;
mod claude_launcher;
mod codex_launcher;
mod commands;
mod debug_log;
mod explorer;
mod foreground;
mod hud_coordinator;
mod hud_probe;
mod hud_window;
mod icon;
mod startup;
mod state;

use state::{add_log, AppState};
use tauri::{
    menu::{MenuBuilder, MenuItemBuilder},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    Emitter, Manager, WindowEvent,
};

pub fn run() {
    let (mut config, config_path) = commands::load_initial_config();
    let start_on_launch = config.app.start_monitoring_on_launch;
    // The `tracing` subscriber backing the `[debug_log]` file sink is
    // installed exactly once here, regardless of whether it starts enabled —
    // see `debug_log`'s module doc for why it must never be re-initialized
    // when the Settings toggle flips later.
    let (debug_log, debug_log_error) = debug_log::init(config.debug_log.enabled);
    if debug_log_error.is_some() {
        // docs/spec.md: an unopenable output destination shows an error in
        // the UI and disables file logging -- that applies at startup too,
        // not only when the Settings toggle is flipped later. Keep the
        // in-memory config's flag in sync with the sink's actual (disabled)
        // state so the Settings toggle isn't shown as on for a sink that
        // isn't running; the on-disk toml is left untouched, so a later save
        // reconciles it. The error itself is logged to the UI log below,
        // once an `AppHandle` exists to emit "log-added" with.
        config.debug_log.enabled = false;
    }
    let app_state = AppState::new(config, config_path, debug_log);

    tauri::Builder::default()
        // Single-instance must be the first plugin registered.
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            if let Some(win) = app.get_webview_window("main") {
                let _ = win.unminimize();
                let _ = win.show();
                let _ = win.set_focus();
            }
        }))
        .plugin(tauri_plugin_dialog::init())
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![
            commands::get_config,
            commands::get_config_path,
            commands::save_config,
            commands::reload_config,
            commands::show_config_file_location,
            commands::get_status,
            commands::get_log_entries,
            commands::get_codex_integration_status,
            commands::respond_to_codex_approval,
            commands::respond_to_claude_approval,
            commands::get_ai_client_state,
            commands::get_ai_display_slots,
            commands::pin_ai_display_slot,
            commands::set_ai_display_slot_auto,
            commands::start_codex_integration,
            commands::launch_codex_cli,
            commands::launch_claude_code,
            commands::get_claude_sessions,
            commands::stop_claude_code,
            commands::list_wsl_distributions,
            commands::stop_codex_integration,
            commands::probe_devices,
            commands::probe_studio_devices,
            commands::read_studio_keymap,
            commands::studio_export_keymap,
            commands::studio_preview_keymap_restore,
            commands::studio_apply_keymap_restore,
            commands::studio_key_catalog,
            commands::resolve_studio_behavior_labels,
            commands::studio_begin_edit,
            commands::studio_set_key,
            commands::studio_add_layer,
            commands::studio_rename_layer,
            commands::studio_remove_layer,
            commands::studio_save_changes,
            commands::studio_discard_changes,
            commands::studio_reset_to_keymap,
            commands::studio_has_unsaved,
            commands::studio_resync_edit_state,
            commands::read_encoder_info,
            commands::read_encoder_bindings,
            commands::read_encoder_layer_bindings,
            commands::studio_set_encoder_bindings,
            commands::studio_encoder_has_unsaved,
            commands::studio_encoder_save,
            commands::studio_encoder_discard,
            commands::studio_encoder_clear_override,
            commands::read_combo_info,
            commands::read_combo,
            commands::studio_set_combo,
            commands::studio_combo_has_unsaved,
            commands::studio_combo_save,
            commands::studio_combo_discard,
            commands::studio_combo_delete,
            commands::studio_combo_reset_to_keymap,
            commands::studio_end_edit,
            commands::studio_abort_edit,
            commands::start_monitoring,
            commands::stop_monitoring,
            commands::refresh_ai_usage,
            commands::get_running_apps,
            commands::get_app_icons,
            commands::get_launch_at_login,
            commands::set_launch_at_login,
            commands::get_key_stats,
            commands::list_key_stats_devices,
            commands::debug_inject_uplink,
        ])
        .setup(move |app| {
            setup_window_icon(app)?;
            setup_tray(app)?;
            setup_hud(app)?;
            let handle = app.handle().clone();
            let state = app.state::<AppState>();
            if let Some(error) = &debug_log_error {
                let message = format!("Debug file log could not be enabled: {error}");
                let entry = add_log(&state.log_entries, &state.log_counter, "error", &message);
                let _ = handle.emit("log-added", entry);
            }
            commands::start_host_link_worker(handle, state.inner(), start_on_launch)
                .map_err(std::io::Error::other)?;
            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { api, .. } = event {
                let _ = window.hide();
                api.prevent_close();
            }
        })
        .run(tauri::generate_context!())
        .expect("failed to start Keylink Studio");
}

/// Verification-only entry point for `--hud-focus-probe` (see `main.rs`).
///
/// Builds its own minimal `tauri::Builder` deliberately kept separate from
/// `run()` above:
/// - No `tauri_plugin_single_instance`: that plugin forwards CLI args to an
///   already-running Studio instance instead of letting a second process
///   start, which would prevent the probe from ever running while Studio is
///   in its normal tray-resident state.
/// - No tray, `AppState`, or `invoke_handler`: the probe doesn't need any
///   of Studio's normal command surface.
///
/// `setup()` hides the `main` window immediately (mirroring the tray-resident
/// state `run()` settles into) and then hands off to `hud_probe::run` on a
/// background thread — `setup()` must return quickly or Tauri's event loop
/// never starts pumping messages, which the probe's WebView2-backed HUD
/// window and Win32 message loop both depend on.
pub fn run_hud_focus_probe() {
    tauri::Builder::default()
        .setup(|app| {
            if let Some(main_window) = app.get_webview_window("main") {
                let _ = main_window.hide();
            }

            let handle = app.handle().clone();
            std::thread::spawn(move || {
                hud_probe::run(handle);
            });

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("failed to start HUD focus probe");
}

/// Creates the HUD window hidden (see `hud_window.rs`'s module doc for why
/// this must happen at startup rather than lazily) and stores the resulting
/// `HudCoordinator` in `AppState` so the host-link monitor thread
/// (`commands.rs`'s `run_monitor_loop`) can push approval updates to it once
/// it starts a few lines below.
fn setup_hud(app: &mut tauri::App) -> tauri::Result<()> {
    let coordinator =
        hud_coordinator::HudCoordinator::create(app.handle()).map_err(std::io::Error::other)?;
    let state = app.state::<AppState>();
    *state.hud.lock().unwrap() = Some(coordinator);
    Ok(())
}

fn setup_window_icon(app: &mut tauri::App) -> tauri::Result<()> {
    if let (Some(window), Some(icon)) = (app.get_webview_window("main"), app.default_window_icon())
    {
        window.set_icon(icon.clone())?;
    }

    Ok(())
}

fn setup_tray(app: &mut tauri::App) -> tauri::Result<()> {
    let start = MenuItemBuilder::with_id("start", "Start monitoring").build(app)?;
    let stop = MenuItemBuilder::with_id("stop", "Stop monitoring").build(app)?;
    let show = MenuItemBuilder::with_id("show", "Show window").build(app)?;
    let quit = MenuItemBuilder::with_id("quit", "Quit").build(app)?;
    let menu = MenuBuilder::new(app)
        .items(&[&start, &stop])
        .separator()
        .items(&[&show, &quit])
        .build()?;

    let _tray = TrayIconBuilder::with_id("main")
        .icon(app.default_window_icon().unwrap().clone())
        .tooltip("Keylink Studio")
        .menu(&menu)
        .on_menu_event(|app, event| match event.id().as_ref() {
            "start" => {
                let state = app.state::<AppState>();
                let _ = commands::begin_monitoring(app.clone(), state.inner());
            }
            "stop" => {
                let state = app.state::<AppState>();
                let _ = commands::stop_monitoring_internal(state.inner());
            }
            "show" => {
                if let Some(win) = app.get_webview_window("main") {
                    let _ = win.unminimize();
                    let _ = win.show();
                    let _ = win.set_focus();
                }
            }
            "quit" => {
                let state = app.state::<AppState>();
                commands::shutdown_codex_integration(state.inner());
                commands::shutdown_host_link_worker(state.inner());
                app.exit(0);
            }
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                let app = tray.app_handle();
                if let Some(win) = app.get_webview_window("main") {
                    if win.is_visible().unwrap_or(false) {
                        let _ = win.hide();
                    } else {
                        let _ = win.unminimize();
                        let _ = win.show();
                        let _ = win.set_focus();
                    }
                }
            }
        })
        .build(app)?;

    Ok(())
}
