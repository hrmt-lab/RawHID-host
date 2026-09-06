use std::{
    fs,
    path::{Path, PathBuf},
};

use serde_json::json;
use thiserror::Error;

use crate::claude_observer::ClaudeObserverConfig;

pub const CLAUDE_TOOL_HOOK_TIMEOUT_SECONDS: u64 = 1;
pub const CLAUDE_PROMPT_HOOK_TIMEOUT_SECONDS: u64 = 2;
pub const CLAUDE_STOP_HOOK_TIMEOUT_SECONDS: u64 = 3;
pub const CLAUDE_SESSION_START_TIMEOUT_SECONDS: u64 = 2;
/// `PermissionRequest` alone gets a much longer hook timeout than every
/// other hook: this is the one hook Studio holds open waiting for a HUD
/// decision (`docs/ai-approval-hud-design.md` §9.2), long enough for a
/// person to actually read the HUD and press a key.
///
/// Raised from an earlier 60s to 600s: `docs/claude-permission-hook-gate-results.md`
/// §Q6 confirmed by an actual round trip that this is safe to extend --
/// Claude Code never waits for the hook itself, it shows its own terminal
/// prompt after ~3s regardless -- so a longer `timeout` here never makes
/// the user wait longer. It only extends how long the *keyboard* answer
/// path stays available, and 60s proved too short in real usage for a
/// person to reliably notice a request and press the HUD in time.
///
/// What is *not* known is the largest `timeout` Claude Code's hook config
/// will actually honor before enforcing some smaller cap of its own and
/// closing the connection early regardless of this value -- that has never
/// been measured. If that happens, the Host-side wait
/// (`claude_decision::CLAUDE_PERMISSION_DECISION_TIMEOUT`, which is kept
/// shorter than this value -- see its own doc comment) still cleans up the
/// stale HUD entry once *it* elapses, so a request never sits on the HUD
/// forever answerable-looking with nowhere for the answer to go -- it just
/// means Claude Code stopped listening some time before that cleanup ran.
pub const CLAUDE_PERMISSION_HOOK_TIMEOUT_SECONDS: u64 = 600;

#[derive(Debug, Clone)]
pub struct ClaudePluginOptions {
    pub plugin_root: PathBuf,
    pub helper_executable: PathBuf,
    pub observer: ClaudeObserverConfig,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaudePluginArtifacts {
    pub plugin_root: PathBuf,
    pub observer_path: PathBuf,
    pub wrapper_path: PathBuf,
}

#[derive(Debug, Error)]
pub enum ClaudePluginError {
    #[error("invalid Claude plugin configuration: {0}")]
    InvalidConfig(String),
    #[error("failed to create Claude plugin directory {path}: {source}")]
    CreateDirectory {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to serialize Claude plugin file {path}: {source}")]
    Serialize {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("failed to write Claude plugin file {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

pub fn write_claude_observer_plugin(
    options: &ClaudePluginOptions,
) -> Result<ClaudePluginArtifacts, ClaudePluginError> {
    if !options.plugin_root.is_absolute() {
        return Err(ClaudePluginError::InvalidConfig(
            "plugin_root must be an absolute path".to_string(),
        ));
    }
    if !options.helper_executable.is_absolute() {
        return Err(ClaudePluginError::InvalidConfig(
            "helper_executable must be an absolute path".to_string(),
        ));
    }
    if options.observer.request_timeout_ms == 0 {
        return Err(ClaudePluginError::InvalidConfig(
            "observer request timeout must be greater than zero".to_string(),
        ));
    }

    let manifest_dir = options.plugin_root.join(".claude-plugin");
    let hooks_dir = options.plugin_root.join("hooks");
    create_directory(&options.plugin_root)?;
    create_directory(&manifest_dir)?;
    create_directory(&hooks_dir)?;

    let observer_path = options.plugin_root.join("observer.json");
    write_json(&observer_path, &options.observer)?;
    write_json(
        &manifest_dir.join("plugin.json"),
        &json!({
            "name": "keylink-claude-observer",
            "description": "Observation-only Claude Code integration for Keylink Studio",
            "version": "0.1.0"
        }),
    )?;
    write_json(
        &hooks_dir.join("hooks.json"),
        &hooks_json(
            &options.helper_executable,
            &observer_path,
            &options.observer,
        ),
    )?;
    let wrapper_path = options.plugin_root.join("keylink-claude-wrapper.ps1");
    write_utf8(&wrapper_path, POWERSHELL_WRAPPER.as_bytes())?;

    Ok(ClaudePluginArtifacts {
        plugin_root: options.plugin_root.clone(),
        observer_path,
        wrapper_path,
    })
}

fn hooks_json(
    helper_executable: &Path,
    observer_path: &Path,
    observer: &ClaudeObserverConfig,
) -> serde_json::Value {
    let session_start = json!({
        "hooks": [{
            "type": "command",
            "command": helper_executable.display().to_string(),
            "args": ["forward", "--observer", observer_path.display().to_string()],
            "timeout": CLAUDE_SESSION_START_TIMEOUT_SECONDS
        }]
    });
    let http_hook = |timeout: u64| {
        json!({
            "hooks": [{
                "type": "http",
                "url": observer.endpoint,
                "headers": {"Authorization": format!("Bearer {}", observer.bearer_token)},
                "timeout": timeout
            }]
        })
    };
    let matched_http_hook = |matcher: &str, timeout: u64| {
        json!({
            "matcher": matcher,
            "hooks": [{
                "type": "http",
                "url": observer.endpoint,
                "headers": {"Authorization": format!("Bearer {}", observer.bearer_token)},
                "timeout": timeout
            }]
        })
    };

    json!({
        "hooks": {
            "SessionStart": [session_start],
            "UserPromptSubmit": [http_hook(CLAUDE_PROMPT_HOOK_TIMEOUT_SECONDS)],
            "PreToolUse": [matched_http_hook("*", CLAUDE_TOOL_HOOK_TIMEOUT_SECONDS)],
            "PermissionRequest": [matched_http_hook("*", CLAUDE_PERMISSION_HOOK_TIMEOUT_SECONDS)],
            "PermissionDenied": [matched_http_hook("*", CLAUDE_TOOL_HOOK_TIMEOUT_SECONDS)],
            "PostToolUse": [matched_http_hook("*", CLAUDE_TOOL_HOOK_TIMEOUT_SECONDS)],
            "PostToolUseFailure": [matched_http_hook("*", CLAUDE_TOOL_HOOK_TIMEOUT_SECONDS)],
            "PostToolBatch": [http_hook(CLAUDE_TOOL_HOOK_TIMEOUT_SECONDS)],
            "Notification": [matched_http_hook("*", CLAUDE_TOOL_HOOK_TIMEOUT_SECONDS)],
            "Stop": [http_hook(CLAUDE_STOP_HOOK_TIMEOUT_SECONDS)],
            "StopFailure": [matched_http_hook("*", CLAUDE_STOP_HOOK_TIMEOUT_SECONDS)],
            "PreCompact": [matched_http_hook("*", CLAUDE_PROMPT_HOOK_TIMEOUT_SECONDS)],
            "PostCompact": [matched_http_hook("*", CLAUDE_PROMPT_HOOK_TIMEOUT_SECONDS)],
            "SessionEnd": [matched_http_hook("*", CLAUDE_TOOL_HOOK_TIMEOUT_SECONDS)],
            "Elicitation": [matched_http_hook("*", CLAUDE_TOOL_HOOK_TIMEOUT_SECONDS)],
            "ElicitationResult": [matched_http_hook("*", CLAUDE_TOOL_HOOK_TIMEOUT_SECONDS)]
        }
    })
}

fn create_directory(path: &Path) -> Result<(), ClaudePluginError> {
    fs::create_dir_all(path).map_err(|source| ClaudePluginError::CreateDirectory {
        path: path.to_path_buf(),
        source,
    })
}

fn write_json(path: &Path, value: &impl serde::Serialize) -> Result<(), ClaudePluginError> {
    let mut bytes =
        serde_json::to_vec_pretty(value).map_err(|source| ClaudePluginError::Serialize {
            path: path.to_path_buf(),
            source,
        })?;
    bytes.push(b'\n');
    write_utf8(path, &bytes)
}

fn write_utf8(path: &Path, bytes: &[u8]) -> Result<(), ClaudePluginError> {
    fs::write(path, bytes).map_err(|source| ClaudePluginError::Write {
        path: path.to_path_buf(),
        source,
    })
}

const POWERSHELL_WRAPPER: &str = r#"param(
    [Parameter(Mandatory = $true)][string]$ClaudeExecutable,
    [Parameter(Mandatory = $true)][string]$ProjectDirectory,
    [Parameter(Mandatory = $true)][string]$PluginRoot,
    [Parameter(Mandatory = $true)][string]$ObserverPath,
    [Parameter(Mandatory = $true)][string]$HelperExecutable,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$ClaudeArguments
)

$observerJson = [System.IO.File]::ReadAllText(
    $ObserverPath,
    [System.Text.UTF8Encoding]::new($false)
)
$claudeExitCode = 1

try {
    Set-Location -LiteralPath $ProjectDirectory
    & $ClaudeExecutable --plugin-dir $PluginRoot @ClaudeArguments
    if ($null -eq $LASTEXITCODE) {
        $claudeExitCode = 0
    } else {
        $claudeExitCode = $LASTEXITCODE
    }
} finally {
    try {
        $observerJson | & $HelperExecutable wrapper-exit --observer-stdin --exit-code $claudeExitCode | Out-Null
    } catch {
        # Observation must never change Claude Code termination.
    }
}

exit $claudeExitCode
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_bom_free_plugin_with_production_timeouts() {
        let root = tempfile::tempdir().unwrap();
        let plugin_root = root.path().join("plugin");
        let helper = root.path().join("keylink-claude-hook.exe");
        let artifacts = write_claude_observer_plugin(&ClaudePluginOptions {
            plugin_root: plugin_root.clone(),
            helper_executable: helper.clone(),
            observer: ClaudeObserverConfig {
                endpoint: "http://127.0.0.1:4567/hooks".to_string(),
                wrapper_exit_endpoint: "http://127.0.0.1:4567/wrapper-exit".to_string(),
                bearer_token: "0123456789abcdef0123456789abcdef".to_string(),
                launch_id: "launch-1".to_string(),
                request_timeout_ms: 500,
            },
        })
        .unwrap();

        assert_eq!(artifacts.plugin_root, plugin_root);
        let hooks_bytes = fs::read(plugin_root.join("hooks/hooks.json")).unwrap();
        assert!(!hooks_bytes.starts_with(&[0xef, 0xbb, 0xbf]));
        let hooks: serde_json::Value = serde_json::from_slice(&hooks_bytes).unwrap();
        assert_eq!(hooks["hooks"]["PreToolUse"][0]["hooks"][0]["timeout"], 1);
        // `PermissionRequest` alone is extended to 600s so a HUD decision has
        // time to arrive; every other hook keeps its short, observation-only
        // timeout (see `CLAUDE_PERMISSION_HOOK_TIMEOUT_SECONDS`'s doc
        // comment).
        assert_eq!(
            hooks["hooks"]["PermissionRequest"][0]["hooks"][0]["timeout"],
            600
        );
        assert_eq!(
            hooks["hooks"]["PermissionDenied"][0]["hooks"][0]["timeout"],
            1
        );
        assert_eq!(
            hooks["hooks"]["UserPromptSubmit"][0]["hooks"][0]["timeout"],
            2
        );
        assert_eq!(hooks["hooks"]["Stop"][0]["hooks"][0]["timeout"], 3);
        assert_eq!(
            hooks["hooks"]["SessionStart"][0]["hooks"][0]["command"],
            helper.display().to_string()
        );

        let wrapper = fs::read_to_string(&artifacts.wrapper_path).unwrap();
        assert!(wrapper.contains("$observerJson ="));
        assert!(wrapper.contains("wrapper-exit --observer-stdin"));
        assert!(!wrapper.as_bytes().starts_with(&[0xef, 0xbb, 0xbf]));

        #[cfg(windows)]
        {
            let wrapper_path = artifacts.wrapper_path.to_string_lossy();
            let wrapper_path = wrapper_path.strip_prefix(r"\\?\").unwrap_or(&wrapper_path);
            let output = std::process::Command::new("powershell.exe")
                .args([
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "$tokens=$null; $errors=$null; [System.Management.Automation.Language.Parser]::ParseFile($env:KEYLINK_WRAPPER_TEST_PATH, [ref]$tokens, [ref]$errors) | Out-Null; if ($errors.Count -ne 0) { $errors | ForEach-Object { [Console]::Error.WriteLine($_.Message) }; exit 1 }",
                ])
                .env("KEYLINK_WRAPPER_TEST_PATH", wrapper_path)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "PowerShell parser failed: stdout={} stderr={}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }

    #[test]
    fn rejects_relative_product_paths() {
        let error = write_claude_observer_plugin(&ClaudePluginOptions {
            plugin_root: PathBuf::from("plugin"),
            helper_executable: PathBuf::from("helper.exe"),
            observer: ClaudeObserverConfig {
                endpoint: "http://127.0.0.1/hooks".to_string(),
                wrapper_exit_endpoint: "http://127.0.0.1/wrapper-exit".to_string(),
                bearer_token: "token".to_string(),
                launch_id: "launch".to_string(),
                request_timeout_ms: 500,
            },
        })
        .unwrap_err();
        assert!(matches!(error, ClaudePluginError::InvalidConfig(_)));
    }
}
