import { useEffect, useState } from "react";
import { onHudApprovalUpdate } from "../api";
import { useLang, type TranslationKey } from "../i18n";
import type { HudApprovalPayload } from "../types";

type TFn = (key: TranslationKey, params?: Record<string, string | number>) => string;

/** Last path segment of `cwd`, shown next to the client name in the
 * heading (docs/ai-approval-hud-design.md §7.2's "セッション見出し").
 * Handles both `\` and `/` separators since Codex/Claude Code sessions on
 * this Host are always Windows paths, but a WSL-launched session's `cwd`
 * can still arrive with forward slashes. */
function projectName(cwd: string | null): string | null {
  if (!cwd) return null;
  const trimmed = cwd.replace(/[\\/]+$/, "");
  const segments = trimmed.split(/[\\/]/).filter(Boolean);
  return segments.length > 0 ? segments[segments.length - 1] : cwd;
}

function clientLabel(client: HudApprovalPayload["client"]): string {
  return client === "codex" ? "Codex" : "Claude Code";
}

/** §7.2: the request kind, in the user's language. Codex's `kind` and
 * Claude's `tool_name` are both raw protocol strings with no fixed enum
 * (`docs/ai-approval-hud-design.md` §9.1 notes Codex's own decision set
 * varies per request), so unrecognized values fall back to the raw string
 * rather than a hardcoded table. */
function describeKind(payload: HudApprovalPayload, t: TFn): string {
  if (payload.client === "codex") {
    if (payload.kind === "command") return t("hud.kind.codex_command");
    return payload.kind ?? t("hud.kind.codex_unknown");
  }
  return payload.kind
    ? t("hud.kind.claude_tool", { tool: payload.kind })
    : t("hud.kind.claude_unknown");
}

/** Codex's `availableDecisions` elements are opaque JSON: a mix of plain
 * strings and single-key objects (e.g. `{"acceptWithExecpolicyAmendment":
 * {...}}`) that must never be reconstructed or reinterpreted (see
 * `pending_approval.rs`'s doc comment on this field). For stage 1's
 * display-only list, this only needs a short human-readable label. */
function decisionLabel(decision: unknown): string {
  if (typeof decision === "string") return decision;
  if (decision && typeof decision === "object") {
    const keys = Object.keys(decision as Record<string, unknown>);
    if (keys.length > 0) return keys[0];
  }
  return JSON.stringify(decision);
}

export default function Hud() {
  const { t } = useLang();
  const [payload, setPayload] = useState<HudApprovalPayload | null>(null);
  // The Host clears the payload the moment an approval resolves, but keeps
  // the window up for HUD_EXIT_ANIMATION_MS (hud_coordinator.rs) so the
  // panel can animate out. Keep drawing the resolved request during that
  // gap instead of blanking instantly, which would make the delayed hide
  // look like a stuck empty window.
  const [leaving, setLeaving] = useState<HudApprovalPayload | null>(null);

  useEffect(() => {
    let cancelled = false;
    let unlisten: (() => void) | null = null;
    onHudApprovalUpdate((next) => {
      setPayload((current) => {
        setLeaving(next === null ? current : null);
        return next;
      });
    }).then((fn) => {
      if (cancelled) {
        fn();
        return;
      }
      unlisten = fn;
    });
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, []);

  // Rendered even with no payload (the Host keeps this window hidden via
  // `HudWindow::hide()` whenever nothing is pending -- see
  // `hud_coordinator.rs` -- so this is only ever visible for the brief gap
  // before the first event arrives).
  const shown = payload ?? leaving;
  if (!shown) {
    // No panel to draw -- leave this transparent (see hud.css) rather than
    // painting bg-surface, since this is only ever visible for the brief
    // gap before the first event arrives (see comment above).
    return <div className="h-full w-full" />;
  }

  const heading = (() => {
    const project = projectName(shown.cwd);
    const client = clientLabel(shown.client);
    return project ? `${client} ・ ${project}` : client;
  })();
  const decisions = shown.available_decisions ?? [];
  // available_decisions is always None on the oversized path (see
  // HudApprovalPayload's doc comment), but guard explicitly rather than
  // relying on that.
  const showDecisions = !shown.oversized && decisions.length > 0;

  return (
    <div className={`flex h-full w-full flex-col overflow-hidden rounded-card border border-border bg-surface/90 ${payload ? "hud-enter" : "hud-leave"}`}>
      <div className="flex-shrink-0 border-b border-background px-4 py-2.5">
        <div className="truncate text-xs font-medium text-muted">{heading}</div>
      </div>

      <div className="min-h-0 flex-1 space-y-2.5 overflow-y-auto px-4 py-3">
        {shown.oversized ? (
          <p className="text-sm text-faint">{t("hud.oversized")}</p>
        ) : (
          <>
            <div className="text-xs font-medium text-accent-deep">{describeKind(shown, t)}</div>

            <div className="rounded-lg bg-plate px-3 py-2">
              <div className="whitespace-pre-wrap break-all font-mono text-xs text-ink">
                {shown.primary_text ?? t("hud.no_primary_text")}
              </div>
            </div>

            {shown.cwd && (
              <div className="text-xs text-muted">
                <span className="text-faint">{t("hud.cwd")}: </span>
                <span className="break-all font-mono">{shown.cwd}</span>
              </div>
            )}

            {shown.reason && (
              <div className="text-xs text-muted">
                <span className="text-faint">{t("hud.reason")}: </span>
                {shown.reason}
              </div>
            )}
          </>
        )}
      </div>

      {showDecisions && (
        <div className="max-h-32 flex-shrink-0 overflow-y-auto px-4 pb-3 pt-1">
          <ul className="space-y-0.5">
            {decisions.map((decision, index) => (
              <li
                key={index}
                className={`rounded px-1 py-0.5 text-xs ${shown.selected_decision_index === index ? "bg-accent/20 text-accent-deep" : "text-ink"}`}
              >
                {shown.selected_decision_index === index && <span aria-hidden="true">› </span>}
                {shown.decision_labels?.[index] ?? decisionLabel(decision)}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
