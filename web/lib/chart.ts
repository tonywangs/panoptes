/** Shared Recharts tooltip styling so dark-mode text isn't black-on-dark. */

export const TOOLTIP_CONTENT_STYLE = {
  background: "var(--surface)",
  border: "1px solid var(--border)",
  borderRadius: 8,
  fontSize: 12,
  color: "var(--foreground)",
} as const;

export const TOOLTIP_ITEM_STYLE = {
  color: "var(--foreground)",
} as const;

export const TOOLTIP_LABEL_STYLE = {
  color: "var(--foreground-muted)",
} as const;
