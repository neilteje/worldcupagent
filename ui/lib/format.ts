export function pct(value: unknown, digits = 0): string {
  const n = typeof value === "number" ? value : Number(value ?? 0);
  if (!Number.isFinite(n)) return "n/a";
  return `${(n * 100).toFixed(digits)}%`;
}

export function signed(value: unknown, digits = 1): string {
  const n = typeof value === "number" ? value : Number(value ?? 0);
  if (!Number.isFinite(n)) return "n/a";
  return `${n >= 0 ? "+" : ""}${(n * 100).toFixed(digits)}pp`;
}

export function money(value: unknown): string {
  const n = typeof value === "number" ? value : Number(value ?? 0);
  if (!Number.isFinite(n)) return "n/a";
  return `$${n.toFixed(2)}`;
}

export function compactDate(value: string | number | null | undefined): string {
  if (!value) return "unknown";
  const date = typeof value === "number" ? new Date(value) : new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("en", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

export function labelize(value: unknown): string {
  if (value === null || value === undefined || value === "") return "none";
  return String(value).replaceAll("_", " ");
}
