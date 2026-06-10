export function JsonBlock({ value }: { value: unknown }) {
  return (
    <pre className="max-h-[28rem] overflow-auto rounded-2xl border border-[var(--line)] bg-[var(--ink)] p-4 text-xs leading-5 text-[var(--paper)]">
      {JSON.stringify(value ?? null, null, 2)}
    </pre>
  );
}
