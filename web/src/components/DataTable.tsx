import type { Confusion, Table } from "@/lib/api";

const HEADERS: Record<string, string> = {
  macro_f1: "Macro-F1",
  accuracy: "Accuracy",
  qwk: "QWK",
  spearman: "Spearman",
  fit_seconds: "Fit (s)",
  n: "Products",
  support: "Support",
  noise_rate: "Noise",
  macro_f1_drop: "F1 drop",
  test_set: "Test set",
};

const header = (c: string) => HEADERS[c] ?? c.replace(/_/g, " ");

function cell(v: string | number | null, col: string) {
  if (v === null || v === undefined || v === "") return "—";
  if (typeof v === "number") {
    if (Math.abs(v) < 5e-4) return "0";
    if (col === "noise_rate") return `${Math.round(v * 100)}%`;
    if (Number.isInteger(v)) return v.toLocaleString();
    return v.toFixed(3);
  }
  return String(v).replace(/_/g, " ");
}

export function DataTable({
  table,
  highlight = "macro_f1",
  sortBy,
  caption,
}: {
  table: Table;
  highlight?: string;
  sortBy?: string;
  caption?: string;
}) {
  if (!table) return null;
  let rows = table.rows;
  const si = sortBy ? table.columns.indexOf(sortBy) : -1;
  if (si >= 0) rows = [...rows].sort((a, b) => Number(b[si] ?? -1) - Number(a[si] ?? -1));
  const hi = table.columns.indexOf(highlight);
  const best = hi >= 0 ? Math.max(...rows.map((r) => (typeof r[hi] === "number" ? (r[hi] as number) : -Infinity))) : null;
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="border-b border-white/20">
            {table.columns.map((c, i) => (
              <th
                key={c}
                className={`whitespace-nowrap py-2 pr-4 text-left text-[11px] font-medium uppercase tracking-wide text-muted ${
                  i === hi ? "text-ink" : ""
                }`}
              >
                {header(c)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, ri) => (
            <tr key={ri} className="border-b border-line">
              {r.map((v, ci) => {
                const num = typeof v === "number";
                const isBest = ci === hi && num && v === best;
                return (
                  <td
                    key={ci}
                    className={`whitespace-nowrap py-1.5 pr-4 ${num ? "font-mono text-[13px]" : ""} ${
                      ci === hi ? "font-medium" : ""
                    } ${isBest ? "text-accent" : ""}`}
                  >
                    {cell(v, table.columns[ci])}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {caption && <p className="mt-2 text-xs text-muted">{caption}</p>}
    </div>
  );
}

export function ConfusionMatrix({ data, caption }: { data: Confusion; caption?: string }) {
  if (!data) return null;
  const n = data.labels.length;
  return (
    <div className="overflow-x-auto">
      <div
        className="inline-grid gap-1 text-[11px]"
        style={{ gridTemplateColumns: `minmax(90px,auto) repeat(${n}, minmax(56px, 1fr))` }}
      >
        <div className="p-1 text-faint">true ↓ / predicted →</div>
        {data.labels.map((l) => (
          <div key={l} className="truncate p-1 text-center text-muted" title={l}>
            {l.replace(/-/g, " ")}
          </div>
        ))}
        {data.matrix.map((row, i) => {
          const total = row.reduce((s, v) => s + v, 0) || 1;
          return (
            <div key={i} className="contents">
              <div className="truncate p-1 pr-2 text-muted" title={data.labels[i]}>
                {data.labels[i].replace(/-/g, " ")}
              </div>
              {row.map((v, j) => {
                const share = v / total;
                return (
                  <div
                    key={j}
                    className={`rounded-md p-1.5 text-center font-mono ${share > 0.55 ? "text-on-accent" : "text-ink"} ${
                      i === j ? "outline outline-1 outline-white/30" : ""
                    }`}
                    style={{ background: `rgba(61, 220, 151, ${0.05 + share * 0.9})` }}
                    title={`${v.toLocaleString()} of ${total.toLocaleString()}`}
                  >
                    {Math.round(share * 100)}%
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
      {caption && <p className="mt-2 text-xs text-muted">{caption}</p>}
    </div>
  );
}
