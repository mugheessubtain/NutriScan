"use client";

import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import { Bar, CountUp, Reveal, TiltCard } from "@/components/motion";
import { api, NOVA, TIERS, pct, pretty, type Analysis, type Driver, type Finding, type Suggestions } from "@/lib/api";

const NOVA_BG = ["bg-nova1", "bg-nova2", "bg-nova3", "bg-nova4"];
const NOVA_TEXT = ["text-nova1", "text-nova2", "text-nova3", "text-nova4"];
const NOVA_GLOW = [
  "shadow-[0_0_24px_2px_rgba(61,220,151,0.55)]",
  "shadow-[0_0_24px_2px_rgba(199,217,74,0.55)]",
  "shadow-[0_0_24px_2px_rgba(244,163,64,0.55)]",
  "shadow-[0_0_24px_2px_rgba(255,93,93,0.55)]",
];
const TIER_BG: Record<string, string> = { good: "bg-nova1", medium: "bg-nova3", poor: "bg-nova4" };
const TIER_TEXT: Record<string, string> = { good: "text-nova1", medium: "text-nova3", poor: "text-nova4" };
const ease = [0.16, 1, 0.3, 1] as const;

function Panel({ title, aside, children, delay = 0 }: { title: string; aside?: React.ReactNode; children: React.ReactNode; delay?: number }) {
  return (
    <Reveal delay={delay} className="glass p-5">
      <div className="mb-4 flex items-baseline justify-between gap-4">
        <h3 className="label">{title}</h3>
        {aside && <div className="text-xs text-faint">{aside}</div>}
      </div>
      {children}
    </Reveal>
  );
}

function Scale({ items, active, colors, glow }: { items: { key: string; label: string }[]; active: number; colors: string[]; glow?: string }) {
  return (
    <div className="grid gap-1.5" style={{ gridTemplateColumns: `repeat(${items.length}, 1fr)` }}>
      {items.map((it, i) => (
        <div key={it.key}>
          <div className="h-2 overflow-hidden rounded-full bg-white/[0.07]">
            {i === active && (
              <motion.div
                initial={{ scaleX: 0 }}
                animate={{ scaleX: 1 }}
                transition={{ duration: 0.8, delay: 0.35, ease }}
                style={{ originX: 0 }}
                className={`h-full rounded-full ${colors[i]} ${glow ?? ""}`}
              />
            )}
          </div>
          <div className={`mt-1.5 text-[11px] leading-tight ${i === active ? "text-ink" : "text-faint"}`}>{it.label}</div>
        </div>
      ))}
    </div>
  );
}

function Verdict({ a }: { a: Analysis }) {
  const g = a.nova.final;
  const t = a.tier.value;
  const tierIdx = TIERS.findIndex((x) => x.id === t);
  const raised = a.nova.final !== a.nova.ml;
  return (
    <TiltCard className="glass overflow-hidden p-6 sm:p-8">
      <div
        aria-hidden
        className={`pointer-events-none absolute -right-24 -top-24 h-72 w-72 rounded-full opacity-25 blur-3xl ${NOVA_BG[g - 1]}`}
      />
      <div className="relative grid gap-10 sm:grid-cols-2">
        <div>
          <div className="label">Processing level · NOVA</div>
          <div className="mt-3 flex items-end gap-4">
            <motion.span
              initial={{ opacity: 0, scale: 0.4, rotateX: -90 }}
              animate={{ opacity: 1, scale: 1, rotateX: 0 }}
              transition={{ type: "spring", stiffness: 180, damping: 14, delay: 0.1 }}
              className={`font-mono text-7xl font-medium leading-none ${NOVA_TEXT[g - 1]}`}
              style={{ textShadow: "0 0 40px currentColor" }}
            >
              {g}
            </motion.span>
            <div className="pb-1.5">
              <div className="text-xl font-semibold">{NOVA[g - 1].name}</div>
              <div className="text-xs text-muted">{NOVA[g - 1].detail}</div>
            </div>
          </div>
          <div className="mt-5">
            <Scale
              items={NOVA.map((n) => ({ key: String(n.group), label: `${n.group} · ${n.name}` }))}
              active={g - 1}
              colors={NOVA_BG}
              glow={NOVA_GLOW[g - 1]}
            />
          </div>
          <p className="mt-4 text-xs text-muted">
            Model confidence <CountUp value={a.nova.confidence * 100} suffix="%" className="font-mono text-ink" />
            {raised ? ` · raised from NOVA ${a.nova.ml} by the rule system` : " · rule system agrees"}
          </p>
        </div>
        <div>
          <div className="label">Health tier · Nutri-Score band</div>
          <div className="mt-3 flex items-end gap-4">
            <motion.span
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, delay: 0.2, ease }}
              className={`text-5xl font-semibold leading-none ${TIER_TEXT[t]}`}
              style={{ textShadow: "0 0 36px currentColor" }}
            >
              {TIERS[tierIdx]?.name}
            </motion.span>
            <span className="pb-1 text-sm text-muted">{TIERS[tierIdx]?.detail}</span>
          </div>
          <div className="mt-5">
            <Scale
              items={TIERS.map((x) => ({ key: x.id, label: x.name }))}
              active={tierIdx}
              colors={TIERS.map((x) => TIER_BG[x.id])}
            />
          </div>
          <p className="mt-4 text-xs text-muted">
            Model confidence <CountUp value={a.tier.confidence * 100} suffix="%" className="font-mono text-ink" />
          </p>
        </div>
      </div>
      <div className="relative mt-8 grid grid-cols-2 gap-4 border-t border-white/[0.07] pt-5 sm:grid-cols-4">
        {[
          ["Food group", pretty(a.category)],
          ["Rules fired", <CountUp key="r" value={a.findings.filter((f) => f.kind === "rule").length} />],
          ["Ingredients", <CountUp key="i" value={a.ingredients.length} />],
          ["Analysed in", `${a.seconds.toFixed(1)} s`],
        ].map(([k, v]) => (
          <div key={k as string}>
            <div className="label">{k}</div>
            <div className="mt-1 font-mono text-lg">{v}</div>
          </div>
        ))}
      </div>
    </TiltCard>
  );
}

const SEVERITY = ["Info", "Low", "Medium", "High"];
const SEV_COLOR = ["bg-white/20", "bg-nova2", "bg-nova3", "bg-nova4"];
const SEV_TEXT = ["text-faint", "text-nova2", "text-nova3", "text-nova4"];

function Findings({ items }: { items: Finding[] }) {
  if (!items.length) return <p className="text-sm text-muted">No ingredients matched a rule or a flagged additive.</p>;
  return (
    <ul className="space-y-2.5">
      {items.map((f, i) => (
        <motion.li
          key={`${f.kind}-${f.id}`}
          initial={{ opacity: 0, x: -16 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.45, delay: i * 0.05, ease }}
          whileHover={{ x: 4 }}
          className="relative overflow-hidden rounded-xl border border-white/[0.06] bg-white/[0.025] py-3 pl-5 pr-4"
        >
          <span className={`absolute inset-y-0 left-0 w-1 ${SEV_COLOR[f.severity] ?? "bg-white/20"}`} />
          <div className="flex items-start justify-between gap-3">
            <p className="text-sm leading-snug">{f.message}</p>
            <span className={`shrink-0 font-mono text-[10px] uppercase tracking-wider ${SEV_TEXT[f.severity] ?? ""}`}>
              {SEVERITY[f.severity] ?? f.severity}
            </span>
          </div>
          <p className="mt-1.5 text-xs text-faint">
            <span className="font-mono uppercase">{f.kind}</span> · Found: <span className="text-muted">{f.evidence.join(", ")}</span> ·
            Source: {f.source}
          </p>
        </motion.li>
      ))}
    </ul>
  );
}

function Drivers({ title, items, color }: { title: string; items: Driver[]; color: string }) {
  const max = Math.max(...items.map((d) => d.value), 0.0001);
  return (
    <div>
      <p className="mb-3 text-xs text-muted">{title}</p>
      {items.length === 0 ? (
        <p className="text-sm text-faint">No single word stands out.</p>
      ) : (
        <ul className="space-y-2">
          {items.slice(0, 6).map((d, i) => (
            <li key={d.term} className="grid grid-cols-[minmax(0,130px)_1fr_40px] items-center gap-3 text-sm">
              <span className="truncate" title={d.term}>
                {d.term}
              </span>
              <span className="h-2 overflow-hidden rounded-full bg-white/[0.06]">
                <Bar value={d.value / max} className={color} delay={0.15 + i * 0.06} />
              </span>
              <span className="text-right font-mono text-xs text-muted">{d.value.toFixed(2)}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function PakistanSuggestions({ id }: { id: string }) {
  const [s, setS] = useState<Suggestions | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let alive = true;
    api
      .suggest(id)
      .then((r) => alive && setS(r))
      .catch(() => alive && setFailed(true));
    return () => {
      alive = false;
    };
  }, [id]);

  // the section only appears when Gemini is configured and answered
  if (failed || (s && (!s.available || s.items.length === 0))) return null;
  return (
    <Panel title="Healthier options in Pakistan" aside="Gemini suggestion" delay={0.1}>
      {!s ? (
        <div className="space-y-2" aria-busy="true">
          {[0, 1, 2].map((i) => (
            <div key={i} className="h-12 animate-pulse rounded-xl bg-white/[0.05]" />
          ))}
        </div>
      ) : (
        <>
          <ul className="space-y-2.5">
            {s.items.map((x, i) => (
              <motion.li
                key={`${x.name}-${x.brand}`}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.45, delay: i * 0.08, ease }}
                whileHover={{ y: -2 }}
                className="rounded-xl border border-white/[0.06] bg-white/[0.025] p-3.5"
              >
                <div className="flex items-start justify-between gap-3">
                  <p className="text-sm font-medium">
                    {x.name}
                    {x.brand && <span className="font-normal text-muted"> · {x.brand}</span>}
                  </p>
                  {x.likely_nova && (
                    <span
                      className={`shrink-0 rounded-full border border-current/30 px-2 py-0.5 font-mono text-[10px] ${NOVA_TEXT[x.likely_nova - 1]}`}
                    >
                      ~NOVA {x.likely_nova}
                    </span>
                  )}
                </div>
                <p className="mt-1 text-xs leading-relaxed text-muted">{x.why_better}</p>
              </motion.li>
            ))}
          </ul>
          <p className="mt-3 text-[11px] text-faint">
            Suggested by {s.model}, not by the trained models. Check the label and availability in store.
          </p>
        </>
      )}
    </Panel>
  );
}

export default function Result({ a }: { a: Analysis }) {
  return (
    <div className="space-y-5">
      <motion.h2
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease }}
        className="text-2xl font-semibold tracking-tight"
      >
        Result
      </motion.h2>

      <motion.div initial={{ opacity: 0, y: 40 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, ease }}>
        <Verdict a={a} />
      </motion.div>

      {a.nova.rule_reasons.map((r) => (
        <Reveal key={r} className="rounded-xl border border-nova3/30 bg-warn-soft px-4 py-3 text-sm text-warn">
          {r}
        </Reveal>
      ))}

      {a.allergens.length > 0 && (
        <Reveal className="rounded-2xl border border-warn/25 bg-warn-soft px-5 py-4">
          <p className="text-sm font-semibold text-warn">Contains declared allergen sources</p>
          <div className="mt-2 flex flex-wrap gap-2">
            {a.allergens.map((g) => (
              <span key={g.group} className="rounded-full border border-warn/30 bg-warn/10 px-3 py-1 text-xs text-warn">
                {pretty(g.group)} <span className="text-warn/70">({g.words.join(", ")})</span>
              </span>
            ))}
          </div>
          <p className="mt-2 text-xs text-muted">Keyword match on the parsed list (EU 1169/2011 Annex II groups).</p>
        </Reveal>
      )}

      <div className="grid gap-5 lg:grid-cols-[1.2fr_1fr]">
        <div className="space-y-5">
          <Panel title="Flags from the expert system" aside={`${a.findings.length} items`}>
            <Findings items={a.findings} />
          </Panel>
          <Panel title="Parsed ingredients" aside="in label order">
            <ol className="flex flex-wrap gap-2">
              {a.ingredients.map((x, i) => (
                <motion.li
                  key={`${x}-${i}`}
                  initial={{ opacity: 0, scale: 0.8 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.3, delay: i * 0.03 }}
                  className="rounded-lg border border-white/[0.08] bg-white/[0.03] px-2.5 py-1 text-sm"
                >
                  <span className="mr-1.5 font-mono text-[11px] text-faint">{i + 1}</span>
                  {x}
                </motion.li>
              ))}
            </ol>
          </Panel>
        </div>

        <div className="space-y-5">
          <Panel title="Why the model decided" aside="SHAP values">
            <div className="space-y-6">
              <Drivers title={`Words pushing toward NOVA ${a.nova.ml}`} items={a.drivers.nova} color="bg-gradient-to-r from-accent/60 to-accent" />
              <Drivers
                title={`Words pushing toward “${a.tier.value}”`}
                items={a.drivers.tier}
                color="bg-gradient-to-r from-sky-400/50 to-sky-300"
              />
            </div>
          </Panel>
          <PakistanSuggestions id={a.id} />
          {!a.alternatives_note && a.alternatives.length > 0 && (
            <Panel title="Similar products in the dataset" aside="same food group">
              <ul className="space-y-2.5">
                {a.alternatives.map((alt) => (
                  <li key={`${alt.name}-${alt.brand}`} className="rounded-xl border border-white/[0.06] bg-white/[0.025] p-3.5">
                    <div className="flex items-baseline justify-between gap-3">
                      <p className="text-sm font-medium">{alt.name}</p>
                      <span className="shrink-0 font-mono text-xs text-muted">{pct(alt.similarity)} similar</span>
                    </div>
                    <p className="text-xs text-muted">
                      {alt.brand} · {alt.reason}
                    </p>
                  </li>
                ))}
              </ul>
            </Panel>
          )}
        </div>
      </div>

      <details className="glass px-5 py-4 text-xs text-muted">
        <summary className="cursor-pointer select-none label">Audit log · model and knowledge-base versions</summary>
        <dl className="mt-3 grid grid-cols-[160px_1fr] gap-y-1 font-mono text-[11px]">
          {Object.entries(a.versions).map(([k, v]) => (
            <div key={k} className="contents">
              <dt className="text-faint">{k}</dt>
              <dd>{String(v ?? "—")}</dd>
            </div>
          ))}
        </dl>
      </details>
    </div>
  );
}
