"use client";

import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import { CountUp, Reveal } from "@/components/motion";
import { api, type Cleaning, type CleaningEntry } from "@/lib/api";

const ease = [0.16, 1, 0.3, 1] as const;

// Every rule the pipeline applies, in order; shown even before a run has recorded its counts.
const RULES: { step: string; dataset: string; issue: string; fix: string; why: string }[] = [
  { step: "prepare", dataset: "Open Food Facts", issue: "8 GB export, 111 columns, all languages", fix: "Read only needed columns and English products with an ingredient list (DuckDB)", why: "The full table does not fit in memory" },
  { step: "prepare", dataset: "Open Food Facts", issue: "Nutri-Score values 'unknown' / 'not-applicable'", fix: "Set to missing", why: "They are not grades and would become a fake tier" },
  { step: "step1", dataset: "Open Food Facts", issue: "Exact duplicate rows", fix: "Drop repeats", why: "A product counted twice weighs twice in training" },
  { step: "step1", dataset: "Open Food Facts", issue: "Same barcode listed more than once", fix: "Keep one row per barcode", why: "One product = one example" },
  { step: "step2", dataset: "Open Food Facts", issue: "Missing or empty ingredient list", fix: "Drop the product", why: "Nothing to classify" },
  { step: "step2", dataset: "Open Food Facts", issue: "Identical ingredient list under different products", fix: "Keep one per normalised list", why: "Identical inputs in train and test would inflate scores" },
  { step: "step2", dataset: "Open Food Facts", issue: "Text shorter than 8 characters / nothing parsed", fix: "Drop the product", why: "Fragments and OCR debris" },
  { step: "step2", dataset: "Open Food Facts", issue: "Missing NOVA, tier or food-group label", fix: "Keep; leave out only of the model that needs that label", why: "Still useful for the other targets" },
  { step: "step3", dataset: "USDA", issue: "Several versions of one product; missing ingredient statements", fix: "Latest version with ingredients, one row per barcode", why: "Older versions are duplicates" },
  { step: "step3", dataset: "Open Food Facts", issue: "Products also present in USDA (about 86%)", fix: "Remove those barcodes from training", why: "Prevents leakage into the USDA test" },
  { step: "step3b", dataset: "USDA", issue: "USDA categories differ from the 11 food groups", fix: "Map with ordered keyword rules", why: "One shared set of labels" },
  { step: "step3b", dataset: "USDA", issue: "No Nutri-Score grade in USDA", fix: "Compute Nutri-Score 2017 from USDA nutrients", why: "Independent health-tier labels" },
];

const TEXT_RULES = [
  ["Upper / lower case", "Everything lower-cased", "SUGAR and Sugar are the same word"],
  ["Percentages", "\"(12%)\" removed", "Amounts are not ingredient names"],
  ["\"May contain\", \"Contains:\" tails", "Removed", "Allergen advice is not an ingredient"],
  ["\"Contains 2% or less of:\"", "Turned into a list separator", "USDA-style labels"],
  ["E-number spelling", "\"E 102\", \"e-102\" → e102", "One spelling per additive"],
  ["Function words", "\"emulsifier: soy lecithin\" → soy lecithin", "The function adds no ingredient"],
  ["Brackets", "Split into sub-ingredients, top-level order kept", "Order ≈ quantity"],
  ["OCR confusions", "0→o, 1→l, |→l inside words", "Typical camera errors"],
] as const;

const fmt = (n: number) => n.toLocaleString();

function RuleCard({ e, i }: { e: CleaningEntry; i: number }) {
  const changed = e.before - e.after;
  const badge =
    e.kind === "kept"
      ? { text: "kept", cls: "bg-sky-400/15 text-sky-300" }
      : changed === 0
        ? { text: "none found", cls: "bg-white/[0.07] text-muted" }
        : { text: e.kind === "fixed" ? "✓ fixed" : "✓ removed", cls: "bg-accent/15 text-accent" };
  return (
    <motion.div
      initial={{ opacity: 0, y: 18 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5, delay: (i % 6) * 0.05, ease }}
      whileHover={{ y: -4 }}
      className="glass flex flex-col p-5"
    >
      <p className="text-sm font-medium leading-snug">{e.issue}</p>
      <p className="mt-0.5 text-[11px] text-faint">
        {e.dataset} · {e.step}
      </p>
      <div className="mt-4 flex items-baseline gap-2.5">
        {e.kind !== "kept" && (
          <>
            <span className="font-mono text-base text-muted line-through decoration-nova4/70">{fmt(e.before)}</span>
            <span className="text-muted">→</span>
          </>
        )}
        <CountUp value={e.after} className="font-mono text-2xl font-semibold text-ink" />
        <span className="text-xs text-faint">{e.unit}</span>
      </div>
      <p className="mt-1 text-xs text-muted">
        {e.kind === "kept"
          ? e.id === "usda_matched"
            ? `of ${fmt(e.before)} USDA products have an OFF label`
            : "rows kept, excluded only from that model"
          : changed > 0
            ? `${fmt(changed)} ${e.kind === "fixed" && e.unit === "values" ? "values corrected" : e.kind === "fixed" ? "of the rows labelled" : "removed"}`
            : "nothing to change"}
      </p>
      <div className="mt-auto pt-4">
        <span className={`inline-block rounded-full px-2.5 py-1 text-[11px] font-medium ${badge.cls}`}>{badge.text}</span>
      </div>
      <p className="mt-3 border-t border-white/[0.06] pt-3 text-xs leading-relaxed text-muted">
        <span className="text-ink/80">Fix:</span> {e.fix}. <span className="text-ink/80">Why:</span> {e.why}.
      </p>
    </motion.div>
  );
}

export default function CleaningPage() {
  const [c, setC] = useState<Cleaning | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api.cleaning().then(setC).catch((e) => setError((e as Error).message));
  }, []);

  return (
    <div className="space-y-12">
      <div className="max-w-3xl pt-4">
        <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="label !text-accent">
          Data preparation
        </motion.p>
        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease }}
          className="mt-2 text-4xl font-semibold tracking-tight sm:text-5xl"
        >
          Data cleaning
        </motion.h1>
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.08, ease }}
          className="mt-4 text-[15px] leading-relaxed text-muted"
        >
          Every problem found in the two datasets, how it was fixed, and why. The cleaned tables
          (<span className="font-mono text-[13px] text-ink/80">data/processed/off_clean.parquet</span>,{" "}
          <span className="font-mono text-[13px] text-ink/80">off_train_disjoint.parquet</span>,{" "}
          <span className="font-mono text-[13px] text-ink/80">usda_labeled.parquet</span>) are the only data the models
          and the evaluation use. Generated by <span className="font-mono text-[13px] text-ink/80">training/steps/step1–step3b</span>.
        </motion.p>
      </div>

      {error && <p className="border-l-2 border-nova4 pl-3 text-sm text-nova4">{error}</p>}
      {!c && !error && <p className="text-sm text-muted">Loading…</p>}

      {c && (
        <>
          {/* funnel of real counts */}
          <section className="space-y-4">
            <h2 className="text-lg font-semibold tracking-tight">From raw export to training data</h2>
            <div className="grid gap-5 lg:grid-cols-2">
              {c.funnel.map((d, di) => {
                const top = d.steps[0].value ?? 1;
                return (
                  <Reveal key={d.dataset} delay={di * 0.08} className="glass p-5">
                    <p className="label">{d.dataset}</p>
                    <ol className="mt-4 space-y-3">
                      {d.steps.map((s, i) => (
                        <li key={s.label}>
                          <div className="flex items-baseline justify-between gap-3 text-sm">
                            <span className="text-muted">{s.label}</span>
                            <span className="shrink-0 font-mono text-ink">
                              {s.value == null ? "—" : `${s.approx ? "~" : ""}${fmt(s.value)}`}
                            </span>
                          </div>
                          <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-white/[0.06]">
                            <motion.div
                              initial={{ width: 0 }}
                              whileInView={{ width: `${Math.max(1.5, ((s.value ?? 0) / top) * 100)}%` }}
                              viewport={{ once: true }}
                              transition={{ duration: 1, delay: 0.2 + i * 0.12, ease }}
                              className="h-full rounded-full bg-gradient-to-r from-accent/50 to-accent"
                            />
                          </div>
                        </li>
                      ))}
                    </ol>
                    <p className="mt-4 text-[11px] text-faint">
                      Sources: {Array.from(new Set(d.steps.map((s) => s.source))).join(", ")}
                    </p>
                  </Reveal>
                );
              })}
            </div>
          </section>

          {/* per-rule counts from the pipeline log */}
          <section className="space-y-4">
            <div className="flex flex-wrap items-baseline justify-between gap-3">
              <h2 className="text-lg font-semibold tracking-tight">Problems found and fixed</h2>
              {c.log && <span className="text-xs text-faint">{c.log.length} rules · from reports/tables/cleaning_log.json</span>}
            </div>
            {c.log && c.log_synthetic && (
              <p className="rounded-xl border border-nova3/30 bg-warn-soft px-4 py-3 text-sm text-warn">
                These counts come from the synthetic sample used for testing, not from the real datasets.
              </p>
            )}
            {c.log ? (
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {c.log.map((e, i) => (
                  <RuleCard key={`${e.step}-${e.id}`} e={e} i={i} />
                ))}
              </div>
            ) : (
              <>
                <p className="rounded-xl border border-sky-400/25 bg-sky-400/[0.06] px-4 py-3 text-sm leading-relaxed text-sky-100/90">
                  Per-rule counts appear here after the cleaning steps run on the real data
                  (<span className="font-mono text-[13px]">python training/run_all.py --from 1 --until 3.5</span>). Each step
                  records how many rows every rule removed or fixed.
                </p>
                <Reveal className="glass overflow-x-auto p-5">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-white/15 text-left text-[11px] uppercase tracking-wide text-muted">
                        <th className="py-2 pr-4 font-medium">Step</th>
                        <th className="py-2 pr-4 font-medium">Problem</th>
                        <th className="py-2 pr-4 font-medium">Fix</th>
                        <th className="py-2 font-medium">Why</th>
                      </tr>
                    </thead>
                    <tbody>
                      {RULES.map((r) => (
                        <tr key={r.issue} className="border-b border-white/[0.06] align-top">
                          <td className="py-2.5 pr-4 font-mono text-xs text-faint">{r.step}</td>
                          <td className="py-2.5 pr-4">
                            {r.issue}
                            <div className="text-[11px] text-faint">{r.dataset}</div>
                          </td>
                          <td className="py-2.5 pr-4 text-muted">{r.fix}</td>
                          <td className="py-2.5 text-muted">{r.why}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </Reveal>
              </>
            )}
          </section>

          {/* text normalisation */}
          <section className="space-y-4">
            <h2 className="text-lg font-semibold tracking-tight">Cleaning the ingredient text</h2>
            <div className="grid gap-5 lg:grid-cols-[1fr_1.4fr]">
              <Reveal className="glass p-5">
                <p className="label">Text rules (parser.py)</p>
                <ul className="mt-3 divide-y divide-white/[0.06]">
                  {TEXT_RULES.map(([k, v, w]) => (
                    <li key={k} className="py-2.5 text-sm">
                      <span className="text-ink">{k}</span>
                      <span className="mx-1.5 text-faint">·</span>
                      <span className="text-muted">{v}</span>
                      <div className="text-[11px] text-faint">{w}</div>
                    </li>
                  ))}
                </ul>
              </Reveal>
              <div className="space-y-4">
                {c.examples.map((ex, i) => (
                  <Reveal key={ex.title} delay={i * 0.06} className="glass p-5">
                    <p className="text-xs text-muted">{ex.title}</p>
                    <p className="mt-2 rounded-lg border border-nova4/20 bg-nova4/[0.06] px-3 py-2 font-mono text-[12px] leading-relaxed text-ink/80">
                      {ex.raw}
                    </p>
                    <div className="my-2 text-center text-faint">↓ parsed</div>
                    <ol className="flex flex-wrap gap-1.5">
                      {ex.ingredients.map((g, j) => (
                        <li
                          key={`${g.name}-${j}`}
                          className={`rounded-md border px-2 py-0.5 text-[13px] ${
                            g.depth > 0
                              ? "border-sky-400/25 bg-sky-400/[0.06] text-sky-100/90"
                              : "border-accent/25 bg-accent/[0.07] text-ink"
                          }`}
                          title={g.depth > 0 ? "sub-ingredient (inside brackets)" : `position ${g.position}`}
                        >
                          {g.depth === 0 && <span className="mr-1 font-mono text-[10px] text-faint">{g.position}</span>}
                          {g.name}
                        </li>
                      ))}
                    </ol>
                  </Reveal>
                ))}
                <p className="text-[11px] text-faint">
                  Green: top-level ingredients with their position (order ≈ quantity). Blue: sub-ingredients from brackets.
                  These examples are parsed live by the same code the models use.
                </p>
              </div>
            </div>
          </section>
        </>
      )}
    </div>
  );
}
