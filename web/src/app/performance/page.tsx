"use client";

import { useEffect, useState } from "react";
import { ConfusionMatrix, DataTable } from "@/components/DataTable";
import { api, type Metrics, type Table } from "@/lib/api";

const TARGETS = [
  { id: "nova_group", name: "NOVA group", note: "4 classes, ordinal" },
  { id: "health_tier", name: "Health tier", note: "good / medium / poor" },
  { id: "category", name: "Food group", note: "11 classes" },
] as const;

const FIGURE_TITLES: Record<string, string> = {
  eda_E2_off_nova: "NOVA class distribution",
  eda_E2_off_tier: "Health tier distribution",
  eda_E3_off_counts_by_nova: "Ingredient and additive counts by NOVA group",
  eda_E4_off_top_additives_nova4: "Most common additives in NOVA 4",
  eda_E6_off_nova_share_by_category: "NOVA share within each food group",
  eda_E10_off_vs_usda_n_ingredients: "Ingredients per product: OFF vs USDA",
  shap_linear_nova_group_4: "SHAP summary: NOVA 4",
  robustness_ocr_noise: "Robustness to OCR noise",
};

function Block({ title, sub, children }: { title: string; sub?: string; children: React.ReactNode }) {
  return (
    <section className="glass p-5 sm:p-6">
      <h2 className="text-lg font-semibold tracking-tight">{title}</h2>
      {sub && <p className="mt-1 max-w-3xl text-sm leading-relaxed text-muted">{sub}</p>}
      <div className="mt-4">{children}</div>
    </section>
  );
}

function headline(t: Table, target: string) {
  if (!t) return null;
  const ti = t.columns.indexOf("target");
  const fi = t.columns.indexOf("macro_f1");
  const ai = t.columns.indexOf("accuracy");
  const row = t.rows.find((r) => r[ti] === target);
  return row ? { f1: row[fi] as number, acc: row[ai] as number, n: row[t.columns.indexOf("n")] as number } : null;
}

export default function PerformancePage() {
  const [m, setM] = useState<Metrics | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [target, setTarget] = useState<(typeof TARGETS)[number]["id"]>("nova_group");

  useEffect(() => {
    api.metrics().then(setM).catch((e) => setError((e as Error).message));
  }, []);

  if (error) return <p className="border-l-2 border-nova4 pl-3 text-sm text-nova4">{error}</p>;
  if (!m) return <p className="text-sm text-muted">Loading results…</p>;

  const cur = m.current;
  const prev = m.previous;
  const unseen = cur?.unseen ?? prev?.unseen ?? null;

  return (
    <div className="space-y-10">
      <div className="max-w-3xl">
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">Model performance</h1>
        <p className="mt-2 text-[15px] leading-relaxed text-muted">
          All scores are on products the models never saw: brands held out from training, and USDA products whose
          barcodes were removed from the training pool. The headline metric is macro-F1, the average F1 over classes,
          because NOVA 4 makes up about two thirds of the data and plain accuracy would hide weak small classes.
        </p>
      </div>

      {!cur && prev && (
        <p className="max-w-3xl rounded-xl border border-nova3/30 bg-warn-soft px-4 py-3 text-sm text-warn">
          Showing the earlier training run (v0.2, TF-IDF + logistic regression). The result tables for the current voting
          ensemble have not been copied to this machine yet (reports/tables).
        </p>
      )}

      {unseen && (
        <div className="grid gap-4 sm:grid-cols-3">
          {TARGETS.map((t) => {
            const h = headline(unseen, t.id);
            return (
              <div key={t.id} className="glass p-5">
                <p className="label">{t.name}</p>
                <p className="mt-2 font-mono text-4xl font-medium text-accent">{h ? h.f1.toFixed(3) : "—"}</p>
                <p className="mt-1 text-xs text-muted">
                  macro-F1 · accuracy {h ? h.acc.toFixed(3) : "—"} · {h ? h.n.toLocaleString() : "—"} unseen USDA products
                </p>
              </div>
            );
          })}
        </div>
      )}

      {cur && (
        <>
          <div className="flex flex-wrap gap-2">
            {TARGETS.map((t) => (
              <button
                key={t.id}
                onClick={() => setTarget(t.id)}
                className={`rounded-full border px-4 py-1.5 text-sm transition-colors ${
                  target === t.id ? "border-accent bg-accent text-on-accent font-medium" : "border-white/10 bg-white/[0.03] text-muted hover:text-ink"
                }`}
              >
                {t.name}
              </button>
            ))}
          </div>
          <Block
            title="Baselines, models, tuning and ensembles"
            sub="Each row is trained on the same brand-held-out split. Baselines show what a trivial rule achieves; the ensemble soft-votes the four tuned models."
          >
            <DataTable table={cur.comparison[target]} sortBy="macro_f1" />
          </Block>
          <div className="grid gap-10 lg:grid-cols-2">
            <Block title="Confusion matrix of the best model" sub="Rows are true classes, shares are row-normalised.">
              <ConfusionMatrix data={cur.confusion[target]} />
            </Block>
            <Block title="Per-class scores" sub="Small classes (NOVA 2) are where imbalance hurts most.">
              <DataTable table={cur.per_class[target]} highlight="f1" />
            </Block>
          </div>
          {cur.imbalance[target] && (
            <Block
              title="Imbalance handling"
              sub="Same model, different ways of dealing with the rare classes. SMOTE is used only on the dense engineered features."
            >
              <DataTable table={cur.imbalance[target]} />
            </Block>
          )}
          {cur.cross_dataset[target] && (
            <Block title="Second dataset: USDA FoodData Central" sub="Trained on Open Food Facts, tested on USDA ingredient statements.">
              <DataTable table={cur.cross_dataset[target]} />
            </Block>
          )}
          {cur.ablation && (
            <Block title="Machine learning vs rules vs hybrid (NOVA)" sub="Contribution of each half of the system.">
              <DataTable table={cur.ablation} />
            </Block>
          )}
          {cur.robustness && (
            <Block title="Robustness to OCR errors" sub="Letters corrupted at random, as a bad photo would.">
              <DataTable table={cur.robustness} />
            </Block>
          )}
          {cur.alternatives && (
            <Block title="Alternatives engine">
              <DataTable table={cur.alternatives} highlight="precision@5" />
            </Block>
          )}
        </>
      )}

      {!cur && prev && (
        <>
          <Block
            title="Baselines and models (NOVA)"
            sub="Internal Open Food Facts test set from the earlier run. The keyword rule and majority class show how much the learned models add."
          >
            <DataTable table={prev.baselines_nova} sortBy="macro_f1" />
          </Block>
          <div className="grid gap-10 lg:grid-cols-2">
            <Block title="Confusion matrix, NOVA (unseen USDA products)">
              <ConfusionMatrix data={prev.confusion_nova} />
            </Block>
            <Block title="Per-class scores, NOVA">
              <DataTable table={prev.per_class_nova} highlight="f1" />
            </Block>
            <Block title="Per-class scores, health tier">
              <DataTable table={prev.per_class_tier} highlight="f1" />
            </Block>
            <Block title="Per-class scores, food group">
              <DataTable table={prev.per_class_category} highlight="f1" />
            </Block>
          </div>
        </>
      )}

      {m.figures.length > 0 && (
        <Block title="Exploratory analysis and explanations">
          <div className="grid gap-6 md:grid-cols-2">
            {m.figures
              .filter((f) => FIGURE_TITLES[f])
              .map((f) => (
                <figure key={f} className="rounded-xl border border-white/10 bg-white p-3">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={`/api/figures/${f}`} alt={FIGURE_TITLES[f]} className="w-full" />
                  <figcaption className="mt-2 text-xs text-neutral-600">{FIGURE_TITLES[f]}</figcaption>
                </figure>
              ))}
          </div>
        </Block>
      )}
    </div>
  );
}
