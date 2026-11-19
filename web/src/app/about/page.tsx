const STEPS = [
  ["Parse", "The label text is cleaned and split into ingredients, keeping brackets and order (earlier means more)."],
  ["Classify", "Three models read the ingredient words: NOVA processing group, Nutri-Score health band and food group. Each is a soft-voting ensemble of logistic regression, a linear SVM, a random forest and gradient boosting."],
  ["Check rules", "An expert system matches 58 documented ingredients and runs 12 IF–THEN rules. Rules can raise the NOVA group to 4, always with a stated reason, and produce the allergen warnings."],
  ["Explain", "SHAP values show which words pushed the model toward its answer."],
  ["Recommend", "Similar products in the same food group with a lower NOVA group or a better health band are ranked by ingredient similarity."],
];

const DATA = [
  ["Open Food Facts", "Crowd-sourced product database with ingredient lists, NOVA groups and Nutri-Score grades. Main training data.", "ODbL"],
  ["USDA FoodData Central (Branded Foods)", "Manufacturer ingredient statements and nutrients. Used as an independent test set and for extra health-tier and food-group labels.", "Public domain"],
  ["Knowledge base", "Our own entries for additives and ingredients, each with a reason and a source (EFSA, EU regulations, WHO, IARC, NOVA papers).", "Project"],
];

const LIMITS = [
  "Open Food Facts derives its NOVA group partly from ingredient markers, so the NOVA model partly re-learns that formula. The health band, which comes from nutrients, is the stronger test of the model.",
  "Most products are ultra-processed (NOVA 4); NOVA 2 is rare, so its scores are the weakest.",
  "The data over-represents some countries and languages; results are English-language labels.",
  "Allergen detection is a keyword match on the parsed list and can miss or over-flag items.",
  "This is a study project. It does not give medical advice.",
];

export default function AboutPage() {
  return (
    <div className="max-w-3xl space-y-10">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">How NutriScan works</h1>
        <p className="mt-2 text-[15px] leading-relaxed text-muted">
          A hybrid system: machine-learning classifiers trained on more than 250,000 products, combined with a
          rule-based expert system that gives cited, auditable reasons.
        </p>
      </div>

      <section>
        <h2 className="label">Pipeline</h2>
        <ol className="glass mt-3 divide-y divide-white/[0.06] px-5">
          {STEPS.map(([name, text], i) => (
            <li key={name} className="grid grid-cols-[32px_110px_1fr] gap-3 py-3 text-sm">
              <span className="font-mono text-accent">{String(i + 1).padStart(2, "0")}</span>
              <span className="font-medium">{name}</span>
              <span className="leading-relaxed text-muted">{text}</span>
            </li>
          ))}
        </ol>
      </section>

      <section>
        <h2 className="label">What the scores mean</h2>
        <div className="mt-3 grid gap-6 text-sm leading-relaxed sm:grid-cols-2">
          <p>
            <span className="font-medium">NOVA</span> groups foods by the extent of industrial processing: 1
            unprocessed, 2 culinary ingredients, 3 processed foods, 4 ultra-processed formulations (Monteiro et al.,
            2019).
          </p>
          <p>
            <span className="font-medium">Health tier</span> groups the Nutri-Score grade, which scores energy, sugars,
            saturated fat and salt against fibre and protein: good (A–B), medium (C), poor (D–E).
          </p>
        </div>
      </section>

      <section>
        <h2 className="label">Data</h2>
        <table className="mt-3 w-full text-sm">
          <tbody>
            {DATA.map(([name, text, lic]) => (
              <tr key={name} className="border-b border-line align-top">
                <td className="py-2.5 pr-4 font-medium">{name}</td>
                <td className="py-2.5 pr-4 leading-relaxed text-muted">{text}</td>
                <td className="whitespace-nowrap py-2.5 text-xs text-faint">{lic}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section>
        <h2 className="label">Privacy</h2>
        <p className="mt-3 text-sm leading-relaxed text-muted">
          No accounts and no personal data. Uploaded photos are read in memory and discarded. The optional question
          box sends only the analysis result, never the photo.
        </p>
      </section>

      <section>
        <h2 className="label">Limitations</h2>
        <ul className="mt-3 list-disc space-y-1.5 pl-5 text-sm leading-relaxed text-muted">
          {LIMITS.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
      </section>
    </div>
  );
}
