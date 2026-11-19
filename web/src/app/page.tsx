"use client";

import dynamic from "next/dynamic";
import { motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import Result from "@/components/scan/Result";
import { api, pretty, type Analysis } from "@/lib/api";

const Scene3D = dynamic(() => import("@/components/Scene3D"), { ssr: false });

const NOVA_HEX = ["#3ddc97", "#c7d94a", "#f4a340", "#ff5d5d"];

const EXAMPLES = [
  {
    name: "Cream biscuits",
    text: "Sugar, wheat flour, palm oil, glucose syrup, emulsifier (soy lecithin), flavouring, colour (E102)",
  },
  {
    name: "Cola",
    text: "Carbonated water, sugar, colour (caramel E150d), phosphoric acid, natural flavourings, caffeine",
  },
  { name: "Rolled oats", text: "Rolled oats" },
  { name: "Tomato sauce", text: "Tomatoes, olive oil, salt" },
  {
    name: "Chicken nuggets",
    text: "Mechanically separated chicken, water, wheat flour, modified maize starch, salt, dextrose, sunflower oil, flavouring, sodium phosphate",
  },
  {
    name: "Banaspati ghee",
    text: "Partially hydrogenated vegetable oil, refined palm oil, antioxidant (TBHQ), vitamin A, vitamin D, natural flavour",
  },
];

const STATS = [
  ["254k", "products trained on"],
  ["3", "AI classifiers"],
  ["58", "documented ingredients"],
  ["12", "expert rules"],
];

const ease = [0.16, 1, 0.3, 1] as const;

export default function ScanPage() {
  const [text, setText] = useState("");
  const [prefs, setPrefs] = useState<string[]>([]);
  const [prefOptions, setPrefOptions] = useState<{ id: string; label: string }[]>([]);
  const [result, setResult] = useState<Analysis | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ocrState, setOcrState] = useState<{ busy: boolean; msg: string | null; bad?: boolean }>({ busy: false, msg: null });
  const fileRef = useRef<HTMLInputElement>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let tries = 0;
    const load = () =>
      api.preferences().then(setPrefOptions).catch(() => {
        if (tries++ < 40) setTimeout(load, 3000);
      });
    load();
  }, []);

  const analyze = async () => {
    if (!text.trim() || busy) return;
    setBusy(true);
    setError(null);
    try {
      const r = await api.analyze(text, prefs);
      setResult(r);
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 80);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const onPhoto = async (file: File | undefined) => {
    if (!file) return;
    setOcrState({ busy: true, msg: "Reading the label…" });
    try {
      const r = await api.ocr(file);
      if (!r.text.trim()) {
        setOcrState({ busy: false, msg: "No text found. Retake the photo closer to the label, or paste the list.", bad: true });
        return;
      }
      setText(r.text);
      setOcrState({
        busy: false,
        msg: r.found_marker
          ? "Text read from the photo. Check it and correct any mistakes before analysing."
          : "No “Ingredients” heading found, so the full text was kept. Remove anything that is not an ingredient.",
      });
    } catch (e) {
      setOcrState({ busy: false, msg: (e as Error).message, bad: true });
    } finally {
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  const sceneColor = result ? NOVA_HEX[result.nova.final - 1] : "#3ddc97";

  return (
    <div className="space-y-12">
      {/* hero */}
      <section className="relative grid items-center gap-6 pt-4 lg:grid-cols-[1.05fr_1fr]">
        <div className="relative z-10">
          <motion.p
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease }}
            className="inline-flex items-center gap-2 rounded-full border border-accent/30 bg-accent/10 px-3 py-1 text-xs text-accent"
          >
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" /> Hybrid AI · machine learning + expert system
          </motion.p>
          <motion.h1
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.08, ease }}
            className="mt-5 text-4xl font-semibold leading-[1.08] tracking-tight sm:text-6xl"
          >
            Know what&apos;s
            <br />
            <span className="glow-text">inside your food.</span>
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.16, ease }}
            className="mt-5 max-w-lg text-[15px] leading-relaxed text-muted"
          >
            Paste the ingredient list from any packaged food, or snap the label. NutriScan predicts how processed it is,
            its Nutri-Score health band and food group, explains why, and flags ingredients with a cited source.
          </motion.p>
          <motion.dl
            initial="hidden"
            animate="show"
            variants={{ show: { transition: { staggerChildren: 0.08, delayChildren: 0.3 } } }}
            className="mt-8 grid max-w-lg grid-cols-4 gap-3"
          >
            {STATS.map(([n, l]) => (
              <motion.div
                key={l}
                variants={{ hidden: { opacity: 0, y: 14 }, show: { opacity: 1, y: 0, transition: { duration: 0.6, ease } } }}
              >
                <dt className="font-mono text-2xl font-medium text-ink">{n}</dt>
                <dd className="mt-0.5 text-[11px] leading-tight text-faint">{l}</dd>
              </motion.div>
            ))}
          </motion.dl>
        </div>
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 1.2, ease }}
          className="relative -my-6 h-[320px] sm:h-[420px]"
        >
          <div
            className="absolute inset-[18%] rounded-full blur-3xl transition-colors duration-1000"
            style={{ background: `${sceneColor}22` }}
          />
          <Scene3D color={sceneColor} busy={busy || ocrState.busy} />
        </motion.div>
      </section>

      {/* input */}
      <motion.section
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.25, ease }}
        className="grid gap-6 lg:grid-cols-[1fr_280px]"
      >
        <div className="glass p-5 sm:p-6">
          <div className="flex items-baseline justify-between">
            <label htmlFor="ingredients" className="label">
              Ingredient list
            </label>
            <span className="font-mono text-[11px] text-faint">{text.length} / 5000</span>
          </div>
          <textarea
            id="ingredients"
            value={text}
            maxLength={5000}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) analyze();
            }}
            rows={5}
            placeholder="e.g. Sugar, wheat flour, palm oil, glucose syrup, emulsifier (soy lecithin), flavouring"
            className="mt-3 w-full resize-y rounded-xl border border-white/[0.08] bg-black/30 p-4 text-[15px] leading-relaxed outline-none transition-colors placeholder:text-faint focus:border-accent/60 focus:bg-black/40"
          />
          {ocrState.msg && (
            <p className={`mt-2 text-sm ${ocrState.bad ? "text-nova4" : "text-muted"}`}>{ocrState.msg}</p>
          )}
          <div className="mt-4 flex flex-wrap items-center gap-3">
            <motion.button
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              onClick={analyze}
              disabled={busy || !text.trim()}
              className="relative overflow-hidden rounded-xl bg-accent px-6 py-3 text-sm font-semibold text-on-accent shadow-[0_8px_30px_-6px_rgba(61,220,151,0.6)] disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none"
            >
              {busy ? (
                <span className="flex items-center gap-2">
                  <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-on-accent/30 border-t-on-accent" />
                  Analysing…
                </span>
              ) : (
                "Analyse ingredients"
              )}
            </motion.button>
            <motion.button
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              onClick={() => fileRef.current?.click()}
              disabled={ocrState.busy}
              className="rounded-xl border border-white/[0.12] bg-white/[0.04] px-5 py-3 text-sm transition-colors hover:border-accent/50 hover:bg-accent/10 disabled:opacity-40"
            >
              {ocrState.busy ? "Reading photo…" : "Upload label photo"}
            </motion.button>
            <input
              ref={fileRef}
              type="file"
              accept="image/png,image/jpeg"
              className="hidden"
              onChange={(e) => onPhoto(e.target.files?.[0])}
            />
            {text && (
              <button
                onClick={() => {
                  setText("");
                  setResult(null);
                  setOcrState({ busy: false, msg: null });
                }}
                className="text-sm text-muted underline-offset-2 hover:text-ink hover:underline"
              >
                Clear
              </button>
            )}
            <span className="text-xs text-faint">Photos are processed in memory and not stored.</span>
          </div>
          {busy && (
            <p className="mt-3 text-sm text-muted">
              Running the parser, three classifiers and the rule system. This takes a few seconds on this machine.
            </p>
          )}
          {error && <p className="mt-3 border-l-2 border-nova4 pl-3 text-sm text-nova4">{error}</p>}
        </div>

        <aside className="glass space-y-6 p-5">
          <div>
            <p className="label">Try an example</p>
            <div className="mt-3 flex flex-wrap gap-2">
              {EXAMPLES.map((ex) => (
                <motion.button
                  key={ex.name}
                  whileHover={{ y: -2 }}
                  whileTap={{ scale: 0.96 }}
                  onClick={() => {
                    setText(ex.text);
                    setOcrState({ busy: false, msg: null });
                  }}
                  className={`rounded-full border px-3 py-1.5 text-xs transition-colors ${
                    text === ex.text
                      ? "border-accent bg-accent/15 text-accent"
                      : "border-white/[0.1] bg-white/[0.03] text-muted hover:border-accent/40 hover:text-ink"
                  }`}
                >
                  {ex.name}
                </motion.button>
              ))}
            </div>
          </div>
          {prefOptions.length > 0 && (
            <div>
              <p className="label">Exclude from alternatives</p>
              <div className="mt-3 space-y-2">
                {prefOptions.map((p) => {
                  const on = prefs.includes(p.id);
                  return (
                    <button
                      key={p.id}
                      onClick={() => setPrefs((s) => (on ? s.filter((x) => x !== p.id) : [...s, p.id]))}
                      className="flex w-full items-center justify-between text-sm"
                    >
                      <span className={on ? "text-ink" : "text-muted"}>{pretty(p.id)}</span>
                      <span className={`relative h-5 w-9 rounded-full transition-colors ${on ? "bg-accent" : "bg-white/10"}`}>
                        <motion.span
                          layout
                          transition={{ type: "spring", stiffness: 500, damping: 32 }}
                          className={`absolute top-0.5 h-4 w-4 rounded-full bg-white ${on ? "right-0.5" : "left-0.5"}`}
                        />
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </aside>
      </motion.section>

      <div ref={resultRef} className="scroll-mt-20">
        {result && <Result key={result.id} a={result} />}
      </div>
    </div>
  );
}
