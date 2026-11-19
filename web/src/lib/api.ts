// Types and fetch helpers for the NutriScan Python API (runtime/api/server.py), proxied under /api.

export type Status = {
  ready: boolean;
  error: string | null;
  load_seconds: number | null;
  version: string;
  data_source: string | null;
  model: string | null;
  trained_rows: number | null;
  trained_at: string | null;
  gallery_size: number;
  gallery_synthetic: boolean | null;
  kb_entries: number | null;
  rules: number | null;
  ocr_available: boolean;
  chat_online: boolean;
  suggestions_available: boolean;
};

export type Finding = {
  kind: "rule" | "additive";
  id: string;
  message: string;
  severity: number;
  source: string;
  evidence: string[];
};

export type Driver = { term: string; value: number };

export type Alternative = {
  name: string;
  brand: string;
  category: string;
  similarity: number;
  nova_group: number | null;
  health_tier: string;
  reason: string;
};

export type Analysis = {
  id: string;
  seconds: number;
  ingredients: string[];
  nova: { final: number; ml: number; confidence: number; rule_reasons: string[] };
  tier: { value: "good" | "medium" | "poor"; confidence: number };
  category: string;
  findings: Finding[];
  allergens: { group: string; words: string[] }[];
  drivers: { nova: Driver[]; tier: Driver[] };
  alternatives: Alternative[];
  alternatives_note: string | null;
  versions: Record<string, string | number | null>;
};

export type Suggestion = { name: string; brand: string; why_better: string; likely_nova: number | null };
export type Suggestions = { available: boolean; items: Suggestion[]; model: string; message: string | null };

export type CleaningEntry = {
  step: string; id: string; dataset: string; issue: string; fix: string; why: string;
  before: number; after: number; unit: string; kind: "removed" | "fixed" | "kept";
};
export type Cleaning = {
  log: CleaningEntry[] | null;
  log_synthetic: boolean | null;
  funnel: { dataset: string; steps: { label: string; value: number | null; approx?: boolean; source: string }[] }[];
  examples: { title: string; raw: string; normalised: string; ingredients: { name: string; position: number; depth: number }[] }[];
};

export type Table = { columns: string[]; rows: (string | number | null)[][] } | null;
export type Confusion = { labels: string[]; matrix: number[][] } | null;

export type Metrics = {
  current: {
    comparison: Record<string, Table>;
    per_class: Record<string, Table>;
    confusion: Record<string, Confusion>;
    imbalance: Record<string, Table>;
    cross_dataset: Record<string, Table>;
    unseen: Table;
    ablation: Table;
    robustness: Table;
    alternatives: Table;
  } | null;
  previous: {
    unseen: Table;
    baselines_nova: Table;
    per_class_nova: Table;
    per_class_tier: Table;
    per_class_category: Table;
    confusion_nova: Confusion;
  } | null;
  figures: string[];
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch {
    throw new Error("Cannot reach the NutriScan API. Is the Python server running on port 8000?");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {}
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  status: () => request<Status>("/api/status", { cache: "no-store" }),
  preferences: () => request<{ id: string; label: string }[]>("/api/preferences"),
  analyze: (text: string, prefs: string[]) =>
    request<Analysis>("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, prefs }),
    }),
  ocr: (file: File) => {
    const form = new FormData();
    form.append("image", file);
    return request<{ text: string; raw: string; found_marker: boolean }>("/api/ocr", { method: "POST", body: form });
  },
  chat: (id: string, question: string) =>
    request<{ answer: string; mode: string }>("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id, question }),
    }),
  suggest: (id: string) =>
    request<Suggestions>("/api/suggest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id }),
    }),
  cleaning: () => request<Cleaning>("/api/cleaning", { cache: "no-store" }),
  metrics: () => request<Metrics>("/api/metrics", { cache: "no-store" }),
};

export const NOVA = [
  { group: 1, name: "Unprocessed", detail: "Whole or minimally processed foods" },
  { group: 2, name: "Culinary ingredient", detail: "Oils, sugar, salt, butter" },
  { group: 3, name: "Processed", detail: "Foods with added salt, sugar or oil" },
  { group: 4, name: "Ultra-processed", detail: "Industrial formulations with additives" },
] as const;

export const TIERS = [
  { id: "good", name: "Good", detail: "Nutri-Score A–B" },
  { id: "medium", name: "Medium", detail: "Nutri-Score C" },
  { id: "poor", name: "Poor", detail: "Nutri-Score D–E" },
] as const;

export const pretty = (s: string) => {
  const t = s.replace(/[-_]/g, " ");
  return t.charAt(0).toUpperCase() + t.slice(1);
};
export const pct = (x: number) => `${Math.round(x * 100)}%`;
