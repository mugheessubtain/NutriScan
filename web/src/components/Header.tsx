"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { api, type Status } from "@/lib/api";

const NAV = [
  { href: "/", label: "Scan" },
  { href: "/cleaning", label: "Data cleaning" },
  { href: "/performance", label: "Model performance" },
  { href: "/about", label: "Method" },
];

export default function Header() {
  const path = usePathname();
  const [status, setStatus] = useState<Status | null>(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    let alive = true;
    const poll = async () => {
      let next = 3000;
      try {
        const s = await api.status();
        if (!alive) return;
        setStatus(s);
        setOffline(false);
        if (s.ready || s.error) next = 30000;
      } catch {
        if (alive) setOffline(true);
      }
      timer = setTimeout(poll, next);
    };
    poll();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, []);

  let dot = "bg-faint";
  let text = "Connecting to API…";
  if (offline) {
    dot = "bg-nova4";
    text = "API not running";
  } else if (status?.error) {
    dot = "bg-nova4";
    text = "Model failed to load";
  } else if (status && !status.ready) {
    dot = "bg-nova3 animate-pulse";
    text = "Loading models…";
  } else if (status?.ready) {
    dot = "bg-nova1";
    text = `${status.model === "voting" ? "Voting ensemble" : status.model} · ${
      status.trained_rows ? `${status.trained_rows.toLocaleString()} training products` : status.data_source
    }`;
  }

  return (
    <header className="sticky top-0 z-40 border-b border-white/[0.06] bg-paper/70 backdrop-blur-xl">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-8 gap-y-2 px-4 py-3 sm:px-6">
        <Link href="/" className="group flex items-center gap-2.5">
          <span className="relative grid h-7 w-7 place-items-center rounded-lg bg-accent/15 ring-1 ring-accent/40">
            <span className="h-2.5 w-2.5 rounded-full bg-accent shadow-[0_0_14px_3px_rgba(61,220,151,0.6)] transition-transform group-hover:scale-125" />
          </span>
          <span className="text-[15px] font-semibold tracking-tight">NutriScan</span>
          <span className="font-mono text-[11px] text-faint">v{status?.version ?? "0.3"}</span>
        </Link>
        <nav className="flex gap-1 rounded-full border border-white/[0.07] bg-white/[0.03] p-1 text-sm">
          {NAV.map((n) => (
            <Link
              key={n.href}
              href={n.href}
              className={`rounded-full px-3.5 py-1.5 transition-colors ${
                path === n.href ? "bg-accent text-on-accent font-medium" : "text-muted hover:text-ink"
              }`}
            >
              {n.label}
            </Link>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-2 text-xs text-muted" title={status?.error ?? undefined}>
          <span className={`inline-block h-1.5 w-1.5 rounded-full ${dot}`} />
          {text}
        </div>
      </div>
    </header>
  );
}
