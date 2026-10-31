import { lazy, Suspense, useEffect, useState } from "react";
import { api } from "./api";
import { AskView } from "./components/AskView";
import { DocumentsView } from "./components/DocumentsView";
import type { Health } from "./types";

// Recharts is most of the bundle; load it only when the Evaluation tab is opened.
const EvalView = lazy(() => import("./components/EvalView").then((m) => ({ default: m.EvalView })));

const TABS = [
  { id: "ask", label: "Ask" },
  { id: "eval", label: "Evaluation" },
  { id: "docs", label: "Documents" },
] as const;
type Tab = (typeof TABS)[number]["id"];

function HealthBadge() {
  const [health, setHealth] = useState<Health | null>(null);
  const [down, setDown] = useState(false);

  useEffect(() => {
    api
      .health()
      .then(setHealth)
      .catch(() => setDown(true));
  }, []);

  if (down) {
    return (
      <span className="rounded-full bg-rose-100 px-2.5 py-1 text-xs font-medium text-rose-700 dark:bg-rose-950 dark:text-rose-300">
        API unreachable
      </span>
    );
  }
  if (!health) return null;
  const ok = health.status === "ok";
  return (
    <span
      title={health.detail ?? `indexes: ${JSON.stringify(health.indexes)}`}
      className={`rounded-full px-2.5 py-1 text-xs font-medium ${
        ok
          ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300"
          : "bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300"
      }`}
    >
      {ok ? "●" : "▲"} {health.generator} · judge {health.judge}
    </span>
  );
}

export default function App() {
  const [tab, setTab] = useState<Tab>("ask");

  return (
    <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6">
      <header className="mb-6 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">
            hy-rag <span className="font-normal text-slate-500">· Northwind Cloud docs</span>
          </h1>
          <p className="text-sm text-slate-500 dark:text-slate-400">
            Hybrid retrieval, verified citations, honest abstention. Runs locally.
          </p>
        </div>
        <HealthBadge />
      </header>

      <nav className="mb-6 flex gap-1 border-b border-slate-200 dark:border-slate-800" role="tablist">
        {TABS.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={tab === t.id}
            onClick={() => setTab(t.id)}
            className={`-mb-px border-b-2 px-4 py-2 text-sm font-medium ${
              tab === t.id
                ? "border-indigo-600 text-indigo-700 dark:text-indigo-300"
                : "border-transparent text-slate-500 hover:text-slate-800 dark:hover:text-slate-200"
            }`}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main>
        {tab === "ask" && <AskView />}
        {tab === "eval" && (
          <Suspense fallback={<p className="text-sm text-slate-500">Loading charts…</p>}>
            <EvalView />
          </Suspense>
        )}
        {tab === "docs" && <DocumentsView />}
      </main>
    </div>
  );
}
