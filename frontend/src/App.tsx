import { lazy, Suspense, useEffect, useState } from "react";
import { api } from "./api";
import { AskView } from "./components/AskView";
import { DocumentsView } from "./components/DocumentsView";
import type { Health } from "./types";

// Recharts is most of the bundle; load it only when the Evaluation tab is opened.
const EvalView = lazy(() => import("./components/EvalView").then((m) => ({ default: m.EvalView })));

const TABS = [
  { id: "ask", label: "Ask" },
  { id: "docs", label: "Documents" },
  { id: "eval", label: "Evaluation" },
] as const;
type Tab = (typeof TABS)[number]["id"];

function Status() {
  const [health, setHealth] = useState<Health | null>(null);
  const [down, setDown] = useState(false);

  useEffect(() => {
    api
      .health()
      .then(setHealth)
      .catch(() => setDown(true));
  }, []);

  if (down) return <span className="text-bad">API not reachable</span>;
  if (!health) return <span>connecting…</span>;
  const chunks = Math.max(0, ...Object.values(health.indexes));
  return (
    <span title={health.detail ?? undefined} className={health.status === "ok" ? "" : "text-warn"}>
      {health.generator} answers, {health.judge} checks · {chunks} passages indexed
    </span>
  );
}

export default function App() {
  const [tab, setTab] = useState<Tab>("ask");

  return (
    <div className="mx-auto max-w-6xl px-5 pb-16">
      <header className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1 border-b border-line py-5">
        <h1 className="font-serif text-2xl text-ink">hy-rag</h1>
        <p className="font-mono text-xs text-muted">
          <Status />
        </p>
      </header>

      <nav className="mt-4 mb-8 flex gap-6 text-sm" role="tablist">
        {TABS.map((t) => (
          <button
            key={t.id}
            role="tab"
            aria-selected={tab === t.id}
            onClick={() => setTab(t.id)}
            className={`pb-1 ${
              tab === t.id
                ? "border-b-2 border-accent text-ink"
                : "border-b-2 border-transparent text-muted hover:text-ink"
            }`}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main>
        {tab === "ask" && <AskView />}
        {tab === "docs" && <DocumentsView />}
        {tab === "eval" && (
          <Suspense fallback={<p className="text-sm text-muted">Loading charts…</p>}>
            <EvalView />
          </Suspense>
        )}
      </main>
    </div>
  );
}
