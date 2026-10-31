import { useState } from "react";
import { api, ApiError } from "../api";
import type { AnswerResponse, AskOverrides, ChunkStrategy, RetrievalMode } from "../types";
import { AnswerPanel } from "./AnswerPanel";

const EXAMPLES = [
  "What error do I get when a batch exceeds ingest.max_batch_size?",
  "How many PTO days carry over to next year?",
  "What should I do when CertExpiringSoon fires?",
  "How much does the Team plan cost?",
];

type Result =
  | { status: "loading" }
  | { status: "done"; response: AnswerResponse }
  | { status: "error"; message: string };

function describe(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  return error instanceof Error ? error.message : String(error);
}

export function AskView() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<RetrievalMode>("hybrid");
  const [strategy, setStrategy] = useState<ChunkStrategy>("recursive");
  const [topK, setTopK] = useState(5);
  const [verify, setVerify] = useState(true);
  const [rerank, setRerank] = useState(true);
  const [compare, setCompare] = useState(false);
  const [results, setResults] = useState<{ label: string; result: Result }[]>([]);

  const busy = results.some((r) => r.result.status === "loading");

  async function run(question: string) {
    const q = question.trim();
    if (!q || busy) return;
    const base: AskOverrides = { strategy, top_k: topK, rerank, verify };
    const runs: { label: string; overrides: AskOverrides }[] = compare
      ? [
          { label: "Hybrid (BM25 + dense)", overrides: { ...base, mode: "hybrid" } },
          { label: "Dense only", overrides: { ...base, mode: "dense" } },
        ]
      : [{ label: "", overrides: { ...base, mode } }];
    setResults(runs.map(({ label }) => ({ label, result: { status: "loading" } })));
    // One after the other: on a single GPU parallel requests would only queue in Ollama.
    for (const [i, { overrides }] of runs.entries()) {
      let result: Result;
      try {
        result = { status: "done", response: await api.ask(q, overrides) };
      } catch (error) {
        result = { status: "error", message: describe(error) };
      }
      setResults((prev) => prev.map((r, j) => (j === i ? { ...r, result } : r)));
    }
  }

  const select =
    "rounded-md border border-slate-300 bg-white px-2 py-1.5 text-sm dark:border-slate-700 dark:bg-slate-900";

  return (
    <div className="space-y-6">
      <form
        className="space-y-3"
        onSubmit={(e) => {
          e.preventDefault();
          void run(query);
        }}
      >
        <div className="flex gap-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask about Northwind Cloud's runbooks, policies, APIs…"
            aria-label="Question"
            className="min-w-0 flex-1 rounded-lg border border-slate-300 bg-white px-4 py-2.5 shadow-sm outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/30 dark:border-slate-700 dark:bg-slate-900"
          />
          <button
            type="submit"
            disabled={busy || !query.trim()}
            className="rounded-lg bg-indigo-600 px-5 py-2.5 font-medium text-white shadow-sm hover:bg-indigo-500 disabled:opacity-50"
          >
            {busy ? "Answering…" : "Ask"}
          </button>
        </div>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm text-slate-600 dark:text-slate-300">
          <label className="flex items-center gap-2">
            Mode
            <select
              aria-label="Retrieval mode"
              value={mode}
              disabled={compare}
              onChange={(e) => setMode(e.target.value as RetrievalMode)}
              className={select}
            >
              <option value="hybrid">hybrid</option>
              <option value="dense">dense</option>
              <option value="sparse">sparse (BM25)</option>
            </select>
          </label>
          <label className="flex items-center gap-2">
            Chunks
            <select
              aria-label="Chunking strategy"
              value={strategy}
              onChange={(e) => setStrategy(e.target.value as ChunkStrategy)}
              className={select}
            >
              <option value="recursive">recursive</option>
              <option value="fixed">fixed</option>
              <option value="semantic">semantic</option>
            </select>
          </label>
          <label className="flex items-center gap-2">
            Passages
            <input
              type="number"
              min={1}
              max={20}
              value={topK}
              onChange={(e) => setTopK(Math.min(20, Math.max(1, Number(e.target.value) || 1)))}
              className={`${select} w-16`}
              aria-label="Number of passages"
            />
          </label>
          <label
            className="flex items-center gap-2"
            title="The cross-encoder reorders the top 20 candidates; turn it off to see raw dense vs hybrid ranking"
          >
            <input type="checkbox" checked={rerank} onChange={(e) => setRerank(e.target.checked)} />
            Rerank
          </label>
          <label className="flex items-center gap-2">
            <input type="checkbox" checked={verify} onChange={(e) => setVerify(e.target.checked)} />
            Verify citations
          </label>
          <label className="flex items-center gap-2 font-medium text-indigo-700 dark:text-indigo-300">
            <input
              type="checkbox"
              checked={compare}
              onChange={(e) => setCompare(e.target.checked)}
            />
            Compare hybrid vs dense
          </label>
        </div>
        <div className="flex flex-wrap gap-2">
          {EXAMPLES.map((example) => (
            <button
              key={example}
              type="button"
              onClick={() => {
                setQuery(example);
                void run(example);
              }}
              disabled={busy}
              className="rounded-full border border-slate-300 px-3 py-1 text-xs text-slate-600 hover:border-indigo-400 hover:text-indigo-700 disabled:opacity-50 dark:border-slate-700 dark:text-slate-300"
            >
              {example}
            </button>
          ))}
        </div>
      </form>

      <div className={results.length > 1 ? "grid gap-6 xl:grid-cols-2" : ""}>
        {results.map(({ label, result }, i) => (
          <div key={label || i} className="min-w-0">
            {result.status === "loading" && (
              <div className="animate-pulse rounded-xl border border-slate-200 bg-white p-6 text-sm text-slate-500 dark:border-slate-800 dark:bg-slate-900">
                {label && <p className="mb-1 font-semibold text-slate-700 dark:text-slate-200">{label}</p>}
                Retrieving, answering and checking citations… local models take 10–30 s.
              </div>
            )}
            {result.status === "error" && (
              <div
                role="alert"
                className="rounded-xl border border-rose-300 bg-rose-50 p-4 text-sm text-rose-800 dark:border-rose-800 dark:bg-rose-950/40 dark:text-rose-200"
              >
                {label && <p className="font-semibold">{label}</p>}
                {result.message}
              </div>
            )}
            {result.status === "done" && (
              <AnswerPanel response={result.response} idPrefix={`r${i}`} title={label || undefined} />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
