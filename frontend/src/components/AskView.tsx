import { useState } from "react";
import { api, ApiError } from "../api";
import type { AnswerResponse, AskOverrides, ChunkStrategy, RetrievalMode } from "../types";
import { AnswerPanel } from "./AnswerPanel";

type Result =
  { status: "loading" } | { status: "done"; response: AnswerResponse } | { status: "error"; message: string };

function describe(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  return error instanceof Error ? error.message : String(error);
}

const field = "border border-line bg-surface px-2 py-1 text-sm text-ink";

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
          {
            label: "Hybrid (BM25 + dense)",
            overrides: { ...base, mode: "hybrid" },
          },
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

  return (
    <div className="space-y-8">
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
            placeholder="Ask a question about your documents"
            aria-label="Question"
            className="min-w-0 flex-1 border border-line bg-surface px-3 py-2 text-ink outline-none placeholder:text-muted focus:border-accent"
          />
          <button
            type="submit"
            disabled={busy || !query.trim()}
            className="bg-accent px-5 py-2 text-surface hover:opacity-90 disabled:opacity-40"
          >
            {busy ? "Answering…" : "Ask"}
          </button>
        </div>
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-sm text-muted">
          <label className="flex items-center gap-2">
            Search
            <select
              aria-label="Retrieval mode"
              value={mode}
              disabled={compare}
              onChange={(e) => setMode(e.target.value as RetrievalMode)}
              className={field}
            >
              <option value="hybrid">hybrid</option>
              <option value="dense">dense</option>
              <option value="sparse">BM25 only</option>
            </select>
          </label>
          <label className="flex items-center gap-2">
            Chunks
            <select
              aria-label="Chunking strategy"
              value={strategy}
              onChange={(e) => setStrategy(e.target.value as ChunkStrategy)}
              className={field}
            >
              <option value="recursive">by heading</option>
              <option value="fixed">fixed size</option>
              <option value="semantic">by topic</option>
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
              className={`${field} w-14`}
              aria-label="Number of passages"
            />
          </label>
          <label
            className="flex items-center gap-1.5"
            title="Rerank the top 20 candidates with the cross-encoder. Turn it off to see the raw retrievers."
          >
            <input type="checkbox" checked={rerank} onChange={(e) => setRerank(e.target.checked)} />
            Rerank
          </label>
          <label className="flex items-center gap-1.5">
            <input type="checkbox" checked={verify} onChange={(e) => setVerify(e.target.checked)} />
            Verify citations
          </label>
          <label className="flex items-center gap-1.5">
            <input type="checkbox" checked={compare} onChange={(e) => setCompare(e.target.checked)} />
            Compare hybrid vs dense
          </label>
        </div>
      </form>

      <div className={results.length > 1 ? "grid gap-10 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]" : ""}>
        {results.map(({ label, result }, i) => (
          <div key={label || i} className="min-w-0">
            {result.status === "loading" && (
              <p className="animate-pulse border-l-2 border-line pl-3 text-sm text-muted">
                {label && <span className="mr-2 text-ink">{label}:</span>}
                Retrieving, answering and checking citations… local models take 10–30 s.
              </p>
            )}
            {result.status === "error" && (
              <div role="alert" className="border-l-2 border-bad bg-bad-soft px-3 py-2 text-sm text-bad">
                {label && <p className="font-medium">{label}</p>}
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
