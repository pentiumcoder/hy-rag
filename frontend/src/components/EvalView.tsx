import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, ApiError } from "../api";
import { pct } from "../format";
import type { EvalSummary } from "../types";

const MODE_COLORS: Record<string, string> = {
  hybrid: "#4f46e5",
  sparse: "#0891b2",
  dense: "#f59e0b",
};

function Card({ title, children, note }: { title: string; children: React.ReactNode; note?: string }) {
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <h3 className="text-sm font-semibold">{title}</h3>
      {note && <p className="mb-3 text-xs text-slate-500 dark:text-slate-400">{note}</p>}
      {children}
    </section>
  );
}

/** recall@5 per strategy, one bar per mode, for the reranked or the raw retrievers. */
function retrievalChartData(summary: EvalSummary, rerank: boolean) {
  const rows = summary.retrieval.configs.filter((c) => c.rerank === rerank);
  const strategies = [...new Set(rows.map((r) => r.strategy))].sort();
  return strategies.map((strategy) => {
    const entry: Record<string, string | number> = { strategy };
    for (const r of rows.filter((x) => x.strategy === strategy)) entry[r.mode] = r.recall_at_5;
    return entry;
  });
}

export function EvalView() {
  const [summary, setSummary] = useState<EvalSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [rerank, setRerank] = useState(true);

  useEffect(() => {
    api
      .evalSummary()
      .then(setSummary)
      .catch((e: unknown) => setError(e instanceof ApiError ? e.message : String(e)));
  }, []);

  if (error) return <p className="text-sm text-rose-600 dark:text-rose-400">{error}</p>;
  if (!summary) return <p className="text-sm text-slate-500">Loading evaluation results…</p>;

  const hasRerank = summary.retrieval.configs.some((c) => c.rerank);
  const showRerank = rerank && hasRerank;
  const chart = retrievalChartData(summary, showRerank);
  const generation = summary.generation;

  return (
    <div className="space-y-6">
      <p className="text-sm text-slate-600 dark:text-slate-300">
        {summary.golden.questions} golden questions (
        {Object.entries(summary.golden.by_type)
          .map(([t, n]) => `${n} ${t.replace("_", "-")}`)
          .join(", ")}
        ), run {summary.generated_at} with{" "}
        {Object.values(summary.models).filter(Boolean).join(", ")}.
      </p>

      <Card
        title="Retrieval recall@5 by chunking strategy and mode"
        note="Share of each answerable question's gold evidence found in the top 5 passages."
      >
        {hasRerank && (
          <label className="mb-2 flex items-center gap-2 text-xs text-slate-600 dark:text-slate-300">
            <input type="checkbox" checked={rerank} onChange={(e) => setRerank(e.target.checked)} />
            with cross-encoder rerank
          </label>
        )}
        <div className="h-64" data-testid="retrieval-chart">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chart} margin={{ left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} opacity={0.4} />
              <XAxis dataKey="strategy" tick={{ fontSize: 12 }} />
              <YAxis
                domain={[0.5, 1]}
                ticks={[0.5, 0.6, 0.7, 0.8, 0.9, 1]}
                tickFormatter={(v: number) => pct(v)}
                tick={{ fontSize: 12 }}
                allowDataOverflow
              />
              <Tooltip formatter={(v: number) => pct(v, 1)} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              {(["hybrid", "sparse", "dense"] as const).map((m) => (
                <Bar key={m} dataKey={m} fill={MODE_COLORS[m]} radius={[3, 3, 0, 0]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>

      {generation && (
        <>
          <Card
            title="Answer quality"
            note={`With the tuned abstention policy: retrieval gate ${generation.policy.abstain_threshold}, judge check ${
              generation.policy.judge_check ? "on" : "off"
            }, confidence floor ${generation.policy.answer_threshold}.`}
          >
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="text-left text-xs uppercase tracking-wide text-slate-500">
                  <tr>
                    <th className="py-2 pr-3">Config</th>
                    <th className="py-2 pr-3 text-right">Correct</th>
                    <th className="py-2 pr-3 text-right">Faithful</th>
                    <th className="py-2 pr-3 text-right">Citations</th>
                    <th className="py-2 pr-3 text-right">Answered</th>
                    <th className="py-2 text-right">Abstention P / R</th>
                  </tr>
                </thead>
                <tbody className="tabular-nums">
                  {generation.configs.map((r) => (
                    <tr key={r.config} className="border-t border-slate-200 dark:border-slate-800">
                      <td className="py-2 pr-3 font-mono text-xs">{r.config}</td>
                      <td className="py-2 pr-3 text-right font-semibold">{pct(r.correctness, 1)}</td>
                      <td className="py-2 pr-3 text-right">{pct(r.faithfulness, 1)}</td>
                      <td className="py-2 pr-3 text-right">{pct(r.citation_accuracy, 1)}</td>
                      <td className="py-2 pr-3 text-right">{pct(r.answered_rate, 1)}</td>
                      <td className="py-2 text-right">
                        {pct(r.abstention_precision)} / {pct(r.abstention_recall)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card
            title="What each abstention layer adds"
            note={`Pooled over all configs. Utility: +1 per correct answer, −${generation.abstention.wrong_answer_cost} per wrong one, per question.`}
          >
            <div className="h-56" data-testid="ablation-chart">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={generation.abstention.ablation}
                  layout="vertical"
                  margin={{ left: 40, right: 20 }}
                >
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} opacity={0.4} />
                  <XAxis type="number" tick={{ fontSize: 12 }} />
                  <YAxis type="category" dataKey="name" width={200} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="wrong" name="answered wrongly" fill="#e11d48" radius={[0, 3, 3, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
