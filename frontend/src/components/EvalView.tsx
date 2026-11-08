import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, ApiError } from "../api";
import { pct } from "../format";
import type { EvalSummary } from "../types";

/** Chart colours come from the theme's CSS variables, so they follow light and dark mode. */
function cssVar(name: string, fallback: string): string {
  if (typeof window === "undefined") return fallback;
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

function Section({ title, children, note }: { title: string; children: React.ReactNode; note?: React.ReactNode }) {
  return (
    <section className="border-t border-line pt-4">
      <h3 className="font-serif text-lg">{title}</h3>
      {note && <p className="mt-1 mb-4 max-w-3xl text-sm text-muted">{note}</p>}
      {children}
    </section>
  );
}

/** Plain names for the ablation rows the evaluation writes. */
function ablationLabel(name: string): string {
  if (name.startsWith("no abstention")) return "no declining";
  if (name.includes("retrieval gate")) return "retrieval threshold + answering model";
  if (name.includes("judge")) return "+ checker: do the docs answer it?";
  if (name === "tuned") return "thresholds tuned on this test set";
  return name;
}

/** recall@5 per chunking strategy, one bar per search mode. */
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

  if (error) return <p className="text-sm text-bad">{error}</p>;
  if (!summary) return <p className="text-sm text-muted">Loading evaluation results…</p>;

  const hasRerank = summary.retrieval.configs.some((c) => c.rerank);
  const chart = retrievalChartData(summary, rerank && hasRerank);
  const compared = chart.filter((row) => row.hybrid !== undefined && row.sparse !== undefined);
  const bm25AsGood = compared.length > 0 && compared.every((row) => Number(row.hybrid) <= Number(row.sparse));
  const generation = summary.generation;
  const unanswerable = summary.golden.by_type.unanswerable ?? 0;
  const colors = {
    hybrid: cssVar("--accent", "#a3471b"),
    sparse: cssVar("--ok", "#2f6b3a"),
    dense: cssVar("--warn", "#8a5a00"),
    bad: cssVar("--bad", "#a12a2a"),
    muted: cssVar("--muted", "#6b665c"),
    line: cssVar("--line", "#dcd6ca"),
  };
  const tick = { fontSize: 12, fill: colors.muted };

  return (
    <div className="space-y-10">
      <p className="max-w-3xl text-sm text-muted">
        {summary.golden.questions} test questions (
        {Object.entries(summary.golden.by_type)
          .map(([t, n]) => `${n} ${t.replace("_", "-")}`)
          .join(", ")}
        ), run {summary.generated_at} with {Object.values(summary.models).filter(Boolean).join(", ")}. The questions and
        the sample documents were written by the same person, so expect real questions to do worse.
      </p>

      <Section
        title="How much of the evidence is in the top 5"
        note={
          bm25AsGood
            ? "On this test set BM25 on its own does as well as hybrid search, or better. The questions reuse a lot of the documents' wording, which suits keyword search; dense search on its own is the weakest."
            : "Share of each answerable question's evidence that shows up in the top 5 passages."
        }
      >
        {hasRerank && (
          <label className="mb-3 flex items-center gap-2 text-sm text-muted">
            <input type="checkbox" checked={rerank} onChange={(e) => setRerank(e.target.checked)} />
            with cross-encoder rerank
          </label>
        )}
        <div className="h-64" data-testid="retrieval-chart">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chart} margin={{ left: -10 }}>
              <CartesianGrid stroke={colors.line} vertical={false} />
              <XAxis dataKey="strategy" tick={tick} stroke={colors.line} />
              <YAxis
                domain={[0, 1]}
                ticks={[0, 0.25, 0.5, 0.75, 1]}
                tickFormatter={(v: number) => pct(v)}
                tick={tick}
                stroke={colors.line}
              />
              <Tooltip formatter={(v: number) => pct(v, 1)} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              {(["hybrid", "sparse", "dense"] as const).map((m) => (
                <Bar key={m} dataKey={m} name={m === "sparse" ? "BM25" : m} fill={colors[m]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Section>

      {generation && (
        <>
          <Section
            title="Answers"
            note={`Correctness is graded by a 7B model against reference answers, and it makes mistakes. "Declined" counts the ${unanswerable} questions the documents can't answer, asked once per setup.`}
          >
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="text-left text-xs text-muted">
                  <tr>
                    <th className="py-2 pr-3 font-normal">Setup</th>
                    <th className="py-2 pr-3 text-right font-normal">Correct</th>
                    <th className="py-2 pr-3 text-right font-normal">Backed by a citation</th>
                    <th className="py-2 pr-3 text-right font-normal">Citations that hold up</th>
                    <th className="py-2 pr-3 text-right font-normal">Answered</th>
                    <th className="py-2 text-right font-normal">Unanswerable declined</th>
                  </tr>
                </thead>
                <tbody className="font-mono text-xs">
                  {generation.configs.map((r) => (
                    <tr key={r.config} className="border-t border-line">
                      <td className="py-2 pr-3">{r.config}</td>
                      <td className="py-2 pr-3 text-right">{pct(r.correctness, 1)}</td>
                      <td className="py-2 pr-3 text-right">{pct(r.faithfulness, 1)}</td>
                      <td className="py-2 pr-3 text-right">{pct(r.citation_accuracy, 1)}</td>
                      <td className="py-2 pr-3 text-right">{pct(r.answered_rate, 1)}</td>
                      <td className="py-2 text-right">
                        {Math.round(r.abstention_recall * unanswerable)} of {unanswerable}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>

          <Section
            title="Wrong answers shown, by declining rule"
            note={`All setups together: ${generation.configs.length} × ${summary.golden.questions} = ${
              generation.configs.length * summary.golden.questions
            } answers. A wrong answer counts for any question the documents can't answer.`}
          >
            <div className="h-48" data-testid="ablation-chart">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={generation.abstention.ablation.map((a) => ({
                    ...a,
                    name: ablationLabel(a.name),
                  }))}
                  layout="vertical"
                  margin={{ left: 40, right: 20 }}
                >
                  <CartesianGrid stroke={colors.line} horizontal={false} />
                  <XAxis type="number" tick={tick} stroke={colors.line} />
                  <YAxis
                    type="category"
                    dataKey="name"
                    width={200}
                    tick={{ ...tick, fontSize: 11 }}
                    stroke={colors.line}
                  />
                  <Tooltip />
                  <Bar dataKey="wrong" name="wrong answers shown" fill={colors.bad} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Section>
        </>
      )}
    </div>
  );
}
