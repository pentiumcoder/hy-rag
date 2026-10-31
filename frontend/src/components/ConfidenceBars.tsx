import { pct } from "../format";
import type { Confidence } from "../types";

const PARTS: { key: keyof Confidence; label: string; weight: number; hint: string }[] = [
  {
    key: "retrieval",
    label: "Retrieval",
    weight: 0.4,
    hint: "How well the best passage matches the question (cross-encoder)",
  },
  {
    key: "citation_coverage",
    label: "Citations",
    weight: 0.4,
    hint: "Share of sentences whose cited passage supports them (judge)",
  },
  {
    key: "completeness",
    label: "Completeness",
    weight: 0.2,
    hint: "Does the answer cover the whole question (judge)",
  },
];

function tone(value: number): string {
  if (value >= 0.75) return "bg-emerald-500";
  if (value >= 0.5) return "bg-amber-500";
  return "bg-rose-500";
}

export function ConfidenceBars({ confidence }: { confidence: Confidence }) {
  return (
    <div className="space-y-2" aria-label="Confidence breakdown">
      <div className="flex items-baseline justify-between">
        <span className="text-sm font-medium text-slate-600 dark:text-slate-300">Confidence</span>
        <span className="text-2xl font-semibold tabular-nums" data-testid="confidence-score">
          {pct(confidence.score)}
        </span>
      </div>
      {PARTS.map(({ key, label, weight, hint }) => {
        const value = confidence[key] as number;
        return (
          <div key={key} title={hint}>
            <div className="mb-0.5 flex justify-between text-xs text-slate-500 dark:text-slate-400">
              <span>
                {label} <span className="opacity-70">× {weight}</span>
              </span>
              <span className="tabular-nums">{pct(value)}</span>
            </div>
            <div className="h-1.5 rounded-full bg-slate-200 dark:bg-slate-800">
              <div
                role="meter"
                aria-label={label}
                aria-valuemin={0}
                aria-valuemax={1}
                aria-valuenow={value}
                className={`h-1.5 rounded-full ${tone(value)}`}
                style={{ width: pct(value) }}
              />
            </div>
          </div>
        );
      })}
      <p className="text-xs text-slate-500 dark:text-slate-400">
        Retrieval signal: {confidence.retrieval_signal}
        {confidence.answerability !== null &&
          ` · judge says the passages answer it: ${
            confidence.answerability === 1 ? "yes" : confidence.answerability === 0 ? "no" : "partly"
          }`}
      </p>
    </div>
  );
}
