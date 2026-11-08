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
    hint: "How many of each sentence's citations hold up when checked; wrong ones count against it",
  },
  {
    key: "completeness",
    label: "Completeness",
    weight: 0.2,
    hint: "Does the answer cover the whole question (judge)",
  },
];

function tone(value: number): string {
  if (value >= 0.75) return "bg-ok";
  if (value >= 0.5) return "bg-warn";
  return "bg-bad";
}

export function ConfidenceBars({ confidence }: { confidence: Confidence }) {
  return (
    <div className="space-y-3 text-sm" aria-label="Confidence breakdown">
      <p className="flex items-baseline justify-between">
        <span className="text-muted">Confidence</span>
        <span className="font-serif text-3xl tabular-nums" data-testid="confidence-score">
          {pct(confidence.score)}
        </span>
      </p>
      {PARTS.map(({ key, label, weight, hint }) => {
        const value = confidence[key] as number;
        return (
          <div key={key} title={hint}>
            <p className="flex justify-between text-xs text-muted">
              <span>
                {label} × {weight}
              </span>
              <span className="font-mono">{pct(value)}</span>
            </p>
            <div className="mt-1 h-1 bg-line">
              <div
                role="meter"
                aria-label={label}
                aria-valuemin={0}
                aria-valuemax={1}
                aria-valuenow={value}
                className={`h-1 ${tone(value)}`}
                style={{ width: pct(value) }}
              />
            </div>
          </div>
        );
      })}
      <p className="text-xs text-muted">
        Retrieval signal: {confidence.retrieval_signal}
        {confidence.answerability !== null &&
          `. Judge says the passages answer it: ${
            confidence.answerability === 1 ? "yes" : confidence.answerability === 0 ? "no" : "partly"
          }.`}
      </p>
    </div>
  );
}
