import { useMemo, useState } from "react";
import { seconds } from "../format";
import type { AnswerResponse } from "../types";
import { AbstentionPanel } from "./AbstentionPanel";
import { AnswerText } from "./AnswerText";
import { ChunkTable, passageId } from "./ChunkTable";
import { ClaimList } from "./ClaimList";
import { ConfidenceBars } from "./ConfidenceBars";

interface Props {
  response: AnswerResponse;
  idPrefix: string;
  title?: string;
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <h3 className="mb-3 text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
        {title}
      </h3>
      {children}
    </section>
  );
}

/** One answer: text with clickable citations, verdicts, confidence and the passages behind it. */
export function AnswerPanel({ response: r, idPrefix, title }: Props) {
  const [highlight, setHighlight] = useState<{ n: number; seq: number } | null>(null);
  const cited = useMemo(() => new Set(r.citations.map((c) => c.number)), [r.citations]);
  const flagged = useMemo(
    () => new Set(r.citations.filter((c) => c.flagged).map((c) => c.number)),
    [r.citations],
  );

  const cite = (n: number) => {
    setHighlight((prev) => ({ n, seq: (prev?.seq ?? 0) + 1 }));
    document
      .getElementById(passageId(idPrefix, n))
      ?.scrollIntoView?.({ behavior: "smooth", block: "center" });
  };

  return (
    <div className="space-y-4" data-testid={`answer-${idPrefix}`}>
      {title && (
        <h2 className="flex items-center gap-2 text-sm font-semibold">
          {title}
          <span className="rounded bg-slate-200 px-1.5 py-0.5 font-mono text-xs font-normal dark:bg-slate-800">
            {r.mode} / {r.strategy}
          </span>
        </h2>
      )}
      <div className="grid gap-4 lg:grid-cols-[1fr_16rem]">
        <Section title="Answer">
          {r.abstained && r.abstention ? (
            <AbstentionPanel abstention={r.abstention} />
          ) : (
            <AnswerText
              text={r.answer ?? ""}
              contextCount={r.contexts.length}
              flagged={flagged}
              onCite={cite}
            />
          )}
          {r.warnings.length > 0 && (
            <ul className="mt-3 space-y-1 text-xs text-amber-700 dark:text-amber-400">
              {r.warnings.map((w) => (
                <li key={w}>⚠ {w}</li>
              ))}
            </ul>
          )}
        </Section>
        <Section title="Confidence">
          <ConfidenceBars confidence={r.confidence} />
        </Section>
      </div>
      {r.claims.length > 0 && (
        <Section title="Citation checks">
          <ClaimList claims={r.claims} checks={r.checks} onCite={cite} />
        </Section>
      )}
      <Section title={`Retrieved passages (${r.contexts.length})`}>
        <ChunkTable
          contexts={r.contexts}
          idPrefix={idPrefix}
          cited={cited}
          flagged={flagged}
          highlight={highlight}
        />
      </Section>
      <p className="text-xs text-slate-500 dark:text-slate-400">
        {[
          r.generator && `generator ${r.generator}`,
          r.judge && `judge ${r.judge}`,
          ...Object.entries(r.timings_ms).map(([k, v]) => `${k} ${seconds(v)}`),
        ]
          .filter(Boolean)
          .join(" · ")}
      </p>
    </div>
  );
}
