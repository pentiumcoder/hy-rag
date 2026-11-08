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
    <section className="border-t border-line pt-3">
      <h3 className="mb-2 text-sm font-medium text-muted">{title}</h3>
      {children}
    </section>
  );
}

/** One answer: text with clickable citations, verdicts, confidence and the passages behind it. */
export function AnswerPanel({ response: r, idPrefix, title }: Props) {
  const [highlight, setHighlight] = useState<{ n: number; seq: number } | null>(null);
  const cited = useMemo(() => new Set(r.citations.map((c) => c.number)), [r.citations]);
  const flagged = useMemo(() => new Set(r.citations.filter((c) => c.flagged).map((c) => c.number)), [r.citations]);

  const cite = (n: number) => {
    setHighlight((prev) => ({ n, seq: (prev?.seq ?? 0) + 1 }));
    document.getElementById(passageId(idPrefix, n))?.scrollIntoView?.({ behavior: "smooth", block: "center" });
  };

  return (
    <div className="@container min-w-0 space-y-6" data-testid={`answer-${idPrefix}`}>
      {title && (
        <h2 className="flex items-baseline gap-3">
          <span className="font-serif text-lg">{title}</span>
          <span className="font-mono text-xs text-muted">
            {r.mode} / {r.strategy}
          </span>
        </h2>
      )}
      <div className="grid gap-x-10 gap-y-6 @2xl:grid-cols-[minmax(0,1fr)_15rem]">
        <div>
          {r.abstained && r.abstention ? (
            <AbstentionPanel abstention={r.abstention} />
          ) : (
            <AnswerText text={r.answer ?? ""} contextCount={r.contexts.length} flagged={flagged} onCite={cite} />
          )}
          {r.warnings.length > 0 && (
            <ul className="mt-4 space-y-1 text-xs text-warn">
              {r.warnings.map((w) => (
                <li key={w}>{w}</li>
              ))}
            </ul>
          )}
        </div>
        <ConfidenceBars confidence={r.confidence} />
      </div>
      {r.claims.length > 0 && (
        <Section title="Citation checks">
          <ClaimList claims={r.claims} checks={r.checks} onCite={cite} />
        </Section>
      )}
      <Section title={`Passages (${r.contexts.length})`}>
        <ChunkTable contexts={r.contexts} idPrefix={idPrefix} cited={cited} flagged={flagged} highlight={highlight} />
      </Section>
      <p className="font-mono text-xs text-muted">
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
