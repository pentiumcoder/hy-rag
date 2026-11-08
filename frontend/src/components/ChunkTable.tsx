import { useState } from "react";
import { num, shortSection } from "../format";
import type { ScoredChunk } from "../types";

interface Props {
  contexts: ScoredChunk[];
  idPrefix: string;
  cited: Set<number>;
  flagged: Set<number>;
  highlight: { n: number; seq: number } | null; // seq restarts the flash on repeat clicks
}

export const passageId = (prefix: string, n: number) => `${prefix}-passage-${n}`;

function Stage({ score, rank, digits }: { score: number | null; rank: number | null; digits: number }) {
  if (score === null) return <span className="text-muted">–</span>;
  return (
    <span className="font-mono text-xs">
      {num(score, digits)}
      {rank !== null && <span className="ml-1 text-muted">#{rank}</span>}
    </span>
  );
}

/** The retrieved passages, numbered as the answer cites them, with every stage's score. */
export function ChunkTable({ contexts, idPrefix, cited, flagged, highlight }: Props) {
  const [open, setOpen] = useState<Set<number>>(new Set());
  const toggle = (n: number) =>
    setOpen((prev) => {
      const next = new Set(prev);
      if (next.has(n)) next.delete(n);
      else next.add(n);
      return next;
    });

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="text-xs text-muted">
          <tr>
            <th className="py-2 pr-3 font-normal">#</th>
            <th className="py-2 pr-3 font-normal">Source</th>
            <th className="py-2 pr-3 text-right font-normal" title="Cosine similarity and rank from dense search">
              Dense
            </th>
            <th className="py-2 pr-3 text-right font-normal" title="BM25 score and rank from keyword search">
              BM25
            </th>
            <th className="py-2 pr-3 text-right font-normal" title="Reciprocal Rank Fusion score">
              RRF
            </th>
            <th className="py-2 text-right font-normal" title="Cross-encoder logit (final order)">
              Rerank
            </th>
          </tr>
        </thead>
        <tbody>
          {contexts.map((sc, i) => {
            const n = i + 1;
            const isOpen = open.has(n);
            return (
              <tr
                key={`${n}-${highlight?.n === n ? highlight.seq : 0}`}
                id={passageId(idPrefix, n)}
                className={`border-t border-line align-top ${
                  highlight?.n === n ? "flash" : ""
                }`}
              >
                <td className="py-2 pr-3 font-mono text-xs">
                  <span
                    className={
                      flagged.has(n) ? "text-bad" : cited.has(n) ? "text-accent" : "text-muted"
                    }
                  >
                    [{n}]
                  </span>
                </td>
                <td className="max-w-md py-2 pr-3">
                  <button
                    type="button"
                    onClick={() => toggle(n)}
                    aria-expanded={isOpen}
                    className="text-left hover:underline"
                  >
                    <span>{sc.chunk.source}</span>
                    {sc.chunk.page !== null && <span className="text-muted"> p.{sc.chunk.page}</span>}
                    <span className="block text-xs text-muted">
                      {shortSection(sc.chunk.section_heading)}
                    </span>
                  </button>
                  {isOpen && (
                    <p className="mt-2 whitespace-pre-wrap border-l-2 border-line pl-3 font-serif text-sm leading-6">
                      {sc.chunk.text}
                    </p>
                  )}
                </td>
                <td className="py-2 pr-3 text-right">
                  <Stage score={sc.dense_score} rank={sc.dense_rank} digits={3} />
                </td>
                <td className="py-2 pr-3 text-right">
                  <Stage score={sc.sparse_score} rank={sc.sparse_rank} digits={2} />
                </td>
                <td className="py-2 pr-3 text-right font-mono text-xs">{num(sc.rrf_score, 4)}</td>
                <td className="py-2 text-right font-mono text-xs">{num(sc.rerank_score, 2)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
