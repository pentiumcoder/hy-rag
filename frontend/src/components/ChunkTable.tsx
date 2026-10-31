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
  if (score === null) return <span className="text-slate-400">–</span>;
  return (
    <span className="tabular-nums">
      {num(score, digits)}
      {rank !== null && <span className="ml-1 text-xs text-slate-400">#{rank}</span>}
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
        <thead className="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">
          <tr>
            <th className="py-2 pr-3">#</th>
            <th className="py-2 pr-3">Source</th>
            <th className="py-2 pr-3 text-right" title="Cosine similarity and rank from dense search">
              Dense
            </th>
            <th className="py-2 pr-3 text-right" title="BM25 score and rank from keyword search">
              BM25
            </th>
            <th className="py-2 pr-3 text-right" title="Reciprocal Rank Fusion score">
              RRF
            </th>
            <th className="py-2 text-right" title="Cross-encoder logit (final order)">
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
                className={`border-t border-slate-200 align-top dark:border-slate-800 ${
                  highlight?.n === n ? "flash" : ""
                }`}
              >
                <td className="py-2 pr-3 font-mono text-xs">
                  <span
                    className={`rounded px-1 ${
                      flagged.has(n)
                        ? "bg-rose-100 text-rose-700 dark:bg-rose-950 dark:text-rose-300"
                        : cited.has(n)
                          ? "bg-indigo-100 text-indigo-700 dark:bg-indigo-950 dark:text-indigo-300"
                          : "text-slate-400"
                    }`}
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
                    <span className="font-medium">{sc.chunk.source}</span>
                    {sc.chunk.page !== null && <span className="text-slate-400"> p.{sc.chunk.page}</span>}
                    <span className="block text-xs text-slate-500 dark:text-slate-400">
                      {shortSection(sc.chunk.section_heading)}
                    </span>
                  </button>
                  {isOpen && (
                    <p className="mt-2 whitespace-pre-wrap rounded bg-slate-100 p-2 text-xs leading-5 text-slate-700 dark:bg-slate-900 dark:text-slate-300">
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
                <td className="py-2 pr-3 text-right tabular-nums">{num(sc.rrf_score, 4)}</td>
                <td className="py-2 text-right tabular-nums">{num(sc.rerank_score, 2)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
