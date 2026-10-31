import type { ClaimLabel } from "./types";

export const pct = (value: number, digits = 0): string => `${(value * 100).toFixed(digits)}%`;

export const num = (value: number | null | undefined, digits = 2): string =>
  value === null || value === undefined ? "–" : value.toFixed(digits);

export const seconds = (ms: number | undefined): string =>
  ms === undefined ? "–" : ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;

/** "a > b > c" section paths are long; the last part is what a reader recognizes. */
export const shortSection = (section: string | null): string =>
  section ? (section.split(" > ").at(-1) ?? section) : "";

export const VERDICT_STYLES: Record<ClaimLabel | "gap", string> = {
  SUPPORTED: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300",
  PARTIAL: "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300",
  UNVERIFIED: "bg-slate-200 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
  UNSUPPORTED: "bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-300",
  UNCITED: "bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-300",
  gap: "bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400",
};

export const ABSTENTION_TITLES: Record<string, string> = {
  no_results: "Nothing was retrieved",
  low_retrieval_confidence: "Off-topic for these documents",
  insufficient_context: "The documents don't answer this",
  unanswerable: "Answer withdrawn: not supported by the documents",
  low_answer_confidence: "Answer withdrawn: confidence too low",
};

/** Split an answer into text and citation markers; "[1, 3]" and "[1-2]" become separate numbers. */
export type AnswerPart = { kind: "text"; text: string } | { kind: "cite"; numbers: number[] };

const CITE = /\[(\d+(?:\s*[,\-–]\s*\d+)*)\]/g;

function expand(group: string): number[] {
  const out: number[] = [];
  for (const part of group.split(/\s*,\s*/)) {
    const [a, b] = part.split(/\s*[-–]\s*/).map(Number);
    if (a === undefined || Number.isNaN(a)) continue;
    if (b !== undefined && !Number.isNaN(b) && b > a && b - a < 20) {
      for (let n = a; n <= b; n++) out.push(n);
    } else {
      out.push(a);
    }
  }
  return out;
}

export function splitCitations(answer: string): AnswerPart[] {
  const parts: AnswerPart[] = [];
  let last = 0;
  for (const match of answer.matchAll(CITE)) {
    const at = match.index ?? 0;
    if (at > last) parts.push({ kind: "text", text: answer.slice(last, at) });
    parts.push({ kind: "cite", numbers: expand(match[1] ?? "") });
    last = at + match[0].length;
  }
  if (last < answer.length) parts.push({ kind: "text", text: answer.slice(last) });
  return parts;
}
