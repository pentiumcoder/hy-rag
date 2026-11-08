import type { ClaimLabel } from "./types";

export const pct = (value: number, digits = 0): string => `${(value * 100).toFixed(digits)}%`;

export const num = (value: number | null | undefined, digits = 2): string =>
  value === null || value === undefined ? "–" : value.toFixed(digits);

export const seconds = (ms: number | undefined): string =>
  ms === undefined ? "–" : ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;

/** "a > b > c" section paths are long; the last part is what a reader recognizes. */
export const shortSection = (section: string | null): string =>
  section ? (section.split(" > ").at(-1) ?? section) : "";

/** How each verdict reads in the UI: a symbol, a word and a colour, not a pill. */
export const VERDICTS: Record<ClaimLabel | "gap", { mark: string; text: string; tone: string }> = {
  SUPPORTED: { mark: "✓", text: "supported", tone: "text-ok" },
  PARTIAL: { mark: "~", text: "partly supported", tone: "text-warn" },
  UNVERIFIED: { mark: "?", text: "not checked", tone: "text-muted" },
  UNSUPPORTED: { mark: "✗", text: "not supported", tone: "text-bad" },
  UNCITED: { mark: "✗", text: "no citation", tone: "text-bad" },
  gap: { mark: "–", text: "not in the documents", tone: "text-muted" },
};

export const ABSTENTION_TITLES: Record<string, string> = {
  no_results: "Nothing was retrieved",
  low_retrieval_confidence: "No passage matched this question well enough",
  insufficient_context: "The documents don't answer this",
  unanswerable: "Answer withdrawn: the passages don't support it",
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
