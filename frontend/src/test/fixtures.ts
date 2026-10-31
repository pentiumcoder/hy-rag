import { vi } from "vitest";
import type { AnswerResponse, EvalSummary, ScoredChunk } from "../types";

function passage(n: number, source: string, text: string, rerank: number): ScoredChunk {
  return {
    chunk: {
      chunk_id: `c${n}`,
      doc_id: `d${n}`,
      source,
      title: source,
      text,
      strategy: "recursive",
      chunk_index: n,
      section_heading: `Doc > Section ${n}`,
      page: null,
    },
    rank: n,
    score: rerank,
    dense_score: n === 3 ? null : 0.6 - n / 100,
    dense_rank: n === 3 ? null : n,
    sparse_score: 7 - n,
    sparse_rank: n,
    rrf_score: 0.016 - n / 1000,
    rerank_score: rerank,
  };
}

export const ANSWER: AnswerResponse = {
  query: "What error do I get when a batch is too large?",
  answer: "You get NW-4012 BATCH_TOO_LARGE [1]. The limit is 500 events [2][3].",
  abstained: false,
  abstention: null,
  claims: [
    { index: 1, text: "You get NW-4012 BATCH_TOO_LARGE.", citations: [1], kind: "fact", inherited: false, label: "SUPPORTED" },
    { index: 2, text: "The limit is 500 events.", citations: [2, 3], kind: "fact", inherited: false, label: "PARTIAL" },
  ],
  checks: [
    { claim: 1, citation: 1, label: "SUPPORTED", reason: "states the error" },
    { claim: 2, citation: 2, label: "PARTIAL", reason: "default only" },
    { claim: 2, citation: 3, label: "UNSUPPORTED", reason: "about webhooks" },
  ],
  citations: [
    { number: 1, chunk_id: "c1", source: "engineering/error-codes.md", title: "Errors", section_heading: null, page: null, flagged: false },
    { number: 2, chunk_id: "c2", source: "api/rest-api-v2.md", title: "API", section_heading: null, page: null, flagged: false },
    { number: 3, chunk_id: "c3", source: "api/webhooks.html", title: "Webhooks", section_heading: null, page: null, flagged: true },
  ],
  confidence: {
    score: 0.86,
    retrieval: 0.99,
    citation_coverage: 0.75,
    completeness: 1,
    retrieval_signal: "rerank",
    answerability: 1,
  },
  contexts: [
    passage(1, "engineering/error-codes.md", "NW-4012 BATCH_TOO_LARGE: HTTP 413.", 5.4),
    passage(2, "api/rest-api-v2.md", "A batch may contain at most 500 events.", 5.0),
    passage(3, "api/webhooks.html", "Deliveries are retried up to 8 times.", -3.2),
  ],
  mode: "hybrid",
  strategy: "recursive",
  generator: "ollama:qwen2.5:7b",
  judge: "ollama:llama3.1:8b",
  timings_ms: { retrieve: 650, generate: 3800, verify: 11400, total: 15850 },
  warnings: ["Passage [3] does not support sentence(s) 2: about webhooks"],
};

export const ABSTAINED: AnswerResponse = {
  ...ANSWER,
  query: "How much does the Team plan cost?",
  answer: null,
  abstained: true,
  abstention: {
    reason: "unanswerable",
    message: "The judge found that the retrieved passages do not answer the question.",
    found: ["Passages that mention: plan, cost."],
    missing: ["The context does not cover a Team plan."],
    suggested_documents: [
      { source: "product/pricing-and-plans.md", title: "Pricing", section_heading: "Pricing > Plan comparison", page: null, score: -2.6 },
    ],
    draft_answer: "The Team plan costs $99 [1].",
  },
  claims: [],
  checks: [],
  citations: [],
  warnings: [],
};

export const SUMMARY: EvalSummary = {
  version: "0.4.0",
  generated_at: "2026-10-30 04:00 UTC",
  mode: "local models",
  golden: { path: "eval/golden.jsonl", questions: 55, by_type: { lookup: 20, multi_hop: 12 } },
  models: { embedder: "BAAI/bge-small-en-v1.5", reranker: null, generator: "qwen2.5:7b", judge: "llama3.1:8b" },
  retrieval: {
    k: 5,
    best: "recursive/hybrid+rerank",
    configs: [
      { config: "recursive/hybrid+rerank", strategy: "recursive", mode: "hybrid", rerank: true, questions: 47, recall_at_1: 0.68, recall_at_5: 0.989, mrr: 0.876, recall_at_5_by_type: {} },
      { config: "recursive/dense+rerank", strategy: "recursive", mode: "dense", rerank: true, questions: 47, recall_at_1: 0.68, recall_at_5: 0.968, mrr: 0.869, recall_at_5_by_type: {} },
      { config: "recursive/dense", strategy: "recursive", mode: "dense", rerank: false, questions: 47, recall_at_1: 0.6, recall_at_5: 0.84, mrr: 0.78, recall_at_5_by_type: {} },
    ],
  },
  generation: {
    configs_run: ["recursive/hybrid"],
    policy: { abstain_threshold: 0.3, answer_threshold: 0, judge_check: true },
    configs: [
      { config: "recursive/hybrid", questions: 55, answered_rate: 0.979, correctness: 0.913, correct_overall: 0.894, key_fact_recall: 0.9, faithfulness: 0.975, citation_accuracy: 0.839, abstention_precision: 0.889, abstention_recall: 1, abstention_f1: 0.94, utility: 0.7, median_latency_s: null },
    ],
    abstention: {
      wrong_answer_cost: 2,
      signal_auroc: { retrieval_confidence: 0.967 },
      ablation: [
        { name: "no abstention", abstention_precision: 1, abstention_recall: 0, abstained: 0, correct: 161.5, wrong: 41, utility: 0.36 },
        { name: "tuned", abstention_precision: 0.886, abstention_recall: 0.969, abstained: 35, correct: 161.5, wrong: 6, utility: 0.68 },
      ],
    },
  },
};

type Route = (init: RequestInit | undefined) => { status?: number; body: unknown };

/** Stub fetch with one handler per "METHOD /path"; returns the mock for assertions. */
export function mockApi(routes: Record<string, Route>) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    const key = `${init?.method ?? "GET"} ${url.replace(/^\/api/, "")}`;
    const route = routes[key];
    if (!route) throw new Error(`unexpected request ${key}`);
    const { status = 200, body } = route(init);
    return new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}
