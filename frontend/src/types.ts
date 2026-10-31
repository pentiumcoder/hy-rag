// Mirrors the pydantic models in src/hyrag/models.py and src/hyrag/api/schemas.py.

export type RetrievalMode = "dense" | "sparse" | "hybrid";
export type ChunkStrategy = "fixed" | "recursive" | "semantic";
export type VerdictLabel = "SUPPORTED" | "PARTIAL" | "UNSUPPORTED" | "UNVERIFIED";
export type ClaimLabel = VerdictLabel | "UNCITED";

export interface Chunk {
  chunk_id: string;
  doc_id: string;
  source: string;
  title: string;
  text: string;
  strategy: ChunkStrategy;
  chunk_index: number;
  section_heading: string | null;
  page: number | null;
}

export interface ScoredChunk {
  chunk: Chunk;
  rank: number;
  score: number;
  dense_score: number | null;
  dense_rank: number | null;
  sparse_score: number | null;
  sparse_rank: number | null;
  rrf_score: number | null;
  rerank_score: number | null;
}

export interface Claim {
  index: number;
  text: string;
  citations: number[];
  kind: "fact" | "gap";
  inherited: boolean;
  label: ClaimLabel | null;
}

export interface CitationCheck {
  claim: number;
  citation: number;
  label: VerdictLabel;
  reason: string;
}

export interface Citation {
  number: number;
  chunk_id: string;
  source: string;
  title: string;
  section_heading: string | null;
  page: number | null;
  flagged: boolean;
}

export interface Confidence {
  score: number;
  retrieval: number;
  citation_coverage: number;
  completeness: number;
  retrieval_signal: "rerank" | "lexical" | "none";
  answerability: number | null;
}

export interface DocumentRef {
  source: string;
  title: string;
  section_heading: string | null;
  page: number | null;
  score: number | null;
}

export interface Abstention {
  reason:
    | "no_results"
    | "low_retrieval_confidence"
    | "insufficient_context"
    | "unanswerable"
    | "low_answer_confidence";
  message: string;
  found: string[];
  missing: string[];
  suggested_documents: DocumentRef[];
  draft_answer: string | null;
}

export interface AnswerResponse {
  query: string;
  answer: string | null;
  abstained: boolean;
  abstention: Abstention | null;
  claims: Claim[];
  checks: CitationCheck[];
  citations: Citation[];
  confidence: Confidence;
  contexts: ScoredChunk[];
  mode: RetrievalMode;
  strategy: ChunkStrategy;
  generator: string | null;
  judge: string | null;
  timings_ms: Record<string, number>;
  warnings: string[];
}

export interface AskOverrides {
  mode?: RetrievalMode;
  strategy?: ChunkStrategy;
  top_k?: number;
  rerank?: boolean;
  verify?: boolean;
}

export interface Health {
  status: "ok" | "degraded";
  version: string;
  llm_provider: string;
  generator: string;
  judge: string;
  indexes: Record<string, number>;
  detail: string | null;
}

export interface DocumentInfo {
  doc_id: string;
  source: string;
  title: string;
  format: string;
  sections: number;
  chars: number;
}

export interface IndexSyncInfo {
  strategy: ChunkStrategy;
  added: number;
  removed: number;
  unchanged: number;
  duplicates: number;
  chunks: number;
}

export interface IngestResponse {
  added: string[];
  updated: string[];
  unchanged: string[];
  skipped: string[];
  indexes: IndexSyncInfo[];
}

export interface RetrievalSummaryRow {
  config: string;
  strategy: ChunkStrategy;
  mode: RetrievalMode;
  rerank: boolean;
  questions: number;
  recall_at_1: number;
  recall_at_5: number;
  mrr: number;
  recall_at_5_by_type: Record<string, number>;
}

export interface GenerationSummaryRow {
  config: string;
  questions: number;
  answered_rate: number;
  correctness: number;
  correct_overall: number;
  key_fact_recall: number;
  faithfulness: number;
  citation_accuracy: number;
  abstention_precision: number;
  abstention_recall: number;
  abstention_f1: number;
  utility: number;
  median_latency_s: number | null;
}

export interface AblationRow {
  name: string;
  abstention_precision: number;
  abstention_recall: number;
  abstained: number;
  correct: number;
  wrong: number;
  utility: number;
}

export interface EvalSummary {
  version: string;
  generated_at: string;
  mode: string;
  golden: { path: string; questions: number; by_type: Record<string, number> };
  models: Record<string, string | null>;
  retrieval: { k: number; configs: RetrievalSummaryRow[]; best: string | null };
  generation?: {
    configs_run: string[];
    policy: { abstain_threshold: number; answer_threshold: number; judge_check: boolean };
    configs: GenerationSummaryRow[];
    abstention: {
      wrong_answer_cost: number;
      signal_auroc: Record<string, number>;
      ablation: AblationRow[];
    };
  };
}
