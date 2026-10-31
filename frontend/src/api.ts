import type {
  AnswerResponse,
  AskOverrides,
  DocumentInfo,
  EvalSummary,
  Health,
  IngestResponse,
} from "./types";

// Same-origin by default: Vite (dev) and nginx (Docker) forward /api to the FastAPI service.
export const API_BASE: string = import.meta.env.VITE_API_BASE ?? "/api";

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new ApiError(0, "Cannot reach the API. Is `hyrag serve` running?");
  }
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
      else if (body.detail) detail = JSON.stringify(body.detail);
    } catch {
      // keep the status text
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request<Health>("/health"),
  ask: (query: string, overrides: AskOverrides = {}) =>
    request<AnswerResponse>("/v1/ask", json({ query, overrides })),
  documents: () => request<DocumentInfo[]>("/v1/documents"),
  ingest: (files: File[]) => {
    const form = new FormData();
    for (const file of files) form.append("files", file);
    return request<IngestResponse>("/v1/ingest", { method: "POST", body: form });
  },
  evalSummary: () => request<EvalSummary>("/v1/eval/summary"),
};
