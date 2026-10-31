import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import App from "../App";
import { mockApi, SUMMARY } from "../test/fixtures";
import { DocumentsView } from "./DocumentsView";
import { EvalView } from "./EvalView";

const DOCS = [
  { doc_id: "a", source: "hr/pto-policy.md", title: "PTO Policy", format: "markdown", sections: 5, chars: 1276 },
  { doc_id: "b", source: "finance/travel-and-expense.pdf", title: "Travel", format: "pdf", sections: 7, chars: 2053 },
];

describe("EvalView", () => {
  it("renders the answer table, the retrieval chart and the ablation", async () => {
    mockApi({ "GET /v1/eval/summary": () => ({ body: SUMMARY }) });
    render(<EvalView />);
    expect(await screen.findByText("recursive/hybrid")).toBeInTheDocument();
    expect(screen.getByText("91.3%")).toBeInTheDocument();
    expect(screen.getByText(/55 golden questions/)).toBeInTheDocument();
    expect(screen.getByText(/retrieval gate 0.3, judge check on/)).toBeInTheDocument();
    expect(screen.getByTestId("retrieval-chart")).toBeInTheDocument();
    expect(screen.getByTestId("ablation-chart")).toBeInTheDocument();
    await userEvent.click(screen.getByLabelText("with cross-encoder rerank"));
    expect(screen.getByLabelText("with cross-encoder rerank")).not.toBeChecked();
  });

  it("explains when there are no results yet", async () => {
    mockApi({
      "GET /v1/eval/summary": () => ({
        status: 404,
        body: { detail: "No evaluation results yet. Run `hyrag eval`." },
      }),
    });
    render(<EvalView />);
    expect(await screen.findByText(/Run `hyrag eval`/)).toBeInTheDocument();
  });
});

describe("DocumentsView", () => {
  it("lists documents and uploads new ones", async () => {
    const fetchMock = mockApi({
      "GET /v1/documents": () => ({ body: DOCS }),
      "POST /v1/ingest": () => ({
        body: {
          added: ["uploads/coffee.md"],
          updated: [],
          unchanged: [],
          skipped: [],
          indexes: [{ strategy: "recursive", added: 1, removed: 0, unchanged: 86, duplicates: 1, chunks: 87 }],
        },
      }),
    });
    render(<DocumentsView />);
    expect(await screen.findByText("2 documents")).toBeInTheDocument();
    expect(screen.getByText("hr/pto-policy.md")).toBeInTheDocument();

    const file = new File(["# Coffee"], "coffee.md", { type: "text/markdown" });
    await userEvent.upload(screen.getByLabelText("Upload documents"), file);

    expect(await screen.findByText(/1 added, 0 updated, 0 unchanged/)).toBeInTheDocument();
    expect(screen.getByText(/recursive: \+1 chunks \(87 total, 1 duplicates\)/)).toBeInTheDocument();
    const upload = fetchMock.mock.calls.find(([, init]) => init?.method === "POST")!;
    const form = upload[1]!.body as FormData;
    expect((form.get("files") as File).name).toBe("coffee.md");
  });

  it("shows upload errors from the API", async () => {
    mockApi({
      "GET /v1/documents": () => ({ body: [] }),
      "POST /v1/ingest": () => ({ status: 415, body: { detail: "Unsupported file 'x.exe'" } }),
    });
    render(<DocumentsView />);
    await screen.findByText("0 documents");
    await userEvent.upload(
      screen.getByLabelText("Upload documents"),
      new File(["x"], "x.md", { type: "text/markdown" }),
    );
    expect(await screen.findByRole("alert")).toHaveTextContent("Unsupported file");
  });
});

describe("App", () => {
  it("shows model health and switches tabs", async () => {
    mockApi({
      "GET /health": () => ({
        body: {
          status: "ok",
          version: "1.0.0",
          llm_provider: "ollama",
          generator: "qwen2.5:7b",
          judge: "llama3.1:8b",
          indexes: { recursive: 86 },
          detail: null,
        },
      }),
      "GET /v1/documents": () => ({ body: DOCS }),
    });
    render(<App />);
    expect(await screen.findByText(/qwen2.5:7b · judge llama3.1:8b/)).toBeInTheDocument();
    expect(screen.getByLabelText("Question")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("tab", { name: "Documents" }));
    expect(await screen.findByText("2 documents")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Documents" })).toHaveAttribute("aria-selected", "true");
  });

  it("says when the API is unreachable", async () => {
    mockApi({});
    render(<App />);
    const header = screen.getByRole("banner");
    expect(await within(header).findByText("API unreachable")).toBeInTheDocument();
  });
});
