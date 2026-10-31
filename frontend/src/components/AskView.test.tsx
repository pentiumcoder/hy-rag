import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { ANSWER, mockApi } from "../test/fixtures";
import { AskView } from "./AskView";

const sent = (init: RequestInit | undefined) => JSON.parse(String(init?.body)) as Record<string, unknown>;

describe("AskView", () => {
  it("asks with the chosen options", async () => {
    const fetchMock = mockApi({ "POST /v1/ask": () => ({ body: ANSWER }) });
    render(<AskView />);
    await userEvent.type(screen.getByLabelText("Question"), "batch too large?");
    await userEvent.selectOptions(screen.getByLabelText("Chunking strategy"), "fixed");
    await userEvent.click(screen.getByLabelText("Verify citations"));
    await userEvent.click(screen.getByLabelText("Rerank"));
    await userEvent.click(screen.getByRole("button", { name: "Ask" }));

    expect(await screen.findByTestId("answer-text")).toHaveTextContent("You get NW-4012");
    expect(sent(fetchMock.mock.calls[0]![1])).toEqual({
      query: "batch too large?",
      overrides: { strategy: "fixed", top_k: 5, rerank: false, verify: false, mode: "hybrid" },
    });
  });

  it("compares hybrid and dense side by side", async () => {
    const fetchMock = mockApi({
      "POST /v1/ask": (init) => {
        const mode = (sent(init).overrides as { mode: string }).mode;
        return { body: { ...ANSWER, mode, answer: `Answer from ${mode} [1].` } };
      },
    });
    render(<AskView />);
    await userEvent.click(screen.getByLabelText("Compare hybrid vs dense"));
    expect(screen.getByLabelText("Retrieval mode")).toBeDisabled();
    await userEvent.click(screen.getByRole("button", { name: /How many PTO days/ }));

    expect(await screen.findByText(/Answer from dense/)).toBeInTheDocument();
    expect(screen.getByText(/Answer from hybrid/)).toBeInTheDocument();
    expect(screen.getByText("Hybrid (BM25 + dense)")).toBeInTheDocument();
    expect(screen.getByText("Dense only")).toBeInTheDocument();
    expect(fetchMock.mock.calls.map(([, init]) => (sent(init).overrides as { mode: string }).mode)).toEqual([
      "hybrid",
      "dense",
    ]);
  });

  it("shows the API's error message", async () => {
    mockApi({
      "POST /v1/ask": () => ({
        status: 503,
        body: { detail: "Cannot reach Ollama at http://localhost:11434. Start it with `ollama serve`." },
      }),
    });
    render(<AskView />);
    await userEvent.type(screen.getByLabelText("Question"), "anything{Enter}");
    expect(await screen.findByRole("alert")).toHaveTextContent("ollama serve");
  });
});
