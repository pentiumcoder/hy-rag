import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ABSTAINED, ANSWER } from "../test/fixtures";
import { AnswerPanel } from "./AnswerPanel";
import { passageId } from "./ChunkTable";

describe("AnswerPanel", () => {
  it("turns [n] into buttons that jump to and highlight passage n", async () => {
    const scroll = vi.fn();
    Element.prototype.scrollIntoView = scroll;
    render(<AnswerPanel response={ANSWER} idPrefix="t" />);

    const answer = screen.getByTestId("answer-text");
    await userEvent.click(within(answer).getByRole("button", { name: "[2]" }));

    const row = document.getElementById(passageId("t", 2))!;
    expect(row).toHaveClass("flash");
    expect(scroll).toHaveBeenCalledWith({ behavior: "smooth", block: "center" });
    expect(within(row).getByText("api/rest-api-v2.md")).toBeInTheDocument();
  });

  it("marks citations the judge flagged", () => {
    render(<AnswerPanel response={ANSWER} idPrefix="t" />);
    const answer = screen.getByTestId("answer-text");
    const flagged = within(answer).getByRole("button", { name: "[3]" });
    expect(flagged).toHaveClass("line-through");
    expect(flagged).toHaveAttribute("title", expect.stringMatching(/does not support/));
    expect(within(answer).getByRole("button", { name: "[1]" })).not.toHaveClass("line-through");
  });

  it("shows verdicts, the confidence breakdown, every stage score and timings", async () => {
    render(<AnswerPanel response={ANSWER} idPrefix="t" />);
    expect(screen.getByText("SUPPORTED")).toBeInTheDocument();
    expect(screen.getByText("PARTIAL")).toBeInTheDocument();
    expect(screen.getByText(/\[3\] does not support this: about webhooks/)).toBeInTheDocument();
    expect(screen.getByTestId("confidence-score")).toHaveTextContent("86%");
    expect(screen.getByRole("meter", { name: "Citations" })).toHaveAttribute("aria-valuenow", "0.75");
    expect(screen.getByText("5.40")).toBeInTheDocument(); // rerank logit of passage 1
    expect(screen.getAllByText("#1").length).toBeGreaterThan(0); // dense and BM25 ranks
    expect(screen.getByText(/verify 11.4 s/)).toBeInTheDocument();
    expect(screen.getByText(/Passage \[3\] does not support sentence/)).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /engineering\/error-codes\.md/ }));
    expect(screen.getByText("NW-4012 BATCH_TOO_LARGE: HTTP 413.")).toBeInTheDocument();
  });

  it("shows an abstention with what was found, what is missing and the withdrawn draft", async () => {
    render(<AnswerPanel response={ABSTAINED} idPrefix="t" />);
    expect(screen.getByText(/Answer withdrawn: not supported/)).toBeInTheDocument();
    expect(screen.getByText("The context does not cover a Team plan.")).toBeInTheDocument();
    expect(screen.getByText("product/pricing-and-plans.md")).toBeInTheDocument();
    await userEvent.click(screen.getByText("Show the withdrawn draft"));
    expect(screen.getByText("The Team plan costs $99 [1].")).toBeVisible();
    expect(screen.queryByText("Citation checks")).not.toBeInTheDocument();
  });
});
