import { describe, expect, it } from "vitest";
import { pct, seconds, shortSection, splitCitations } from "./format";

describe("splitCitations", () => {
  it("separates text from citation groups", () => {
    expect(splitCitations("Five days [2]. Expires March 31 [1][3].")).toEqual([
      { kind: "text", text: "Five days " },
      { kind: "cite", numbers: [2] },
      { kind: "text", text: ". Expires March 31 " },
      { kind: "cite", numbers: [1] },
      { kind: "cite", numbers: [3] },
      { kind: "text", text: "." },
    ]);
  });

  it("expands lists and ranges like the backend does", () => {
    const cites = splitCitations("a [1, 4] b [2-4] c [5–6]").filter((p) => p.kind === "cite");
    expect(cites).toEqual([
      { kind: "cite", numbers: [1, 4] },
      { kind: "cite", numbers: [2, 3, 4] },
      { kind: "cite", numbers: [5, 6] },
    ]);
  });

  it("leaves text without citations alone", () => {
    expect(splitCitations("No sources.")).toEqual([{ kind: "text", text: "No sources." }]);
  });
});

describe("formatting", () => {
  it("formats percentages, durations and section paths", () => {
    expect(pct(0.9134, 1)).toBe("91.3%");
    expect(seconds(650)).toBe("650 ms");
    expect(seconds(11400)).toBe("11.4 s");
    expect(seconds(undefined)).toBe("–");
    expect(shortSection("Runbook > When it fires")).toBe("When it fires");
    expect(shortSection(null)).toBe("");
  });
});
