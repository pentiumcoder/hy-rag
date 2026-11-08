import { splitCitations } from "../format";

interface Props {
  text: string;
  contextCount: number;
  flagged: Set<number>;
  onCite: (n: number) => void;
}

const BULLET = /^\s*[-*•]\s+/;

/** The answer with every [n] turned into a button that jumps to passage n. Lines starting with "- " become a list. */
export function AnswerText({ text, contextCount, flagged, onCite }: Props) {
  const inline = (line: string, key: string) =>
    splitCitations(line).map((part, i) =>
      part.kind === "text" ? (
        <span key={`${key}-${i}`}>{part.text}</span>
      ) : (
        <span key={`${key}-${i}`} className="whitespace-nowrap">
          {part.numbers.map((n) => {
            const exists = n >= 1 && n <= contextCount;
            const bad = flagged.has(n) || !exists;
            return (
              <button
                key={n}
                type="button"
                disabled={!exists}
                onClick={() => onCite(n)}
                title={
                  !exists
                    ? `Passage ${n} does not exist`
                    : flagged.has(n)
                      ? `Passage ${n} does not support every sentence citing it`
                      : `Show passage ${n}`
                }
                className={`ml-0.5 align-super font-mono text-[0.7rem] ${
                  bad ? "text-bad line-through decoration-bad" : "text-accent underline-offset-2 hover:underline"
                } disabled:cursor-not-allowed`}
              >
                [{n}]
              </button>
            );
          })}
        </span>
      ),
    );

  // Group consecutive bullet lines into one list; everything else stays a paragraph.
  const blocks: { list: boolean; lines: string[] }[] = [];
  for (const line of text.split("\n")) {
    const list = BULLET.test(line);
    if (!list && !line.trim()) continue;
    const last = blocks[blocks.length - 1];
    if (last && last.list && list) last.lines.push(line.replace(BULLET, ""));
    else blocks.push({ list, lines: [list ? line.replace(BULLET, "") : line] });
  }

  return (
    <div data-testid="answer-text" className="space-y-3 font-serif text-[1.05rem] leading-8 text-ink">
      {blocks.map((b, i) =>
        b.list ? (
          <ul key={i} className="list-disc space-y-1 pl-6 marker:text-muted">
            {b.lines.map((line, j) => (
              <li key={j}>{inline(line, `${i}-${j}`)}</li>
            ))}
          </ul>
        ) : (
          <p key={i}>{inline(b.lines[0]!, `${i}`)}</p>
        ),
      )}
    </div>
  );
}
