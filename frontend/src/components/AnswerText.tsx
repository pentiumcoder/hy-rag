import { splitCitations } from "../format";

interface Props {
  text: string;
  contextCount: number;
  flagged: Set<number>;
  onCite: (n: number) => void;
}

/** The answer with every [n] turned into a button that jumps to passage n. */
export function AnswerText({ text, contextCount, flagged, onCite }: Props) {
  return (
    <div data-testid="answer-text" className="whitespace-pre-wrap leading-7 text-slate-800 dark:text-slate-100">
      {splitCitations(text).map((part, i) =>
        part.kind === "text" ? (
          <span key={i}>{part.text}</span>
        ) : (
          <span key={i} className="whitespace-nowrap">
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
                  className={`mx-0.5 rounded px-1 align-baseline font-mono text-xs font-semibold transition ${
                    bad
                      ? "bg-rose-100 text-rose-700 line-through decoration-rose-500 hover:bg-rose-200 dark:bg-rose-950 dark:text-rose-300"
                      : "bg-indigo-100 text-indigo-700 hover:bg-indigo-200 dark:bg-indigo-950 dark:text-indigo-300"
                  } disabled:cursor-not-allowed`}
                >
                  [{n}]
                </button>
              );
            })}
          </span>
        ),
      )}
    </div>
  );
}
