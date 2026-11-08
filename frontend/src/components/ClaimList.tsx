import { VERDICTS } from "../format";
import type { CitationCheck, Claim } from "../types";

interface Props {
  claims: Claim[];
  checks: CitationCheck[];
  onCite: (n: number) => void;
}

/** Each sentence of the answer, and what the judge said about each of its citations. */
export function ClaimList({ claims, checks, onCite }: Props) {
  if (claims.length === 0) return null;
  return (
    <ol className="space-y-3">
      {claims.map((claim) => {
        const verdict = VERDICTS[claim.label ?? "gap"];
        const mine = checks.filter((c) => c.claim === claim.index);
        const failed = mine.filter((c) => c.label === "UNSUPPORTED");
        return (
          <li key={claim.index} className="grid grid-cols-[8.5rem_1fr] gap-3 text-sm">
            <span className={verdict.tone}>
              {verdict.mark} {verdict.text}
            </span>
            <div className="min-w-0">
              <p>
                {claim.text}{" "}
                {claim.citations.map((n) => {
                  const check = mine.find((c) => c.citation === n);
                  const v = check ? VERDICTS[check.label] : VERDICTS.UNVERIFIED;
                  return (
                    <button
                      key={n}
                      type="button"
                      onClick={() => onCite(n)}
                      title={`Passage ${n}: ${v.text}`}
                      className={`mr-1 font-mono text-xs hover:underline ${v.tone}`}
                    >
                      {v.mark}[{n}]
                    </button>
                  );
                })}
                {claim.inherited && (
                  <span
                    className="text-xs text-muted"
                    title="The model cited only the end of this list; the citation was applied to each line and still checked"
                  >
                    (citation from the end of the list)
                  </span>
                )}
              </p>
              {failed.map((c) => (
                <p key={c.citation} className="text-xs text-bad">
                  [{c.citation}] does not support this{c.reason ? `: ${c.reason}` : ""}
                </p>
              ))}
            </div>
          </li>
        );
      })}
    </ol>
  );
}
