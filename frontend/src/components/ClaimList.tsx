import { VERDICT_STYLES } from "../format";
import type { CitationCheck, Claim } from "../types";

interface Props {
  claims: Claim[];
  checks: CitationCheck[];
  onCite: (n: number) => void;
}

/** Each sentence of the answer with the judge's verdict on its citations. */
export function ClaimList({ claims, checks, onCite }: Props) {
  if (claims.length === 0) return null;
  return (
    <ol className="space-y-2">
      {claims.map((claim) => {
        const label = claim.label ?? "gap";
        const failed = checks.filter((c) => c.claim === claim.index && c.label === "UNSUPPORTED");
        return (
          <li key={claim.index} className="flex gap-3 text-sm">
            <span
              className={`h-fit shrink-0 rounded px-1.5 py-0.5 text-[10px] font-semibold tracking-wide ${VERDICT_STYLES[label]}`}
            >
              {label === "gap" ? "NOT COVERED" : label}
            </span>
            <div className="min-w-0">
              <p className="text-slate-700 dark:text-slate-200">
                {claim.text}{" "}
                {claim.citations.map((n) => (
                  <button
                    key={n}
                    type="button"
                    onClick={() => onCite(n)}
                    className="font-mono text-xs text-indigo-600 hover:underline dark:text-indigo-400"
                  >
                    [{n}]
                  </button>
                ))}
                {claim.inherited && (
                  <span
                    className="ml-1 text-xs text-slate-400"
                    title="The model cited only the end of this list; the citation was applied to each line and still checked"
                  >
                    (citation from list end)
                  </span>
                )}
              </p>
              {failed.map((c) => (
                <p key={c.citation} className="text-xs text-rose-600 dark:text-rose-400">
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
