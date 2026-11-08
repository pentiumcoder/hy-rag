import { ABSTENTION_TITLES, shortSection } from "../format";
import type { Abstention } from "../types";

export function AbstentionPanel({ abstention }: { abstention: Abstention }) {
  return (
    <div className="border-l-2 border-warn pl-4">
      <h3 className="font-serif text-lg">{ABSTENTION_TITLES[abstention.reason] ?? "No answer"}</h3>
      <p className="mt-1 text-sm text-muted">{abstention.message}</p>
      <dl className="mt-4 grid grid-cols-[8rem_1fr] gap-x-3 gap-y-2 text-sm">
        {abstention.found.length > 0 && (
          <>
            <dt className="text-muted">Found</dt>
            <dd>
              {abstention.found.map((f) => (
                <p key={f}>{f}</p>
              ))}
            </dd>
          </>
        )}
        {abstention.missing.length > 0 && (
          <>
            <dt className="text-muted">Missing</dt>
            <dd>
              {abstention.missing.map((m) => (
                <p key={m}>{m}</p>
              ))}
            </dd>
          </>
        )}
        {abstention.suggested_documents.length > 0 && (
          <>
            <dt className="text-muted">Closest documents</dt>
            <dd>
              {abstention.suggested_documents.map((d) => (
                <p key={d.source}>
                  {d.source}
                  {d.section_heading && (
                    <span className="text-muted"> › {shortSection(d.section_heading)}</span>
                  )}
                </p>
              ))}
            </dd>
          </>
        )}
      </dl>
      {abstention.draft_answer && (
        <details className="mt-4 text-sm">
          <summary className="cursor-pointer text-muted">Show the withdrawn draft</summary>
          <p className="mt-2 whitespace-pre-wrap font-serif text-muted line-through">
            {abstention.draft_answer}
          </p>
        </details>
      )}
    </div>
  );
}
