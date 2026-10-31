import { ABSTENTION_TITLES, shortSection } from "../format";
import type { Abstention } from "../types";

export function AbstentionPanel({ abstention }: { abstention: Abstention }) {
  return (
    <div className="rounded-lg border border-amber-300 bg-amber-50 p-4 dark:border-amber-800 dark:bg-amber-950/40">
      <h3 className="font-semibold text-amber-900 dark:text-amber-200">
        {ABSTENTION_TITLES[abstention.reason] ?? "No confident answer"}
      </h3>
      <p className="mt-1 text-sm text-amber-900/80 dark:text-amber-100/80">{abstention.message}</p>
      <dl className="mt-3 space-y-2 text-sm">
        {abstention.found.length > 0 && (
          <div>
            <dt className="font-medium text-slate-700 dark:text-slate-200">Found</dt>
            {abstention.found.map((f) => (
              <dd key={f} className="text-slate-600 dark:text-slate-300">
                {f}
              </dd>
            ))}
          </div>
        )}
        {abstention.missing.length > 0 && (
          <div>
            <dt className="font-medium text-slate-700 dark:text-slate-200">Missing</dt>
            {abstention.missing.map((m) => (
              <dd key={m} className="text-slate-600 dark:text-slate-300">
                {m}
              </dd>
            ))}
          </div>
        )}
        {abstention.suggested_documents.length > 0 && (
          <div>
            <dt className="font-medium text-slate-700 dark:text-slate-200">Closest documents</dt>
            {abstention.suggested_documents.map((d) => (
              <dd key={d.source} className="text-slate-600 dark:text-slate-300">
                {d.source}
                {d.section_heading && (
                  <span className="text-slate-400"> › {shortSection(d.section_heading)}</span>
                )}
              </dd>
            ))}
          </div>
        )}
      </dl>
      {abstention.draft_answer && (
        <details className="mt-3 text-sm">
          <summary className="cursor-pointer text-slate-600 dark:text-slate-300">
            Show the withdrawn draft
          </summary>
          <p className="mt-2 whitespace-pre-wrap rounded bg-white/60 p-2 text-slate-600 line-through decoration-slate-400 dark:bg-slate-900/60 dark:text-slate-400">
            {abstention.draft_answer}
          </p>
        </details>
      )}
    </div>
  );
}
