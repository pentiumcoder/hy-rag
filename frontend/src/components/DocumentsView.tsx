import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "../api";
import type { DocumentInfo, IngestResponse } from "../types";

const ACCEPT = ".md,.markdown,.txt,.html,.htm,.pdf";

export function DocumentsView() {
  const [docs, setDocs] = useState<DocumentInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [report, setReport] = useState<IngestResponse | null>(null);
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  const load = useCallback(() => {
    api
      .documents()
      .then(setDocs)
      .catch((e: unknown) => setError(e instanceof ApiError ? e.message : String(e)));
  }, []);

  useEffect(load, [load]);

  async function upload(files: File[]) {
    if (files.length === 0) return;
    setUploading(true);
    setError(null);
    setReport(null);
    try {
      setReport(await api.ingest(files));
      load();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="space-y-6">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          void upload([...e.dataTransfer.files]);
        }}
        className={`rounded-xl border-2 border-dashed p-8 text-center transition ${
          dragging
            ? "border-indigo-500 bg-indigo-50 dark:bg-indigo-950/40"
            : "border-slate-300 dark:border-slate-700"
        }`}
      >
        <p className="text-sm text-slate-600 dark:text-slate-300">
          Drop Markdown, text, HTML or PDF files here, or{" "}
          <button
            type="button"
            onClick={() => input.current?.click()}
            className="font-medium text-indigo-600 hover:underline dark:text-indigo-400"
          >
            choose files
          </button>
          . They are chunked, deduplicated, embedded and indexed for every strategy.
        </p>
        <input
          ref={input}
          type="file"
          multiple
          accept={ACCEPT}
          aria-label="Upload documents"
          className="hidden"
          onChange={(e) => void upload([...(e.target.files ?? [])])}
        />
        {uploading && <p className="mt-3 text-sm text-slate-500">Indexing…</p>}
      </div>

      {error && (
        <p role="alert" className="text-sm text-rose-600 dark:text-rose-400">
          {error}
        </p>
      )}

      {report && (
        <div className="rounded-xl border border-emerald-300 bg-emerald-50 p-4 text-sm dark:border-emerald-800 dark:bg-emerald-950/40">
          <p className="font-medium text-emerald-900 dark:text-emerald-200">
            {report.added.length} added, {report.updated.length} updated,{" "}
            {report.unchanged.length} unchanged
            {report.skipped.length > 0 && `, ${report.skipped.length} skipped (no text)`}
          </p>
          <p className="mt-1 text-emerald-800/80 dark:text-emerald-200/80">
            {report.indexes
              .map((i) => `${i.strategy}: +${i.added} chunks (${i.chunks} total, ${i.duplicates} duplicates)`)
              .join(" · ")}
          </p>
        </div>
      )}

      <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-900">
        <h3 className="mb-3 text-sm font-semibold">{docs ? `${docs.length} documents` : "Documents"}</h3>
        {docs && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-xs uppercase tracking-wide text-slate-500">
                <tr>
                  <th className="py-2 pr-3">Source</th>
                  <th className="py-2 pr-3">Title</th>
                  <th className="py-2 pr-3">Format</th>
                  <th className="py-2 pr-3 text-right">Sections</th>
                  <th className="py-2 text-right">Characters</th>
                </tr>
              </thead>
              <tbody>
                {docs.map((d) => (
                  <tr key={d.doc_id} className="border-t border-slate-200 dark:border-slate-800">
                    <td className="py-2 pr-3 font-mono text-xs">{d.source}</td>
                    <td className="py-2 pr-3">{d.title}</td>
                    <td className="py-2 pr-3 uppercase text-slate-500">{d.format}</td>
                    <td className="py-2 pr-3 text-right tabular-nums">{d.sections}</td>
                    <td className="py-2 text-right tabular-nums">{d.chars.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
