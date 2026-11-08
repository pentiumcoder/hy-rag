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
        className={`border border-dashed p-6 ${dragging ? "border-accent bg-accent-soft" : "border-line"}`}
      >
        <p className="text-sm text-muted">
          Drop Markdown, text, HTML or PDF files here, or{" "}
          <button
            type="button"
            onClick={() => input.current?.click()}
            className="text-accent underline-offset-2 hover:underline"
          >
            choose files
          </button>
          . Each file is split into passages and added to the search indexes.
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
        {uploading && <p className="mt-2 text-sm text-muted">Indexing…</p>}
      </div>

      {error && (
        <p role="alert" className="text-sm text-bad">
          {error}
        </p>
      )}

      {report && (
        <div className="border-l-2 border-ok pl-3 text-sm">
          <p>
            {report.added.length} added, {report.updated.length} updated,{" "}
            {report.unchanged.length} unchanged
            {report.skipped.length > 0 && `, ${report.skipped.length} skipped (no text)`}
          </p>
          <p className="mt-1 text-muted">
            {report.indexes
              .map((i) => `${i.strategy}: +${i.added} chunks (${i.chunks} total, ${i.duplicates} duplicates)`)
              .join(" · ")}
          </p>
        </div>
      )}

      <section className="border-t border-line pt-4">
        <h3 className="mb-3 font-serif text-lg">{docs ? `${docs.length} documents` : "Documents"}</h3>
        {docs && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-xs text-muted">
                <tr>
                  <th className="py-2 pr-3 font-normal">Source</th>
                  <th className="py-2 pr-3 font-normal">Title</th>
                  <th className="py-2 pr-3 font-normal">Format</th>
                  <th className="py-2 pr-3 text-right font-normal">Sections</th>
                  <th className="py-2 text-right font-normal">Characters</th>
                </tr>
              </thead>
              <tbody>
                {docs.map((d) => (
                  <tr key={d.doc_id} className="border-t border-line">
                    <td className="py-2 pr-3 font-mono text-xs">{d.source}</td>
                    <td className="py-2 pr-3">{d.title}</td>
                    <td className="py-2 pr-3 text-muted">{d.format}</td>
                    <td className="py-2 pr-3 text-right font-mono text-xs">{d.sections}</td>
                    <td className="py-2 text-right font-mono text-xs">{d.chars.toLocaleString()}</td>
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
