"""Turning eval runs into ``summary.json``, ``report.md`` and charts."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hyrag.evaluation.dataset import QUESTION_TYPES
from hyrag.evaluation.metrics import abstention_scores, auroc, mean
from hyrag.evaluation.policy import GenerationRecord, Policy, PolicyResult, evaluate_policy
from hyrag.evaluation.runner import RetrievalRow

_CHECK_SCORES = {"SUPPORTED": 1.0, "PARTIAL": 0.5, "UNVERIFIED": 0.5, "UNSUPPORTED": 0.0}


def summarize_retrieval(rows: Sequence[RetrievalRow]) -> list[dict[str, Any]]:
    """Recall@1, recall@5 and MRR per configuration, over answerable questions only."""
    by_config: dict[str, list[RetrievalRow]] = defaultdict(list)
    for row in rows:
        if row.answerable:
            by_config[row.config].append(row)
    out: list[dict[str, Any]] = []
    for config, group in by_config.items():
        first = group[0]
        by_type = {
            t: round(mean([r.recall_at_5 for r in group if r.type == t]), 4)
            for t in QUESTION_TYPES
            if any(r.type == t for r in group)
        }
        out.append(
            {
                "config": config,
                "strategy": first.strategy.value,
                "mode": first.mode.value,
                "rerank": first.rerank,
                "questions": len(group),
                "recall_at_1": round(mean([r.recall_at_1 for r in group]), 4),
                "recall_at_5": round(mean([r.recall_at_5 for r in group]), 4),
                "mrr": round(mean([r.reciprocal_rank for r in group]), 4),
                "recall_at_5_by_type": by_type,
            }
        )
    out.sort(key=lambda r: (-r["recall_at_5"], -r["mrr"], r["config"]))
    return out


def summarize_generation(
    records: Sequence[GenerationRecord], policy: Policy
) -> list[dict[str, Any]]:
    """Per configuration, with ``policy`` deciding which answers are shown."""
    by_config: dict[str, list[GenerationRecord]] = defaultdict(list)
    for r in records:
        by_config[r.config].append(r)
    out: list[dict[str, Any]] = []
    for config, group in by_config.items():
        result = evaluate_policy(policy, group)
        shown = [r for r in group if policy.decide(r) is None]
        shown_answerable = [r for r in shown if r.answerable]
        answerable = [r for r in group if r.answerable]
        checks: dict[str, int] = defaultdict(int)
        for r in shown:
            for label, n in r.checks.items():
                checks[label] += n
        total_checks = sum(checks.values())
        graded = [r for r in shown_answerable if r.correctness is not None]
        correct = [r.confidence for r in graded if (r.correctness or 0) >= 1.0]
        incorrect = [r.confidence for r in graded if (r.correctness or 0) < 1.0]
        # Replies served from the LLM cache take well under 50 ms; only real calls count.
        latency = [r.generate_ms + r.verify_ms for r in group if r.generate_ms >= 50]
        out.append(
            {
                "config": config,
                "questions": len(group),
                "answered_rate": round(len(shown_answerable) / len(answerable), 4)
                if answerable
                else 0.0,
                "correctness": round(mean([r.correctness or 0.0 for r in graded]), 4),
                "correct_overall": round(
                    sum(r.correctness or 0.0 for r in graded) / len(answerable), 4
                )
                if answerable
                else 0.0,
                "key_fact_recall": round(mean([r.key_fact_recall or 0.0 for r in graded]), 4),
                "faithfulness": round(mean([r.citation_coverage for r in shown]), 4),
                "citation_accuracy": round(
                    sum(_CHECK_SCORES.get(k, 0.0) * n for k, n in checks.items()) / total_checks, 4
                )
                if total_checks
                else 0.0,
                "abstention_precision": round(result.abstention_precision, 4),
                "abstention_recall": round(result.abstention_recall, 4),
                "abstention_f1": round(result.abstention_f1, 4),
                "utility": round(result.utility, 4),
                "confidence_correct": round(mean(correct), 4),
                "confidence_incorrect": round(mean(incorrect), 4),
                "median_latency_s": round(statistics.median(latency) / 1000, 1)
                if latency
                else None,
                "grader": graded[0].grade_source if graded else None,
            }
        )
    out.sort(key=lambda r: (-r["utility"], -r["correctness"], r["config"]))
    return out


def signal_separation(records: Sequence[GenerationRecord]) -> dict[str, float]:
    """AUROC of each signal for telling answerable from unanswerable questions (1.0 = perfect)."""
    pos = [r for r in records if r.answerable]
    neg = [r for r in records if not r.answerable]

    def score(name: str, get: Any) -> float:
        p = [v for v in (get(r) for r in pos) if v is not None]
        n = [v for v in (get(r) for r in neg) if v is not None]
        return round(auroc(p, n), 4)

    return {
        "retrieval_confidence": score("retrieval", lambda r: r.retrieval),
        "generator_answered": score("gen", lambda r: 0.0 if r.generator_abstained else 1.0),
        "judge_answerability": score(
            "ans", lambda r: None if r.generator_abstained else (r.answerability or 0.0)
        ),
        "final_confidence": score("conf", lambda r: 0.0 if r.generator_abstained else r.confidence),
    }


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Evaluation report",
        "",
        f"Generated {summary['generated_at']} in **{summary['mode']}** mode "
        f"from `{summary['golden']['path']}` ({summary['golden']['questions']} questions: "
        + ", ".join(f"{n} {t.replace('_', '-')}" for t, n in summary["golden"]["by_type"].items())
        + ").",
        "",
        "Models: " + ", ".join(f"{k} `{v}`" for k, v in summary["models"].items() if v) + ".",
        "",
        "## Retrieval",
        "",
        "Recall@k is the share of a question's gold evidence found in the top k chunks; MRR is "
        "the mean reciprocal rank of the first relevant chunk. Answerable questions only.",
        "",
        "| config | recall@1 | recall@5 | MRR | lookup | multi-hop | exact-term | ambiguous |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in summary["retrieval"]["configs"]:
        t = r["recall_at_5_by_type"]
        lines.append(
            f"| {r['config']} | {_pct(r['recall_at_1'])} | **{_pct(r['recall_at_5'])}** | "
            f"{r['mrr']:.3f} | "
            + " | ".join(
                _pct(t.get(k, 0.0)) for k in ("lookup", "multi_hop", "exact_term", "ambiguous")
            )
            + " |"
        )
    generation = summary.get("generation")
    if generation:
        policy = generation["policy"]
        grader = (
            f"`{summary['models']['judge']}`"
            if generation["configs"][0]["grader"] == "judge"
            else "key facts (`must_include`)"
        )
        lines += [
            "",
            "## Answers",
            "",
            f"With the tuned abstention policy (retrieval gate {policy['abstain_threshold']}, "
            f"judge check {'on' if policy['judge_check'] else 'off'}, confidence floor "
            f"{policy['answer_threshold']}). Correctness is graded by {grader} against the "
            "reference answers. Latency counts only calls that reached the model; replies "
            "served from the LLM cache show as cached.",
            "",
            "| config | answered | correct (answered) | correct (all answerable) | faithfulness "
            "| citation accuracy | abstention P / R | median latency |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in generation["configs"]:
            lines.append(
                f"| {r['config']} | {_pct(r['answered_rate'])} | **{_pct(r['correctness'])}** | "
                f"{_pct(r['correct_overall'])} | {_pct(r['faithfulness'])} | "
                f"{_pct(r['citation_accuracy'])} | {_pct(r['abstention_precision'])} / "
                f"{_pct(r['abstention_recall'])} | "
                + (f"{r['median_latency_s']} s" if r["median_latency_s"] is not None else "cached")
                + " |"
            )
        abst = generation["abstention"]
        lines += [
            "",
            "## Abstention",
            "",
            "How well each signal separates answerable from unanswerable questions (AUROC, "
            "1.0 is perfect, 0.5 is chance), pooled over all configurations:",
            "",
            "| signal | AUROC |",
            "|---|---|",
            *(f"| {k.replace('_', ' ')} | {v:.3f} |" for k, v in abst["signal_auroc"].items()),
            "",
            "What each layer adds, pooled over all configurations. Utility counts a correct "
            "answer as +1 and a wrong one (including any answer to an unanswerable question) "
            f"as -{abst['wrong_answer_cost']:g}, per question.",
            "",
            "| policy | abstention P / R | answered correctly | answered wrongly | utility |",
            "|---|---|---|---|---|",
            *(
                f"| {row['name']} | "
                + (
                    f"{_pct(row['abstention_precision'])} / {_pct(row['abstention_recall'])}"
                    if row["abstained"]
                    else "never abstains"
                )
                + f" | {row['correct']:.1f} | {row['wrong']:.1f} | "
                f"{row['utility']:+.3f} |"
                for row in abst["ablation"]
            ),
        ]
    if summary.get("charts"):
        lines += ["", "## Charts", ""]
        lines += [f"![{Path(c).stem}]({c})" for c in summary["charts"]]
    return "\n".join(lines) + "\n"


def write_charts(summary: dict[str, Any], out_dir: Path) -> list[str]:
    """PNG charts next to the report. Skipped (empty list) when matplotlib is missing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib ships with the dev extra
        return []
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    configs = [r for r in summary["retrieval"]["configs"]]
    strategies = sorted({r["strategy"] for r in configs})
    columns = sorted({(r["mode"], r["rerank"]) for r in configs}, key=lambda c: (c[0], not c[1]))
    grid = [[0.0] * len(columns) for _ in strategies]
    for r in configs:
        grid[strategies.index(r["strategy"])][columns.index((r["mode"], r["rerank"]))] = r[
            "recall_at_5"
        ]
    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * len(columns), 1.2 + 0.7 * len(strategies)))
    low = min(0.5, min(min(row) for row in grid))  # zoom in: the differences are at the top
    image = ax.imshow(grid, cmap="Blues", vmin=low, vmax=1, aspect="auto")
    ax.set_xticks(range(len(columns)), [m + ("\n+rerank" if rr else "") for m, rr in columns])
    ax.set_yticks(range(len(strategies)), strategies)
    for i, row in enumerate(grid):
        for j, value in enumerate(row):
            ax.text(
                j,
                i,
                f"{value:.0%}",
                ha="center",
                va="center",
                color="white" if value > (low + 1) / 2 else "black",
                fontsize=9,
            )
    ax.set_title("Retrieval recall@5")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(charts_dir / "retrieval_recall.png", dpi=150)
    plt.close(fig)
    written.append("charts/retrieval_recall.png")

    generation = summary.get("generation")
    if generation:
        rows = generation["configs"]
        metrics = [
            ("correctness", "correct (answered)"),
            ("faithfulness", "faithfulness"),
            ("citation_accuracy", "citation accuracy"),
            ("answered_rate", "answered"),
        ]
        fig, ax = plt.subplots(figsize=(8, 3.6))
        width = 0.8 / len(rows)
        for i, row in enumerate(rows):
            xs = [m + i * width for m in range(len(metrics))]
            ax.bar(xs, [row[key] for key, _ in metrics], width, label=row["config"])
        ax.set_xticks([m + 0.4 - width / 2 for m in range(len(metrics))], [n for _, n in metrics])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("share")
        ax.set_title("Answer quality by configuration")
        ax.legend(fontsize=8, ncols=len(rows), loc="upper center", bbox_to_anchor=(0.5, -0.12))
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(charts_dir / "answer_quality.png", dpi=150)
        plt.close(fig)
        written.append("charts/answer_quality.png")

        sweep = generation["abstention"]["floor_sweep"]
        fig, ax = plt.subplots(figsize=(6, 3.4))
        xs = [p["answer_threshold"] for p in sweep]
        ax.plot(xs, [p["correct"] for p in sweep], label="answered correctly")
        ax.plot(xs, [p["wrong"] for p in sweep], label="answered wrongly")
        ax.axvline(
            generation["policy"]["answer_threshold"],
            color="grey",
            ls="--",
            lw=1,
            label="tuned floor",
        )
        ax.set_xlabel("confidence floor (answer_threshold)")
        ax.set_ylabel("questions (all configs)")
        ax.set_title("Abstention trade-off")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(charts_dir / "abstention_tradeoff.png", dpi=150)
        plt.close(fig)
        written.append("charts/abstention_tradeoff.png")
    return written


def abstention_section(
    records: Sequence[GenerationRecord],
    tuned: PolicyResult,
    ablation_rows: Sequence[tuple[str, PolicyResult]],
    wrong_answer_cost: float,
) -> dict[str, Any]:
    floor_sweep = []
    for step in range(0, 17):
        threshold = round(step * 0.05, 2)
        result = evaluate_policy(
            Policy(tuned.policy.abstain_threshold, threshold, tuned.policy.judge_check), records
        )
        floor_sweep.append(
            {
                "answer_threshold": threshold,
                "correct": round(result.correct, 2),
                "wrong": round(result.wrong, 2),
                "utility": round(result.utility, 4),
            }
        )
    return {
        "wrong_answer_cost": wrong_answer_cost,
        "signal_auroc": signal_separation(records),
        "ablation": [
            {
                "name": name,
                **{k: v for k, v in r.as_dict().items() if k != "by_layer"},
                "by_layer": r.by_layer,
            }
            for name, r in ablation_rows
        ],
        "floor_sweep": floor_sweep,
        "overall": {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in abstention_scores(
                [(not r.answerable, tuned.policy.decide(r) is not None) for r in records]
            ).__dict__.items()
        },
    }


def write_outputs(
    out_dir: Path,
    summary: dict[str, Any],
    retrieval_rows: Sequence[RetrievalRow],
    generation_records: Sequence[GenerationRecord],
    *,
    charts: bool = True,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = out_dir / "runs"
    runs.mkdir(exist_ok=True)
    (runs / "retrieval.jsonl").write_text(
        "".join(r.model_dump_json() + "\n" for r in retrieval_rows), encoding="utf-8"
    )
    if generation_records:
        (runs / "generation.jsonl").write_text(
            "".join(r.model_dump_json() + "\n" for r in generation_records), encoding="utf-8"
        )
    summary["charts"] = write_charts(summary, out_dir) if charts else []
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(render_markdown(summary), encoding="utf-8")
    return [out_dir / "summary.json", out_dir / "report.md"]
