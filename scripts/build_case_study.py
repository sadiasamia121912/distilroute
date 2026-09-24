"""Fill the case-study page with the current numbers (roadmap 6.8).

    python scripts/build_case_study.py      # docs/case-study/template.html -> index.html

Every figure on the page comes from a file in results/ (or data/labels/ for the teacher), so
the write-up cannot drift from the experiments: re-run this after any experiment and the page
follows. Sections whose results do not exist yet (robustness, the second dataset) render as
"in progress" instead of guessing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distilroute.data import DOCS, RESULTS, ROOT, load_labels, load_split, rel  # noqa: E402

PAGE = DOCS / "case-study"
STUDENTS = {  # run -> display name, parameter count
    "tfidf_lr_teacher": ("TF-IDF + LR", None),
    "tinybert_ft_teacher": ("TinyBERT", "14M"),
    "minilm_ft_teacher": ("MiniLM, fine-tuned", "22M"),
    "minilm_frozen_teacher": ("MiniLM, frozen", "22M"),
    "distilbert_ft_teacher": ("DistilBERT", "67M"),
}
SERVED = "minilm_ft_teacher"


def mean_by_n(curve: dict) -> list[list[float]]:
    by: dict[int, list[float]] = {}
    for r in curve["rows"]:
        by.setdefault(r["n"], []).append(r["accuracy"])
    return [[n, float(np.mean(v)), float((max(v) - min(v)) / 2)] for n, v in sorted(by.items())]


def teacher_accuracy(name: str) -> float:
    lab = load_labels(name)
    return float((lab.teacher == lab.gold).mean())


def main() -> None:
    cost = json.loads((RESULTS / "cost.json").read_text())
    latency = json.loads((RESULTS / "latency.json").read_text())
    cascade = json.loads((RESULTS / "cascade.json").read_text())
    vm_per_s = cost["prices"]["vm"]["usd_per_hour"] / 3600
    llm_usd = cost["rows"]["teacher_single"]["usd_per_1m"]
    teacher = teacher_accuracy("test")

    students = []
    for run, (label, params) in STUDENTS.items():
        m = json.loads((RESULTS / f"{run}.json").read_text())
        students.append(
            {
                "run": run,
                "label": label,
                "params": params,
                "acc": m["accuracy"],
                "p50": latency[run]["p50_ms"],
                "usd": 1e6 * latency[run]["mean_ms"] / 1000 * vm_per_s,
            }
        )
    served = next(s for s in students if s["run"] == SERVED)

    systems = {tuple(s["tiers"]): s for s in cascade["systems"]}
    two = systems[(SERVED,)]
    three = systems[("tfidf_lr_teacher", SERVED)]
    cascade_curve = [
        {"budget": float(b), "llm": v["llm"], "acc": v["acc"], "usd": v["usd"]}
        for b, v in two["budgets"].items()
    ]
    best = max(cascade_curve, key=lambda r: r["acc"])

    curves = {p.stem: json.loads(p.read_text()) for p in (RESULTS / "curves").glob("*.json")}
    oos = json.loads((RESULTS / "oos.json").read_text())

    data = {
        "teacher": {
            "acc": teacher,
            "usd": llm_usd,
            "batched_usd": cost["rows"]["teacher_batched"]["usd_per_1m"],
        },
        "leak": {"sorted": teacher_accuracy("test.sorted_batches"), "shuffled": teacher},
        "n_test": len(load_split("test")),
        "n_train_labels": int(json.loads((RESULTS / f"{SERVED}.json").read_text())["n_train"]),
        "students": students,
        "served": served,
        "gold_reference": json.loads((RESULTS / "minilm_ft_gold.json").read_text())["accuracy"],
        "cascade": {
            "curve": cascade_curve,
            "best": best,
            "two_tier_match": {k: two["match_llm"][k] for k in ("acc", "llm", "usd")},
            "three_tier_match": {
                k: three["match_llm"][k] for k in ("acc", "llm", "usd", "ms", "first")
            },
            "seeds": cascade["seeds"],
        },
        "data_curve": {
            "gold": mean_by_n(curves["minilm_gold"]),
            "teacher": mean_by_n(curves["minilm_teacher"]),
        },
        "active": {
            "random": mean_by_n(curves["active_random"]),
            "diverse": mean_by_n(curves["active_diverse"]),
            "uncertainty": mean_by_n(curves["active_uncertainty"]),
        },
        "oos": {
            "caught_min": min(r["caught_entropy_at_10pct"] for r in oos),
            "caught_max": max(r["caught_entropy_at_10pct"] for r in oos),
            "auroc_min": min(r["auroc_entropy"] for r in oos),
            "auroc_max": max(r["auroc_entropy"] for r in oos),
        },
        "robustness": None,
        "second_domain": None,
    }

    rob = RESULTS / "robustness.json"
    if rob.exists():
        r = json.loads(rob.read_text())
        # The headline quotes the served int8 graph (clean text, serving code path), not fp32.
        for row in r["students"]:
            if row["model"] == SERVED:
                data["served"]["acc_int8"] = row["kinds"]["clean"]["acc"]
        if len(r["students"]) >= len(STUDENTS):  # a finished run, not a partial one
            data["robustness"] = r

    # The second dataset lives under its own subdirectories (DISTILROUTE_DATASET=clinc150).
    clinc = ROOT / "results" / "clinc150"
    clinc_labels = ROOT / "data" / "labels" / "clinc150" / "test.jsonl"
    if (clinc / "minilm_frozen_teacher.json").exists() and clinc_labels.exists():
        lab = [json.loads(x) for x in clinc_labels.open(encoding="utf-8")]
        data["second_domain"] = {
            "teacher": float(np.mean([x["teacher"] == x["gold"] for x in lab])),
            "n_teacher": len(lab),
            **{
                k: json.loads((clinc / f"{k}.json").read_text())["accuracy"]
                for k in (
                    "tfidf_lr_gold",
                    "minilm_frozen_gold",
                    "tfidf_lr_teacher",
                    "minilm_frozen_teacher",
                )
                if (clinc / f"{k}.json").exists()
            },
        }

    template = (PAGE / "template.html").read_text(encoding="utf-8")
    page = template.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    (PAGE / "index.html").write_text(page, encoding="utf-8")
    print(f"-> {rel(PAGE)}/index.html")


if __name__ == "__main__":
    main()
