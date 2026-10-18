"""The README's accuracy-vs-cost chart, as a dependency-free SVG.

    python scripts/readme_chart.py          # -> docs/accuracy_vs_cost.svg

Same sources as the case study (`build_case_study.py`): students from results/<run>.json and
latency.json, cascades from cascade.json, the teacher from its test labels. The served model
is plotted at its int8 accuracy through the serving code (robustness.json), as in the README
headline; the other students at their recorded accuracy, as in the README table.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distilroute.calibration import apply_temperature  # noqa: E402
from distilroute.data import DOCS, RESULTS, load_labels, load_split, rel  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_case_study import SERVED, STUDENTS, teacher_accuracy  # noqa: E402
from evaluate import threshold_cascade  # noqa: E402

OUT = DOCS / "accuracy_vs_cost.svg"
THRESHOLD = 0.8  # the README table's cascade row
# Reference categorical palette, slots 1-3 in fixed order (validated set; text never wears them).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#52514e", "#8a8984", "#e6e5e0"
W, H = 900, 500
LEFT, RIGHT, TOP, BOTTOM = 70, 30, 96, 70
X_MIN, X_MAX = 0.001, 1000.0  # $ per 1M, log scale
Y_MIN, Y_MAX = 0.78, 0.88
# Label placement per point: (dx, dy, anchor), picked by eye against the rendered chart.
PLACE = {
    "tfidf_lr_teacher": (-12, 5, "end"),
    "tinybert_ft_teacher": (-12, 5, "end"),
    "minilm_ft_teacher": (-12, 5, "end"),
    SERVED: (-12, -10, "end"),
    "minilm_frozen_teacher": (12, -6, "start"),
    "distilbert_ft_teacher": (12, 14, "start"),
    "best": (0, -14, "end"),
    "three": (-12, 5, "end"),
    "teacher": (0, 24, "middle"),
}


def x_of(usd: float) -> float:
    span = math.log10(X_MAX) - math.log10(X_MIN)
    return LEFT + (math.log10(usd) - math.log10(X_MIN)) / span * (W - LEFT - RIGHT)


def y_of(acc: float) -> float:
    return TOP + (Y_MAX - acc) / (Y_MAX - Y_MIN) * (H - TOP - BOTTOM)


def points() -> list[dict]:
    cost = json.loads((RESULTS / "cost.json").read_text())
    latency = json.loads((RESULTS / "latency.json").read_text())
    cascade = json.loads((RESULTS / "cascade.json").read_text())
    robust = json.loads((RESULTS / "robustness.json").read_text())
    vm_per_s = cost["prices"]["vm"]["usd_per_hour"] / 3600
    int8 = {r["model"]: r["kinds"]["clean"]["acc"] for r in robust["students"]}

    out = []
    for run, (label, _) in STUDENTS.items():
        acc = json.loads((RESULTS / f"{run}.json").read_text())["accuracy"]
        if run == SERVED:
            acc, label = int8[run], "MiniLM, 10k labels (served)"
        usd = 1e6 * latency[run]["mean_ms"] / 1000 * vm_per_s
        out.append({"key": run, "label": label, "acc": acc, "usd": usd, "color": BLUE})

    # The README table's two cascade rows, so chart and table show the same systems.
    systems = {tuple(s["tiers"]): s for s in cascade["systems"]}
    three = systems[("tfidf_lr_teacher", SERVED)]
    llm_usd = cost["rows"]["teacher_single"]["usd_per_1m"]
    served_usd = next(p["usd"] for p in out if p["key"] == SERVED)
    out += [
        {"key": "best", **threshold_point(llm_usd, served_usd), "color": AQUA},
        {
            "key": "three",
            "label": "TF-IDF → MiniLM → LLM, cheapest match",
            **_au(three["match_llm"]),
            "color": AQUA,
        },
        {
            "key": "teacher",
            "label": "LLM alone",
            "acc": teacher_accuracy("test"),
            "usd": cost["rows"]["teacher_single"]["usd_per_1m"],
            "color": ORANGE,
        },
    ]
    return out


def threshold_point(llm_usd: float, served_usd: float) -> dict:
    """The README table's cascade row: escalate below 0.8 calibrated confidence, as evaluate.py
    scores it (fp32 probabilities, the teacher's test labels on the escalated rows)."""
    z = np.load(RESULTS / f"{SERVED}_test_probs.npz", allow_pickle=False)
    temp = json.loads((RESULTS / f"{SERVED}.json").read_text())["temperature"]
    pred = z["classes"][z["proba"].argmax(axis=1)]
    conf = apply_temperature(z["proba"], temp).max(axis=1)
    row = next(
        r
        for r in threshold_cascade(
            pred, conf, load_split("test").category.values, load_labels("test").teacher
        )
        if r["thr"] == THRESHOLD
    )
    return {
        "label": f"MiniLM → LLM below {THRESHOLD}",
        "acc": row["mixed"],
        "usd": served_usd + row["escalated"] * llm_usd,
    }


def _au(row: dict) -> dict:
    return {"acc": row["acc"], "usd": row["usd"]}


def svg(pts: list[dict]) -> str:
    font = "font-family='ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif'"
    s = [
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {W} {H}' width='{W}' "
        f"height='{H}' {font} role='img' aria-labelledby='t d'>",
        "<title id='t'>Accuracy against cost for every system</title>",
        "<desc id='d'>Accuracy on 3,080 Banking77 test queries against dollars per million "
        "requests, log scale. "
        + "; ".join(f"{p['label']}: {p['acc']:.3f} at ${p['usd']:.2f}" for p in pts)
        + ".</desc>",
        f"<rect width='{W}' height='{H}' fill='{SURFACE}'/>",
        f"<text x='{LEFT}' y='34' font-size='20' font-weight='600' fill='{INK}'>"
        "Accuracy against cost for every system</text>",
    ]
    # Legend, one row under the title.
    lx = LEFT
    for name, color in (
        ("Students alone", BLUE),
        ("Cascades (student, then LLM)", AQUA),
        ("LLM alone", ORANGE),
    ):
        s.append(f"<circle cx='{lx + 5}' cy='59' r='5' fill='{color}'/>")
        s.append(f"<text x='{lx + 15}' y='64' font-size='13' fill='{INK2}'>{name}</text>")
        lx += 30 + 7.2 * len(name)
    # Grid and axes: recessive.
    for k in range(6):
        acc = Y_MIN + k * 0.02
        y = y_of(acc)
        s.append(
            f"<line x1='{LEFT}' x2='{W - RIGHT}' y1='{y:.1f}' y2='{y:.1f}' "
            f"stroke='{GRID}' stroke-width='1'/>"
        )
        s.append(
            f"<text x='{LEFT - 10}' y='{y + 4:.1f}' font-size='12' fill='{MUTED}' "
            f"text-anchor='end'>{acc:.2f}</text>"
        )
    for e in range(-3, 4):
        x = x_of(10.0**e)
        tick = f"${10.0**e:g}"
        s.append(
            f"<text x='{x:.1f}' y='{H - BOTTOM + 22}' font-size='12' fill='{MUTED}' "
            f"text-anchor='middle'>{tick}</text>"
        )
    s.append(
        f"<text x='{W - RIGHT}' y='{H - BOTTOM + 46}' font-size='12' fill='{MUTED}' "
        "text-anchor='end'>$ per 1M requests (log scale)</text>"
    )
    s.append(
        f"<text x='{LEFT - 10}' y='{TOP - 14}' font-size='12' fill='{MUTED}' "
        "text-anchor='end'>accuracy</text>"
    )
    # Marks: 5 px radius (10 px across) with a 2 px surface ring; every point direct-labelled.
    for p in pts:
        x, y = x_of(p["usd"]), y_of(p["acc"])
        dx, dy, anchor = PLACE[p["key"]]
        tip = escape(f"{p['label']}: accuracy {p['acc']:.3f}, ${p['usd']:.2f} per 1M")
        s.append(
            f"<g><title>{tip}</title><circle cx='{x:.1f}' cy='{y:.1f}' r='6' "
            f"fill='{p['color']}' stroke='{SURFACE}' stroke-width='2'/></g>"
        )
        s.append(
            f"<text x='{x + dx:.1f}' y='{y + dy:.1f}' font-size='13' fill='{INK}' "
            f"text-anchor='{anchor}'>{escape(p['label'])}</text>"
        )
    s.append(
        f"<text x='{LEFT}' y='{H - 8}' font-size='11' fill='{MUTED}'>Banking77, 3,080 "
        "held-out queries vs human labels. Students never see a human label "
        "(3,000 LLM labels; served: 10,003, int8). scripts/readme_chart.py</text>"
    )
    s.append("</svg>")
    return "\n".join(s) + "\n"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # the labels carry arrows
    pts = points()
    OUT.write_text(svg(pts), encoding="utf-8")
    for p in pts:
        print(f"  {p['label']:30} {p['acc']:.3f}  ${p['usd']:.3f}")
    print(f"-> {rel(OUT)}")


if __name__ == "__main__":
    main()
