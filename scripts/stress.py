"""Two questions in one message, and a question buried in a long ticket (roadmap 7.6)

    python scripts/stress.py                  # the teacher-label students, ~10 min on CPU
    python scripts/stress.py --models minilm_ft_teacher --pairs 200 --padded 150

Banking77 queries are one short sentence with one intent. Real tickets are not:

- **Joined pairs.** Two test queries with different intents, sent as one message ("A B" and
  "A Also, b"). A router picks one intent, so the fair questions are: is its top-1 at least one
  of the two, are **both** in its top-2 (a multi-intent UI could offer the second), which of the
  two wins, and does its confidence drop, so that the cascade escalates the mixed message?
- **Ticket length.** Test queries padded with intent-neutral filler (greetings, apologies,
  "I've been a customer for years") to ~50 and ~120 words, with the question at the start, at
  the end, or in the middle. The fine-tuned students read up to 512 tokens as served; until
  2026-09-28 they read 64, and `minilm_ft_teacher@64` re-runs the served model with that old
  limit, so the doc shows what it cost. Accuracy is against the same queries unpadded.

Everything runs through `distilroute.students`, the code path the service uses, with the
cascade's calibrated 0.8 threshold for "escalated". No LLM calls. Writes results/stress.json and
docs/stress.md.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distilroute import students  # noqa: E402
from distilroute.data import DOCS, RESULTS, load_split, rel  # noqa: E402

MODELS = [
    "tfidf_lr_teacher",
    "minilm_frozen_teacher",
    "minilm_ft_teacher",  # the served model
    "minilm_ft_teacher@64",  # ... with the 64-token limit it was served with until 2026-09-28
    "tinybert_ft_teacher",
    "distilbert_ft_teacher",
]
THRESHOLD = 0.8  # the cascade threshold of 1b.2 / 6.5
JOINS = {"space": "{a} {b}", "also": "{a} Also, {b}"}
LENGTHS = [50, 120]  # words, filler included
WHERE = ["start", "middle", "end"]  # where the question sits in the padded ticket
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
FOUND = 0.5  # a sentence's answer counts as a detected intent at this calibrated confidence

# Sentences a customer might wrap around any question. None names a product, a payment or an
# action on an account, so none of them points at an intent.
FILLER = [
    "Hi there, I hope you are having a good day.",
    "Sorry to bother you with this.",
    "I have been a customer for about three years now and have always been happy.",
    "I tried looking through the help pages first but could not find an answer.",
    "I am writing this on my phone so please excuse any mistakes.",
    "My name is Alex and I live in Manchester.",
    "This is the first time I have had to contact support.",
    "I would really appreciate a quick reply if possible.",
    "A friend of mine recommended you to me last year.",
    "I am not very good with technology so please bear with me.",
    "I have a busy week ahead so I wanted to get this sorted today.",
    "Please let me know if you need any more information from me.",
    "I am sure this is a simple question but I am a bit confused.",
    "Thank you in advance for your help.",
    "Kind regards and have a nice weekend.",
    "I asked my partner about it but they did not know either.",
    "Apologies if this has already been asked before.",
    "I am usually able to work these things out myself.",
]


def words(text: str) -> int:
    return len(text.split())


def pad(text: str, target: int, where: str, seed: int) -> str:
    """`text` with filler sentences around it until the message has ~`target` words."""
    rng = random.Random(seed)
    pool = FILLER[:]
    rng.shuffle(pool)
    before, after = [], []
    for sentence in pool * 3:
        if words(" ".join([*before, text, *after])) >= target:
            break
        if where == "start":
            after.append(sentence)
        elif where == "end" or len(before) <= len(after):
            before.append(sentence)
        else:
            after.append(sentence)
    return " ".join([*before, text, *after])


def route(router: students.Router, texts: list[str]) -> tuple[np.ndarray, list, np.ndarray]:
    routed = [router.route(t) for t in texts]
    return (
        np.array([r.intent for r in routed]),
        [r.ranked for r in routed],
        np.array([r.confidence for r in routed]),
    )


def sentences(text: str) -> list[str]:
    return [x for x in SENTENCE_END.split(text.strip()) if x]


def entropy(p: np.ndarray) -> float:
    return float(-(p * np.log(p + 1e-12)).sum())


class SplitRouter:
    """The fix to test: route each sentence alone, answer with the most confident one.

    Each sentence is routed on its own, and the set of confident per-sentence answers is
    a cheap multi-intent signal. With `max_entropy` set, sentences above it are dropped first as
    off-topic (the out-of-scope test of 6.2, applied per sentence); if none is left, the whole
    message is routed as before. `cache` is shared between routers of one model: the filler
    repeats, so this costs one route per *distinct* sentence.
    """

    def __init__(self, router: students.Router, cache: dict, max_entropy: float | None = None):
        self.router, self.cache, self.max_entropy = router, cache, max_entropy

    def one(self, text: str) -> tuple[students.Routed, float]:
        if text not in self.cache:
            self.cache[text] = (self.router.route(text), entropy(self.router.proba(text)[1]))
        return self.cache[text]

    def route(self, texts: list[str]) -> tuple[np.ndarray, list[set], np.ndarray]:
        top1, found, conf = [], [], []
        for t in texts:
            scored = [self.one(x) for x in sentences(t)]
            if self.max_entropy is not None:
                scored = [s for s in scored if s[1] < self.max_entropy] or [self.one(t)]
            routed = [r for r, _ in scored]
            best = max(routed, key=lambda r: r.confidence)
            top1.append(best.intent)
            conf.append(best.confidence)
            found.append({r.intent for r in routed if r.confidence >= FOUND})
        return np.array(top1), found, np.array(conf)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--pairs", type=int, default=400)
    ap.add_argument("--padded", type=int, default=300)
    args = ap.parse_args()

    test = load_split("test")
    rng = np.random.default_rng(0)
    a = rng.choice(len(test), args.pairs * 2, replace=False)
    pairs = [
        (i, j) for i, j in zip(a[::2], a[1::2], strict=True) if test.category[i] != test.category[j]
    ]
    pad_idx = rng.choice(len(test), args.padded, replace=False)
    texts, gold = test.text.values, test.category.values

    def join(i: int, j: int, how: str) -> str:
        b = texts[j]
        return JOINS[how].format(a=texts[i], b=b[0].lower() + b[1:] if how == "also" else b)

    joined = {how: [join(i, j, how) for i, j in pairs] for how in JOINS}
    padded = {
        (n, w): [pad(texts[i], n, w, seed=int(i)) for i in pad_idx] for n in LENGTHS for w in WHERE
    }
    ga, gb = gold[[i for i, _ in pairs]], gold[[j for _, j in pairs]]
    singles = sorted({k for p in pairs for k in p} | set(pad_idx.tolist()))
    print(
        f"{len(pairs)} pairs, {args.padded} padded queries; padded length "
        + ", ".join(
            f"{n}: median {np.median([words(t) for t in padded[(n, 'end')]]):.0f} words"
            for n in LENGTHS
        )
    )

    rows = []
    for name in args.models.split(","):
        base, _, tokens = name.partition("@")  # name@N: an ONNX student reading N tokens
        if base not in students.available():
            print(f"skip {name}: not under models/")
            continue
        router = students.load(base)
        if tokens:
            router.max_tokens = int(tokens)
        t0 = time.time()
        pred1, _, conf1 = route(router, [texts[k] for k in singles])
        # Off-topic cut: the entropy above which 10 % of clean queries fall (6.2's budget).
        cache: dict = {}
        naive = SplitRouter(router, cache)
        max_entropy = float(np.quantile([naive.one(texts[k])[1] for k in singles], 0.9))
        splits = {"split": naive, "split_filtered": SplitRouter(router, cache, max_entropy)}
        single = dict(zip(singles, pred1, strict=True))
        single_conf = dict(zip(singles, conf1, strict=True))
        a_ok = np.array([single[i] == gold[i] for i, _ in pairs])
        b_ok = np.array([single[j] == gold[j] for _, j in pairs])
        row = {
            "model": name,
            "kind": router.kind,
            "max_entropy": max_entropy,
            "filler_passing_filter": sum(naive.one(f)[1] < max_entropy for f in FILLER),
            "pairs": {
                "singles_both_right": float((a_ok & b_ok).mean()),
                "singles_escalated": float(
                    np.mean([single_conf[k] < THRESHOLD for p in pairs for k in p])
                ),
            },
            "padded": {},
        }
        for how, xs in joined.items():
            top1, ranked, conf = route(router, xs)
            both2 = np.array([ga[k] in r[:2] and gb[k] in r[:2] for k, r in enumerate(ranked)])
            row["pairs"][how] = {
                "top1_either": float(((top1 == ga) | (top1 == gb)).mean()),
                "top1_first": float((top1 == ga).mean()),
                "top1_second": float((top1 == gb).mean()),
                "both_in_top2": float(both2.mean()),
                "both_in_top2_when_singles_right": float(both2[a_ok & b_ok].mean()),
                "escalated": float((conf < THRESHOLD).mean()),
            }
            for key, split in splits.items():
                top1, found, conf = split.route(xs)
                row["pairs"][how][key] = {
                    "top1_either": float(((top1 == ga) | (top1 == gb)).mean()),
                    "both_found": float(
                        np.mean([ga[k] in f and gb[k] in f for k, f in enumerate(found)])
                    ),
                    "escalated": float((conf < THRESHOLD).mean()),
                }
        clean_ok = np.array([single[i] == gold[i] for i in pad_idx])
        clean_esc = np.array([single_conf[i] < THRESHOLD for i in pad_idx])
        row["padded"]["clean"] = {
            "acc": float(clean_ok.mean()),
            "escalated": float(clean_esc.mean()),
        }
        for (n, w), xs in padded.items():
            top1, _, conf = route(router, xs)
            cell = row["padded"][f"{n}_{w}"] = {
                "acc": float((top1 == gold[pad_idx]).mean()),
                "escalated": float((conf < THRESHOLD).mean()),
            }
            for key, split in splits.items():
                s1, _, sconf = split.route(xs)
                cell[f"{key}_acc"] = float((s1 == gold[pad_idx]).mean())
                cell[f"{key}_escalated"] = float((sconf < THRESHOLD).mean())
        rows.append(row)
        p, q = row["pairs"]["space"], row["padded"]
        print(
            f"{name:24} pairs: either {p['top1_either']:.2f}, both in top-2 "
            f"{p['both_in_top2']:.2f}, escalated {p['escalated']:.2f}   padded clean "
            f"{q['clean']['acc']:.3f} -> "
            + " ".join(
                f"{n}/{w} {q[f'{n}_{w}']['acc']:.3f}|{q[f'{n}_{w}']['split_acc']:.3f}"
                f"|{q[f'{n}_{w}']['split_filtered_acc']:.3f}"
                for n in LENGTHS
                for w in WHERE
            )
            + f"  ({time.time() - t0:.0f}s)",
            flush=True,
        )

    out = {
        "threshold": THRESHOLD,
        "n_pairs": len(pairs),
        "n_padded": int(args.padded),
        "examples": {
            **{f"pair_{how}": xs[0] for how, xs in joined.items()},
            **{f"padded_{n}_{w}": xs[0] for (n, w), xs in padded.items()},
        },
        "padded_words": {
            str(n): float(np.median([words(t) for t in padded[(n, "end")]])) for n in LENGTHS
        },
        "students": rows,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "stress.json").write_text(json.dumps(out, indent=2) + "\n")
    write_doc(out)
    print(f"-> {rel(RESULTS)}/stress.json, {rel(DOCS)}/stress.md")


def takeaways(rows: list[dict]) -> str:
    """The findings, read off the numbers (so they stay true if the run changes)."""
    by = {r["model"]: r for r in rows}
    out = []
    if "minilm_ft_teacher" in by and "minilm_frozen_teacher" in by:
        ft, fr = by["minilm_ft_teacher"], by["minilm_frozen_teacher"]
        e = f"{LENGTHS[-1]}_end"
        old = by.get("minilm_ft_teacher@64")
        if old:
            out.append(
                f"Reading only 64 tokens, as it was served until 2026-09-28, the served model "
                f"fell from {old['padded']['clean']['acc']:.3f} to {old['padded'][e]['acc']:.3f} "
                f"when the question ends a {LENGTHS[-1]}-word ticket (the cascade escalated "
                f"{old['padded'][e]['escalated']:.0%} of those). Reading its full 512 tokens, it "
                f"keeps {ft['padded'][e]['acc']:.3f} there and the same "
                f"{ft['padded']['clean']['acc']:.3f} on short queries, which never reach 64."
            )
        else:
            out.append(
                f"The served model keeps {ft['padded'][e]['acc']:.3f} when the question ends a "
                f"{LENGTHS[-1]}-word ticket (clean {ft['padded']['clean']['acc']:.3f})."
            )
        out.append(
            f"The frozen MiniLM, which reads 256 tokens, keeps {fr['padded'][e]['acc']:.3f} on "
            "the same tickets, and routed sentence by sentence with the off-topic filter it keeps "
            f"{fr['padded'][e]['split_filtered_acc']:.3f} "
            f"(clean {fr['padded']['clean']['acc']:.3f})."
        )
    e = f"{LENGTHS[-1]}_end"
    per = "; ".join(
        f"{r['model']} {r['padded'][e]['acc']:.3f} → {r['padded'][e]['split_filtered_acc']:.3f} "
        f"({r['filler_passing_filter']} of {len(FILLER)} filler sentences pass)"
        for r in rows
    )
    out.append(
        "Sentence routing helps most where the filter recognises chit-chat (TF-IDF is weak on "
        "short sentences either way) — whole "
        f"message → sentence routing with filter, question at the end of {LENGTHS[-1]} words: "
        f"{per}. A student that is confidently wrong on small talk lets a filler sentence "
        "outscore the real question. It does work as multi-intent detection: see the last column "
        "against the whole-message top-2 above."
    )
    return " ".join(out)


def write_doc(o: dict) -> None:
    rows = o["students"]
    lines = [
        "# Stress text: two questions at once, and a question inside a long ticket",
        "",
        "_Generated by `scripts/stress.py`. Teacher-label students through the serving code path "
        f"(`distilroute.students`, int8 ONNX for the fine-tuned ones); escalated = calibrated "
        f"confidence below the cascade's {o['threshold']}. No LLM calls._",
        "",
        "## Two questions in one message",
        "",
        f"{o['n_pairs']} pairs of test queries with different intents, sent as one message. "
        f'Example: _"{o["examples"]["pair_also"]}"_',
        "",
        "| model | join | top-1 is one of the two | first wins | second wins | both in top-2 "
        "| both in top-2, when each alone is right | escalated (vs alone) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        for how in JOINS:
            p = r["pairs"][how]
            lines.append(
                f"| {r['model']} | {how} | {p['top1_either']:.0%} | {p['top1_first']:.0%} | "
                f"{p['top1_second']:.0%} | **{p['both_in_top2']:.0%}** | "
                f"{p['both_in_top2_when_singles_right']:.0%} | {p['escalated']:.0%} "
                f"({r['pairs']['singles_escalated']:.0%}) |"
            )
    lines += [
        "",
        "## A question inside a long ticket",
        "",
        f"{o['n_padded']} test queries padded with intent-neutral filler to a median of "
        + " and ".join(f"{int(v)}" for v in o["padded_words"].values())
        + " words, the question at the start, in the middle or at the end. Example (120 words, "
        f'question at the end): _"{o["examples"]["padded_120_end"]}"_',
        "",
        "| model | clean | " + " | ".join(f"{n} words, {w}" for n in LENGTHS for w in WHERE) + " |",
        "|---|---:|" + "---:|" * len(LENGTHS) * len(WHERE),
    ]
    for r in rows:
        q = r["padded"]
        lines.append(
            f"| {r['model']} | {q['clean']['acc']:.3f} | "
            + " | ".join(
                f"{q[f'{n}_{w}']['acc']:.3f} ({q[f'{n}_{w}']['escalated']:.0%} esc.)"
                for n in LENGTHS
                for w in WHERE
            )
            + " |"
        )
    lines += [
        "",
        "Cells: accuracy vs gold on the same queries, and the share the cascade would escalate. "
        "Clean-column escalation: "
        + ", ".join(f"{r['model']} {r['padded']['clean']['escalated']:.0%}" for r in rows)
        + ".",
        "",
        "## Routing sentence by sentence",
        "",
        "Split the message on sentence ends, route each sentence alone, answer with the most "
        "confident one (escalate if even that is below the threshold). A sentence whose "
        f"answer reaches {FOUND} calibrated confidence counts as a detected intent — a cheap "
        "multi-intent signal, which the service returns as `also` (see the pairs columns). "
        "**naive** routes every sentence; **+ off-topic filter** first drops sentences whose "
        "entropy is above the level only 10 % of clean queries reach (the out-of-scope test of "
        "[6.2](oos.md), per sentence), and routes the whole message if nothing is left. Same "
        "models, no retraining.",
        "",
        "| model | variant | clean (whole message) | "
        + " | ".join(f"{n} words, {w}" for n in LENGTHS for w in WHERE)
        + " | pairs: one of the two | pairs: both detected |",
        "|---|---|---:|" + "---:|" * (len(LENGTHS) * len(WHERE) + 2),
    ]
    for r in rows:
        q = r["padded"]
        for key, label in [("split", "naive"), ("split_filtered", "+ off-topic filter")]:
            sp = r["pairs"]["also"][key]
            lines.append(
                f"| {r['model']} | {label} | {q['clean']['acc']:.3f} | "
                + " | ".join(
                    f"{q[f'{n}_{w}'][f'{key}_acc']:.3f} "
                    f"({q[f'{n}_{w}'][f'{key}_escalated']:.0%} esc.)"
                    for n in LENGTHS
                    for w in WHERE
                )
                + f" | {sp['top1_either']:.0%} | {sp['both_found']:.0%} |"
            )
    lines += [
        "",
        f"**What it shows.** {takeaways(rows)}",
        "",
        'Pairs columns use the "Also," join, where the two questions are separate sentences '
        "(the plain join often has no sentence break to split on).",
        "",
        "**Caveats.** The filler is written to be intent-neutral and is the same 18 sentences "
        "throughout; real tickets carry context that points *toward* an intent (and sometimes "
        "away from it). Joined pairs are two independent test queries, not how one person "
        "writes two questions. Both sets are rule-built, like the noise of "
        "[robustness](robustness.md).",
    ]
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "stress.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
