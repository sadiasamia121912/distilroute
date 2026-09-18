# distilroute — Roadmap

_Last updated: 2026-09-18. Project 2 of `../tabaudit/AI_ML_Portfolio_Projects.md`. Budget: **$0**._

## The pitch

Teams run a frontier LLM to route support tickets into a few dozen queues. It works, but it is
100–1000× the cost and latency of a small classifier. The production pattern is
**LLM-as-teacher → small student**: label once with the big model, train a tiny model on those
labels, serve the tiny model. This repo does that end to end on a public dataset and publishes
the one table every team actually wants: *accuracy vs. teacher · p50/p95 latency · cost per
1M requests*.

**Résumé bullet target:** "Distilled a 120B-parameter LLM ticket router into a 66M-parameter
model retaining **X %** of teacher accuracy at **Z×** lower latency and **$0** per 1M requests."

## Dataset — Banking77 (decided 2026-09-17)

PolyAI's Banking77: 13,083 real customer queries from a banking app, human-labelled into 77
intents. Chosen over the Kaggle/Bitext ticket sets because it is (a) real, not synthetic,
(b) a standard benchmark with published numbers to compare against (fine-tuned BERT ≈ 93 %,
DistilBERT ≈ 92 %), and (c) small enough to label for free.

Checked 2026-09-17: 10,003 train / 3,080 test · 77 intents in both · 35–187 examples per class
(median 127) · median query 47 chars, p95 151 · **0 duplicates, 0 train/test overlap**.
Source: `PolyAI-LDN/task-specific-datasets` on GitHub (plain CSVs; the HF copy needs a legacy
loader script, so we skip it).

## Protocol (the part an interviewer will ask about)

- **Gold labels are for evaluation only.** The student never sees a human label. It trains on
  teacher labels for the train split; every accuracy number is measured against gold on the
  untouched test split.
- **Teacher is zero-shot**: intent names plus a one-line *description* each
  (`data/intent_descriptions.json`, written from the names, no dataset queries — a test checks
  that). Example queries would leak gold labels into the "LLM-labelled" data and muddy the
  story. Decided 2026-09-17 on a 200-query comparison: names-only 0.860 → with descriptions
  0.885; `reasoning_effort` medium adds ~1 pt for 40 % more tokens, so **low + descriptions**.
- **Three numbers per model**: accuracy vs. gold (test), agreement with teacher (test), and
  macro-F1 vs. gold (77 classes are imbalanced 5:1, so accuracy alone can hide weak classes).
- **Teacher self-agreement**: relabel a 300-query sample a second time; report agreement. A
  teacher that changes its mind is noise the student cannot learn.

## Teacher — free tier (decided 2026-09-17)

Primary **Groq** (`openai/gpt-oss-120b` — the Llama models had left Groq's free catalogue by
2026-09-17; `GET /openai/v1/models` shows what a key can use), fallback **Google AI Studio**
(Gemini Flash).
Both are free with rate + daily caps, no card. Labelling is therefore a *resumable* script:
queries are sent in batches of 20 with the 77 intent names in the prompt, every result is
appended to a JSONL checkpoint, and re-running picks up where it stopped. Once labelled, the
API is never needed again — the label files are committed.

**Measured limits (2026-09-17, `x-ratelimit-*` headers on gpt-oss-120b):** 8,000 tokens/min,
1,000 requests/day, and an unadvertised ~200k tokens/day (waits jumped to minutes at 207k).
With descriptions the prompt is ~1,350 fixed + ~27 tokens per query, so batch size sets the
daily throughput: batch 20 ≈ 94 tok/query ≈ 2,000 queries/day; batch 50 ≈ 54 ≈ 3,700/day;
batch 100 ≈ 40 ≈ 5,000/day. Bigger batches must pass the 200-query accuracy gate first.
Plan: label a 3,000-row train subset first (1b.3 wants subsets anyway), start students on it,
keep filling in the rest daily.

Why not local (Ollama)? This laptop is 8 GB RAM / no GPU: a 3B model would be a weak teacher
and a 70B one does not fit. See `docs/teacher.md` once written.

---

## Phase 1 — Data + teacher labels  (2 days, mostly waiting on rate limits)

- [x] **1.1** `scripts/download_data.py` → `data/raw/{train,test}.csv`, `categories.json`, with the sanity checks above. _(2026-09-17)_
- [x] **1.2** `distilroute/teacher.py`: provider-agnostic client (Groq / Gemini via plain HTTP), batched zero-shot prompt, strict parsing against the 77-name label set, per-item retry for anything unparseable. Tests use a fake transport — no key needed. _(2026-09-17)_
- [x] **1.3** `scripts/label.py`: resumable labelling with JSONL checkpoint, rate-limit backoff, `--split test|train --limit N`. _(2026-09-17)_
- [x] **1.4** Get a Groq key (free, https://console.groq.com) → `.env` as `GROQ_API_KEY`. _(2026-09-17)_
- [x] **1.4b** Teacher config comparison on 200 test queries (`data/labels/test.cmp_*.jsonl`) — see Protocol. _(2026-09-17)_
- [~] **1.5** Label the **test** split first (3,080) → teacher accuracy vs. gold. This is the ceiling every student is measured against; if it is below ~85 % switch teacher before labelling train.
  _2026-09-17: v1 run stopped at 2,460 (kept as `test.v1_partial.jsonl`; file is intent-sorted so it covers ~60 of 77 intents — not a random sample). It exposed that 33 descriptions mis-described the dataset's actual intent semantics (`get_physical_card` = PIN questions, 0 % correct). v2 descriptions written from TRAIN examples. Next: `scripts/gates.ps1` (v2 / top-3 / batch 50 / batch 100 on the 200-query sample), then relabel test with the winner._
- [ ] **1.6** Label the **train** split (10,003). Commit `data/labels/*.jsonl`.
- [~] **1.7** `scripts/teacher_report.py` → `docs/teacher.md` (accuracy, macro-F1, parse-failure rate, top-k coverage, weakest intents, confusions; every gate run in one table). _(script done 2026-09-18; regenerate when test completes)_ Still to do: self-agreement run (300 test queries relabelled).

## Phase 1b — Make it advanced, for free  (decided 2026-09-17)

The user's brief: as advanced as possible at $0. These are the levers, ordered by when they
must be decided. 1b.1 has to land *before* 1.6 because relabelling 10k rows on a rate-limited
free tier costs a day.

- [x] **1b.1 Soft labels.** _(gated 2026-09-18: top-1 unchanged at 0.905, gold in top-3 for 96 %; +25 % tokens. Adopted.)_ Teacher returns its **top-3 ranked intents** per query, same call
  count, ~2× output tokens. Students train on a soft target (rank-weighted, Hinton-style
  distillation) as well as the hard top-1; report both. Gate: top-1 accuracy on the 200-query
  sample must not drop vs. the single-label prompt.
- [ ] **1b.2 Confidence cascade.** Student answers when confident, escalates to the LLM
  otherwise. Curve: accuracy and LLM-cost vs. escalation fraction. Needs a *calibrated*
  student → report ECE, apply temperature scaling.
- [~] **1b.3 Data-efficiency curve.** Student accuracy vs. number of teacher labels
  (500 / 1k / 2k / 5k / 10k). "How many LLM calls do you actually need?" _(`scripts/data_curve.py`,
  2026-09-18, on gold: frozen MiniLM 0.61 / 0.74 / 0.84 / 0.89 / 0.92 / 0.93 at 250 / 500 / 1k / 2k /
  5k / 10k; TF-IDF 0.44 → 0.91 over the same sizes — the pretrained encoder is worth ~10 pts at 1k
  labels, 2 pts at 10k. Re-run with `--labels teacher` once train is labelled.)_
- [ ] **1b.4 Model-size Pareto.** TinyBERT (14M) / MiniLM-L6 (22M) / DistilBERT (66M) on
  Colab; accuracy vs. params vs. CPU latency. Then ONNX + int8 quantisation of the winner.
- [ ] **1b.5 tabaudit on the teacher labels.** Run `tabaudit audit` on the LLM-labelled train
  set; does dropping the flagged label-noise rows help the student? Cross-project.
- [ ] **1b.6 (later)** Second teacher (Gemini Flash) on the test split: agreement as a noise
  signal, and "which free teacher is best".

## Phase 2 — Students  (2 days)

- [x] **2.1** Baseline: TF-IDF (word + char n-grams) + logistic regression. Trained on **gold** first as the reference (what supervised learning gets), then on **teacher** labels (the distilled version). _(2026-09-17: gold-trained baseline done; teacher-trained waits on 1.6)_
- [~] **2.2** `scripts/setfit_student.py` (all-MiniLM-L6-v2, 22M): `--mode frozen` (embeddings + LR head, all rows) and `--mode setfit` (contrastive few-shot, `--per-class 16`). _(2026-09-18: **frozen on gold = 0.930 / 0.930 macro-F1, 15 ms** — beats TF-IDF 0.913 and matches published fine-tuned DistilBERT with no fine-tuning. SetFit few-shot 16/intent (1,232 rows, 624 s CPU) = **0.867** — few-shot costs ~6 pts vs the frozen encoder on all 10k rows.)_
- [~] **2.3** `scripts/finetune.py` (distilbert 66M / minilm 22M / tinybert 14M; hard or `--soft`
  top-3 targets; `--export-onnx` → fp32 + dynamic-int8 graphs, int8 accuracy and latency recorded)
  + `notebooks/finetune_colab.ipynb` that clones the repo, runs every config on a T4 and zips
  `results/` + `models/` back. _(2026-09-18: written and smoke-tested on CPU end to end (TinyBERT
  int8: 2.1 ms p50, 14.6 MB). Still to do: run it on Colab — gold now, teacher once 1.6 lands.)_
- [x] **2.4** `scripts/evaluate.py` → `docs/results.md`: every run in `results/` (metrics JSON + test probabilities, the contract in `baseline.py::save_run`) × {acc, macro-F1, agreement w/ teacher, ECE, latency} + the cascade preview table (1b.2). _(2026-09-18; fills in as runs land)_

## Phase 3 — Serving + the cost/latency table  (1–2 days)

- [ ] **3.1** FastAPI `POST /route` with a `model=` switch (teacher | tfidf | setfit | distilbert); same request/response schema for all.
- [ ] **3.2** `scripts/bench_latency.py`: 500 requests per model, p50/p95, on this laptop (CPU). Teacher latency measured end-to-end through the free-tier API.
- [ ] **3.3** Cost per 1M requests: teacher from the provider's *paid* price list (the free tier is not a production option — say so), students from CPU-seconds on a priced cloud VM.
- [ ] **3.4** Dockerfile (student only — ~300 MB image), `docker run` → `/route` works. README with the final table + the pitch.

## Phase 4 — Publish  (½ day)

- [ ] **4.1** README results table, résumé bullet numbers into `../tabaudit/AI_ML_Portfolio_Projects.md`.
- [ ] **4.2** Public repo, LinkedIn post.

## Phase 5 — Stretch: prove it is a method, not a banking demo  (optional, ~1 day)

- [ ] Second dataset from a very different domain (toxic-comment detection or news topics),
  run through the *same* scripts with only a download script + descriptions file added;
  second results table. Answers "does this only work for banking?" with a number.
- **Scope, on the record:** distillation into a fixed-label classifier covers "read a short
  text, pick a bucket" — routing, moderation, triage, sentiment. It does not cover free-text
  outputs (summaries, replies). The README says so.

---

## Rules of thumb

- Keep the venv lean — this laptop has ~20 GB free. Torch/SetFit go in a separate
  `requirements-train.txt` and get installed only when Phase 2.2 starts; DistilBERT trains on
  Colab, never here.
- Commit the label files; never commit `.env`.
- Every number in `docs/` comes from a script in `scripts/` that can be re-run.

## Session log

**2026-09-18 (curves, fine-tune script)** — Test labelling restarted as a detached process
(`logs/label_test.log`; 380 → 1,080 and running). Wrote `scripts/data_curve.py` (1b.3) and ran
it on gold for both CPU students: MiniLM frozen reaches 0.84 with 1k random labels and 0.92
with 5k; TF-IDF needs 5k to reach 0.89. `evaluate.py` renders the curves into `docs/results.md`.
Wrote `scripts/finetune.py` (2.3 / 1b.4: DistilBERT / MiniLM / TinyBERT full fine-tune, soft
top-3 targets, ONNX + int8 export) with a Colab wrapper notebook; smoke-tested on CPU.
Next: run the notebook on Colab for the gold rows; `bench_latency.py` + FastAPI (3.1–3.2) can
be built against the gold-trained models while teacher labels fill in.

**2026-09-17 (kickoff)** — Repo created, Banking77 chosen and checked, teacher client +
resumable labeller written with fake-transport tests, TF-IDF baseline on gold labels run.

**2026-09-18 (students, pause)** — 550B second-teacher gate final: 0.835 / 7.5 % unparsed vs
gpt-oss-120b 0.905 → gpt-oss stays. Installed torch-cpu + sentence-transformers + setfit
(`requirements-train.txt`). **Frozen MiniLM-L6 + LR on gold: 0.930**, the best student so far
and the likely deployment candidate. SetFit few-shot (16/intent) on gold: 0.867. Groq
test-split labelling at 360/3,080 when the session paused — it only progresses while a
labeller process is running, so **start it in a normal terminal, not inside the assistant**
(one process per label file at a time; it resumes from the checkpoint):

```powershell
cd C:\Users\User\dev\distilroute
.\.venv\Scripts\python.exe scripts/label.py --split test --descriptions --top-k 3
```

Next session: re-run the SetFit few-shot on gold, keep labelling test → 3k train subset, then
`--labels teacher` for every student, then the Colab notebook for DistilBERT/TinyBERT.

**2026-09-18 (second host, reports)** — Cerebras and OpenRouter no longer serve gpt-oss-120b
free; main teacher stays on Groq (~9 days of background labelling). OpenRouter's free
nemotron-3-ultra-550b wired in as the second teacher — its gate exposed three failure modes of
free reasoning endpoints (null content, empty batches, 200-with-error bodies), all now handled
with tests; at 100/200 it reads 0.76 with 15 % unparsed, i.e. a weaker teacher. Wrote
`distilroute/data.py`, `scripts/teacher_report.py` (→ docs/teacher.md) and
`scripts/evaluate.py` (→ docs/results.md, with ECE and the cascade preview); baseline now
saves test probabilities under a shared contract.

**2026-09-18 (gates)** — Same 200 queries: v1 0.885 → **v2 0.905** (macro-F1 0.851 → 0.882);
top-3 keeps top-1 at 0.905 with gold in the list 96 % of the time; batch 50 and 100 both drop
to 0.885 — rejected, the teacher is the ceiling. Config: **v2 + top-3 + batch 20**, ~140
tokens/query → ~1,400 queries/day on Groq alone (~9 days). Proposed: Cerebras free tier serves
the same gpt-oss-120b with a far larger daily allowance; add as a provider, gate agreement on
the same 200, then split the work. Test split relabelling started on Groq meanwhile.

**2026-09-17 (descriptions v2)** — Partial v1 test labels (2,460) read 0.943 but the file is
sorted by intent, so that is optimistic; what it did show was four intents at 0–35 % because
the *names* mislead (`get_physical_card` is about the PIN). Wrote v2 descriptions from three
random train examples per intent (33 changed), stopped the v1 run to save budget for the v2
gates. Budget maths: ~200k tokens/day → gates (80k) tomorrow, then test at batch ≥50 (~170k),
then a 3k train subset, then the rest.

**2026-09-17 (teacher online)** — First smoke run 404'd: Groq's free catalogue no longer has
Llama; switched to `openai/gpt-oss-120b` and made non-429 4xx fail fast instead of retrying.
Two label-name quirks bit the parser (`Refund_not_showing_up`, `reverted_card_payment?`) —
now matched loosely, answered canonically, with tests. The 40-query smoke read 77 %, which was
the bug plus a tiny sample; the 200-query comparison says 86–89 %. Chose low reasoning +
descriptions (0.885, cheapest per token). Full test-split labelling started (1.5).
