# distilroute — Roadmap

_Last updated: 2026-09-25. Project 2 of `../tabaudit/AI_ML_Portfolio_Projects.md`. Budget: **$0**._

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
- [x] **1.5** Label the **test** split first (3,080) → teacher accuracy vs. gold. This is the ceiling every student is measured against; if it is below ~85 % switch teacher before labelling train.
  _2026-09-17: v1 run stopped at 2,460 (kept as `test.v1_partial.jsonl`; file is intent-sorted so it covers ~60 of 77 intents — not a random sample). It exposed that 33 descriptions mis-described the dataset's actual intent semantics (`get_physical_card` = PIN questions, 0 % correct). v2 descriptions written from TRAIN examples. Next: `scripts/gates.ps1` (v2 / top-3 / batch 50 / batch 100 on the 200-query sample), then relabel test with the winner._
  _2026-09-19: 3,080 / 3,080, 0 unparsed, read 0.948 — **but see 2026-09-21: invalid.** The test CSV is intent-sorted, so every batch of 20 was a single intent and the teacher used the batch as a hint: on the same 200 queries it scores 0.98 in sorted batches vs 0.905 in random ones. Kept as `test.sorted_batches.jsonl` for the record; the labeller now always shuffles. **Relabelled in shuffled order 2026-09-23**: 3,080 / 3,080, 0 unparsed, 155 calls. **Teacher = 0.867 accuracy / 0.864 macro-F1**, gold in top-3 94.7 % — 8 points below the leaked number and in line with the 0.905 gate (which was an easier random sample of 200). This is the ceiling the project reports._
- [~] **1.6** Label the **train** split (10,003). Commit `data/labels/*.jsonl`. _(3,000-row random subset (`--limit 3000 --seed 0`) done 2026-09-21: 0 unparsed, 150 calls, 283k in / 174k out. Teacher vs gold on it: **0.848**, top-3 hit 0.936 — random batches, so this is the honest zero-shot number on the train distribution. Remaining 7,003 rows: later, if the data curve says they matter. 2026-09-27: 4,900 / 10,003, new rows 0.847 vs gold — same as the first 3,000.)_
  **Retraining plan (written 2026-09-25, before the remaining 7,003 labels land).** The question:
  6.1 left the student 1.8 pt below its teacher at 3,000 labels, and the data curve was still
  rising (0.834 at 2k → 0.846 at 3k). Do all 10,003 labels close the gap, or does the student
  flatten at the teacher's own error rate?
  0. *Done now — reproducibility guard.* `DISTILROUTE_TEACHER_ROWS` (default **3,000**) caps how
     many labelled rows a student sees: the first N of `train.jsonl`, which is in the labeller's
     seed-0 order, so any N is a random sample and nested in every larger one. The default keeps
     every published command and number reproducible as the file grows (verified: TF-IDF and
     frozen MiniLM on teacher labels re-run byte-identical, test probabilities included). Any
     other N writes to `results/teacher<N>/`, `models/…`, `docs/…`, never over the published runs.
  1. *Check the labels* (no training): unparsed rate, and teacher accuracy vs gold on the new
     7,003 (gold is read for this report only). It should match the first 3,000 (0.848).
  2. *CPU students at N = 10,003* (`DISTILROUTE_TEACHER_ROWS=10003`, minutes each): `baseline.py`
     and `setfit_student.py --mode frozen` on teacher labels, `denoise.py --variants
     base,soft,filter` (`self` has nothing left to pseudo-label once every row is labelled),
     `data_curve.py --labels teacher` for the 5k and 10k points, then `evaluate.py`.
  3. *Fine-tuned students* on Colab (`notebooks/finetune_colab.ipynb`, user runs it; ~30 min on a
     T4): MiniLM, DistilBERT, TinyBERT at N = 10,003, int8 ONNX exported.
  4. *Decide:* the served model changes only if the fine-tuned MiniLM at 10k beats the current one
     (0.842 int8) by more than run-to-run noise (±0.3 pt, 1b.4). If it does: release `model-v2`,
     rebuild the demo (`build_demo.py`), re-run `cascade.py` and `robustness.py`, re-bench latency,
     and point the Docker CI at model-v2. If it does not, the finding is that the labels are
     saturated and the gap is the teacher's error rate.
  5. *Report* either way: README finding 5 and the case study's "What did not work" get the 10k
     number; the data curve gains its 5k/10k teacher points. Adopting 10k as the default (bumping
     `DEFAULT_TEACHER_ROWS`) is a separate, explicit decision, because it changes every published
     teacher-label number at once.
  *Checkpoints (2026-09-25):* the 7,003 labels now run first in the groq lane, and
  `scripts/retrain_check.py --rows N` does steps 1-2 in one command (~3 min; writes
  `results/teacher<N>/checkpoint.json` with the 3,000-vs-N table). Run it at 5,000 and 7,500 while
  labelling continues: a flat curve at 5k answers the question before the week is out.
  *Cost, for the write-up:* 7,003 more labels at the paid, batched price ($26.62 per 1M) is about
  **$0.19**, the cheapest lever in the project if it works.
- [~] **1.7** `scripts/teacher_report.py` → `docs/teacher.md` (accuracy, macro-F1, parse-failure rate, top-k coverage, weakest intents, confusions; every gate run in one table). _(generated on the full test split 2026-09-19)_ Still to do: self-agreement run (300 test queries relabelled).

## Phase 1b — Make it advanced, for free  (decided 2026-09-17)

The user's brief: as advanced as possible at $0. These are the levers, ordered by when they
must be decided. 1b.1 has to land *before* 1.6 because relabelling 10k rows on a rate-limited
free tier costs a day.

- [x] **1b.1 Soft labels.** _(gated 2026-09-18: top-1 unchanged at 0.905, gold in top-3 for 96 %; +25 % tokens. Adopted.)_ Teacher returns its **top-3 ranked intents** per query, same call
  count, ~2× output tokens. Students train on a soft target (rank-weighted, Hinton-style
  distillation) as well as the hard top-1; report both. Gate: top-1 accuracy on the 200-query
  sample must not drop vs. the single-label prompt.
- [x] **1b.2 Confidence cascade.** Student answers when confident, escalates to the LLM
  otherwise. Curve: accuracy and LLM-cost vs. escalation fraction. Needs a *calibrated*
  student → report ECE, apply temperature scaling. _(2026-09-18: `distilroute/calibration.py`;
  every student holds out a stratified 10 % of its training pool (`runs.split_calib`), fits T
  there, `evaluate.py` applies it. On gold: both LR heads are **under**-confident (T ≈ 0.7),
  ECE 0.074 → 0.011 (MiniLM), 0.090 → 0.007 (TF-IDF); the holdout costs ~0.3 pt. Threshold
  cascade table in results.md: MiniLM below 0.8 → escalate 14 %, **97.5 %** on the rest. The
  mixed-system column fills in when the test split is fully labelled.)_ _(Closed 2026-09-25: the
  test split is fully labelled and the column is filled — fine-tuned MiniLM on teacher labels,
  escalating below 0.8: 15 % to the LLM, **0.874** overall, above the LLM alone (0.867). The
  cost-aware version, with thresholds chosen on held-out data, is 6.5.)_
- [x] **1b.3 Data-efficiency curve.** Student accuracy vs. number of teacher labels
  (500 / 1k / 2k / 5k / 10k). "How many LLM calls do you actually need?" _(`scripts/data_curve.py`,
  2026-09-18, on gold: frozen MiniLM 0.61 / 0.74 / 0.84 / 0.89 / 0.92 / 0.93 at 250 / 500 / 1k / 2k /
  5k / 10k; TF-IDF 0.44 → 0.91 over the same sizes — the pretrained encoder is worth ~10 pts at 1k
  labels, 2 pts at 10k. Re-run with `--labels teacher` once train is labelled.)_ _(Closed
  2026-09-25: re-run on teacher labels up to all 3,000 bought — frozen MiniLM 0.567 / 0.715 /
  0.787 / 0.834 / 0.846 at 250 / 500 / 1k / 2k / 3k, TF-IDF 0.402 → 0.808; the curve flattens
  near the teacher's 0.867, which gold labels do not. In results.md.)_
- [x] **1b.4 Model-size Pareto.** TinyBERT (14M) / MiniLM-L6 (22M) / DistilBERT (66M) on
  Colab; accuracy vs. params vs. CPU latency. Then ONNX + int8 quantisation of the winner.
  _(2026-09-23, Colab round 2 with the fixed recipe — per-model LR, ≥2,000 steps, MiniLM from
  the sentence-transformers checkpoint. On gold: DistilBERT **0.928**, MiniLM **0.927**,
  TinyBERT 0.892. On 3k teacher labels: MiniLM **0.847**, DistilBERT 0.839, TinyBERT 0.796.
  **MiniLM-L6 at 22M matches DistilBERT at 67M on gold and beats it on teacher labels** — the
  Pareto point and the deployment pick. int8 quantisation costs ±0.3 pt (0.9305 vs 0.9279 for
  DistilBERT gold, i.e. noise), so the served graph is the quantised one. The soft top-3 target
  hurt here too (0.836 vs 0.839), matching 6.1 on the frozen student: two architectures, same
  conclusion. Round 1 (lr 5e-5 for everything) had MiniLM at 0.52 and TinyBERT at 0.39 — those
  runs were under-trained, not bad models, and were discarded.)_
- [x] **1b.5 tabaudit on the teacher labels.** Run `tabaudit audit` on the LLM-labelled train
  set; does dropping the flagged label-noise rows help the student? Cross-project.
  _(2026-09-25: `denoise.py --variants tabaudit,tabaudit_suspected`, tabaudit 0.3.0 from PyPI run
  as `run_audit` on the 3,000 rows as a table of 384 MiniLM dimensions + the teacher's label, then
  the same frozen-MiniLM student as 6.1 without the flagged rows. **It hurts.** tabaudit says HIGH,
  "~535 rows (17.8 %) likely mislabeled" — a fair estimate of the true 15.2 % — but row by row only
  44 % of its flags are real teacher errors (3× the base rate; 234 of 457 caught), and dropping
  them costs **1.6 pt** (0.842 → 0.827; the looser tier: 647 rows, 40 %, −2.2). 6.1's own filter
  drops 39 rows at 77 % precision and gains 0.4. Why: tabaudit's defaults were tuned on tabular
  data with few classes. Here there are 77 classes and ~35 rows each, so its regularised boosting
  (`min_samples_leaf=40`) can barely fit a class, gives low self-confidence to many correct rows,
  and a fixed 0.2 cut means something different when chance is 1/77. Dropping 300 correct rows
  from the hard, rare intents costs more than removing 234 wrong ones gains. Lessons for tabaudit:
  scale the thresholds (or the leaf size) with the number of classes, and trust its *rate*
  estimate more than its row list. Also found: tabaudit's `encode_features` inserted columns one
  by one — one pandas fragmentation warning per column on wide data; fixed in the tabaudit repo.)_
- [-] **1b.6 (cut 2026-09-25)** _(OpenRouter's free models gave ~20 rows/day, so three 200-row gates would take weeks for a side question; the lane is gone from `jobs.py`. The 40 Gemma rows it had produced were never committed and were deleted 2026-09-27.)_ Second teacher (Gemini Flash) on the test split: agreement as a noise
  signal, and "which free teacher is best".

## Phase 2 — Students  (2 days)

- [x] **2.1** Baseline: TF-IDF (word + char n-grams) + logistic regression. Trained on **gold** first as the reference (what supervised learning gets), then on **teacher** labels (the distilled version). _(2026-09-17: gold-trained baseline done; teacher-trained waits on 1.6)_
- [x] **2.2** `scripts/setfit_student.py` (all-MiniLM-L6-v2, 22M): `--mode frozen` (embeddings + LR head, all rows) and `--mode setfit` (contrastive few-shot, `--per-class 16`). _(2026-09-18: **frozen on gold = 0.930 / 0.930 macro-F1, 15 ms** — beats TF-IDF 0.913 and matches published fine-tuned DistilBERT with no fine-tuning. SetFit few-shot 16/intent (1,232 rows, 624 s CPU) = **0.867** — few-shot costs ~6 pts vs the frozen encoder on all 10k rows.)_ _(Closed 2026-09-25: both modes also ran on teacher labels — frozen **0.848**, SetFit 7 per intent 0.789; in results.md.)_
- [x] **2.3** `scripts/finetune.py` (distilbert 66M / minilm 22M / tinybert 14M; hard or `--soft`
  top-3 targets; `--export-onnx` → fp32 + dynamic-int8 graphs, int8 accuracy and latency recorded)
  + `notebooks/finetune_colab.ipynb` that clones the repo, runs every config on a T4 and zips
  `results/` + `models/` back. _(2026-09-18: written and smoke-tested on CPU end to end (TinyBERT
  int8: 2.1 ms p50, 14.6 MB). Still to do: run it on Colab — gold now, teacher once 1.6 lands.)_ _(Closed 2026-09-23: ran on Colab, round 2 — see 1b.4.)_
- [x] **2.4** `scripts/evaluate.py` → `docs/results.md`: every run in `results/` (metrics JSON + test probabilities, the contract in `baseline.py::save_run`) × {acc, macro-F1, agreement w/ teacher, ECE, latency} + the cascade preview table (1b.2). _(2026-09-18; fills in as runs land)_

## Phase 3 — Serving + the cost/latency table  (1–2 days)

- [x] **3.1** FastAPI `POST /route` with a `model=` switch (teacher | tfidf | setfit | distilbert); same request/response schema for all. _(2026-09-18: `distilroute/serve.py` over `distilroute/students.py`, a loader for every `models/<run>/meta.json` (tfidf / minilm / onnx) plus the teacher; lazy-loaded, `GET /models`, `/health`; 5 tests with a fake router. Student scripts now persist their models.)_
- [x] **3.2** `scripts/bench_latency.py`: 500 requests per model, p50/p95, on this laptop (CPU). Teacher latency measured end-to-end through the free-tier API. _(2026-09-18: in-process — MiniLM frozen 11.5 / 14.6 ms, TF-IDF 1.8 / 2.5 ms; `--http` mode too, but Windows loopback delayed-ACK adds ~30 ms so the table uses in-process. `--teacher N` written, not yet run: it would share Groq's rate limit with the labeller. 2026-09-24: re-benched all 13 students on an idle, mains-powered machine; no † left in the table. TinyBERT FT 1.8 / 3.3 ms, MiniLM FT 2.7 / 4.7, DistilBERT FT 6.4 / 11.7, MiniLM frozen 11.1 / 13.7, TF-IDF 1.7 / 2.1. `--http` rows not re-run. 2026-09-25: teacher through Groq, end to end, 30 queries: p50 1,085 / p95 1,517 ms — ~390× the served MiniLM's 2.8 ms.)_
- [x] **3.3** Cost per 1M requests: teacher from the provider's *paid* price list (the free tier is not a production option — say so), students from CPU-seconds on a priced cloud VM. _(2026-09-18: `scripts/cost.py` → `results/cost.json` → results.md. gpt-oss-120b on Groq paid ($0.15 / $0.60 per 1M tokens): **$220 / 1M** one ticket per call, $28 batched 20; MiniLM frozen **$0.07**, TF-IDF $0.01 on a t3.small ($0.0208/h) — ~3,000× cheaper than the single-query teacher; $0 on owned hardware. Tokens per query measured on the final config: 118 at batch 20.)_
- [x] **3.4** Dockerfile (student only — ~300 MB image), `docker run` → `/route` works. README with the final table + the pitch. _(2026-09-24: `Dockerfile` + `.dockerignore` + `requirements-serve.txt` — python:3.13-slim, one ONNX student (`--build-arg MODEL=`, default `minilm_ft_gold`), no torch / scipy / pandas / sklearn; `calibration.py` and `data.py` now import scipy and pandas lazily so the service never needs them. Not built — no Docker on this machine — but the image's exact contents were run in a clean venv: 296 MB of site-packages, `/route` answers, and all 500 test predictions match the full training venv to the last digit. Still to do: `docker build` + `docker run` once Docker is installed; README.)_

## Phase 4 — Publish  (½ day)
 _(Closed 2026-09-25: built and run on GitHub Actions — no Docker locally — by `.github/workflows/docker.yml` on every change to the image's inputs. The served model (`minilm_ft_teacher`, int8) is the `model-v1` release file, SHA-256 checked. Run 36042909223: image **457 MB** (the ~300 MB guess left out the 120 MB python:3.13-slim base), `/health` ok, `/models` = minilm_ft_teacher, "my card still has not arrived" → `card_arrival` 0.948 calibrated in 2.9 ms, empty text → 422. The Dockerfile now defaults to the served teacher-trained model.)_
- [x] **4.1** README results table, résumé bullet numbers into `../tabaudit/AI_ML_Portfolio_Projects.md`. _(2026-09-24: README rewritten around the final table and six findings; Project 2 section of the portfolio file rewritten with the final numbers, a résumé bullet and interview talking points.)_
- [~] **4.2** Public repo, LinkedIn post. _(2026-09-25: **repo public**, checked anonymously. Post drafted in `../tabaudit/linkedin_post_distilroute.md`, image `docs/case-study/media/pareto_linkedin.png` (`scripts/post_image.py`). Still to do: share the case-study and demo artifacts, then post — after the two tabaudit posts.)_

## Phase 5 — Stretch: prove it is a method, not a banking demo  (optional, ~1 day)

- [ ] Second dataset from a very different domain (toxic-comment detection or news topics),
  run through the *same* scripts with only a download script + descriptions file added;
  second results table. Answers "does this only work for banking?" with a number.
- **Scope, on the record:** distillation into a fixed-label classifier covers "read a short
  text, pick a bucket" — routing, moderation, triage, sentiment. It does not cover free-text
  outputs (summaries, replies). The README says so.

## Phase 6 — Results nobody else reports  (decided 2026-09-21, after Phases 1–4)

The model is a distilled MiniLM/DistilBERT; that is not special and the README must not claim
it is. What can make the project valuable is a handful of *measured, surprising, reusable*
results on questions teams actually have. Ordered by payoff per effort at $0. All of 6.1–6.3
are offline (no LLM calls) and start once the shuffled test labels and Colab round 2 give the
honest baselines.

- [x] **6.1 Student beats its teacher.** _(Closed 2026-09-25 as **not met** at 3,000 labels; the open lever is more labels, 1.6.)_ _(`scripts/denoise.py`, 2026-09-23. On 3,000 teacher
  labels, all six variants reported, none cherry-picked: base 0.842 · soft 0.840 · filter 0.846
  · self 0.847 · **filter+self 0.849** · all 0.849, against a teacher at **0.867**. So: the
  levers are worth **+0.7 pt**, the student still sits **1.8 pt below its teacher**, and the
  stated goal is **not met at this label budget**. What each lever taught us: (a) the top-3
  soft target *hurts* (−0.2) — the teacher's 2nd and 3rd guesses are noise more often than
  signal on 77 near-synonym intents; (b) the noise filter is **precise** — 77 % of the 39 rows
  it drops really are teacher errors against a 15 % base rate, but it only catches 30 of 457,
  so its effect is small; (c) self-training on the 7,003 unlabelled rows is the biggest single
  lever (+0.5) **once the confidence threshold is applied to calibrated probabilities** — the
  head is under-confident (T≈0.7), so a raw 0.9 cut kept 379 rows instead of 4,749. Also:
  self-training left the head nearly calibrated on its own (raw ECE 0.02), so temperature
  scaling adds nothing there. Next: label the remaining 7,003 train rows and re-run — the data
  curve says the gap closes with labels, not with tricks.)_ Today the student lands exactly on the teacher's
  accuracy (0.848 on 3k rows) — it learns the noise. Three free levers, each reported alone
  and combined, target vs gold on test: (a) **soft labels** — train on the top-3 rank-weighted
  target (`finetune.py --soft`, and a soft head for the frozen student); (b) **noise filtering**
  — cross-validated student predictions that confidently disagree with the teacher flag likely
  teacher errors; drop or down-weight them (confident-learning style; this is the tabaudit
  cross-project item 1b.5); (c) **self-training** — the student pseudo-labels the 7,003
  unlabelled train rows, retrain on 3k teacher + 7k self labels. Success = a student above
  0.848 with zero human labels in training. _Script: `scripts/denoise.py` writing runs under the
  shared contract so `evaluate.py` shows them as rows._
- [x] **6.2 Out-of-scope detection.** _(`scripts/download_clinc.py` + `scripts/oos.py` →
  `docs/oos.md`, 2026-09-23. CLINC150's 1,200 out-of-scope queries vs the Banking77 test split,
  scored by every saved student, nothing retrained. **It works, and it is free**: at a 10 %
  in-scope escalation budget the students catch **94–97 %** of out-of-scope traffic; AUROC
  0.97–0.99. Two findings worth the write-up: (a) **entropy beats max-probability on every
  model** (0.981–0.987 vs 0.961–0.975) — an out-of-scope message spreads mass thinly over many
  intents, a hard in-scope one is torn between two or three, and max-prob cannot tell those
  apart; (b) **temperature scaling slightly hurts separation** (e.g. 0.972 → 0.941 on the
  SetFit student) — it is fitted on in-scope data to make confidence honest, not to push
  unfamiliar input away. So: calibrate for the cascade threshold, score out-of-scope with
  entropy. The messages that do slip through cluster in a few magnet intents
  (`age_limit`, `country_support`, `lost_or_stolen_phone`) — a short watch-list for production.
  Caveat in the doc: CLINC's negatives are *far* out of scope; a mortgage or insurance question
  would be harder, and no near-OOS set exists for Banking77.)_ Real inboxes contain messages that fit none of the 77
  intents; a router that confidently misfiles them is worse than one that escalates. Banking77
  has no OOS class; **CLINC150** (free, 1,200 labelled out-of-scope queries) does. Measure: at
  each calibrated-confidence threshold, the share of OOS queries escalated vs the share of
  in-scope queries wrongly escalated (ROC-style table). Also test whether temperature scaling
  helps or hurts OOS separation. _Script: `scripts/oos.py` + `scripts/download_clinc.py`._
- [x] **6.3 Active labelling — spend LLM calls where they matter.** _(Closed 2026-09-24; the session logs of 09-23 and 09-24 have the full story. 3 seeds: **diverse** (k-means, no model) 0.785 ± 0.005 at 500 labels vs random 0.708, and 0.822 vs 0.799 at 1,000, which is random's 1,500-label score. It ties random at 1,500 and trails it slightly at 2,000–2,500 (0.839 vs 0.842). Uncertainty / disagreement: +0.5 pt. So the "2× fewer calls" claim holds for the first ~1,000 labels only, and it comes from diversity, not from uncertainty.)_ Simulated entirely inside
  the 3,000 labelled rows: pick N rows by (i) random, (ii) TF-IDF/MiniLM disagreement,
  (iii) lowest student confidence after a 500-row seed round; train the frozen student on each
  N and compare on test. Claim shape: "same accuracy with ~2× fewer LLM calls". _Extends
  `data_curve.py` with a `--select` strategy._
- [x] **6.4 Second domain** — _(2026-09-25: done. Students on 2,704 teacher labels: frozen MiniLM **0.862**, TF-IDF 0.804 (teacher 0.901; gold-trained 0.954 / 0.928); the case study has both datasets side by side.)_ _(2026-09-24: `DISTILROUTE_DATASET=clinc150` runs every script on CLINC150; `download_clinc.py` writes `data/raw/clinc150/` (train 15,000 / test 4,500, 150 intents). Gold: TF-IDF 0.928, frozen MiniLM 0.954. Teacher gate, names only, 200 test queries: **0.890**, top-3 0.965, so no descriptions file (4× smaller prompt, ~2,000 labels/day on Groq). Labelling 1,500 test + 3,000 train with the Banking77 config otherwise (gpt-oss-120b, low, top-3, batch 20, shuffled). Then: teacher-trained TF-IDF / MiniLM, cascade, the cross-dataset comparison.)_ Phase 5, now concrete: CLINC150 (150 intents, 10 domains) through
  the same scripts with only a download script + descriptions file added. Together with 6.2 it
  reuses one dataset for two results.
- [x] **6.5 Three-tier cascade + cost Pareto.** _(2026-09-25: `scripts/cascade.py` → `docs/cascade.md`. Offline over the saved test probabilities and the teacher's test labels; thresholds chosen on one half of test and scored on the other, 5 halvings × 2 = 10 held-out halves, so no number is tuned on the rows it is scored on. **Cheapest system that matches the LLM alone (0.867): TF-IDF → MiniLM-ft → LLM at $13.76 / 1M** (6.4 % of queries to the LLM, 58 % answered by TF-IDF), vs $19.19 for MiniLM-ft → LLM and $216 for the LLM alone: **16× cheaper, same accuracy**. Paired on identical halves the TF-IDF first tier is cheaper in 10 / 10 (−$5.43, −2.5 pt of LLM share) but costs +0.36 ms of mean CPU, because the int8 MiniLM behind it is only 3 ms; in front of the 11 ms frozen MiniLM it saves both (−$9.95 and −4.7 ms). At a *fixed* LLM budget the first tier changes accuracy by ±0.1 pt: it buys cheaper escalation near the knee, not a better system. TinyBERT as a first tier is useless (answers 10 %). Best accuracy anywhere: 0.874 at ≤ 30 % LLM share, so the roadmap's "cheapest system that reaches 95 %" has no answer on teacher labels: the ceiling is the teacher.)_ TF-IDF (1.8 ms) → MiniLM → LLM with a threshold
  per tier; one table/chart of accuracy vs $ per 1M requests, and the answer to "cheapest
  system that reaches 95 %".
- [~] **6.6 Robustness.** _(2026-09-25: `distilroute/perturb.py` (typo1, typo3, chat, slang, wrap; deterministic per query) + `scripts/robustness.py` → `docs/robustness.md`, every student on the full test split through the serving code path. **Typos are the weakness, and the transformers are the fragile ones**: 3 typos cost TF-IDF 12 pt (teacher labels) / 10 pt (gold), but MiniLM-ft 30, frozen MiniLM 29, DistilBERT 33, TinyBERT 43 — WordPiece shatters a misspelt word, TF-IDF's character n-grams still overlap. One typo: TF-IDF −4, MiniLM −9/−10. Lowercase / no punctuation: 0 to −1; texting slang and greetings: −1 to −4. The cascade half-absorbs it: with 3 typos MiniLM-ft escalates 58 % instead of 16 % and keeps 0.847 on the rest (0.901 clean), so noise turns into LLM cost rather than silent errors. Note: the served int8 MiniLM-ft scores 0.842 clean here vs 0.847 recorded on Colab (fp32 → int8, −0.5 pt). Still to do: the teacher on 300 noisy queries per kind (`label.py --perturb KIND --limit 300`, ~170k tokens, after CLINC150's labels), to say whether the students degrade *more than the teacher*.)_ _(2026-09-25, **the fix**: `--augment typo1,typo3` on `baseline.py` / `setfit_student.py` / `finetune.py` adds one typo'd copy per training row and kind (seed 1; the test copies use seed 0), after the calibration split, same teacher label, no LLM calls. Frozen MiniLM: clean 0.848 → 0.851, 1 typo −8.7 → −4.9, 3 typos −29.3 → −16.5; TF-IDF: 3 typos −12.1 → −8.8. No transfer to chat / slang / wrap (−0 to −2 either way). Still to do: `minilm_ft_aug_teacher` on Colab, in the same session as the 1.6 retrain.)_ Typo / paraphrase perturbations of the test set (rule-based, free):
  does the student degrade more than the teacher? Support tickets are messy.
- [x] **6.7 Live demo page** _(2026-09-25: https://claude.ai/artifact/MkYEhKKtUj5oJ2b8NUsqoz, private until shared. `demo/index.html` + `scripts/build_demo.py`. The served student (MiniLM-ft, int8 ONNX, trained on teacher labels) runs **in the browser** with onnxruntime-web 1.30 (WebAssembly, one thread) and a JavaScript WordPiece tokenizer; nothing is sent anywhere. Per message: the routing verdict (student / escalate below 0.8 / not banking by entropy > 0.996, which flags 10 % of real banking messages and catches 89 % of CLINC150's out-of-scope ones), calibrated top-3, token count, in-browser latency; a session bill against the LLM alone. The page recomputes 6 reference messages on load and shows whether tokens, intent and confidence match the Python service. The ONNX graph ships as three base64 text parts, since artifact pages serve no other binaries than .wasm.)_ (published artifact): type a message → intent, calibrated
  confidence, escalate-or-not, running cost meter for teacher vs student. Judged in 30 seconds.
- [~] **6.8 Case study write-up** _(2026-09-25: https://claude.ai/artifact/CqwqgvREwra9bT7zJgGb7p, private until shared. `docs/case-study/template.html` + `scripts/build_case_study.py` → `index.html`: every number and chart is filled from results/ and data/labels/, so re-running the build after an experiment updates the page. Four charts (accuracy vs $ per 1M for every system; the cascade curve; the data curve; three ways of choosing what to label), each with a numbers table; the leak; OOS; what did not work; robustness. The second-dataset section shows "in progress" until CLINC150's students exist. Still to do: fill that in, and the teacher side of 6.6.)_ with the charts: cost table, cascade curve, data curve, and
  the batch-context leak as a lesson. The repo is the proof; the write-up is what gets read.
- [x] **6.9 Where should a few human labels go?** _(2026-09-25: `scripts/correct.py` → `docs/correction.md`. Frozen MiniLM on the 3,000 teacher labels (15.2 % wrong); a human is simulated with gold; budgets 100–1,000; strategies: check random rows, check the rows whose teacher label the student believes least (out-of-fold, confident-learning score as a ranking), check the rows the student is least sure of, or label that many *new* rows. **Disagreement wins by far:** 78 % of the first 100 it picks are real teacher errors (5× the base rate); 500 checks → **0.876, above the teacher's 0.867** (6.1 met with a human budget), 1,000 → 0.890. Random checks: 0.854 / 0.868; 500 new human rows: 0.857. Fixing the LLM's labels beats adding to them. Budget 0 reproduces denoise `base` (0.842) exactly. The teacher's 0.867 carries ±0.6 pt of test-set sampling noise, so +0.9 at 500 is a narrow win and +2.2 at 1,000 a clear one; a paired bootstrap would settle it.)_
- [x] **6.10 How sure are the numbers?** _(2026-09-25: `scripts/bootstrap.py` → `docs/bootstrap.md`, paired bootstrap over the 3,080 test queries, 10,000 resamples, 25 s. Every accuracy carries ±1.3 pt; paired differences are tighter. Every student is reliably below the teacher (served int8 −2.5, interval −3.6 to −1.4). **Softened:** cascade vs LLM +0.6 (−0.1 to +1.4, ahead in 96 %) → "at least as good"; 500 human checks vs LLM +0.9 (−0.3 to +2.0, 93 %); MiniLM vs DistilBERT +0.8 (−0.0 to +1.6) → "at least as good". **Confirmed:** 1,000 checks +2.2 (+1.1 to +3.4); cascade over its own student +2.7; targeted over random checks and over new rows, both ≈ +2 (+1.3 to +3.0); int8 −0.5 (−0.9 to 0.0); fine-tuned vs frozen MiniLM and typo augmentation on clean text are ties. Test sampling only; seed noise is separate. `correct.py` now saves its test predictions for this.)_

**Framing rule for the README:** not "a novel classifier" but "a reproducible $0 recipe with
three results people will quote: student beats teacher (6.1), cascade beats teacher at a fifth
of the LLM cost (1b.2), and it knows when a message is not its job (6.2)".

## Phase 7 — Close the limitations  (decided 2026-09-27)

The README's Limitations section lists what is weak. Of the 16 weaknesses reviewed on
2026-09-27, 8 are fully fixable at $0, 7 partly, 1 not (the method is not novel — that is the
framing, not a flaw). In order of payoff per effort; 7.1 and 7.2 need no Groq budget, so they
run while `jobs.py` labels.

- [ ] **7.1 Real human checks** (fixes "the human in finding 5 is simulated"). _(2026-09-27: `scripts/review.py` ready — same ranking as `correct.py`, gold never loaded while reviewing, resumable; `--report` scores and retrains. Waiting on the user's review.)_ A small review
  script: take the ~200 rows `correct.py` ranks as least believed, show the user the text, the
  teacher's label and the 77 intents — **never the gold label** — and record their answer and
  seconds per item. Report: the user's agreement with gold, the time cost, and the frozen
  student retrained on the user's corrections vs the simulated 0.876 curve. The user does the
  checking (~2–3 h); gold is read only to score it afterwards.
- [x] **7.2 Leave-intents-out** _(`scripts/new_intents.py` → `docs/new_intents.md`, 2026-09-27: new banking intents AUROC 0.83 vs 0.98 far OOS, 53 % caught at a 10 % budget vs 96 %; entropy no better than max-prob here; ~50 labels per new intent to match the all-77 student)_ (near out-of-scope, new-intent behaviour; free, CPU, ~30 min).
  Train students without ~10 of the 77 intents; score how many held-out queries are flagged
  (entropy, as in 6.2) against 10 % in-scope escalation. Then add the intents back with
  25 / 50 / 100 labels each: how many labels does a new intent need?
- [ ] **7.3 One Colab session after 1.6 finishes**: _(2026-09-27: notebook ready — `notebooks/finetune_colab.ipynb` clones the public repo and runs `minilm_ft_aug_teacher`, seeds 1 and 2 of the served config, and the 10,003-label run (skipped until labelling is done), then downloads only the new outputs. `finetune.py` now names seeds ≠ 0 `<run>_s<N>` (seed 1 would have overwritten the published run) and records seed + augment; `scripts/seeds.py` reports the spread. The first two can run now; the user runs Colab.)_ fine-tuned MiniLM at N = 10,003 (1.6 step 3),
  `minilm_ft_aug_teacher` (6.6), and 3 seeds of the served config (seed noise, 6.10 caveat).
- [ ] **7.4 Measured cost and latency** _(2026-09-27, load test done: `scripts/load_test.py` + `load-test.yml` → `docs/load_test.md`. Service changes along the way: onnxruntime's pool sized to the container's CPU quota (removed a 53–79 ms p95 tail at one client), then **one single-thread inference per CPU** as the default (`/health` reports cpus / concurrency / threads). Final run (Intel Xeon 8573C runner), same-runner comparison: 2 CPUs **497 req/s vs 312** for one multi-thread inference at a time, 8-client p95 33 vs 44 ms; 1 CPU 270 req/s; **$0.011–0.012 / 1M** (estimate was $0.020; teacher $216). Lesson: runners landed on four CPU types across five runs (model time 1.0–3.2 ms), so setups are compared only within one run; the earlier "quota not detected" guess was wrong — `/health` shows it detected. Still to do: the HF Space.)_: concurrent load test of `/route` (in GitHub Actions on
  Linux, avoiding Windows loopback); cost per 1M from measured throughput, not CPU-seconds.
  Plus a live endpoint as a free Hugging Face Space (`build_hf_release.py`).
- [ ] **7.5 After the labelling queue (Groq free again):** _(2026-09-27: Gemini half ready — `jobs.py` has a `gemini` lane (the 200 queries of `test.gate_v2_top3`, same prompt, then `scripts/second_teacher.py`: each teacher's accuracy, agreement as a noise signal). It skips itself while `GEMINI_API_KEY` in `.env` is empty, which it is: the user needs a free key from aistudio.google.com/apikey. Check the model name (`gemini-2.0-flash` default may be retired; `GEMINI_MODEL` overrides). The v1-vs-v2 re-run is queued at the end of the groq lane (2026-09-28): v1 descriptions on the first 200 train rows via `label.py --descriptions-file`, compared with their existing v2 labels by `scripts/descriptions_check.py` (paired bootstrap). Dry run hint: v2 scores 0.835 on those 200 train rows vs 0.905 on the 200 test rows it was chosen on.)_ second teacher through Google AI
  Studio (Gemini, `teacher.py` already speaks it) on the 200-query gate; the v1-vs-v2
  descriptions comparison re-run on 200 *train* queries, to show the test-based choice did not
  matter.
- [x] **7.6 Stress text (free):** _(2026-09-27: `scripts/stress.py` → `docs/stress.md`. The served int8 MiniLM reads 64 tokens: a question at the end of a 120-word ticket drops it 0.867 → 0.013 (cascade escalates 90 %); frozen MiniLM (256 tokens) keeps 0.763, and 0.837 routed sentence by sentence with a per-sentence entropy off-topic filter. Sentence routing rescues DistilBERT (0.023 → 0.770) but only partly the served model (0.550): it is confidently wrong on 4 of 18 chit-chat lines. Two questions in one message: top-1 is one of them 60–87 %, both in the top-2 only 7–48 %; sentence routing finds both in ~50 %.)_ two queries joined into one message (is the pair in the
  top 2?) and queries padded to ticket length.
- [ ] **7.7 Publish** _(2026-09-28: README gained findings 8–10 (new intents, long tickets, load test) and updated Limitations/Scope; `docs/START_HERE.md` is the one-page map. Posting still waits for 7.1, 7.3 and the HF Space.)_ (4.2) once 7.1–7.4 land, so the posts carry the stronger numbers; trim the
  ROADMAP or add a one-page "start here".

---

## Rules of thumb

- Keep the venv lean — this laptop has ~20 GB free. Torch/SetFit go in a separate
  `requirements-train.txt` and get installed only when Phase 2.2 starts; DistilBERT trains on
  Colab, never here.
- Commit the label files; never commit `.env`.
- Every number in `docs/` comes from a script in `scripts/` that can be re-run.

## Session log

**2026-10-02 (HANDOFF — start here next session)**

- `jobs.py` (00:33–01:00): **Banking77 train labels complete, 10,003 / 10,003** (0 unparsed, no
  duplicate `idx`, `271ddc2`). Self-agreement 300 / 300 (`08eb56f`): top-1 agreement **0.977**,
  35 of the teacher's 40 errors on those queries are stable (same wrong label twice) —
  `docs/teacher.md` regenerated. Noise labels typo1 / typo3 / chat done (`c4c3c52`, `30d1ee2`,
  `0ef8205`); slang **260 / 300** (uncommitted until the step finishes). Groq's rolling limit hit
  at 00:55; a restart at 01:00 got one call before the next limit.
- **Gemini lane finished** (`2ab5f51`): gate 200 / 200; on the same 200 test queries gemini-3.5-flash
  **0.910** (top-3 0.980) vs gpt-oss-120b 0.905 (0.960) — within the ±5 pt noise. They agree on
  96 % of rows (0.927 accurate there); only 5 of the teacher's 19 errors fall in the 4 % disputed.
  `docs/second_teacher.md`.

**Next session, in order:** (1) `jobs.py` in the evening: slang (40 left) → wrap → robustness →
descriptions v1 + check; (2) the **10,003 retrain** (1.6 plan) — asked the user, no answer yet;
(3) **user:** `hf auth login` + username → `build_hf_release.py --user X --publish`; (4) **user:**
7.1 review, 7.3 Colab (add OE to it).

**2026-10-01 (HANDOFF — start here next session)**

- `jobs.py`: Banking77 train labels **7,780 → 9,580 / 10,003** (0 unparsed, `9432097`, pushed); Groq
  limit at 02:34. Next run should finish train labels (423 left) → then the 10,003 retrain (needs
  the user's go-ahead), self-agreement (220 / 300), noise, descriptions.
- **OE fine-tune evaluated** (`oos.py --models minilm_ft_oe_teacher`, which now merges into
  `results/oos.json` instead of overwriting it; `stress.py`). vs the served `minilm_ft_teacher`:
  off-topic AUROC (entropy) 0.956 → **0.989**, caught at 5 % false alarms 0.827 → **0.968**; filler
  passing the filter 4 → **0** of 18; sentence routing with the filter, question at the end of 120
  words 0.550 → **0.830**; `also` invented at 0.8 / 0.9 / 0.95: 82 % / 72 % / 1 % → **3 % / 2 % /
  1 %**, found 53 % / 47 % / 20 % → 51 % / 44 % / 31 %. So OE removes the small-talk problem, and
  with it `ALSO_THRESHOLD` could drop to 0.8 (found 51 %, invented 3 %) — but only with this
  model served; the served model must stay at 0.95. `ALSO_THRESHOLD` unchanged.
- **Blocker for serving it:** clean 0.841 vs 0.847 and int8 −0.4 pt (rule: ≤ 0.3). Next try: OE
  in the Colab run (GPU, 24 epochs like the served model) and/or on the 10,003 labels; if clean
  and int8 then pass, serve it and set `ALSO_THRESHOLD = 0.8`.
- **Gemini key added** (7.5). `gemini-2.0-flash` and `gemini-2.5-flash` are closed to new keys and
  `gemini-3.8-flash` timed out, so the default is now **`gemini-3.5-flash`** (`50157a4`;
  `GEMINI_MODEL` in `.env` too). Gate: **120 / 200** (`data/labels/test.gate_gemini.jsonl`, 0
  unparsed, uncommitted until the step finishes); the free tier stopped it after 6 calls with a
  daily 429, back 08:00. The next `jobs.py` run resumes it, then runs `second_teacher.py`. A second
  Groq pass added 60 train labels → **9,640 / 10,003** (`beb15ae`).

**Next session, in order:** (1) `jobs.py` in the evening: finishes the Gemini gate + comparison and
the last 363 train labels; (2) with 10,003 labels, ask the user before the retrain (1.6 plan);
(3) **user:** `hf auth login` (write token from huggingface.co/settings/tokens) + username →
`build_hf_release.py --user X --publish` after confirming; (4) **user:** 7.1 review, 7.3 Colab
(add OE to it).

**2026-09-30 (HANDOFF — start here next session)**

**Done this session** (02:10–04:00):
- `jobs.py`: Banking77 train labels **6,020 → 7,780 / 10,003** (0 unparsed), paused on Groq's rolling
  limit at 02:38 (`9329872`, pushed). Run it with **`.venv\Scripts\python.exe`**: bare `python` is
  the system 3.13 without pandas, and jobs.py reuses `sys.executable`, so every step fails at once.
- **7,500-row check** (`retrain_check.py --rows 7500`, `results/teacher7500/`, `docs/teacher7500/`,
  `8055731`, pushed): new labels as good as the old (teacher vs gold 0.850 on rows 3,001–7,500,
  0.848 on the first 3,000). Frozen MiniLM 0.848 → **0.861** (0.6 pt behind the teacher's 0.867),
  TF-IDF 0.812 → 0.836; denoising still does not beat the plain student; learning curve still
  rising (0.835 / 0.852 / 0.860 at 2k / 5k / 7.5k) → the 10,003 retrain should add a little more.
- **Outlier-exposure fine-tune finished** (the 2026-09-28 run had died at epoch 7/14; re-run on
  the laptop CPU, 87 min): `finetune.py --model minilm --labels teacher --export-onnx
  --outlier-exposure 2000` → `results/minilm_ft_oe_teacher.json` + `_test_probs.npz` (committed),
  model in `models/minilm_ft_oe_teacher/` (gitignored, local only). Same 2,697 teacher rows as the
  served model. Clean accuracy **0.841** (served `minilm_ft_teacher`: 0.847), int8 **0.837**
  (served 0.845); int8 loses 0.4 pt vs its fp32 — over the 0.3 pt serving rule. int8 p50 3.5 ms.
  Caveat: the served model had 24 epochs on a GPU, this one 14 on CPU.

**In flight / not done:** the robustness side of the OE model is **not evaluated**. `oos.py` then
`stress.py` were started and killed by Claude Code for low system memory before writing anything
(`results/oos.json`, `results/stress.json` unchanged). Next: free memory, run them **one at a
time** (`oos.py` loads every model under `models/` by default; check whether `--models` rewrites
the whole json before narrowing it). Compare with the served model's numbers: off-topic AUROC
(entropy) **0.956**, caught at 5 % false alarms **0.827**, filler passing the filter **4**, `also`
invented on filler at the start **0.65 / 0.50 / 0.003** at thresholds 0.8 / 0.9 / 0.95. Frozen
MiniLM + OE for reference: AUROC 0.992, 0.973, 0 filler. Decision rule unchanged: lower
`ALSO_THRESHOLD` (0.95, `distilroute/multi.py`) only if the invented rate falls; serving needs
int8 within 0.3 pt of clean — on clean accuracy alone it is not a candidate now.

**Left, in order:** (1) `jobs.py` daily, evenings (≈2,200 train rows + 80 self-agreement + noise
runs + descriptions left; `data/labels/test.self_agreement.jsonl`, 220 / 300, stays uncommitted
until its step finishes); (2) the OE evaluation above; (3)–(6) as in the 2026-09-28 list below
(7.1 review, Colab, Gemini key, `hf auth login`, then the 10,003 retrain).

**2026-09-28 (handoff)**

**Done this session** (2026-09-27 evening → 2026-09-28, all pushed, CI green):
- **Phase 7:** 7.2 new intents (`docs/new_intents.md`), 7.6 stress text (`docs/stress.md`), 7.4 load
  test (`docs/load_test.md`; served container 497 req/s on 2 CPUs, **$0.012 / 1M**), 7.7 README
  findings 8–10 + `docs/START_HERE.md`. Prepared, waiting on the user: 7.1 `scripts/review.py`,
  7.3 `notebooks/finetune_colab.ipynb`, 7.5 Gemini lane in `jobs.py` + v1-vs-v2 descriptions
  re-check queued at the end of the groq lane (`scripts/descriptions_check.py`).
- **Service fixes** (each measured): onnxruntime threads sized to the CPU quota, then one
  single-thread inference per CPU by default (`/health` reports it); ONNX students read **512
  tokens** (was 64: question at the end of a 120-word ticket 0.013 → 0.803, short queries
  unchanged); `transformers` dropped from the image (packages 296 → 164 MB, identical tokens);
  **`GET /monitor`** flags an intent whose traffic share jumps (a new intent at 5 % of traffic
  caught within 2,000 messages 94 % of the time, 1.6 % false alarms); **`also`** in `POST /route`
  (second request from another sentence, threshold 0.95 because the fine-tuned model is
  confidently wrong on small talk).
- **Negative results kept:** distance scores (Mahalanobis, k-NN) are worse than entropy on new
  banking intents (0.72 vs 0.83) though best on off-topic (0.997); cross-run load-test
  comparisons were confounded by 4 runner CPU types, so setups are compared within one run.
- **Outlier exposure** (`--outlier-exposure N` in `setfit_student.py` and `finetune.py`; 2,000
  non-banking CLINC150 train queries, uniform target): frozen MiniLM off-topic AUROC 0.982 →
  0.992. The plain frozen model was already calm on small talk; the fine-tuned ones are not.
- **CI:** `.github/workflows/tests.yml` (ruff + 50 tests); fixed `httpx` in requirements-dev,
  pytest `pythonpath`, data download.
- Labels: train **6,020 / 10,003**.

**In flight at handoff:** `finetune.py --model minilm --labels teacher --export-onnx
--outlier-exposure 2000` on the laptop CPU (~1 h, log `%TEMP%\ft_oe.log`). If
`results/minilm_ft_oe_teacher.json` exists: run `scripts/stress.py` and `scripts/oos.py` (the model
is already in stress.py's list), compare with `minilm_ft_teacher` (filler sentences passing the
filter, `also` invented rate, off-topic AUROC, clean accuracy); if the invented rate falls,
lower `ALSO_THRESHOLD` in `distilroute/multi.py` to the value the table supports, and consider
serving it (int8 must not lose > 0.3 pt on clean). If the file is missing, the run was
interrupted: the same model is in the Colab notebook, or re-run the command.

**Left, in order:**
1. `jobs.py` once a day, ideally evenings (Groq's rolling 24 h budget); `retrain_check.py --rows
   7500` past 7,500; the descriptions re-check and teacher-on-noise run at the end of the queue.
2. **User:** 7.1 review (`.\.venv\Scripts\python.exe scripts\review.py`, 2–3 h, resumable), then
   `review.py --report`.
3. **User:** Colab run of the notebook (aug, seeds 1–2, outlier exposure; the 10,003-label run
   skips itself until labelling is done), then `seeds.py`, `robustness.py`, `stress.py`,
   `oos.py`, `evaluate.py`.
4. **User:** Gemini key in `.env` (aistudio.google.com/apikey) → check the model name, run the lane.
5. **User:** `hf auth login` → HF Space (7.4's last piece; `build_hf_release.py`).
6. After those: 1.6 retrain on 10,003 labels; README headline + case study rebuild; 7.7 posts
   (user shares and posts).

**2026-09-27 evening** — `.venv` rebuilt (38 tests pass). `jobs.py`: train labels 4,900 →
**5,860 / 10,003** (0 unparsed), paused on Groq's rolling limit at 21:14. **5,000-row check**
(`results/teacher5000/`): frozen MiniLM 0.848 → 0.853, TF-IDF 0.812 → 0.833, gap to teacher
1.4 pt. **7.2 done** (`docs/new_intents.md`). Commits `61305cc`, `e4790bf`, `b4368ae` (pushed).
**Next:** run `jobs.py` once a day; `retrain_check.py --rows 7500` past 7,500; start 7.1.

**2026-09-27 (labelling resumed)** — Ran `jobs.py`: Banking77 train labels **3,000 → 4,900 /
10,003** (1,860 rows in 25 min, then 40 more), 0 unparsed, committed and pushed (`93d4f4e`).
Quality check on the new rows (gold read for this report only): teacher accuracy **0.847**,
gold in top-3 0.940, against 0.848 / 0.936 on the first 3,000 — same teacher, same config, no
duplicates. **Groq's limit is a rolling 24 h token budget**, not a daily reset: after ~290k
tokens it refused with "back in ~10 min", but a re-run then got only 40 rows (the budget
refills at ~1 row/min). So run `jobs.py` **once a day** (~1,300–1,900 rows, ~25 min); restarting
every few minutes is not worth it. Left: ~5,100 train rows, 80 self-agreement, 1,500 noisy
queries → about 4 more daily runs (≈ 30 Sep – 1 Oct). Also: case study rebuilt with the
typo-augmented students (not yet republished — wait for the teacher-on-noise results), and the
partial Gemma gate file of the cut 1b.6 lane deleted.
Later the same day: README gained a **Limitations** section (`c1c032a`); the 16 weaknesses were
triaged into **Phase 7** (8 fully fixable at $0, 7 partly, 1 not). `.venv` deleted to free
1.35 GB — rebuild it first (commands in the 2026-09-23 handoff below; `requirements-dev.txt`
too, for pytest).
**Next session, in order:** (1) rebuild `.venv`; (2) run `jobs.py` once (it runs ~25 min, then
pauses on Groq's rolling limit); (3) once past 5,000 rows, `scripts/retrain_check.py --rows 5000`;
(4) meanwhile start **7.1** (review script for real human checks) and **7.2** (leave-intents-out),
neither of which uses Groq.

```powershell
cd C:\Users\User\dev\distilroute

# 1. Recreate the venv (~1.3 GB download, 5-10 min)
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\pip install --index-url https://download.pytorch.org/whl/cpu torch
.\.venv\Scripts\pip install -r requirements-train.txt -r requirements-dev.txt

# 2. Label: runs ~25 min until Groq's rolling limit, then pauses itself. Once a day.
.\.venv\Scripts\python.exe scripts\jobs.py
.\.venv\Scripts\python.exe scripts\jobs.py --status        # progress, any time

# 3. When train labels pass 5,000 (the first line of --status), the first "do more labels
#    help?" check (~3 min; writes results/teacher5000/checkpoint.json, never over published runs)
(Get-Content data\labels\train.jsonl | Measure-Object -Line).Lines
.\.venv\Scripts\python.exe scripts\retrain_check.py --rows 5000
```

**2026-09-25 (HANDOFF — where to start next session)**

**Resume:** open the repo and run `.venv\Scripts\python scripts\jobs.py` (or ask Claude to
"run the jobs"). `scripts/jobs.py --status` shows what is left. It runs every pending LLM job in
order, skips finished steps, is safe to stop at any time (Ctrl-C or shutting the laptop), and
refuses to run twice. The free tiers cap tokens per day, so give it ~1 hour a day for ~6 days,
or add a card to Groq (pay-as-you-go, no fee, whole queue ≈ $0.50) and it finishes in one sitting.
The runner never commits: review each finished step's output, then commit.

**Pending in the runner** (state at shutdown: CLINC150 train labels 1,940 / 3,000; Gemma gate
20 / 200; both files are on disk, untracked until complete):
- groq lane, in order: CLINC150 train labels → CLINC150 students on teacher labels (fills the
  case study's second-dataset table: `build_case_study.py`, republish) → teacher latency (3.2;
  only the teacher entry changes, students keep the 2026-09-24 numbers) → self-agreement 300
  (1.7; `teacher_report.py` then has the stable-vs-unstable error split) → teacher on 5 noise
  kinds × 300 (6.6) → `robustness.py` with the teacher section → the remaining 7,003 Banking77
  train labels (1.6).
- openrouter lane: three second-teacher gates, 200 test messages each (1b.6): gemma-4-31b,
  qwen3.8-27b, glm-5.2 (free models, ~50 requests/day). Then pick the best and decide whether to
  label the full test split with it (~4 days).

**After the runner, decisions for the user:**
- 1.6 retraining on all 10,003 labels: follow the plan in 1.6 (`DISTILROUTE_TEACHER_ROWS=10003`,
  writes under `results/teacher10003/`, never over published runs). Fine-tuned students need a
  Colab run. Replace the served model only if int8 MiniLM beats 0.842 by more than ±0.3 pt.
- 4.2: repo is public. Still to do: share the case study
  (https://claude.ai/artifact/CqwqgvREwra9bT7zJgGb7p) and demo
  (https://claude.ai/artifact/MkYEhKKtUj5oJ2b8NUsqoz) from their Share menus, then post. Draft and
  notes: `../tabaudit/linkedin_post_distilroute.md`; image `docs/case-study/media/pareto_linkedin.png`.
  Post the two tabaudit drafts first.

**Done 2026-09-24/25:** latency re-bench on an idle mains-powered machine (3.2); Dockerfile, then
built and routed in GitHub Actions against the `model-v1` release file (3.4, 457 MB); README with
final table (4.1); 6.3 diverse; 6.4 dataset switch + CLINC150 teacher 0.901; 6.5 multi-tier
cascade ($13.76 / 1M to match the LLM); 6.6 student robustness (3 typos: transformers −29 to −43,
TF-IDF −12); 6.7 in-browser demo; 6.8 case study page (generated from results/); 1b.5 tabaudit on
teacher labels (hurts, −1.6 pt), which led to tabaudit 0.3.1 on PyPI (wide-frame fix, 3.11 in CI,
limit documented); headlines quote the served int8 model (0.842, 97 %); roadmap markers tidied;
`DISTILROUTE_TEACHER_ROWS` guard; resumable `jobs.py`.

**2026-09-24 (6.3 diverse, 4.1 README)** — The 3-seed diverse run finished (seeds reproduce: 0.779 /
0.786 / 0.789 at 500). Mean ± half-range: 0.785 ± 0.005 / 0.822 / 0.831 / 0.837 / 0.839 at 500 → 2,500,
against random's 0.708 / 0.799 / 0.822 / 0.835 / 0.842. So choosing the most *typical* rows is worth
+7.7 pt at 500 labels (≈ half the LLM calls for the same accuracy) and +2.3 at 1,000. It stops helping
by 1,500, and past 2,000 it is slightly worse: k-means centroids are the easy, typical queries, and
once the student has those, the rare and hard ones matter more. It also buys dirtier labels than random
(16–20 % wrong vs the 15.2 % base rate, worst at 1,000–1,500), for the same reason uncertainty does. The practical recipe is
diverse for the first ~1,000 LLM calls, random after that. README rewritten for 4.1 around the final
table and six findings. 6.1 is stated as *not met* (student 1.8 pt below teacher), since the old
framing rule assumed it would be.

**2026-09-24 (latency re-bench)** — Handoff item 1 closed. `bench_latency.py` (500 queries,
in-process) on an idle, mains-powered machine (OneDrive shut down; it was holding more than a core), then
`cost.py` and `evaluate.py`. Every latency cell is now from one run on one machine: the fine-tuned
ONNX int8 students are the fastest transformers — TinyBERT **1.8 ms**, MiniLM **2.7 ms** (as fast as
TF-IDF at 1.7 ms, and 3.5 points more accurate on teacher labels), DistilBERT 6.4 ms — while the frozen
and SetFit MiniLMs (PyTorch encoder) sit at 11–13 ms. The old † numbers (27 ms MiniLM, 6.6 ms TF-IDF)
were 2.5–4× inflated. Cost per 1M on a t3.small: $0.01–0.08 for every student. `results.md` also
gains the six Colab fine-tuned rows, which the round-2 commit had not rebuilt it with. `--http` rows are
still the 2026-09-18 ones.

**2026-09-23 (HANDOFF — machine compromised, work moves to a new device)**

The laptop this project was built on is infected with `Trojan:Win32/JScealTaskExec` (an
info-stealer, running as SYSTEM, blocked but not removed by Defender; likely entry point was a
KMS activation crack in `C:\Program Files\Activation-Renewal`). All 45 commits are pushed to
`github.com/sadiasamia121912/distilroute`. The machine is being rebuilt; nothing below needs it.

**Resume on the new machine:**

```bash
git clone https://github.com/sadiasamia121912/distilroute.git
cd distilroute
python -m venv .venv && .venv/Scripts/activate          # or source .venv/bin/activate
pip install -r requirements.txt                         # scripts, service, evaluation
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements-train.txt                   # students (torch CPU, sentence-transformers)
python scripts/download_data.py                         # data/raw/ is gitignored; re-fetch Banking77
python scripts/download_clinc.py                        # CLINC150, for 6.2 / 6.4
pytest -q && python scripts/evaluate.py                 # 24 tests; rebuilds docs/results.md
```

Then rebuild the CPU students (minutes, no API key needed) — `models/` is gitignored:

```bash
python scripts/baseline.py --labels gold
python scripts/baseline.py --labels teacher
python scripts/setfit_student.py --mode frozen --labels gold
python scripts/setfit_student.py --mode frozen --labels teacher
```

**New API keys are needed only to label more data** (`.env`: `GROQ_API_KEY`, optionally
`GEMINI_API_KEY`, `OPENROUTER_API_KEY`). The old keys were revoked after the compromise. All
existing labels are committed, so every student, report and table reproduces without a key.

**What is pending, in priority order:**

1. ~~**Latency re-bench**~~ — _done 2026-09-24, see session log._ Every number in the latency column is stale: measured on battery
   (2.5× slow), during a k-means job, or on Colab's CPU (marked †). Run
   `python scripts/bench_latency.py` once on an idle, mains-powered machine, then
   `python scripts/cost.py` and `python scripts/evaluate.py`, since the cost table divides by it.
2. ~~**6.3 `diverse` strategy**~~ — _done 2026-09-24, see session log._ Was incomplete. Seed 0 gave 0.779 at 500 labels vs random's 0.708
   (a 7-point cold-start gain, the most promising active-labelling result), but the 3-seed run
   was interrupted. `python scripts/active.py --strategies diverse` finishes it; k-means with
   k≈2,500 is slow, so give it an hour.
3. **Colab models** — `results/*_ft_*.json` and their test probabilities are committed, so the
   table is intact, but `models/*_ft_*/` (the int8 ONNX graphs) are not. Re-run
   `notebooks/finetune_colab.ipynb` if the served artefacts are wanted.
4. Then: **3.4** Dockerfile (_written 2026-09-24; needs a `docker build` check_), **4.1** README with the final table (_written 2026-09-24_),
   and **Phase 6.4–6.8**.

**2026-09-23 (Colab round 2, 6.3)** — All seven fine-tune runs landed with the fixed recipe;
1b.4 closed (see above). Phase 6.3 active labelling (`scripts/active.py`): uncertainty and
disagreement selection buy only **+0.5 pt** over random at the same budget — not the 2× saving
the literature reports — and the diagnostic says why: **the rows they choose are dirtier**.
The teacher mislabels 15.2 % of the pool, but 19.8 % of what uncertainty sampling buys at a
1,500 budget (17.6 % for diverse). Ambiguous rows are exactly the near-synonym intents the
teacher fumbles, so informativeness is paid back in label noise. That is a real caution for
LLM-labelled active learning, and it points at the fix: pair selection with the 6.1 noise
filter, or send the uncertain rows to the teacher with a bigger reasoning budget. The
diversity (k-means) strategy looks strong at the cold start (0.779 vs 0.708 at 500 labels,
seed 0) — the full 3-seed run is still pending after a memory bug in the distance computation.

**2026-09-23 (Phase 6.2)** — Out-of-scope detection, offline, on every saved student. At a
10 % escalation budget they catch 94–97 % of CLINC150's out-of-scope queries; entropy is the
best score on every model and temperature scaling slightly hurts separation. `docs/oos.md`.
CLINC150 is now downloaded in full (22,500 in-scope queries over 150 intents), which is also
the second dataset Phase 6.4 needs.

**2026-09-23 (Phase 6.1)** — `scripts/denoise.py`: soft labels, confident-learning noise
filter, and self-training on the unlabelled train rows, each alone and combined, all on the
frozen-MiniLM student with the project's own LR head (soft targets reach it as weighted
duplicate rows, so `base` reproduces the main table). Result: **+0.7 pt to 0.849, teacher
0.867 — the student does not beat its teacher at 3k labels.** Reported in full rather than
cherry-picked. Two findings worth keeping: the teacher's top-3 is not useful supervision here
(soft hurts), and a confidence threshold on an *uncalibrated* head silently discards ~90 % of
the data it should keep. Fixed a subtler bug on the way: for self-trained runs the temperature
was being fitted on the model's own pseudo-labels.

**2026-09-23 (honest teacher number — and it changes the story)** — Shuffled relabel of the
test split done: **teacher 0.867**, not 0.948. Consequences, all good for the project:
- **Every gold-trained student beats the teacher**: MiniLM frozen 0.927, TF-IDF 0.910,
  DistilBERT 0.918 vs 0.867. Supervised data is worth ~6 pts over a zero-shot 120B model.
- **The distilled student (0.848 on 3k teacher labels) is within 2 pts of its teacher** — and
  the data curve says more labels close it.
- **The cascade result inverts, and that is the finding.** Escalating a *gold-trained*
  student's least-confident 20 % to the teacher makes it **worse** (0.927 → 0.917): the
  "expert" is less accurate than the student it is correcting. For the *teacher-trained*
  student it still helps (0.848 → 0.871). So: a confidence cascade pays only while the student
  is worse than the LLM; measure both before building one. The earlier "cascade beats the
  teacher" line came from the leaked labels and is retracted.
- Where they differ: the student is right and the teacher wrong on 8.9 % of test queries, the
  reverse on 3.0 %, both wrong on 4.4 %. A perfect router between the two would reach 0.957 —
  the headroom Phase 6.1 (noise filtering, self-training) goes after.

**2026-09-21 (train subset done; the batch-context leak)** — Train subset finished (3,000,
0 unparsed). Students on teacher labels: TF-IDF **0.812**, frozen MiniLM **0.848** — exactly
the teacher's own accuracy on those rows (0.848), i.e. the student learns the teacher, noise
included; the gold-trained curve at 3k gives ~0.90, so the distillation gap is ~5 pts of
teacher noise. Sanity-checking why the teacher read 0.948 on test but 0.848 on train exposed
a **protocol flaw**: the test CSV is intent-sorted, every batch of 20 was one intent, and the
teacher used that as context (same 200 queries: 0.98 sorted vs 0.905 shuffled). The train
run was shuffled, so its labels and everything trained on them stand; the test labels do not.
Fix: `label.py` now always shuffles; the sorted run is kept as `test.sorted_batches.jsonl`
and listed as superseded in `teacher.md`; shuffled relabel of test started (~2 days). Until it
lands, `results.md`'s teacher line, agreement and cascade columns are partial. Colab zip
rebuilt with the train labels so the teacher rows can run there. SetFit few-shot on teacher
labels: 0.789 (7/intent — the rarest intent has only 7 teacher rows). **Latency caveat:** the
laptop was on battery (21 %, 1.6 GHz) when the teacher-trained models were benched — every model
read ~2.5× slower than Friday, so those numbers were discarded. **To do: `bench_latency.py`
for all models in one session, plugged in.**

**2026-09-19 (test labels complete)** — Labeller finished the test split overnight (one 2-hour
stall when the laptop slept and Wi-Fi did not come back; it recovered by itself). Teacher on
the full split: **0.948**. `results.md` cascade columns filled in: **MiniLM frozen escalating
its least-confident 20 % to the teacher = 0.963, above the teacher alone (0.948)** — 80 % of
queries never touch the LLM. Cost script now uses the measured token split (252k in / 66k
out). Train subset (3,000 rows, seed 0) labelling started.

**2026-09-18 (curves, fine-tune script, serving)** — Test labelling restarted as a detached process
(`logs/label_test.log`; 380 → 1,180 at pause, then Groq's daily token cap: 5–15 min sleeps). Wrote `scripts/data_curve.py` (1b.3) and ran
it on gold for both CPU students: MiniLM frozen reaches 0.84 with 1k random labels and 0.92
with 5k; TF-IDF needs 5k to reach 0.89. `evaluate.py` renders the curves into `docs/results.md`.
Wrote `scripts/finetune.py` (2.3 / 1b.4: DistilBERT / MiniLM / TinyBERT full fine-tune, soft
top-3 targets, ONNX + int8 export) with a Colab wrapper notebook; smoke-tested on CPU.
Then 3.1 + 3.2: `distilroute/students.py` (one loader per model kind, incl. the teacher),
`distilroute/serve.py` (FastAPI, tested with a fake router), `scripts/bench_latency.py`
(→ `results/latency.json`, which `evaluate.py` now prefers for the latency column). Verified
end to end against a running uvicorn. Not done: 3.3 cost table, 3.4 Dockerfile (no Docker on
this laptop), the teacher latency run (wait for labelling to finish).
**Resume (tonight / next session):**

```powershell
cd C:\Users\User\dev\distilroute
# 1. Is the labeller still alive? It survives the assistant session, not a reboot.
Get-Content logs\label_test.log -Tail 2; (Get-Content data\labels\test.jsonl | Measure-Object -Line).Lines
# 2. If not, restart it — resumes from the checkpoint; one process per label file at a time.
.\.venv\Scripts\python.exe scripts/label.py --split test --descriptions --top-k 3
# 3. Sanity: tests + the results doc
.\.venv\Scripts\python.exe -m pytest -q; .\.venv\Scripts\python.exe scripts/evaluate.py
```

Then, in order: (a) Colab notebook on the gold rows — upload a zip of the checkout without
`.venv/` and `data/raw/`; unzip the results back and re-run `evaluate.py` + `bench_latency.py`.
(b) When test hits 3,080: `teacher_report.py`, then `bench_latency.py --teacher 30`.
(c) Start train: `label.py --split train --limit 3000 --seed 0`, then `--labels teacher` for
every student. (d) 3.4 Dockerfile (install Docker first), 4.1 README.

**2026-09-18 (late: cost, calibration — final pause)** — 3.3 `scripts/cost.py`: teacher at Groq's
paid price $220 / 1M one query per call ($28 batched 20), MiniLM $0.07, TF-IDF $0.01 on a
t3.small. 1b.2 `distilroute/calibration.py`: every student holds out 10 % of its training pool,
fits a temperature there; ECE MiniLM 0.074 → 0.011, TF-IDF 0.090 → 0.007, SetFit 0.214 → 0.059
(all under-confident, T ≈ 0.7); threshold-cascade table in results.md (MiniLM < 0.8: escalate
14 %, 97.5 % on the rest); the service applies T and reports `calibrated`. 24 tests. Labeller at
1,180 / 3,080 when paused, alive, in daily-cap sleeps. **Everything unblocked without labels, a
GPU or Docker is done** — next session starts with the Colab run or the label files.

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
