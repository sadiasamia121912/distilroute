# distilroute — Roadmap

_Last updated: 2026-09-17. Project 2 of `../tabaudit/AI_ML_Portfolio_Projects.md`. Budget: **$0**._

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

Why not local (Ollama)? This laptop is 8 GB RAM / no GPU: a 3B model would be a weak teacher
and a 70B one does not fit. See `docs/teacher.md` once written.

---

## Phase 1 — Data + teacher labels  (2 days, mostly waiting on rate limits)

- [x] **1.1** `scripts/download_data.py` → `data/raw/{train,test}.csv`, `categories.json`, with the sanity checks above. _(2026-09-17)_
- [x] **1.2** `distilroute/teacher.py`: provider-agnostic client (Groq / Gemini via plain HTTP), batched zero-shot prompt, strict parsing against the 77-name label set, per-item retry for anything unparseable. Tests use a fake transport — no key needed. _(2026-09-17)_
- [x] **1.3** `scripts/label.py`: resumable labelling with JSONL checkpoint, rate-limit backoff, `--split test|train --limit N`. _(2026-09-17)_
- [x] **1.4** Get a Groq key (free, https://console.groq.com) → `.env` as `GROQ_API_KEY`. _(2026-09-17)_
- [x] **1.4b** Teacher config comparison on 200 test queries (`data/labels/test.cmp_*.jsonl`) — see Protocol. _(2026-09-17)_
- [ ] **1.5** Label the **test** split first (3,080) → teacher accuracy vs. gold. This is the ceiling every student is measured against; if it is below ~85 % switch teacher before labelling train.
- [ ] **1.6** Label the **train** split (10,003). Commit `data/labels/*.jsonl`.
- [ ] **1.7** Self-agreement run: 300 test queries relabelled → `docs/teacher.md` (accuracy, macro-F1, self-agreement, confusion pairs, parse-failure rate, wall-clock and calls used).

## Phase 2 — Students  (2 days)

- [x] **2.1** Baseline: TF-IDF (word + char n-grams) + logistic regression. Trained on **gold** first as the reference (what supervised learning gets), then on **teacher** labels (the distilled version). _(2026-09-17: gold-trained baseline done; teacher-trained waits on 1.6)_
- [ ] **2.2** SetFit (`sentence-transformers/paraphrase-MiniLM-L3-v2` or `all-MiniLM-L6-v2`), CPU-trainable here.
- [ ] **2.3** DistilBERT fine-tune on Colab/Kaggle free GPU (`notebooks/distilbert.ipynb`); save the model, download to `models/`.
- [ ] **2.4** `scripts/evaluate.py` → `docs/results.md`: one table, all models × {acc vs gold, agreement w/ teacher, macro-F1}, plus "gold-trained vs teacher-trained" for the same architecture — the distillation gap.

## Phase 3 — Serving + the cost/latency table  (1–2 days)

- [ ] **3.1** FastAPI `POST /route` with a `model=` switch (teacher | tfidf | setfit | distilbert); same request/response schema for all.
- [ ] **3.2** `scripts/bench_latency.py`: 500 requests per model, p50/p95, on this laptop (CPU). Teacher latency measured end-to-end through the free-tier API.
- [ ] **3.3** Cost per 1M requests: teacher from the provider's *paid* price list (the free tier is not a production option — say so), students from CPU-seconds on a priced cloud VM.
- [ ] **3.4** Dockerfile (student only — ~300 MB image), `docker run` → `/route` works. README with the final table + the pitch.

## Phase 4 — Publish  (½ day)

- [ ] **4.1** README results table, résumé bullet numbers into `../tabaudit/AI_ML_Portfolio_Projects.md`.
- [ ] **4.2** Public repo, LinkedIn post.

---

## Rules of thumb

- Keep the venv lean — this laptop has ~20 GB free. Torch/SetFit go in a separate
  `requirements-train.txt` and get installed only when Phase 2.2 starts; DistilBERT trains on
  Colab, never here.
- Commit the label files; never commit `.env`.
- Every number in `docs/` comes from a script in `scripts/` that can be re-run.

## Session log

**2026-09-17 (kickoff)** — Repo created, Banking77 chosen and checked, teacher client +
resumable labeller written with fake-transport tests, TF-IDF baseline on gold labels run.

**2026-09-17 (teacher online)** — First smoke run 404'd: Groq's free catalogue no longer has
Llama; switched to `openai/gpt-oss-120b` and made non-429 4xx fail fast instead of retrying.
Two label-name quirks bit the parser (`Refund_not_showing_up`, `reverted_card_payment?`) —
now matched loosely, answered canonically, with tests. The 40-query smoke read 77 %, which was
the bug plus a tiny sample; the 200-query comparison says 86–89 %. Chose low reasoning +
descriptions (0.885, cheapest per token). Full test-split labelling started (1.5).
