# distilroute

**Distil an LLM support-ticket router into a 22M-parameter model: 98 % of the LLM's accuracy,
2.8 ms on a CPU, ~12,000× cheaper than one LLM call per ticket, built for $0.**

Teams route support tickets with a frontier LLM. It works, at orders of magnitude more cost and
latency than a small classifier. The production pattern is *LLM-as-teacher → small student*:
label once with the big model, train a tiny model on those labels, serve the tiny model. This
repo does that end to end on [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets)
(13k real banking-app queries, 77 intents) using a free-tier hosted LLM as the teacher, and
publishes the table that decides whether you needed the LLM at all.

It is not a novel classifier. It is a reproducible $0 recipe, and a handful of measured
results on questions teams actually have: how much accuracy distillation costs, when to still
call the LLM, whether the student knows a message is not its job, and which tickets are worth
paying the LLM to label.

## The table

The student never sees a human label: it trains on **3,000 teacher-labelled queries**. Every
accuracy is against the human labels on the untouched 3,080-query test split.

| model | params | accuracy | macro-F1 | p50 / p95 latency (CPU) | $ per 1M requests |
|---|---:|---:|---:|---:|---:|
| **Teacher**: gpt-oss-120b, zero-shot (Groq) | 120B | **0.867** | 0.864 | API | 216.15 (1 query / call) · 26.62 (batched 20) |
| **MiniLM-L6 fine-tuned, int8 ONNX** ← served | 22M | **0.847** | 0.845 | **2.8 / 4.9 ms** | **0.02** |
| MiniLM-L6 frozen + logistic head | 22M | 0.848 | 0.846 | 11.0 / 13.9 ms | 0.07 |
| DistilBERT fine-tuned, int8 ONNX | 67M | 0.839 | 0.837 | 6.2 / 11.6 ms | 0.04 |
| TinyBERT fine-tuned, int8 ONNX | 14M | 0.796 | 0.794 | 1.9 / 3.2 ms | 0.01 |
| TF-IDF + logistic regression | — | 0.812 | 0.809 | 1.6 / 2.0 ms | 0.01 |
| **Cascade**: MiniLM, escalate to the LLM below 0.8 confidence | — | **0.874** | — | 2.8 ms for 85 % of queries | ≈ 32 |

- **Teacher cost** is Groq's *paid* list price. The free tier the labels were made with is
  rate-capped and not a production option. **Student cost** is CPU time on an AWS t3.small at
  on-demand price, one request at a time: an upper bound, and $0 on hardware you already own.
- **Latency** is one query at a time, in-process, on a laptop CPU (`scripts/bench_latency.py`).
- **Accuracy of the fine-tuned rows is the fp32 model's;** latency and cost are the int8 graph's,
  which is what gets served. int8 costs 0.1–0.7 pt: the served MiniLM scores **0.842** on this
  CPU (DistilBERT 0.838, TinyBERT 0.792).
- **Reference, trained on all 10,003 human labels:** MiniLM 0.927, DistilBERT 0.928,
  TF-IDF 0.910. The ~8-point gap to the distilled rows is the teacher's own error rate, passed
  on to the student.

Full tables (calibration, cascade thresholds, data curves, every variant tried):
[docs/results.md](docs/results.md).

## What was found

1. **Distillation keeps 98 % of the teacher's accuracy** (0.847 vs 0.867) at 2.8 ms and
   $0.02 per 1M requests. MiniLM-L6 at 22M parameters matches DistilBERT at 67M on human labels
   (0.927 vs 0.928) and beats it on teacher labels, so it is the model to ship. int8
   quantisation costs 0.1–0.7 pt (0.847 → 0.842 for the served MiniLM) for a 4× smaller graph.
2. **The cascade beats the teacher using 15 % of its calls.** Calibrate the student
   (temperature scaling, fitted on a held-out slice of its training labels), answer when it is
   ≥ 0.8 confident, and send the rest to the LLM: 0.874 overall, above the LLM alone, at about
   a seventh of its cost. Calibration is what makes "0.8" mean something.
3. **It knows when a message is not its job.** On CLINC150's 1,200 out-of-scope queries, the
   students flag **94–97 %** of them while escalating only 10 % of genuine banking queries
   (AUROC 0.97–0.99). Score this with the entropy of the prediction, not max-probability:
   entropy wins on every model. Temperature scaling slightly *hurts* this separation, so
   calibrate for the cascade and use entropy for out-of-scope. [docs/oos.md](docs/oos.md)
4. **The student does not beat its teacher, yet.** Soft top-3 targets, a confident-learning
   noise filter and self-training on the 7,003 unlabelled queries are together worth +0.7 pt
   (0.849), still 1.8 pt below the teacher at this label budget. The soft target *hurts* on 77
   near-synonym intents, on both the frozen and the fine-tuned student. The data curve says the
   gap closes with more labels, not with tricks.
5. **Which tickets to pay the LLM for matters only at the start.** Picking the 500 most
   *typical* queries (k-means over the embeddings, no model needed) scores **0.785**, against
   0.708 for 500 random ones: about what random reaches with ~900 labels, so the cold start costs
   half the LLM calls. The edge fades by 1,500 labels and is gone by 2,000. Once there is a model
   to choose with, uncertainty and committee-disagreement selection buy only +0.5 pt, not the
   "2× fewer calls" the literature suggests, because the ambiguous tickets they pick are exactly
   the ones the teacher mislabels (19.8 % wrong vs 15.2 % on average).
6. **A lesson in leakage.** The first teacher run scored **0.948**. The test file is sorted by
   intent, so every batch of 20 queries shared one intent and the LLM used the batch as a hint.
   Relabelled in shuffled order: **0.867**. The labeller now always shuffles.
   [docs/teacher.md](docs/teacher.md)

## Protocol

- **Human labels are for evaluation only.** Students train on teacher labels. The
  gold-trained rows exist only as the supervised reference.
- **Zero-shot teacher**: the prompt carries the 77 intent names and a one-line description
  each, no example queries (examples would leak gold labels into the "LLM-labelled" data).
  It returns its top 3, parsed strictly: an answer outside the 77 names is a failure, never a
  guess (0 of 3,080 failed).
- **Resumable labelling** under free-tier rate limits: every batch is checkpointed to JSONL,
  and the label files are committed, so every number here reproduces without an API key.
- **Every number in `docs/` comes from a script in `scripts/`** that can be re-run.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt                      # service, scripts, pytest
python scripts/download_data.py                          # Banking77 → data/raw/
pytest -q
python scripts/baseline.py --labels teacher              # TF-IDF student, ~1 min on CPU
python scripts/evaluate.py                               # rebuilds docs/results.md
```

The transformer students need PyTorch (CPU is enough for the frozen MiniLM; the fine-tunes run
on a free Colab T4 via `notebooks/finetune_colab.ipynb`):

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements-train.txt
python scripts/setfit_student.py --mode frozen --labels teacher
```

Serve a student, same schema for every model (`GET /models` lists them):

```bash
uvicorn distilroute.serve:app --port 8000
curl -X POST 127.0.0.1:8000/route -H 'content-type: application/json' \
     -d '{"text": "my card still has not arrived"}'
# {"intent": "card_arrival", "confidence": 0.977, "ranked": [...], "calibrated": true, ...}
```

Or as a container: one ONNX student, no torch, no API key (`--build-arg MODEL=` picks it; build
from a checkout that has `models/<MODEL>/`):

```bash
docker build -t distilroute .
docker run --rm -p 8000:8000 distilroute
```

Labelling more data needs a free key (no card): copy `.env.example` to `.env`, set
`GROQ_API_KEY` (https://console.groq.com/keys), then `python scripts/label.py --split train`.

## Scope

Distillation into a fixed-label classifier covers "read a short text, pick a bucket": routing,
moderation, triage, sentiment. It does not cover free-text outputs such as summaries or
replies. The out-of-scope test uses CLINC150, whose negatives are *far* from banking; a
mortgage or insurance question would be harder, and no near-out-of-scope set exists for
Banking77.

_Status and the full task list: [ROADMAP.md](ROADMAP.md). Project write-up:
[docs/PROJECT.md](docs/PROJECT.md)._
