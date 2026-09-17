# distilroute

**Distil an LLM support-ticket router into a tiny model — and publish the accuracy / latency /
cost table that decides whether you needed the LLM at all.**

Teams route support tickets with a frontier LLM. It works, at 100–1000× the cost and latency
of a small classifier. The production pattern is *LLM-as-teacher → small student*: label once
with the big model, train a tiny model on those labels, serve the tiny model. This repo does
that end to end on [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets)
(13k real banking-app queries, 77 intents) for **$0**, using free-tier hosted LLMs as the
teacher.

_Status: in progress — see [ROADMAP.md](ROADMAP.md)._

## Results so far

| model | trained on | acc vs gold | macro-F1 | p50 / p95 latency (CPU) |
|---|---|---|---|---|
| TF-IDF + logistic regression | gold (10,003) | **0.913** | 0.913 | 1.9 / 4.1 ms |
| TF-IDF + logistic regression | teacher labels | — | — | — |
| SetFit (MiniLM) | teacher labels | — | — | — |
| DistilBERT | teacher labels | — | — | — |
| Teacher: Llama 3.3 70B (zero-shot) | — | — | — | — |

All accuracies are against the human labels on the untouched 3,080-query test split. The
student never sees a human label; "gold" rows are the supervised reference.

## Protocol

- **Zero-shot teacher**: the prompt lists the 77 intent names, no example queries. Examples
  would leak gold labels into the "LLM-labelled" data.
- **Strict parsing**: an answer that is not one of the 77 names is recorded as a failure,
  never guessed. The failure rate is reported.
- **Resumable labelling** within free-tier rate limits: every batch is checkpointed to JSONL.
- **Self-agreement**: a 300-query sample is labelled twice to measure how much of the
  teacher's error is noise.

## Setup

```powershell
python -m venv .venv; .\.venv\Scripts\activate
pip install -r requirements-dev.txt
python scripts/download_data.py
python scripts/baseline.py --labels gold      # ~40 s on CPU
pytest -q
```

Labelling needs a free key (no card) — copy `.env.example` to `.env` and fill in
`GROQ_API_KEY` (https://console.groq.com/keys) or `GEMINI_API_KEY`.

```powershell
python scripts/label.py --split test           # teacher accuracy ceiling first
python scripts/label.py --split train          # then the training labels
python scripts/baseline.py --labels teacher    # the distilled baseline
```
