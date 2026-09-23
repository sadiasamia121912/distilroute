# distilroute — project description

_Written 2026-09-17. Living document: the "Status" section is updated as work lands.
Task-level detail lives in [`../ROADMAP.md`](../ROADMAP.md)._

## 1. The problem

Support teams increasingly route incoming tickets with a large language model: paste the
ticket into a prompt, get back which queue it belongs to. It works well and needs no training
data — but every ticket costs an LLM call, which is roughly 100–1000× the cost and latency of
a small classifier, and it ties a core workflow to a third-party API.

The pattern production teams actually converge on is **distillation**:

1. Use the LLM once, as a *teacher*, to label a few thousand historical tickets.
2. Train a small *student* model on those labels.
3. Serve the student. Optionally, escalate the few tickets it is unsure about back to the LLM.

This project builds that pipeline end to end on a public dataset and produces the one table
every team wants before committing to it: **accuracy vs. the teacher · p50/p95 latency · cost
per million requests**, for the teacher and for each student.

Total budget: **$0**. The teacher is a free-tier hosted LLM; the students train on CPU or on
Colab's free GPU; nothing is deployed to a paid cloud.

## 2. What it is not

Distillation into a fixed-label classifier covers every "read a short text, pick a bucket"
task — routing, moderation, triage, sentiment, lead qualification. It does not cover free-text
outputs (summaries, replies). The banking domain here is only the benchmark; the code is
domain-agnostic (any CSV with `text` and `category` columns plus a label list).

## 3. Dataset — Banking77

[Banking77](https://github.com/PolyAI-LDN/task-specific-datasets) (PolyAI, 2020): 13,083
real customer queries to a banking app, human-labelled into 77 fine-grained intents.

| | |
|---|---|
| train / test | 10,003 / 3,080 queries |
| intents | 77, in both splits |
| examples per intent (train) | 35 – 187, median 127 |
| query length | median 47 chars, p95 151 |
| duplicates / train–test overlap | 0 / 0 (checked) |

Chosen because it is real (not synthetic), a standard benchmark with published results to
compare against (fine-tuned BERT ≈ 93 %, DistilBERT ≈ 92 %), and small enough to label for
free. Its 77 intents include near-synonyms (`declined_transfer` vs `failed_transfer`,
`card_not_working` vs `declined_card_payment`), which is exactly what makes zero-shot LLM
routing imperfect and distillation interesting.

## 4. Protocol — the rules that make the numbers defensible

- **Human labels are for evaluation only.** No student ever trains on a gold label. Every
  accuracy figure is measured against gold on the untouched test split. Students trained on
  gold are reported separately, as the supervised reference ("what you would get with a
  labelled dataset").
- **The teacher is zero-shot.** Its prompt contains the 77 intent names and a one-line
  description of each (`data/intent_descriptions.json`). The descriptions were written from
  the names, then revised after reading three random **train** examples per intent — never
  test — because several names are misleading in this dataset (`get_physical_card` is about
  the PIN). That is what a human writing a labelling guide from historical tickets would do.
  The prompt never contains an example query — a test asserts that no description matches a
  dataset row — and the model is asked in batches of 20 *shuffled* queries, since batches of
  one intent let it use the batch as a hint (+7 pts, found and fixed 2026-09-21).
- **Parsing is strict.** An answer that is not one of the 77 names (after case/punctuation
  normalisation) is recorded as a failure, never guessed; the failure rate is reported.
- **Three numbers per model:** accuracy vs. gold, macro-F1 vs. gold (77 classes are imbalanced
  5:1), and agreement with the teacher.
- **Every number in `docs/` comes from a script in `scripts/` that can be re-run.**

## 5. The teacher

**Model:** `openai/gpt-oss-120b` via Groq's free tier. (The plan said Llama 3.3 70B; it had
left Groq's free catalogue by the time the key was issued. gpt-oss-120b is larger and is the
strongest model the key can reach.) Fallback: Gemini Flash via Google AI Studio, also free.

**Prompt design, decided on a 200-query experiment (same queries, four configurations):**

| reasoning effort / prompt | accuracy | macro-F1 | tokens per 200 queries |
|---|---|---|---|
| low / names only | 0.860 | 0.814 | 14k |
| **low / names + descriptions** | **0.885** | **0.851** | 22k |
| medium / names only | 0.865 | 0.831 | 23k |
| medium / names + descriptions | 0.895 | 0.847 | 30k |

Descriptions are worth ~2.5 points; extra reasoning is worth ~1 point (two queries — noise at
this sample size) for 40 % more tokens and 3× the wall-clock. Chosen: **low + descriptions**.

**Free-tier mechanics.** Groq allows 8,000 tokens/min, 1,000 requests/day and roughly 200k
tokens/day for this model. Labelling is therefore a *resumable* process: queries go in
batches, every batch is appended to a JSONL checkpoint the moment it returns, and re-running
the same command continues where it stopped. Once labelled, the API is never needed again —
the label files are committed to the repo. The batch size trades throughput against
accuracy (the 77 descriptions are a fixed ~1,350 tokens per call); larger batches are gated
by the same 200-query test before use.

**Soft labels.** The teacher can be asked for its **top-3 ranked intents** per query instead
of one. The first entry is the hard label; the full list lets a student be trained on a soft
target (Hinton-style distillation), which is most useful exactly on the near-synonym intents
where a single hard label is a coin flip.

## 6. The students

| student | params | trains on | why it is in the table |
|---|---|---|---|
| TF-IDF (word + char n-grams) + logistic regression | ~0 | CPU, 40 s | the baseline everyone should beat; already 91.3 % on gold |
| SetFit on MiniLM | 22M | CPU | few-shot contrastive fine-tuning; the "no GPU at all" option |
| DistilBERT | 66M | Colab free GPU | the standard distillation target |
| TinyBERT / MiniLM-L6 | 14M / 22M | Colab | the small end of the size/accuracy curve |

Each transformer student is exported to **ONNX and int8-quantised** on Colab so the laptop
needs only `onnxruntime` (not PyTorch) to measure real CPU latency.

Every student is trained twice where it makes sense — on gold and on teacher labels — so the
table shows the **distillation gap** for the same architecture.

## 7. Beyond the basic pipeline (all free)

- **Confidence cascade.** The student answers when confident and escalates to the LLM
  otherwise. Curve: accuracy and LLM cost vs. escalation fraction. Requires a calibrated
  student → expected-calibration error is reported and temperature scaling applied. This is
  the deployment pattern most teams actually want, and its table is the strongest result the
  project can produce.
- **Data-efficiency curve.** Student accuracy vs. number of teacher labels (500 → 10k):
  "how many LLM calls do you actually need?"
- **Model-size Pareto.** Accuracy vs. parameters vs. CPU latency across the students, then
  quantisation of the winner.
- **Auditing the teacher's labels with tabaudit** (project 1): does dropping the rows flagged
  as likely label noise improve the student?
- **Teacher self-agreement:** a 300-query sample labelled twice; disagreement is noise the
  student cannot learn.
- Later: a second free teacher on the same split (agreement as a noise signal; "which free
  teacher is best"), and a second domain dataset run through the same scripts.

## 8. Deliverables

1. The results table (README + `docs/results.md`): every model × accuracy, macro-F1,
   agreement with teacher, p50/p95 latency, cost per 1M requests.
2. The cascade curve.
3. A FastAPI service with one `POST /route` endpoint and a `model=` switch, Dockerised
   (student-only image).
4. Committed label files, so anyone can reproduce the students without an API key.
5. Résumé line, filled in from the table: *"Distilled a 120B-parameter LLM ticket router into
   a 66M-parameter model retaining X % of teacher accuracy at Z× lower latency and $0 per
   1M requests."*

## 9. Repository layout

```
distilroute/
├── distilroute/teacher.py      LLM client: prompt, strict parsing, top-k, providers over plain HTTP
├── distilroute/students.py     one loader per model kind (tfidf / minilm / onnx / teacher)
├── distilroute/serve.py        FastAPI POST /route with a model= switch
├── scripts/
│   ├── download_data.py        Banking77 → data/raw/, with sanity checks
│   ├── label.py                resumable teacher labelling → data/labels/*.jsonl
│   ├── baseline.py             TF-IDF + LR, --labels gold|teacher → results/*.json + models/
│   ├── setfit_student.py       frozen MiniLM + LR head, or SetFit few-shot
│   ├── finetune.py             DistilBERT / MiniLM / TinyBERT fine-tune, ONNX int8 (Colab)
│   ├── data_curve.py           accuracy vs. number of training labels
│   ├── bench_latency.py        p50/p95 per model on this CPU → results/latency.json
│   └── evaluate.py             every run → docs/results.md
├── notebooks/finetune_colab.ipynb  runs finetune.py on a free T4
├── data/
│   ├── intent_descriptions.json  one line per intent (no dataset queries)
│   └── labels/                   committed teacher labels + the 200-query comparison runs
├── results/                    one JSON per model run
├── tests/                      teacher tests run against a fake transport (no key needed)
├── ROADMAP.md                  phases, decisions, session log
└── docs/PROJECT.md             this file
```

## 10. Status (updated 2026-09-21)

**Done**
- Data downloaded and checked; repo, 24 tests, lint.
- Teacher client: strict parsing, v2 intent descriptions (revised from train examples after
  v1 exposed misleading intent names), top-3 ranked answers, providers Groq / OpenRouter /
  NVIDIA / Gemini, resumable labeller hardened against free-endpoint failure modes.
- Teacher configuration study on 200 queries: v1 0.885 → v2 0.905; top-3 keeps top-1 and puts
  gold in the list 96 % of the time; batch 50/100 lose 2 points (rejected); second teacher
  nemotron-550b 0.835 (rejected). Final: gpt-oss-120b, v2, top-3, batch 20.
- `scripts/teacher_report.py` → `docs/teacher.md`; `scripts/evaluate.py` → `docs/results.md`
  (accuracy, macro-F1, agreement, ECE, cascade preview) from a shared run contract.
- Students on gold (supervised reference): TF-IDF + LR **0.913** (1.8 ms); frozen
  MiniLM-L6 + LR **0.930** (11.5 ms), no fine-tuning; SetFit few-shot 16/intent **0.867**.
- Data-efficiency curve on gold (`scripts/data_curve.py`): frozen MiniLM 0.84 at 1k random
  labels, 0.92 at 5k; TF-IDF 0.74 / 0.89 at the same sizes.
- `scripts/finetune.py` (DistilBERT / MiniLM / TinyBERT, hard or soft top-3 targets, ONNX +
  int8 export) and the Colab notebook that runs it; smoke-tested on CPU.
- Serving: `distilroute/serve.py` (FastAPI `POST /route`, `model=` switch, lazy loading) over
  one loader per model kind; `scripts/bench_latency.py` → `results/latency.json`.
- Cost per 1M requests (`scripts/cost.py`, paid list prices): teacher **$220** one ticket per
  call ($28 batched), MiniLM frozen **$0.07**, TF-IDF $0.01 on a t3.small.
- Calibration (`distilroute/calibration.py`): temperature fit on a 10 % training-pool holdout,
  never on test; ECE 0.074 → 0.011 for MiniLM. Threshold cascade: escalate below 0.8
  confidence → 14 % to the teacher, 97.5 % accuracy on what the student keeps.
- Train subset labelled (3,000 random rows, 0 unparsed). Teacher on it: **0.848** vs gold.
- **Distillation gap on 3k labels:** frozen MiniLM 0.848 on teacher labels vs ~0.90 on gold
  (data curve) — the student matches its teacher; TF-IDF 0.812 vs 0.86.
- **Found and fixed a protocol flaw (2026-09-21):** the intent-sorted test CSV meant every
  teacher batch was one intent, inflating the teacher's test accuracy to 0.948 (0.98 vs 0.905
  on the same 200 queries, sorted vs shuffled). The labeller now always shuffles; the test
  split was relabelled 2026-09-23. The train labels were shuffled and stand.
- **Teacher (honest): 0.867 accuracy, 0.864 macro-F1, gold in top-3 for 94.7 %.** Every
  gold-trained student beats it (MiniLM frozen 0.927, DistilBERT 0.918, TF-IDF 0.910); the
  student distilled from 3k teacher labels reaches 0.848, within 2 pts of its teacher.
- **Consequence for the cascade:** escalating a gold-trained student's least-confident 20 % to
  the teacher *lowers* accuracy (0.927 → 0.917), because the LLM is the weaker of the two. It
  still helps the teacher-trained student (0.848 → 0.871). The deployable lesson is to measure
  the teacher on held-out data before building an escalation path to it.

**Paused with**
- The labeller running as a detached process (log: `logs/label_test.log`). It survives the
  assistant session but not a reboot; restarting is safe, it resumes from the checkpoint:
  `.\.venv\Scripts\python.exe scripts/label.py --split test --descriptions --top-k 3`
- `models/` holds `tfidf_lr_gold` and `minilm_frozen_gold` (gitignored, rebuilt by the
  scripts); the service serves both: `uvicorn distilroute.serve:app --port 8000`.

**Next, in order**
1. Colab: `notebooks/finetune_colab.ipynb` on a T4 with the gold rows (zip the checkout for
   cell 1 until the repo is public). Unzip the download into the checkout, run
   `scripts/evaluate.py` and `scripts/bench_latency.py`.
2. Test labels complete → `scripts/teacher_report.py` (the real teacher ceiling), the cascade
   table fills in, teacher latency (`bench_latency.py --teacher 30`, once the labeller is idle).
3. `label.py --split train --limit 3000 --seed 0` → every student with `--labels teacher`
   (baseline, setfit_student, data_curve, finetune on Colab) → the distillation gap.
4. Dockerfile (3.4, needs Docker installed), README table (4.1).
