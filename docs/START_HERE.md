# Start here

A one-page map of distilroute. The pitch and the headline results are in the
[README](../README.md); the full history of every decision is in [ROADMAP.md](../ROADMAP.md).
This page says where each answer lives and what is still open.

## Which question, which page

| question | page | made by |
|---|---|---|
| How good is each student, and what does it cost? | [results.md](results.md) | `scripts/evaluate.py` |
| How good is the teacher, and where does it go wrong? | [teacher.md](teacher.md) | `scripts/teacher_report.py` |
| How sure are the numbers? | [bootstrap.md](bootstrap.md) | `scripts/bootstrap.py` |
| When should the router still call the LLM? | [cascade.md](cascade.md) | `scripts/cascade.py` |
| Does it notice off-topic messages? | [oos.md](oos.md) | `scripts/oos.py` |
| Does it notice a *new* banking intent, and how many labels does one need? | [new_intents.md](new_intents.md) | `scripts/new_intents.py` |
| Does it survive typos, slang and greetings? | [robustness.md](robustness.md) | `scripts/robustness.py` |
| Long tickets, two questions in one message? | [stress.md](stress.md) | `scripts/stress.py` |
| Where should a few human labels go? | [correction.md](correction.md) | `scripts/correct.py` |
| What does it take under real load? | [load_test.md](load_test.md) | `scripts/load_test.py`, `.github/workflows/load-test.yml` |
| Does it transfer to another dataset? | [clinc150/](clinc150/) | the same scripts, `DISTILROUTE_DATASET=clinc150` |
| Do more teacher labels help? | [teacher5000/](teacher5000/) | `scripts/retrain_check.py` |
| The whole story, for a reader | [case-study/](case-study/) | `scripts/build_case_study.py` |

Every number on these pages is written by the script next to it; none is typed by hand.

## The code

- `distilroute/` — the package: data loading (`data.py`), the LLM teacher (`teacher.py`), every
  student behind one interface (`students.py`), calibration, text noise (`perturb.py`), and
  the FastAPI service (`serve.py`).
- `scripts/` — one script per experiment (table above), plus `label.py` (ask the teacher) and
  `jobs.py` (every pending LLM job, resumable, one lane per provider).
- `notebooks/finetune_colab.ipynb` — the GPU runs, on Colab's free T4.
- `Dockerfile` — the served student alone: int8 ONNX, no torch, no API key.
- `data/labels/` — every teacher answer ever paid for, so every result reproduces without a key.

## Open work

- **1.6 / 7.3** — the teacher is labelling the rest of the training split (`jobs.py`, once a
  day on Groq's free tier). Then one Colab session retrains the served student on all 10,003
  labels; the typo-augmented model and seed noise can run before that.
- **7.1** — a real person checks the teacher's least-believed labels (`scripts/review.py`),
  replacing the simulated human of finding 5.
- **7.4** — a live endpoint as a free Hugging Face Space.
- **7.5** — a second teacher (Gemini) on 200 queries, once a key is in `.env`.
- **7.7** — the write-up and posts, with the numbers above.
