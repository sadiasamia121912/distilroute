# distilroute

**Distil an LLM support-ticket router into a 22M-parameter model: 99 % of the LLM's accuracy,
4 ms on a CPU, ~7,000× cheaper than one LLM call per ticket, built for $0.**

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

The student never sees a human label: the served model trains on **all 10,003 teacher-labelled
queries**; the comparison rows below it on the first **3,000**, the budget every model and
experiment was first compared at. Every accuracy is against the human labels on the untouched
3,080-query test split.

| model | params | teacher labels | accuracy | macro-F1 | p50 / p95 latency (CPU) | $ per 1M requests |
|---|---:|---:|---:|---:|---:|---:|
| **Teacher**: gpt-oss-120b, zero-shot (Groq) | 120B | — | **0.867** | 0.864 | API | 216.15 (1 query / call) · 26.62 (batched 20) |
| **MiniLM-L6 fine-tuned + outlier exposure, int8 ONNX** ← served | 22M | 10,003 | **0.863** | 0.860 | **4.0 / 7.9 ms** | **0.03** |
| MiniLM-L6 fine-tuned, int8 ONNX | 22M | 10,003 | 0.862 | 0.858 | 4.0 / 7.6 ms | 0.03 |
| MiniLM-L6 fine-tuned, int8 ONNX | 22M | 3,000 | 0.847 | 0.845 | 4.0 / 8.1 ms | 0.03 |
| MiniLM-L6 frozen + logistic head | 22M | 3,000 | 0.848 | 0.846 | 12.6 / 16.7 ms | 0.08 |
| DistilBERT fine-tuned, int8 ONNX | 67M | 3,000 | 0.839 | 0.837 | 11.9 / 24.7 ms | 0.08 |
| TinyBERT fine-tuned, int8 ONNX | 14M | 3,000 | 0.796 | 0.794 | 2.2 / 4.2 ms | 0.01 |
| TF-IDF + logistic regression | — | 3,000 | 0.812 | 0.809 | 2.0 / 2.9 ms | 0.01 |
| **Cascade**: served MiniLM, escalate to the LLM below 0.8 confidence | — | 10,003 | **0.872** | — | 4.0 ms for 86 % of queries | ≈ 30 |
| **Cascade**: TF-IDF → served MiniLM → LLM, cheapest thresholds that match the LLM | — | 3,000 / 10,003 | 0.863 ± 0.008 | — | 4.0 ms mean; 2 % also wait for the LLM | **5.26** ± 9.96 |

- **Teacher cost** is Groq's *paid* list price. The free tier the labels were made with is
  rate-capped and not a production option. **Student cost** is CPU time on an AWS t3.small at
  on-demand price, one request at a time: an upper bound, and $0 on hardware you already own.
- **Latency** is one query at a time, in-process, on a consumer CPU (`scripts/bench_latency.py`,
  every row re-measured on one machine on 2026-10-17; the previous machine was ~30 % faster).
  **Measured under load** (finding 10), the served container on 2 CPUs sustains **497 requests
  per second**, which puts its cost at **$0.012 per 1M**, below the estimate above.
- **Accuracy of the fine-tuned rows is the fp32 model's;** latency and cost are the int8 graph's,
  which is what gets served. int8 costs 0.0–0.7 pt: the served MiniLM scores **0.859** through
  the serving code (DistilBERT 0.838, TinyBERT 0.792).
- **The three-tier row** picks its thresholds on one half of the test split and is scored on the
  other (5 random halvings, mean ± half-range), so it is never tuned on the queries it is scored
  on; it matches the LLM to within that noise, and the cost varies widely between halvings
  because the LLM share is so small (2.4 % ± 4.6 %). [docs/cascade.md](docs/cascade.md)
- **Reference, trained on all 10,003 human labels:** MiniLM 0.927, DistilBERT 0.928,
  TF-IDF 0.910. The 6–8-point gap to the distilled rows is the teacher's own error rate, passed
  on to the student.

- **How sure:** 3,080 test queries put a ±1.3 pt 95 % interval on every accuracy above (paired
  bootstrap, 10,000 resamples). Differences between two systems scored on the same queries are
  tighter: the served student is 0.8 pt below the teacher (−1.7 to +0.1), so within test noise
  of it; every 3,000-label student is reliably below it; and every claim below of "beats" or
  "ties" is checked the same way.
  [docs/bootstrap.md](docs/bootstrap.md)

Full tables (calibration, cascade thresholds, data curves, every variant tried):
[docs/results.md](docs/results.md).

## What was found

1. **Distillation keeps 99 % of the teacher's accuracy** (0.859 vs 0.867, served int8) at 4 ms
   and $0.03 per 1M requests. Labelling all 10,003 training queries instead of 3,000 is worth
   **+1.5 pt** (+0.6 to +2.4) for the fine-tuned MiniLM, for about $0.19 of LLM calls at paid
   prices. MiniLM-L6 at 22M parameters matches DistilBERT at 67M on human labels (0.927 vs
   0.928) and is at least as good on teacher labels (+0.8 pt, interval −0.0 to +1.6), so it is
   the model to ship. int8 quantisation costs 0.0–0.7 pt (0.863 → 0.859 for the served MiniLM,
   −0.8 to +0.1) for a 4× smaller graph.
2. **The cascade matches or beats the teacher using 14 % of its calls.** Calibrate the student
   (temperature scaling, fitted on a held-out slice of its training labels), answer when it is
   ≥ 0.8 confident, and send the rest to the LLM: 0.872 overall, +0.9 pt over the student alone
   (+0.2 to +1.7) and +0.5 pt over the LLM alone, at about a seventh of its cost. That last gap
   is within test noise (−0.0 to +1.1; ahead in 96 % of resamples), so the honest claim is
   *at least as good as the LLM*. Calibration is what makes "0.8" mean something. With thresholds
   chosen on held-out data, the cheapest system that **matches the LLM costs about $5 per 1M
   requests, ~40× less**: TF-IDF answers 58 % of queries, the served MiniLM most of the rest,
   and 2.4 % reach the LLM. With so few escalations the cost swings between halvings
   (± $9.96), so read it as "tens of times cheaper", not an exact factor. On 3,000 labels the
   same system needed 6.4 % of queries on the LLM and cost $13.76: a better student escalates
   less. The TF-IDF first tier saves LLM calls in 9 of 10 paired halves but changes accuracy by
   only ±0.1 pt at a fixed LLM budget; TinyBERT is no use as a first tier.
   [docs/cascade.md](docs/cascade.md)
3. **It knows when a message is not its job.** On CLINC150's 1,200 out-of-scope queries, the
   served model flags **98 %** of them while escalating only 10 % of genuine banking queries
   (AUROC 0.987); students without outlier exposure flag 86–97 % (AUROC 0.94–0.99; TinyBERT only
   67 %). Outlier
   exposure means 2,000 non-banking CLINC150 training queries were added to its training data
   with a uniform target ("none of these intents"); no extra LLM calls, and it costs nothing on
   banking queries (+0.1 pt, −0.5 to +0.7). Caveat: those queries come from the same dataset as
   the out-of-scope test set (a different split), so 98 % is the in-distribution figure; the
   86–97 % of the plain students is the safer guide to unseen kinds of off-topic text. Score
   with the entropy of the prediction, not max-probability: entropy wins on every model.
   Temperature scaling slightly *hurts* this separation, so calibrate for the cascade and use
   entropy for out-of-scope. [docs/oos.md](docs/oos.md)
4. **Typos are the weak spot, and the transformers are the fragile ones.** Re-scored on noisy
   copies of the test set, three typos per message cost TF-IDF 12 pt but the served MiniLM
   30, DistilBERT 33 and TinyBERT 43: WordPiece shatters a misspelt word into unfamiliar pieces,
   while TF-IDF's character n-grams still overlap with the right spelling. One typo: TF-IDF −4,
   served MiniLM −8. More labels did not help (3,000 or 10,003: −30 either way). The teacher
   barely notices: on the same 300 queries, three typos cost it **2.7 pt** (0.870 → 0.843)
   against 31 for the served model. Lowercase without punctuation, texting slang ("u", "pls",
   "acct") and greetings cost everyone 0–4 pt. The cascade absorbs part of it, because the
   students also get less sure: with three typos the served MiniLM escalates 54 % of messages
   instead of 14 % and stays 0.844 accurate on the rest, so noise shows up as LLM cost rather
   than silent misroutes. **The fix is free:** train on typo'd copies of the same
   teacher-labelled rows (`--augment typo1,typo3`, different random draws from the test copies,
   no extra LLM calls). On the fine-tuned MiniLM at 3,000 labels it cuts the three-typo loss
   from 30 to **10.5 pt** (one typo: 10 → 3) with clean accuracy unchanged (0.848 vs 0.847); on
   the frozen MiniLM from 29 to 16 pt; on TF-IDF from 12 to 9. It does not carry over to the
   other kinds of noise, which barely hurt anyway. Not yet combined with the 10,003 labels of the
   served model (next on the roadmap). [docs/robustness.md](docs/robustness.md)
5. **With no human labels the student does not beat its teacher; with 500 it does.** All
   10,003 LLM labels bring the fine-tuned student to 0.863 (fp32), a tie with the teacher's
   0.867 by test noise (−0.4 pt, −1.3 to +0.5) but not ahead of it. On 3,000 labels, soft
   top-3 targets, a confident-learning noise filter and self-training on the 7,003 unlabelled
   queries are together worth +0.7 pt (0.849), still 1.8 pt below the teacher. The soft target
   *hurts* on 77 near-synonym intents. What works is a small human budget spent in the right
   place: rank the teacher's labels by how little the student believes them (out-of-fold, the
   same confident-learning score) and have a human check the top of the list. 78 % of the first
   100 checked are real teacher errors, 5× the base rate. **500 checks lift the frozen student
   from 0.842 to 0.876, above the teacher's 0.867 (+0.9 pt, −0.3 to +2.0: ahead in 93 % of
   resamples); 1,000 reach 0.890, clearly above it (+2.2 pt, +1.1 to +3.4).** The same 500 human labels
   spent on random rows give 0.854, and on 500 *new* rows 0.857: fixing the LLM's labels is
   worth far more than adding to them (both gaps about +2 pt, intervals +1.3 to +3.0).
   [docs/correction.md](docs/correction.md)
6. **Which tickets to pay the LLM for matters only at the start.** Picking the 500 most
   *typical* queries (k-means over the embeddings, no model needed) scores **0.785**, against
   0.708 for 500 random ones: about what random reaches with ~900 labels, so the cold start costs
   half the LLM calls. The edge fades by 1,500 labels and is gone by 2,000. Once there is a model
   to choose with, uncertainty and committee-disagreement selection buy only +0.5 pt, not the
   "2× fewer calls" the literature suggests, because the ambiguous tickets they pick are exactly
   the ones the teacher mislabels (19.8 % wrong vs 15.2 % on average).
7. **A lesson in leakage.** The first teacher run scored **0.948**. The test file is sorted by
   intent, so every batch of 20 queries shared one intent and the LLM used the batch as a hint.
   Relabelled in shuffled order: **0.867**. The labeller now always shuffles.
   [docs/teacher.md](docs/teacher.md)
8. **A new banking intent slips past per-message checks, so the service watches the traffic
   mix instead.** Retrained without 10 of the 77 intents (3 random draws), the student catches
   only **53 %** of their queries at the same 10 % escalation budget that catches 95 % of
   CLINC150's off-topic traffic. Distance scores on the embeddings, near-perfect on off-topic
   messages (AUROC 0.997), are *worse* here (0.72): a new banking intent sits among the known
   ones, and no per-message score tried beats entropy by more than noise. What gives it away is
   that one neighbour absorbs it (`failed_transfer` → `declined_transfer`, 88 %). So the service
   now has a **traffic monitor** (`GET /monitor`): it compares the mix of routed intents with a
   reference and names any intent whose share jumps. In simulation it flags a new intent at 5 %
   of traffic within a 2,000-message window 94 % of the time, at 10 % within 500 messages 97 %
   of the time, with 1.6 % false alarms. Adding the intent back with 5 labelled examples routes
   39 % of it correctly; **50 match the student that had it from the start**.
   [docs/new_intents.md](docs/new_intents.md)
9. **Long tickets used to break the served model; now it reads them, and small talk no longer
   fools it.** The first served model was cut at the 64 tokens it was trained on, so a question
   at the end of a 120-word ticket scored **0.013** (the cascade escalated 90 % of those, so they
   cost LLM calls rather than misroutes). Reading 512 tokens fixed most of that (0.803).
   DistilBERT recovers the same way (0.023 → 0.783); TinyBERT does not cope with long padding at
   all. Routing each sentence alone is the other lever, but plain fine-tuned students are
   confidently wrong on small talk ("hi there, hope you are well" → `card_arrival`), so a filler
   sentence outscores the real question. Outlier exposure (finding 3) removes that: no filler
   sentence passes the off-topic filter, and sentence routing lifts the served model on that
   120-word ticket from 0.733 to **0.830** (the same model without it: 0.517). It also makes the
   `also` field of `POST /route` (a second question in the same message) usable: at the service's
   0.9 threshold both questions come back in **50 %** of two-question messages, and only 2 % of
   single-question long tickets get an invented second intent. A 10,003-label model *without*
   outlier exposure, briefly served, invented one in 41 % even at 0.95, which is why the served
   model has it. [docs/stress.md](docs/stress.md)
10. **Load-tested, the router costs $0.012 per 1M requests.** The Docker image on a Linux
    runner, real queries, 1–32 concurrent clients, no errors: 270 req/s on 1 CPU, 497 on 2. Two
    service fixes came out of it: size onnxruntime's thread pool to the container's CPU quota
    (it had been throttled into a 53–79 ms p95 at a single client), then run one single-thread
    inference per CPU (1.6× the throughput of one multi-thread inference at a time, compared on
    the same runner; runner CPUs changed between runs and moved the model's speed 3×). Measured
    on the first served model; the current one has the same architecture and graph size.
    [docs/load_test.md](docs/load_test.md)

## Protocol

- **Human labels are for evaluation only.** Students train on teacher labels. The
  gold-trained rows exist only as the supervised reference.
- **Zero-shot teacher**: the prompt carries the 77 intent names and a one-line description
  each, no example queries (examples would leak gold labels into the "LLM-labelled" data;
  how the descriptions were written is under [Limitations](#limitations)).
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

Or as a container: the served student (int8 ONNX, trained on teacher labels only), no torch,
no API key. The model is a [release file](https://github.com/sadiasamia121912/distilroute/releases/tag/model-v3);
CI builds this image and calls `/route` on every change.

```bash
curl -fsSLO https://github.com/sadiasamia121912/distilroute/releases/download/model-v3/minilm_ft_oe_10k_teacher.zip
unzip minilm_ft_oe_10k_teacher.zip          # -> models/minilm_ft_oe_10k_teacher/
docker build -t distilroute .
docker run --rm -p 8000:8000 distilroute
```

Labelling more data needs a free key (no card): copy `.env.example` to `.env`, set
`GROQ_API_KEY` (https://console.groq.com/keys), then `python scripts/label.py --split train`.

## Limitations

- **Human labels still win, by about 6 pt.** A MiniLM trained on 10,003 human labels scores
  0.927, above every system in the table, the LLM included. Distillation is the answer when those
  labels do not exist yet or cannot be made fast enough; it is not a replacement for them. 500
  targeted human checks (finding 5) close only part of the gap.
- **The human in finding 5 is simulated.** The gold label stands in for the reviewer's answer, so
  every check is correct and free. Real reviewers disagree, make mistakes and cost time, so
  0.876 at 500 checks is an upper bound, and even that is within test noise of the teacher.
- **The teacher saw a little human-labelled information.** The prompt carries no example
  queries, but 33 of the 77 descriptions were rewritten after reading three training queries per
  intent, because some names mislead (`get_physical_card` is about PINs). On the 200-query gate
  that took the teacher from 0.885 to 0.905. The prompt settings (descriptions, reasoning effort,
  batch size, top-3) were also chosen on those 200 *test* queries. Both are disclosed choices, not
  a strict zero-shot protocol.
- **Benchmark text, not tickets.** Banking77 queries are short (median 47 characters), English,
  one intent each and one snapshot in time. Long tickets, two questions in one message and new
  intents are simulated (findings 8 and 9) with rule-built text and held-out intents, not real
  traffic; personal data and slow drift are not measured at all.
- **Cost is list prices on stand-in hardware.** Teacher cost is a list price. Student cost is
  measured throughput (finding 10) on GitHub's shared runners, priced as a t3.small, which is a
  different and burstable machine; runner CPUs vary between runs, so throughput moves by up to
  3× from run to run.
- **One teacher, mostly one training seed.** Every student learns from gpt-oss-120b with one
  prompt. A second teacher on the same 200-query gate, gemini-3.5-flash, scores about the same
  (0.910 vs 0.905, agreeing on 96 % of queries; [docs/second_teacher.md](docs/second_teacher.md)),
  but no student was trained on its labels, so how the findings transfer to other LLMs is
  untested. The bootstrap intervals cover test sampling, not training randomness: three seeds of
  the 3,000-label fine-tune span **0.6 pt** (0.841–0.847; [docs/seeds.md](docs/seeds.md)), and
  the served model is a single seed. Read differences under ~1 pt as ties.
- **Still open.** Most comparison rows use 3,000 labels; only the fine-tuned MiniLM was retrained
  on all 10,003. The typo-augmented version of the served model is not trained yet, so the
  served model still loses 30 pt on three-typo messages (finding 4).

## Scope

Distillation into a fixed-label classifier covers "read a short text, pick a bucket": routing,
moderation, triage, sentiment. It does not cover free-text outputs such as summaries or
replies. The out-of-scope test uses CLINC150, whose negatives are *far* from banking; the
near case is simulated by holding intents out of Banking77 itself (finding 8), and a mortgage or
insurance question, outside Banking77 altogether, is not measured.

_A one-page map of every result and script: [docs/START_HERE.md](docs/START_HERE.md). Status and
the full task list: [ROADMAP.md](ROADMAP.md). Project write-up: [docs/PROJECT.md](docs/PROJECT.md)._
