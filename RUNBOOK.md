# Runbook: carrying on without Claude

Everything below runs in PowerShell from the repo folder (`C:\Users\User\dev\distilroute`).
Always use the venv's Python, `.\.venv\Scripts\python.exe`: plain `python` is the system 3.13
without pandas, and every step fails at once.

Where things stand: the top entry of **ROADMAP.md → Session log**. A future Claude session
should read that first ("continue" is enough).

## 0. Setting up on a new machine (e.g. after the 2026-10 SSD swap)

```powershell
mkdir C:\Users\User\dev; cd C:\Users\User\dev      # keep this exact path: Claude's memory folder is named after it
git clone https://github.com/sadiasamia121912/distilroute
git clone https://github.com/sadiasamia121912/portfolio-notes   # private: notes, Claude memory, models backup
cd distilroute
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
.\.venv\Scripts\python.exe scripts\download_data.py
.\.venv\Scripts\python.exe scripts\download_clinc.py
copy .env.example .env      # then paste NEW keys (Groq, Gemini, OpenRouter); the old ones are compromised
```

- **Claude's memory:** copy `portfolio-notes\claude-memory\distilroute\*` into
  `C:\Users\User\.claude\projects\C--Users-User-dev-distilroute\memory\` (same for `tabaudit` →
  `...-dev-tabaudit\memory\`, `dev` → `...-dev\memory\`). Or ask Claude to do it.
- **Trained models:** download the `models-backup-*` release zip from portfolio-notes
  (`gh release download -R sadiasamia121912/portfolio-notes`) and unzip it here, giving `models\`.
- Then open Claude Code in `distilroute` and say "continue".

## 1. Daily LLM jobs (Groq)

```powershell
.\.venv\Scripts\python.exe scripts\jobs.py --status   # what is done / pending, with row counts
.\.venv\Scripts\python.exe scripts\jobs.py            # run until done or the daily limit
```

- Run it once a day, evenings are best. It resumes where it stopped, so stopping it is harmless.
- It ends by itself with `daily limit reached in <step>, back at <time>; lane paused`
  (see `logs\queue.log`). Then it is safe to close the laptop.
- Never run `label.py` by hand while `jobs.py` runs: two writers duplicate rows.
- Left as of 2026-10-04: noise `wrap` → robustness with the teacher → descriptions v1 (200)
  → descriptions check.

**After a step shows `done`**, commit what it wrote and push:

```powershell
git status --short                      # new/changed files under data/labels, results, docs
git add data/labels results docs
git commit -m "Labels: teacher on noise wrap (6.6), 300 test queries (jobs.py)"
git push
```

## 2. The 10,003-label retrain (roadmap 1.6)

**Laptop part:** `scripts\retrain_check.py`. Already run on 2026-10-04: results are in
`results\teacher10003\` and the summary in ROADMAP. To redo it: `.\.venv\Scripts\python.exe
scripts\retrain_check.py` (~5 min).

**Colab part (GPU, ~20 min):**

1. Open https://colab.research.google.com → File → Open notebook → GitHub →
   `sadiasamia121912/distilroute` → `notebooks/finetune_colab.ipynb`.
2. Runtime → Change runtime type → **T4 GPU**. Then Runtime → **Run all**.
3. At the end it downloads `distilroute_runs.zip`. Back on the laptop:

```powershell
tar -xf $HOME\Downloads\distilroute_runs.zip   # adds results/* and models/* only
.\.venv\Scripts\python.exe scripts\seeds.py
.\.venv\Scripts\python.exe scripts\robustness.py
.\.venv\Scripts\python.exe scripts\stress.py
.\.venv\Scripts\python.exe scripts\oos.py
.\.venv\Scripts\python.exe scripts\evaluate.py
$env:DISTILROUTE_TEACHER_ROWS = "10003"; .\.venv\Scripts\python.exe scripts\evaluate.py; Remove-Item Env:DISTILROUTE_TEACHER_ROWS
git add results docs; git commit -m "Colab 7.3 runs: aug, seeds, OE, 10,003 labels"; git push
```

(`models\` is gitignored: the model files stay on the laptop, which is fine.)

## 3. Should the served model change? (the rule)

The served model is `minilm_ft_teacher` (int8 ONNX, **0.842** on the test split through the
serving code). A new model replaces it **only if its int8 accuracy is above 0.845** (0.842 +
0.3 pt run-to-run noise) **and** int8 loses at most 0.3 pt against its own fp32. The numbers are in
`results\teacher10003\minilm_ft_teacher.json` and `minilm_ft_oe_teacher.json`.

- **It passes:** the swap touches several files (`SERVED`/`MODEL` in `scripts\build_demo.py`,
  `build_case_study.py`, `build_hf_release.py`, the `Dockerfile`, `.github\workflows\docker.yml`
  and `load-test.yml`, plus a `model-v2` GitHub release). Best done in a Claude session.
  With the OE model served, `ALSO_THRESHOLD` in `distilroute\multi.py` can drop from 0.95 to 0.8.
- **It does not pass:** that is a result too. The labels are saturated and the gap is the
  teacher's own error rate. Write it into ROADMAP 1.6 and keep the served model.

## 4. Hugging Face release

```powershell
hf auth login     # paste a *write* token from huggingface.co/settings/tokens
.\.venv\Scripts\python.exe scripts\build_hf_release.py --user YOUR_HF_USERNAME            # build + check
.\.venv\Scripts\python.exe scripts\build_hf_release.py --user YOUR_HF_USERNAME --publish  # upload
```

## 5. If something breaks

- `jobs.py` step `FAILED`: read `logs\queue_<step>.log`. A `FAILED ... exit 1` right after you
  pressed Ctrl+C is just the stop, not a real failure.
- Tests: `.\.venv\Scripts\python.exe -m pytest -q`.
- Undo uncommitted changes to one file: `git restore <file>`. Nothing pushed is ever lost.
