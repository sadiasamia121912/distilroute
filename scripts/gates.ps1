# Tomorrow's first job, once the Groq daily budget has reset (~40 calls, ~80k tokens):
# the same 200 test queries (seed 1) under the v2 descriptions, gating three changes.
# Compare against test.cmp_low_desc.jsonl (v1 descriptions, batch 20, top-1: 0.885).
$py = ".\.venv\Scripts\python.exe"
& $py scripts/label.py --split test --limit 200 --seed 1 --descriptions --run gate_v2
& $py scripts/label.py --split test --limit 200 --seed 1 --descriptions --run gate_v2_top3 --top-k 3
& $py scripts/label.py --split test --limit 200 --seed 1 --descriptions --run gate_v2_b50 --batch-size 50
& $py scripts/label.py --split test --limit 200 --seed 1 --descriptions --run gate_v2_b100 --batch-size 100
