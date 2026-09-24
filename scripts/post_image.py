"""Render the case study's first chart as a LinkedIn image (roadmap 4.2).

    python scripts/build_case_study.py && python scripts/post_image.py

Takes docs/case-study/index.html — the published page, numbers and all — hides everything but
the first figure (accuracy against cost for every system), adds a one-line source note, and
screenshots it with headless Edge (or Chrome) at 1200 x 627, LinkedIn's landscape size, at 2x,
in the page's light theme (a LinkedIn feed is white; headless browsers follow the OS theme).
The chart is drawn by the page's own code from the page's own data, so the image cannot drift
from the write-up. Writes docs/case-study/media/pareto_linkedin.png.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distilroute.data import ROOT, rel  # noqa: E402

PAGE = ROOT / "docs" / "case-study" / "index.html"
OUT = ROOT / "docs" / "case-study" / "media" / "pareto_linkedin.png"
BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "msedge",
    "google-chrome",
    "chromium",
]
WIDTH, HEIGHT = 1200, 627  # LinkedIn landscape, 1.91 : 1

NOTE = (
    "Banking77 support messages, 77 queues. Accuracy on 3,080 held-out messages against human "
    "labels; students trained on 3,000 LLM labels only. distilroute"
)
ONLY_FIRST_FIGURE = """
<script>document.documentElement.dataset.theme = "light";</script>
<style>
  body { background: var(--bg); }
  .page { max-width: 1200px; padding: 34px 44px 0; gap: 0; }
  .page > header, .tiles, footer { display: none !important; }
  article > :not(figure:first-of-type) { display: none !important; }
  article { gap: 0; }
  figure:first-of-type figcaption, figure:first-of-type details { display: none; }
  figure:first-of-type .fig-title { font-size: 26px; max-width: none; }
  figure:first-of-type .legend { font-size: 15px; }
  figure:first-of-type::after {
    content: "__NOTE__";
    font: 400 13px/1.4 var(--mono); color: var(--muted);
  }
</style>
"""


def browser() -> str:
    for b in BROWSERS:
        if Path(b).exists() or shutil.which(b):
            return b
    raise SystemExit("no Edge / Chrome found; install one or add its path to BROWSERS")


def main() -> None:
    inject = ONLY_FIRST_FIGURE.replace("__NOTE__", NOTE)
    html = PAGE.read_text(encoding="utf-8").replace("<style>", inject + "<style>", 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        shot = Path(tmp) / "shot.html"
        shot.write_text(html, encoding="utf-8")
        subprocess.run(
            [
                browser(),
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--force-device-scale-factor=2",
                f"--window-size={WIDTH},{HEIGHT}",
                "--virtual-time-budget=8000",  # let the web fonts load before the shot
                f"--screenshot={OUT}",
                shot.as_uri(),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
    print(f"-> {rel(OUT)}")


if __name__ == "__main__":
    main()
