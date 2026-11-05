"""Smoke test for patch 122iv — banner tool icons on their doc sections.

The banner's three tool tiles (pose smoothing, ROI analysis, classifiers)
are standalone SVGs in docs/images/ and head their sections, as in
neurosuite-3: the README's "The tools" section has one <details> block per
tool (icon + bold name in the <summary>, bullets, doc links), and the
matching headings in docs/workflows.md and docs/README.md carry the icon.
"""
from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


icons = ["smoothing", "roi", "classifiers"]
for n in icons:
    p = REPO / "docs" / "images" / f"icon_{n}.svg"
    try:
        root = ET.parse(p).getroot()
        ok = root.get("viewBox") == "0 0 106 106"
    except (OSError, ET.ParseError):
        ok = False
    check(f"icon_{n}.svg exists and is a 106x106 SVG", ok)

readme = (REPO / "README.md").read_text()
sec = readme[readme.find("## The tools"):readme.find("## Running Mufasa")]
summaries = re.findall(r"<summary>(.*?)</summary>", sec)
check("README has a 'The tools' section before 'Running Mufasa' with three "
      "<details> blocks", readme.find("## The tools") > 0 and len(summaries) == 3)
check("each summary leads with its tool icon and a bold name",
      [re.match(r'<img src="docs/images/icon_(\w+)\.svg" width="28" align="center" alt=""> <b>', s)
       and re.match(r'<img src="docs/images/icon_(\w+)\.svg"', s).group(1) for s in summaries] == icons)
links = re.findall(r"\]\((docs/[^)#]+)", sec)
check("every doc link in the section resolves",
      links and all((REPO / l).is_file() for l in links))
check("details bodies are separated by blank lines (GitHub renders the markdown)",
      sec.count("</summary>\n\n") == 3 and sec.count("\n\n</details>") == 3)

wf = (REPO / "docs" / "workflows.md").read_text()
want = {"Smooth pose": "smoothing", "Stage 4: ROI definitions and analysis": "roi",
        "Stage 5: Feature extraction": "classifiers", "Stage 6: Behavior labelling": "classifiers",
        "Stage 7: Train machine model": "classifiers", "Stage 8: Run machine model": "classifiers"}
check("workflows.md section headings carry their tool icon",
      all(re.search(rf'^#+ <img src="images/icon_{n}\.svg" width="24" align="center" alt=""> {re.escape(h)}', wf, re.M)
          for h, n in want.items()))
idx = (REPO / "docs" / "README.md").read_text()
check("docs index: the Kalman design-track heading carries the smoothing icon",
      re.search(r'^## <img src="images/icon_smoothing\.svg"[^>]*> Kalman smoother — design track', idx, re.M))

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122iv_docs_section_icons: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
