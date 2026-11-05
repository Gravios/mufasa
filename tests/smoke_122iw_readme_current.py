"""Smoke test for patch 122iw — README matches the current project.

Facts the README states that drifted from the code before this patch, each
checked against its source of truth so they cannot drift again silently:

* the console commands in the table are exactly pyproject's [project.scripts];
* every workbench page title in the README's page table is a real page;
* every pose importer named in the README is offered by the import form;
* the install line installs the ``qt`` extra (PySide6 is not a base dep);
* the project-layout tree names the real ROI file, stage directories and
  the models / backups folders; the layout doc agrees;
* the license named is the one in LICENSE;
* the standalone smoother command and its flags exist.
"""
from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


readme = (REPO / "README.md").read_text()
py = tomllib.loads((REPO / "pyproject.toml").read_text())

scripts = set(py["project"]["scripts"])
rows = set(re.findall(r"^\| `(mufasa[\w-]*)` \|", readme, re.M))
check(f"README command table == [project.scripts] ({sorted(scripts)})", rows == scripts)

titles = set()
for f in (REPO / "mufasa" / "ui_qt" / "pages").glob("*_page.py"):
    titles |= set(re.findall(r'add_page\("([^"]+)"', f.read_text()))
sec = readme[readme.find("The workbench sidebar"):readme.find("The Kalman v2 smoother also")]
named = {t.strip() for row in re.findall(r"^\| ([^|]+) \|", sec, re.M)[1:]
         for t in row.split("·")}
check("every page in the README's page table exists", named and named <= titles)
check("…and every workbench page is in the table", titles <= named)

imp = (REPO / "mufasa" / "ui_qt" / "forms" / "pose_import.py").read_text()
for name in ["FreeDLC", "DLC", "SLEAP", "MARS", "TRK", "FaceMap", "YOLO-pose", "SuperAnimal-TopView"]:
    check(f"importer {name} named in README is offered by the import form",
          name in readme and name in imp)

check("install line uses the qt extra (PySide6 is optional in pyproject)",
      'pip install -e ".[qt]"' in readme
      and not any("PySide6" in d for d in py["project"]["dependencies"])
      and any("PySide6" in d for d in py["project"]["optional-dependencies"]["qt"]))
check("every extra in the README's extras table exists",
      set(re.findall(r"^\| `(\w+)` \| ", readme, re.M)) - scripts
      == set(py["project"]["optional-dependencies"]) - {"qt"})

layout = (REPO / "docs" / "v1_project_layout.md").read_text()
for doc, text in [("README", readme), ("v1_project_layout.md", layout)]:
    check(f"{doc}: ROI file is rois.json, not ROI_definitions.h5 in the tree",
          "measures/rois.json" in text or "rois.json" in text.split("```")[1])
    check(f"{doc}: tree shows smoothed/kalman_v2, per-family features, backups",
          "smoothed/kalman_v2/<run_id>/" in text
          and "features/<family>/<video>.parquet" in text and "backups/" in text)
pl = (REPO / "mufasa" / "project_layout.py").read_text()
check("those paths are what the code uses",
      '"rois.json"' in pl and 'KALMAN_V2 = "kalman_v2"' in pl
      and '"backups"' in (REPO / "mufasa" / "model" / "marker_rename.py").read_text())
check("layout doc no longer claims both layouts are supported",
      "Both layouts remain supported" not in layout)

lic = (REPO / "LICENSE").read_text().splitlines()[0]
check("README names the license in LICENSE", "BSD 3-Clause" in lic and "BSD 3-Clause" in readme
      and "GPL" not in readme)

sm = (REPO / "mufasa" / "data_processors" / "kalman_pose_smoother_v2.py").read_text()
check("standalone smoother command and flags exist",
      "python -m mufasa.data_processors.kalman_pose_smoother_v2" in readme
      and all(f'"{f}"' in sm for f in ["--beside-input", "--save-model", "--load-model"]))

links = re.findall(r"\]\(((?:docs/|CHANGELOG|LICENSE)[^)#]*)\)", readme)
check("every local link in the README resolves", links and all((REPO / l).exists() for l in links))

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122iw_readme_current: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
