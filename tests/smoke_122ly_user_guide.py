"""Smoke test for patch 122ly — the user guide (``docs/guide/``) and its developer chapters.

User: "Can you add a docs folder and document the current features?"; for "both users and
developers".

The guide is checked against the code, so it can't drift silently:

* every chapter the index lists exists, and every chapter is in the index;
* every relative link resolves, anchors included (GitHub's heading slugs);
* every workbench page, and every section of every page, is named in the guide;
* every **Page › Section** the guide names is a real page and one of its sections;
* every ``mufasa`` command, ``mufasa list`` subject, console tool and ``mufasa run`` step is in the
  command-line chapter;
* the developer chapters name APIs that exist (``project_layout``, ``OperationForm``'s methods,
  ``SECTIONS``, ``COMMANDS``, ``FEATURE_FAMILIES``, ``process_one_video``, ``STEPS`` / ``CHAIN``);
* the docs index and the README point to the guide.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
GUIDE = REPO / "docs" / "guide"

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


def slug(heading: str) -> str:
    """GitHub's anchor for a heading."""
    s = heading.strip().lower()
    s = re.sub(r"[`*_]", "", s)
    s = re.sub(r"[^\w\- ]", "", s)
    return s.replace(" ", "-")


def anchors(path: Path) -> set[str]:
    out, seen = set(), {}
    in_code = False
    for line in path.read_text().splitlines():
        if line.startswith("```"):
            in_code = not in_code
        if in_code:
            continue
        m = re.match(r"^#{1,6} (.+)$", line)
        if m:
            a = slug(m.group(1))
            n = seen.get(a, 0)
            out.add(a if n == 0 else f"{a}-{n}")
            seen[a] = n + 1
    return out


def pages_and_sections() -> dict[str, set[str]]:
    pages: dict[str, set[str]] = {}
    for p in sorted((REPO / "mufasa/ui_qt/pages").glob("*_page.py")):
        tree = ast.parse(p.read_text())
        for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
            title, secs = None, []
            for n in ast.walk(fn):
                attr = getattr(getattr(n, "func", None), "attr", "")
                if isinstance(n, ast.Call) and attr == "add_page":
                    title = n.args[0].value
                if (isinstance(n, ast.Call) and attr in ("add_section", "add_section_widget") and n.args
                        and isinstance(n.args[0], ast.Constant)):
                    secs.append(n.args[0].value)
            if title:
                pages.setdefault(title, set()).update(secs)
    return pages


def main() -> int:
    files = sorted(GUIDE.rglob("*.md"))
    index = (GUIDE / "README.md").read_text()
    listed = {(GUIDE / m).resolve() for m in re.findall(r"\]\(((?:developers/)?[\w.-]+\.md)\)", index)}
    chapters = {f.resolve() for f in files if f.name != "README.md"}
    check(f"the guide: {len(chapters)} chapters ({sum(1 for f in chapters if 'developers' in str(f))} for "
          f"developers), each in the index and each the index lists present",
          len(chapters) >= 17 and listed == chapters)

    broken = []
    n_links = 0
    for f in files:
        text = re.sub(r"```.*?```", "", f.read_text(), flags=re.S)
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            if re.match(r"^[a-z]+://", target):
                continue
            n_links += 1
            path, _, anchor = target.partition("#")
            dest = (f.parent / path).resolve() if path else f.resolve()
            if not dest.exists():
                broken.append(f"{f.relative_to(GUIDE)} → {target}")
            elif anchor and dest.suffix == ".md" and anchor not in anchors(dest):
                broken.append(f"{f.relative_to(GUIDE)} → {target} (no such heading)")
    check(f"every relative link resolves, anchors included ({n_links} links)"
          + (f"; broken: {broken[:5]}" if broken else ""), not broken and n_links > 60)

    pages = pages_and_sections()
    text = " ".join(re.sub(r"\s+", " ", f.read_text()) for f in files)
    missing = [p for p in pages if p not in text]
    missing_s = [f"{p} › {s}" for p, ss in pages.items() for s in ss if s not in text]
    check(f"every workbench page ({len(pages)}) and every section ({sum(len(s) for s in pages.values())}) is named "
          f"in the guide" + (f"; missing {missing + missing_s}" if missing or missing_s else ""),
          len(pages) >= 20 and not missing and not missing_s)

    wrong = []
    n_refs = 0
    for f in files:
        for m in re.finditer(r"\*\*([^*\n›]+?) › ([^*\n›]+?)(?: › [^*\n]+)?\*\*", f.read_text()):
            page, sec = m.group(1).strip(), m.group(2).strip()
            if page in ("File", "Help", "Page"):          # the menus, and the convention's own example
                continue
            n_refs += 1
            if page not in pages or sec not in pages[page]:
                wrong.append(f"{f.name}: {page} › {sec}")
    check(f"each **Page › Section** named ({n_refs}) is a real page and one of its sections"
          + (f"; wrong: {wrong}" if wrong else ""), not wrong and n_refs > 40)

    from mufasa.cli.commands import COMMANDS, LIST_SUBJECTS, _tools
    from mufasa.cli.run import STEPS
    cli = (GUIDE / "13-command-line.md").read_text()
    tools = [t["command"] for t in _tools()]
    miss = ([c for c in COMMANDS if f"mufasa {c}" not in cli] + [s for s in LIST_SUBJECTS if s not in cli]
            + [t for t in tools if f"`{t}`" not in cli and t != "mufasa"])
    steps_ok = all(s in cli or s in (GUIDE / "05-3d-pose.md").read_text() + (GUIDE / "06-body-model-and-dynamics.md").read_text()
                   for s in STEPS)
    check(f"the command-line chapter: every command ({len(COMMANDS)}), list subject ({len(LIST_SUBJECTS)}) and tool "
          f"({len(tools)}); every `mufasa run` step in chapters 5, 6 or 13"
          + (f"; missing {miss}" if miss else ""), not miss and steps_ok)

    dev = " ".join((GUIDE / "developers" / n).read_text() for n in ("architecture.md", "extending.md",
                                                                   "testing-and-patches.md"))
    import mufasa.project_layout as PL
    import mufasa.section_provenance as SP
    from mufasa.cli import run as R
    from mufasa.feature_extractors import feature_subset_orchestration as FO
    from mufasa.feature_extractors import feature_subsets as FS
    named = {"project_paths_from_config": hasattr(PL, "project_paths_from_config"),
             "project_metadata_from_config": hasattr(PL, "project_metadata_from_config"),
             "find_project_config": hasattr(PL, "find_project_config"),
             "generate_run_id": hasattr(PL, "generate_run_id"), "is_run_id": hasattr(PL, "is_run_id"),
             "publish_to_stage": hasattr(PL, "publish_to_stage"),
             "global_model_cache_dir": hasattr(PL, "global_model_cache_dir"),
             "stage_run_dir": hasattr(PL.ProjectPaths, "stage_run_dir"),
             "latest_run": hasattr(PL.ProjectPaths, "latest_run"),
             "record_run": hasattr(SP, "record_run"), "SECTIONS": hasattr(SP, "SECTIONS"),
             "FEATURE_FAMILIES": hasattr(FS, "FEATURE_FAMILIES"), "process_one_video": hasattr(FO, "process_one_video"),
             "SIGNED_FAMILIES": hasattr(FO, "SIGNED_FAMILIES"), "STEPS": hasattr(R, "STEPS"),
             "CHAIN": hasattr(R, "CHAIN"), "LIST_SUBJECTS": True, "COMMANDS": True}
    src = (REPO / "mufasa/ui_qt/workbench.py").read_text()
    for meth in ("build", "collect_args", "target", "add_page", "add_section", "add_section_widget",
                 "add_tools_action"):
        named[meth] = f"def {meth}(" in src
    absent = [k for k, ok in named.items() if not ok or k not in dev]
    check(f"the developer chapters name {len(named)} APIs, each of which exists"
          + (f"; not found: {absent}" if absent else ""), not absent)
    pl_src = Path(PL.__file__).read_text()
    keys = ("project_root", "video_dir", "input_pose_dir", "models_dir", "derived_features_dir", "logs_dir")
    check("the architecture chapter's project_paths_from_config keys are the real ones (" + ", ".join(keys) + ")",
          all(k in dev and f'"{k}":' in pl_src for k in keys))

    docs_index = (REPO / "docs" / "README.md").read_text()
    readme = (REPO / "README.md").read_text()
    check("the docs index and the README point to the guide",
          "guide/README.md" in docs_index and "docs/guide/README.md" in readme)
    return report()


def report() -> int:
    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122ly_user_guide: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
