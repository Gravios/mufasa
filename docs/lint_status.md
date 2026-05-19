# Lint status — codebase audit (patch 122dg)

**Audience:** maintainers planning future lint / typing sweeps.

**Scope:** snapshot of `ruff check` findings across the whole codebase after the 122dg targeted sweep, with disposition for each category.

---

## Tool + config

* **Tool:** `ruff` 0.15.13 (current as of 122dg).
* **Config:** `[tool.ruff]` section of `pyproject.toml`.
  * Target: Python 3.11.
  * Line length: 100.
  * Excludes: `build/`, `dist/`, `.venv/` (the previous `mufasa/ui` entry was removed in 122dg — the directory itself was deleted in 122de).
  * Rules selected: `E F W I UP B SIM` (pycodestyle + pyflakes + isort + pyupgrade + bugbear + simplify).
  * Rules ignored: `E501` (line length is handled by formatter, not lint).

---

## Total findings (post-122dg snapshot)

```
ruff check mufasa/   → 9765 errors (down from 9846 pre-122dg)
```

The 81-error drop is the F401/W292/W293 sweep on `mufasa/ui_qt/`. The remaining 9765 are concentrated in legacy SimBA backend code and largely consist of modernization warnings (UP-prefix rules) that haven't been swept yet.

### By directory

| Directory | Errors | Disposition |
|---|---:|---|
| `mufasa/ui_qt/` | ~650 | **Active code; needs lint work.** 122dg fixed the safe subset (F401/W292/W293 = 81 errors). The remaining ~650 are mostly pyupgrade (UP045 modernization to `X \| None`) and would be worth a follow-up sweep. |
| `mufasa/mixins/` | 1658 | Legacy SimBA backend. Mix of pyupgrade modernization and real issues. |
| `mufasa/utils/` | 1036 | Mix of utility code (active) and SimBA legacy. Per-file triage needed. |
| `mufasa/video_processors/` | ~1200 | Legacy SimBA backend with heavy old-style typing. Largely modernization. |
| `mufasa/feature_extractors/` | ~800 | Legacy SimBA backend. |
| `mufasa/data_processors/` | ~600 | Legacy SimBA backend (much of it unreachable from Qt — see `hardwired_paths_audit.md`). |
| `mufasa/project_layout.py` | 63 | Active code; small enough to sweep manually. |
| `mufasa/cli/` | 28 | Small surface; safe to sweep. |
| `mufasa/legacy_layout.py` | 16 | By-design legacy; very small surface. |
| Other | ~3400 | Pose importers, plotting, model code, etc. Mostly legacy SimBA. |

### By rule (top 20)

| Rule | Count | Description | Auto-fixable? | Disposition |
|---|---:|---|---|---|
| UP045 | 2081 | Use `X \| None` instead of `Optional[X]` | Yes | Modernization. Safe to auto-fix per-directory. |
| UP006 | 1695 | Use `list` instead of `List` (PEP 585) | Yes | Modernization. Safe. |
| UP007 | 1275 | Use `X \| Y` instead of `Union[X, Y]` | Yes | Modernization. Safe. |
| E701 | 716 | Multiple statements on one line (colon) | Some | Style; existing codebase pattern. |
| UP035 | 469 | Import from typing is deprecated | Yes | Modernization. Safe. |
| UP032 | 458 | Use f-string instead of `.format()` | Yes | Modernization. Mostly safe (some `.format()` calls have dynamic kwargs that don't translate). |
| F401 | 334 | Unused imports (down from 415) | Yes | Always safe in non-`__init__.py`. 122dg cleared these in `ui_qt/`. |
| I001 | 377 | Unsorted imports | Yes | Style; large noisy diff if applied broadly. |
| B007 | 365 | Loop variable not used | No | Often legitimate (using index but not value). Per-site triage. |
| E402 | 224 | Module-level imports not at top | No | Often intentional (lazy imports after path setup). |
| SIM118 | 167 | Use `key in dict` instead of `key in dict.keys()` | Yes | Style. Safe. |
| E702 | 149 | Multiple statements on one line (semicolon) | No | Style; existing pattern. |
| B905 | 142 | `zip()` without explicit strict= | No | Per-site triage (could mask bugs). |
| B904 | 113 | Use `raise from` in except | No | Per-site triage. |
| F541 | 109 | f-string without placeholders | Yes | Safe. |
| F841 | 108 | Unused local variable | No | Sometimes intentional (held for stack-trace context). |
| E722 | 103 | Bare except | No | Often legitimate fallback handling. Per-site triage. |
| W292 | 0 (down from 75) | No newline at end of file | Yes | 122dg cleared in `ui_qt/`. |
| F405 | 67 | Possibly undefined name from wildcard | No | From `from X import *`; harder to fix. |
| W293 | 6 (down from 66) | Trailing whitespace | Yes | 122dg cleared in `ui_qt/`; 6 remaining in legacy. |

---

## 122dg sweep — what landed

**Targeted scope:** `mufasa/ui_qt/` only.

**Rules applied:** `F401 W292 W293` (unused imports + final newline + trailing whitespace). The narrowest possible safe set.

**Files touched:** 39.

**Errors eliminated:** 81 (all 81 reported by `ruff check mufasa/ui_qt --select F401,W292,W293`).

**Why this narrow scope:** the pyupgrade rules (UP045/UP006/UP007) are mechanically safe but produce noisy diffs (~270 issues in `ui_qt/` alone, 5051 across the codebase). Isort (I001) reformats every import block in every touched file. Both are fine ideas but warrant their own focused patches with maintainer review. 122dg restricts itself to "delete things that aren't used" — narrowly defensible per-file.

**Manual fix in this patch:** `mufasa/ui_qt/input_source_picker.py` had a `from PySide6.QtCore import Qt, Signal` inside a try/except headless guard. `Qt` was actually unused (verified by grep); ruff didn't auto-fix because the import was inside try/except. Removed manually.

**Config tweak:** removed the `mufasa/ui` entry from `[tool.ruff].extend-exclude` since that directory was deleted in 122de.

---

## Recommended follow-up sweeps

Each is a candidate for its own focused patch. Order is approximate; pick what matters.

### Tier 1 — Safe + valuable

* **`ui_qt/` pyupgrade sweep** (~270 errors). UP045 / UP006 / UP007 / UP035 / UP032. Modernizes Qt code to PEP 604 / 585 syntax (`X | None`, `list[T]`, etc.). Diff is mechanical but touches most files. Worth a single patch with maintainer review on the diff style.
* **`ui_qt/` isort sweep** (~114 errors). I001. Standardizes import grouping. Cosmetic but unifies style across the directory. Bundle with the pyupgrade sweep or do separately depending on diff-review preference.
* **`project_layout.py` + `cli/` full sweep** (~90 errors total). Small surface; can fix everything in one patch.

### Tier 2 — Per-file triage required

* **`mixins/` and `utils/` selective sweep.** 1658 + 1036 errors. Most are pyupgrade-modernization candidates but the directories include some classifier-training and feature-extraction code that's behavior-sensitive — sweep file-by-file rather than directory-wide.

### Tier 3 — Legacy code with low ROI

* **`video_processors/`, `feature_extractors/`, `data_processors/`, pose importers, plotting modules.** Combined ~6000 errors. These are mostly inherited SimBA backend with old-style typing. Unless a workflow modernization is planned, sweeping these is large diff cost for marginal benefit. Per-file modernization can happen incrementally as files are touched for other reasons.

### Tier 4 — Not recommended

* **`legacy_layout.py`** (16 errors). The file's purpose IS to represent the legacy layout; modernizing it has no value. Could `# noqa` the file or add a `per-file-ignores` entry.

---

## What this audit does NOT cover

* **Type checking.** Ruff's lint covers syntax / unused / style; it doesn't run `mypy` or `pyright`. A real typing audit would need a separate tool and config (no `[tool.mypy]` section in `pyproject.toml` currently). Quick estimate: thousands of typing errors expected on first run, mostly from third-party-library stubs (numpy, pandas, cv2, networkx) needing `Any` annotations.
* **Dead code.** Tools like `vulture` can find unused functions/classes. Not run here; would be a separate audit.
* **Security lints.** `bandit` covers things like hardcoded secrets, insecure subprocess use. Not run; if relevant, a separate sweep.
* **Format.** `ruff format` (or `black`) would normalize whitespace / quote style / line breaks. Not run in 122dg; a format sweep would touch nearly every file.

---

## How to reproduce

Install ruff via pip (already a dev tool in many setups; no entry in `pyproject.toml`'s deps):

```bash
pip install --break-system-packages ruff
ruff check mufasa/                          # everything
ruff check mufasa/ui_qt/                    # Qt code only
ruff check mufasa/ui_qt/ --select F401      # specific rule
ruff check mufasa/ --fix --diff             # preview broader fix
ruff check mufasa/ui_qt/ --statistics       # rule counts only
```

For typing, the simplest start:

```bash
pip install --break-system-packages mypy
mypy mufasa/project_layout.py mufasa/ui_qt/qt_confirm.py   # narrow start
```

(Expect failures from missing stubs; add `[tool.mypy]` config to `pyproject.toml` with `ignore_missing_imports = true` to start.)
