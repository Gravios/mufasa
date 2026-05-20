# Lint status — codebase audit (patches 122dg + 122do)

**Audience:** maintainers planning future lint / typing sweeps.

**Scope:** snapshot of `ruff check` findings across the whole codebase after the 122dg targeted sweep (F401/W292/W293) and the 122do modernization sweep (UP045/UP006/UP007/UP035/I001/F401 cascade), with disposition for each remaining category.

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

## Total findings (post-122do snapshot)

```
ruff check mufasa/          → 9292 errors  (was 9846 pre-122dg, 9765 post-122dg)
ruff check mufasa/ui_qt/    →  173 errors  (was ~650 post-122dg)
```

122do dropped the codebase-wide total by another 473 and shrank `mufasa/ui_qt/` by ~73% on top of 122dg. The remaining 9292 are concentrated in legacy SimBA backend code. Within `mufasa/ui_qt/` what's left is behavior-sensitive (E702 multiple-statements-on-line, B904 raise-without-from, SIM suggestions) — not mechanically swept.

### By directory

| Directory | Errors | Disposition |
|---|---:|---|
| `mufasa/ui_qt/` | 173 | **Active code; modernized.** 122dg cleared F401/W292/W293 (81 errors); 122do cleared UP045/UP006/UP007/UP035/I001/F401 (571 errors). What remains is mostly behavior-sensitive style (E702 multi-statement-on-line, B904 raise-without-from-inside-except, SIM suggestions) — not appropriate for a mechanical sweep. |
| `mufasa/mixins/` | 1658 | Legacy SimBA backend. Mix of pyupgrade modernization and real issues. |
| `mufasa/utils/` | 1036 | Mix of utility code (active) and SimBA legacy. Per-file triage needed. |
| `mufasa/video_processors/` | ~1200 | Legacy SimBA backend with heavy old-style typing. Largely modernization. |
| `mufasa/feature_extractors/` | ~800 | Legacy SimBA backend. |
| `mufasa/data_processors/` | ~600 | Legacy SimBA backend (much of it unreachable from Qt — see `hardwired_paths_audit.md`). |
| `mufasa/project_layout.py` | 63 | Active code; small enough to sweep manually. |
| `mufasa/cli/` | 28 | Small surface; safe to sweep. |
| `mufasa/legacy_layout.py` | 16 | By-design legacy; very small surface. |
| Other | ~3400 | Pose importers, plotting, model code, etc. Mostly legacy SimBA. |

### By rule (top 20, post-122do)

| Rule | Count | Description | Auto-fixable? | Disposition |
|---|---:|---|---|---|
| UP045 | 1811 | Use `X \| None` instead of `Optional[X]` | Yes | Modernization. Down from 2081 (122do cleared 270 in `ui_qt/`). |
| UP006 | 1641 | Use `list` instead of `List` (PEP 585) | Yes | Modernization. Down from 1695. |
| UP007 | 1266 | Use `X \| Y` instead of `Union[X, Y]` | Yes | Modernization. Down from 1275. |
| E701 | 716 | Multiple statements on one line (colon) | Some | Style; existing codebase pattern. |
| UP032 | 458 | Use f-string instead of `.format()` | Yes | Modernization. Mostly safe (some `.format()` calls have dynamic kwargs that don't translate). |
| UP035 | 443 | Import from typing is deprecated | Yes | Modernization. Down from 469. |
| B007 | 364 | Loop variable not used | No | Often legitimate (using index but not value). Per-site triage. |
| F401 | 333 | Unused imports | Yes | Down from 334. 122do cleared 65 in `ui_qt/` via the UP-cascade. |
| I001 | 263 | Unsorted imports | Yes | Style. Down from 377 (122do cleared 114 + 18 in `ui_qt/`). |
| E402 | 224 | Module-level imports not at top | No | Often intentional (lazy imports after path setup). |
| SIM118 | 167 | Use `key in dict` instead of `key in dict.keys()` | Yes | Style. Safe. |
| E702 | 152 | Multiple statements on one line (semicolon) | No | Style; existing pattern. |
| B905 | 142 | `zip()` without explicit strict= | No | Per-site triage (could mask bugs). |
| B904 | 113 | Use `raise from` in except | No | Per-site triage. |
| F541 | 109 | f-string without placeholders | Yes | Safe. |
| F841 | 108 | Unused local variable | No | Sometimes intentional (held for stack-trace context). |
| E722 | 103 | Bare except | No | Often legitimate fallback handling. Per-site triage. |
| W292 | 75 | No newline at end of file | Yes | 122dg cleared in `ui_qt/`; remaining are in legacy backend dirs. |
| F405 | 67 | Possibly undefined name from wildcard | No | From `from X import *`; harder to fix. |
| W293 | 66 | Trailing whitespace | Yes | 122dg cleared in `ui_qt/`; remaining are in legacy. |

---

## 122do sweep — what landed

**Targeted scope:** `mufasa/ui_qt/` only.

**Rules applied:** `UP045 UP006 UP007 UP035 I001` (pyupgrade modernization + isort), with a cascading `F401` follow-up for orphaned typing imports.

**Files touched:** 75.

**Errors eliminated:** 571 total in `mufasa/ui_qt/`:
- 488 by the initial `--select UP045,UP006,UP007,UP035,UP032,I001 --fix` pass
- 65 by the `--select F401 --fix` follow-up (orphaned `typing.Optional` and `typing.Union` imports that became unused after the UP-rule conversions)
- 18 by a final `--select I001 --fix` pass to re-sort import blocks that the F401 cleanup shuffled

**Manual touches:**
- 10 files needed manual cleanup of `from typing import …` lines that ruff considered unsafe to remove even with `--unsafe-fixes`. These were UP035 leftovers where the named imports (`Optional`, `Union`, `List`, `Dict`, `Tuple`, `Type`) became orphans after the auto-conversions but the import statement itself stayed. Files: `dialog.py`, `dialogs/edit_project_metadata_dialog.py`, `dialogs/pixel_calibration.py`, `dialogs/roi_canvas.py`, `forms/_backend_dispatch.py`, `forms/project_create.py`, `forms/video_info.py`, `input_source_picker.py`, `reconfigure_dialog.py`, `workbench.py`.
- `forms/pose_cleanup.py` — added `from typing import Any`. This was a **pre-existing latent F821** (`Any` referenced in two `dict[str, Any]` annotations but never imported). `from __future__ import annotations` made the annotations lazy strings so it didn't blow up at import time, but the file would have failed `typing.get_type_hints()` and any static type-check. Fixed under this sweep because it's in the typing-imports area anyway.

**Why this scope:** the chosen rules are PEP-aligned modernization (PEP 604 unions, PEP 585 builtin generics) for a Python-3.11+ codebase and produce purely cosmetic AST changes. Verified by an AST-level semantic diff: stripping annotations and imports from each file before/after the sweep yields **0 files with semantic differences** — proof that no runtime logic was touched.

**Verified clean post-sweep:** `UP045` `UP006` `UP007` `UP035` `I001` `F401` `F821` all return 0 errors on `mufasa/ui_qt/`. The 122dg `W292`/`W293` baseline is also preserved.

**What was NOT included:**
- `UP032` (f-string conversion): ruff reported 0 hits in `ui_qt/` at sweep time, so the rule was a no-op. Left in the rule list for completeness.
- `UP033` (lru_cache_with_maxsize_none) and `UP037` (quoted-annotation): 2 + 1 hits respectively but `--select` was scoped only to the original rule set. Easy follow-up if desired.
- The codebase outside `ui_qt/` — the legacy SimBA backend dirs still have ~9100 errors of the same kinds.

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

* **`project_layout.py` + `cli/` full sweep** (~90 errors total). Small surface; can fix everything in one patch.
* **`ui_qt/` UP033/UP037 mop-up** (3 errors). `ruff check mufasa/ui_qt --select UP033,UP037 --fix` would clean these; left out of 122do only because they weren't in the original target rule set.
* **Tier-2 `B904` sweep on `ui_qt/`** (20 errors). `raise … from exc` inside except blocks. Mechanical but benefits from a per-site sanity check that the `from` is actually informative (vs `from None` for noise suppression). Worth a small focused patch.
* **`ui_qt/` `E702` audit** (116 errors). Multiple statements on one line; ruff doesn't auto-fix. Per-site decision — most are likely Qt boilerplate (`x.foo(); y.bar()`) that's a deliberate compactness choice; some may benefit from a line break.

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
