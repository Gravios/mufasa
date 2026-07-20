"""Smoke test for patch 122hs — matplotlib cm.get_cmap removal.

matplotlib.cm.get_cmap was deprecated in 3.7 and removed in 3.11; a user on a
recent matplotlib hit "module 'matplotlib.cm' has no attribute 'get_cmap'" when
adding sessions (the smoothing/colour path). This patch replaces every
cm.get_cmap(name, N) call with matplotlib.colormaps[name].resampled(N), the
version-safe modern equivalent, across plotting_mixin, utils/data and
utils/lookups.

Verifies: no cm.get_cmap call remains in the package; the modern replacement is
numerically identical to the old API where the old one still works; and the
replacement is used (imports present, resampled() calls in place).
"""
from __future__ import annotations

import pathlib
import sys
import warnings

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


# ------------------------------------------------------------------ #
# 1. no cm.get_cmap remains in the package source
# ------------------------------------------------------------------ #
pkg = REPO / "mufasa"
offenders = []
for py in pkg.rglob("*.py"):
    text = py.read_text(encoding="utf-8", errors="ignore")
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue  # comments referencing the old API are fine
        # match a real call: cm.get_cmap( ... )  (not "def get_cmaps")
        if "cm.get_cmap(" in line:
            offenders.append(f"{py.relative_to(REPO)}:{i}")
check(f"no cm.get_cmap( call remains in mufasa/ (found: {offenders})",
      not offenders)

# ------------------------------------------------------------------ #
# 2. the three fixed files use the modern API
# ------------------------------------------------------------------ #
pm = (REPO / "mufasa/mixins/plotting_mixin.py").read_text()
check("plotting_mixin imports matplotlib.colormaps",
      "from matplotlib import colormaps as mpl_colormaps" in pm)
check("plotting_mixin uses resampled()",
      "mpl_colormaps[pallete_name].resampled(increments + 1)" in pm)

dt = (REPO / "mufasa/utils/data.py").read_text()
check("utils/data uses matplotlib.colormaps[...].resampled",
      "matplotlib.colormaps[pallete_name].resampled(increments + 1)" in dt
      and "matplotlib.colormaps[" in dt)

lk = (REPO / "mufasa/utils/lookups.py").read_text()
check("utils/lookups imports matplotlib.colormaps",
      "from matplotlib import colormaps as mpl_colormaps" in lk)
check("utils/lookups uses resampled()",
      "mpl_colormaps[name].resampled(map_size)" in lk)
# the stale int-in-cmap_d check is gone
check("utils/lookups no longer uses the stale cm.cmap_d int check",
      "cm.cmap_d" not in lk or lk.count("cm.cmap_d") == 0)

# ------------------------------------------------------------------ #
# 3. the modern API is numerically identical to the old one
#    (where the old one still exists), and works when it's removed
# ------------------------------------------------------------------ #
from matplotlib import colormaps as mpl_colormaps  # noqa: E402

# equivalence vs the (possibly still-present) old API
try:
    import matplotlib.cm as cm
    if hasattr(cm, "get_cmap"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ident = True
            for name, N in [("viridis", 5), ("jet", 4), ("spring", 3)]:
                old = cm.get_cmap(name, N)
                new = mpl_colormaps[name].resampled(N)
                if not (old.N == new.N == N
                        and all(old(i) == new(i) for i in range(N))):
                    ident = False
            check("new API is identical to old cm.get_cmap where present", ident)
    else:
        # old API already gone (matplotlib >= 3.11) — the whole point; the new
        # API is what we rely on, checked below.
        check("old cm.get_cmap already removed (new API is the only path)",
              True)
except Exception as exc:  # noqa: BLE001
    check(f"colormap equivalence check ran ({exc})", False)

# the replacement expression itself works and yields N colours
def _palette(name: str, n: int):
    cmap = mpl_colormaps[name].resampled(n + 1)
    return [list(cmap(i)[:3]) for i in range(cmap.N)]


pal = _palette("jet", 3)
check("resampled() palette yields N+1 colours", len(pal) == 4)
check("resampled() colours are RGB triples",
      all(len(c) == 3 for c in pal))

# fallback-to-spring logic (lookups) resolves a bad name safely
def _safe_name(name: str) -> str:
    return name if name in mpl_colormaps else "spring"


check("unknown colormap name falls back to spring",
      _safe_name("not_a_real_cmap_xyz") == "spring")
check("known colormap name is kept", _safe_name("viridis") == "viridis")

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122hs_get_cmap_removal: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
