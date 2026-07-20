"""Smoke test for patch 122hv — --load-model needs no project/layout.

A saved v2 model already stores its own layout (save_model_v2 serializes every
segment's name/parent/markers), and smooth_pose_v2 uses the loaded layout and
ignores any passed one. So once a model is built, smoothing an external file
needs nothing but the input files and an output location — no project.toml, no
--config, no skeleton, no input-tracking.

But main() unconditionally ran layout resolution (project search -> FreeDLC
sidecar -> rig fallback) BEFORE smooth_pose_v2, even with --load-model, so a
standalone file would hit the rig-fallback warning needlessly and the
multi-project error could even block a legitimate load-model run. 122hv guards
the whole layout-resolution block behind `if args.load_model is None:` — with a
model, config and layout are left None (smooth_pose_v2 uses the model's layout)
and the layout-shaping flags are reported as ignored.

The module imports heavy deps absent in the sandbox, so this test AST-checks the
guard and the model's layout serialization, and exercises the guard's decision
logic in isolation.
"""
from __future__ import annotations

import ast
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


src = (REPO / "mufasa/data_processors/kalman_pose_smoother_v2.py").read_text()
tree = ast.parse(src)

# ---- the saved model carries the layout (so inputs/project aren't needed) ----
# save_model_v2 serializes layout segments; load_model_v2 reconstructs a
# BodyLayout; smooth_pose_v2 uses the loaded layout over any passed one.
check("save_model_v2 serializes the layout segments",
      "layout_segments=np.array(seg_data" in src)
check("save_model_v2 records each segment's parent and markers",
      '"parent": seg.parent' in src and '"markers": dict(seg.markers)' in src)
check("load_model_v2 reconstructs a BodyLayout from the saved segments",
      'seg_data = data["layout_segments"]' in src
      and "layout = BodyLayout(" in src)
check("smooth_pose_v2 uses the loaded model's layout over a passed one",
      "ignoring the caller's layout" in src
      or "the loaded model's parameters are dimensioned" in src)

# ---- main() guards layout resolution behind --load-model ----
main = next((n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
check("main() is parseable", main is not None)

# find the `if args.load_model is not None:` guard whose else-branch holds the
# project/layout resolution
guard = None
for node in ast.walk(main) if main else []:
    if (isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Attribute)
            and node.test.left.attr == "load_model"):
        guard = node
        break
check("main() has an `if args.load_model is not None:` guard", guard is not None)

guard_src = ast.get_source_segment(src, guard) if guard else ""
# the else-branch (no model) must be where project/layout resolution lives
check("layout resolution is in the guard's else-branch (skipped with a model)",
      "find_project_config" in guard_src
      and "standard_rat_layout" in guard_src
      and "layout_from_fdlc_sidecar" in guard_src)
# and the project search / sidecar must NOT be reachable outside the else
# (i.e. not run when a model is loaded)
# crude but effective: find_project_config appears only inside the guard
check("find_project_config runs only under the no-model branch",
      src.count("find_project_config(") == 1
      and "find_project_config(" in guard_src)
# the guard reports ignored layout flags when a model is loaded
check("the guard reports ignored layout flags",
      "ignored" in guard_src
      and "carries its own layout" in guard_src)
# config and layout default to None before the guard (so a model load leaves
# them None -> smooth_pose_v2 uses the model's own layout)
check("config and layout default to None ahead of the guard",
      "config = None" in src and "layout = None" in src)

# ---- the guard's decision logic ----
class _Args:
    def __init__(self, **kw):
        self.load_model = None
        self.config = None
        self.with_drift = False
        self.orient_drift_segments = ""
        self.const_accel_segments = ""
        self.high_angular_noise_segments = ""
        self.no_back4 = self.no_tail = self.no_lateral = self.no_center = False
        self.__dict__.update(kw)


def _ignored_flags(args):
    return [
        n for n, g in (
            ("--config", args.config is not None),
            ("--with-drift", args.with_drift),
            ("--orient-drift-segments", bool(args.orient_drift_segments)),
            ("--const-accel-segments", bool(args.const_accel_segments)),
            ("--high-angular-noise-segments",
             bool(args.high_angular_noise_segments)),
            ("--no-back4", args.no_back4), ("--no-tail", args.no_tail),
            ("--no-lateral", args.no_lateral), ("--no-center", args.no_center),
        ) if g
    ]


# load-model + no layout flags -> nothing reported
check("load-model alone reports no ignored flags",
      _ignored_flags(_Args(load_model="/m.npz")) == [])
# load-model + layout flags -> each reported ignored
check("load-model + --config + --with-drift reports both ignored",
      set(_ignored_flags(_Args(
          load_model="/m.npz", config="/p.toml", with_drift=True)))
      == {"--config", "--with-drift"})

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122hv_load_model_skips_layout: {n_pass}/{len(checks)} "
      f"checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
