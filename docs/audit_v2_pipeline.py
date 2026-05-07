"""AST audit of mufasa.data_processors.kalman_pose_smoother_v2.

Output: a structured map of the v2 smoother pipeline, focused on
identifying every site where introducing local pair distance
constraints would require code change.

Sections produced:
  1. Module-level structure: top-level functions and classes
  2. Class structure: methods per class
  3. Call graph: who calls whom (intra-module only)
  4. Pipeline stages: functions reachable from smooth_pose_v2,
     grouped by role (layout / kinematics / dynamics / filter /
     smoother / m-step / observation / io)
  5. Pair-distance touch points: the specific functions that would
     need modification for each candidate design
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

SRC = Path(
    "/home/claude/work_119a/mufasa/data_processors/"
    "kalman_pose_smoother_v2.py"
)


# Manual role tagging by name pattern. Audited once vs. the file's
# actual content; refined where automatic guess was wrong.
ROLE_PATTERNS = [
    # role,                           name substring(s)
    ("layout",         ["BodyLayout", "BodySegment", "_compute_topo",
                        "marker_attachment", "slice_root", "slice_segment",
                        "slice_marker_drift", "drift_block_slice",
                        "standard_rat_layout", "_segment_distal_marker",
                        "_pack_state_layout_indices"]),
    ("kinematics",     ["forward_kinematics", "state_to_marker_positions",
                        "state_to_marker_jacobian", "ForwardKinematicsResult",
                        "_R", "marker_positions_batch",
                        "marker_jacobian_batch"]),
    ("dynamics",       ["build_F_v2", "build_Q_v2"]),
    ("noise_params",   ["NoiseParamsV2", "FittedLengths",
                        "fit_body_lengths", "PerspectiveModelV2",
                        "apply_fitted_offsets_to_layout"]),
    ("filter",         ["forward_filter", "_predict", "_update",
                        "FilterResultV2", "_h_obs", "_obs_residual"]),
    ("smoother",       ["rts_smoother", "RTSResultV2",
                        "_lag_one_cross"]),
    ("m_step",         ["_MStepStatsV2", "_PerSessionFitV2",
                        "_accum_session_stats", "finalize_m_step",
                        "_m_step", "_session_em_iteration"]),
    ("observation",    ["_build_observation", "_observation_vector",
                        "_observation_likelihood"]),
    ("io",             ["save_model", "load_model", "_arrays_to_df",
                        "_build_and_write", "_load_csv",
                        "_import_io_helpers"]),
    ("worker_pool",    ["_pool_init", "_POOL_WORKER_STATE",
                        "_pool_worker_"]),
    ("orchestrator",   ["smooth_pose_v2"]),
    ("perspective",    ["fit_perspective", "_persp", "perspective_"]),
]


def classify(name: str) -> str:
    for role, patterns in ROLE_PATTERNS:
        for p in patterns:
            if p in name:
                return role
    return "uncategorized"


def collect(tree: ast.Module) -> Dict[str, dict]:
    """Walk top-level + class-level definitions. For each, record:
        - name
        - line range
        - kind (function / class / classmethod)
        - parent class (if any)
        - parameters
        - direct callees (Name and Attribute calls)
        - direct attribute reads on `self` (helps spot state access)
    """
    items: Dict[str, dict] = {}

    def record_callees(node: ast.AST) -> Set[str]:
        callees: Set[str] = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                f = sub.func
                if isinstance(f, ast.Name):
                    callees.add(f.id)
                elif isinstance(f, ast.Attribute):
                    callees.add(f.attr)
        return callees

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            items[node.name] = dict(
                name=node.name,
                kind="function",
                parent=None,
                lineno=node.lineno,
                end_lineno=node.end_lineno,
                params=[a.arg for a in node.args.args],
                callees=record_callees(node),
                role=classify(node.name),
            )
        elif isinstance(node, ast.ClassDef):
            items[node.name] = dict(
                name=node.name,
                kind="class",
                parent=None,
                lineno=node.lineno,
                end_lineno=node.end_lineno,
                params=[],
                callees=set(),
                role=classify(node.name),
            )
            for cn in node.body:
                if isinstance(cn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qual = f"{node.name}.{cn.name}"
                    items[qual] = dict(
                        name=qual,
                        kind="method",
                        parent=node.name,
                        lineno=cn.lineno,
                        end_lineno=cn.end_lineno,
                        params=[a.arg for a in cn.args.args],
                        callees=record_callees(cn),
                        role=classify(qual),
                    )
    return items


def transitive_reach(
    items: Dict[str, dict],
    start: str,
    name_lookup: Dict[str, str],
) -> Set[str]:
    """Set of qualified names reachable from `start` via the
    intra-module call graph. name_lookup maps a callee name (Name
    or Attribute) to a qualified definition name when there's a
    unique match; otherwise the name is dropped (call resolves
    out-of-module).
    """
    reach: Set[str] = set()
    work = [start]
    while work:
        cur = work.pop()
        if cur in reach or cur not in items:
            continue
        reach.add(cur)
        for c in items[cur].get("callees", set()):
            if c in name_lookup:
                target = name_lookup[c]
                if target not in reach:
                    work.append(target)
    return reach


def main() -> int:
    src = SRC.read_text()
    tree = ast.parse(src)
    items = collect(tree)

    # Build name → qualified definition lookup. Pick top-level
    # definitions first; methods only resolve when there's a unique
    # method by that name.
    name_lookup: Dict[str, str] = {}
    method_by_short: Dict[str, List[str]] = defaultdict(list)
    for qual, info in items.items():
        short = info["name"].split(".")[-1]
        if info["kind"] in ("function", "class"):
            name_lookup[short] = qual
        else:
            method_by_short[short].append(qual)
    for short, quals in method_by_short.items():
        if short in name_lookup:
            continue
        if len(quals) == 1:
            name_lookup[short] = quals[0]

    # ----- 1. Structure summary -----
    print("=" * 70)
    print("1. MODULE STRUCTURE")
    print("=" * 70)
    by_role: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for qual, info in items.items():
        if info["kind"] != "method":
            by_role[info["role"]].append((qual, info["lineno"]))
    for role in [
        "orchestrator", "io", "layout", "noise_params",
        "kinematics", "dynamics", "filter", "smoother",
        "m_step", "observation", "perspective",
        "worker_pool", "uncategorized",
    ]:
        if role in by_role and by_role[role]:
            print(f"\n[{role}]")
            for name, ln in sorted(by_role[role], key=lambda x: x[1]):
                kind = items[name]["kind"]
                print(f"  L{ln:4d}  {kind:<8s}  {name}")

    # ----- 2. Class structure -----
    print()
    print("=" * 70)
    print("2. CLASS METHODS")
    print("=" * 70)
    for qual, info in items.items():
        if info["kind"] == "class":
            methods = [
                (q, items[q]["lineno"])
                for q in items
                if items[q].get("parent") == qual
            ]
            if methods:
                print(f"\n[{qual}]  L{info['lineno']}")
                for m, ml in sorted(methods, key=lambda x: x[1]):
                    print(f"    L{ml:4d}  {m}")

    # ----- 3. Reachability from orchestrator -----
    print()
    print("=" * 70)
    print("3. PIPELINE REACHABILITY FROM smooth_pose_v2")
    print("=" * 70)
    if "smooth_pose_v2" not in items:
        print("  smooth_pose_v2 not found")
    else:
        reach = transitive_reach(items, "smooth_pose_v2", name_lookup)
        # Group by role
        rg: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        for q in reach:
            rg[items[q]["role"]].append((q, items[q]["lineno"]))
        n_total = len(reach)
        print(f"  {n_total} reachable definitions, by role:")
        for role in [
            "orchestrator", "io", "layout", "noise_params",
            "kinematics", "dynamics", "filter", "smoother",
            "m_step", "observation", "perspective",
            "worker_pool", "uncategorized",
        ]:
            if role in rg:
                print(f"\n  [{role}]")
                for n, ln in sorted(rg[role], key=lambda x: x[1]):
                    print(f"    L{ln:4d}  {n}")

    # ----- 4. Pair-distance touch points -----
    print()
    print("=" * 70)
    print("4. PAIR-DISTANCE TOUCH POINTS")
    print("=" * 70)
    # The functions that read marker offsets via marker_attachment
    # are where rigid-body marker positions are computed; pair
    # distances would also need to be computed at these sites.
    print("\n[A] Readers of layout.marker_attachment "
          "(FK / observation / Jacobian):")
    for qual, info in items.items():
        if info["kind"] == "method":
            continue
        if "marker_attachment" in info["callees"]:
            print(f"    L{info['lineno']:4d}  {qual}")

    print("\n[B] Functions that build the observation vector / "
          "Jacobian H:")
    obs_signals = {
        "state_to_marker_positions",
        "state_to_marker_jacobian",
    }
    for qual, info in items.items():
        if info["kind"] == "method":
            continue
        if any(s in info["callees"] for s in obs_signals):
            print(f"    L{info['lineno']:4d}  {qual}")

    print("\n[C] M-step sufficient-stats accumulator (would gain "
          "pair-distance residuals if pseudo-obs added):")
    for qual, info in items.items():
        if "MStepStatsV2" in qual:
            print(f"    L{info['lineno']:4d}  {qual}")

    print("\n[D] save / load (would round-trip new pair-distance "
          "params):")
    for qual, info in items.items():
        if items[qual]["role"] == "io" and "model" in qual.lower():
            print(f"    L{info['lineno']:4d}  {qual}")

    print("\n[E] Layout slice helpers (would gain a new helper "
          "for pair distance pseudo-observation indexing if state "
          "augmentation is chosen):")
    for qual, info in items.items():
        if "slice_" in qual and info["kind"] == "method":
            print(f"    L{info['lineno']:4d}  {qual}")

    # ----- 5. State-vector construction sites -----
    print()
    print("=" * 70)
    print("5. STATE-VECTOR CONSTRUCTION / INITIAL STATE SITES")
    print("=" * 70)
    # Look for functions that allocate np.zeros(state_dim)-shaped
    # arrays or call _pack_state_layout_indices
    print("\nFunctions that touch state-dim-sized arrays "
          "(state init / shape checks):")
    for qual, info in items.items():
        if info["kind"] == "method":
            continue
        if "state_dim" in info["callees"] or "initial_state" in qual:
            print(f"    L{info['lineno']:4d}  {qual}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
