"""
tests/smoke_join_transition_cluster.py
======================================

Patch 122t: regression guard for the join/transition cluster
in ``mufasa/ui_qt/forms/video_join.py``.

Scope of this patch:

1. **CrossfadeVideosForm (NEW)** — strict 2-video crossfade
   transition. Replaces :class:`CrossfadeVideosPopUp`. Pulls the
   method list from ``mufasa.utils.lookups.get_ffmpeg_crossfade_methods``
   so the UI stays in sync with the canonical 18+ ffmpeg xfade
   modes. Calls
   ``mufasa.video_processors.video_processing.crossfade_two_videos``.

2. **JoinVideosForm — normalize stub removed** — the form
   previously collected a ``normalize`` flag from a checkbox
   but never passed it to any backend (vapourware). Dropped.

3. **CrossfadeVideosPopUp** excised from
   ``mufasa/ui/pop_ups/video_processing_pop_up.py`` with the
   standard deletion-record marker.

4. **SimBA.py wiring** scrubbed: import + 'Cross-fade videos'
   menu entry.

5. **Video Processing page** registers CrossfadeVideosForm
   alongside JoinVideosForm in the existing "Join & transition"
   section (multi-form section).

AST-only — the forms transitively import PySide6 / ffmpeg via
the backend module, neither available in the sandbox.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


CHECKS_PASSED = 0
CHECKS_TOTAL = 0


def check(label: str, cond: bool, *, detail: str = "") -> None:
    global CHECKS_PASSED, CHECKS_TOTAL
    CHECKS_TOTAL += 1
    if cond:
        CHECKS_PASSED += 1
    else:
        print(f"FAIL: {label}" + (f" — {detail}" if detail else ""))


def _find_class(tree: ast.Module, name: str):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _methods(cls: ast.ClassDef) -> dict[str, ast.FunctionDef]:
    return {
        n.name: n for n in cls.body
        if isinstance(n, ast.FunctionDef)
    }


def main() -> int:
    # ==================================================================
    # 1. CrossfadeVideosForm shape
    # ==================================================================
    vj_path = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
               / "video_join.py")
    vj_src = vj_path.read_text()
    vj_tree = ast.parse(vj_src)

    cls = _find_class(vj_tree, "CrossfadeVideosForm")
    check("CrossfadeVideosForm class defined", cls is not None)

    if cls is not None:
        class_src = ast.unparse(cls)
        bases = [
            (b.id if isinstance(b, ast.Name)
             else getattr(b, "attr", ""))
            for b in cls.bases
        ]
        check(
            "CrossfadeVideosForm extends OperationForm",
            "OperationForm" in bases,
        )

        for attr in (
            "self.video_1_edit", "self.video_2_edit",
            "self.method_cb", "self.duration",
            "self.offset_edit", "self.quality",
            "self.format_cb",
        ):
            check(
                f"CrossfadeVideosForm sets {attr}",
                attr in class_src,
            )

        # Method list sourced from the canonical lookup
        check(
            "CrossfadeVideosForm pulls methods from "
            "get_ffmpeg_crossfade_methods",
            "get_ffmpeg_crossfade_methods" in class_src,
        )

        # Output format options
        check(
            "CrossfadeVideosForm format combo includes mp4/avi/webm",
            "'mp4'" in class_src and "'avi'" in class_src
            and "'webm'" in class_src,
        )

        methods = _methods(cls)
        # collect_args validates same-file rejection, HH:MM:SS offset,
        # non-empty paths
        if "collect_args" in methods:
            ca_src = ast.unparse(methods["collect_args"])
            check(
                "CrossfadeVideosForm.collect_args rejects empty "
                "Video 1 / Video 2",
                "Pick a Video 1" in ca_src
                and "Pick a Video 2" in ca_src,
            )
            check(
                "CrossfadeVideosForm.collect_args rejects identical "
                "Video 1 + Video 2",
                "same file" in ca_src,
            )
            check(
                "CrossfadeVideosForm.collect_args validates HH:MM:SS "
                "offset format",
                "_OFFSET_RE" in ca_src or "HH:MM:SS" in ca_src,
            )

        # target() calls crossfade_two_videos with all kwargs
        if "target" in methods:
            target_src = ast.unparse(methods["target"])
            check(
                "CrossfadeVideosForm.target calls crossfade_two_videos",
                "crossfade_two_videos(" in target_src,
            )
            for kw in ("video_path_1", "video_path_2",
                       "crossfade_duration", "crossfade_method",
                       "crossfade_offset", "quality", "out_format"):
                check(
                    f"CrossfadeVideosForm.target passes {kw}= to "
                    "the backend",
                    f"{kw}=" in target_src,
                )

    # ==================================================================
    # 2. JoinVideosForm — normalize stub removed
    # ==================================================================
    cls = _find_class(vj_tree, "JoinVideosForm")
    check("JoinVideosForm class still defined", cls is not None)

    if cls is not None:
        class_src = ast.unparse(cls)
        check(
            "JoinVideosForm no longer has a `self.normalize` attribute "
            "(was a stub — never passed to any backend)",
            "self.normalize" not in class_src,
        )
        methods = _methods(cls)
        if "collect_args" in methods:
            ca_src = ast.unparse(methods["collect_args"])
            check(
                "JoinVideosForm.collect_args no longer returns "
                "`normalize`",
                "'normalize'" not in ca_src
                and "normalize=" not in ca_src,
            )
        if "target" in methods:
            t_src = ast.unparse(methods["target"])
            check(
                "JoinVideosForm.target signature no longer takes "
                "`normalize`",
                "normalize:" not in t_src
                and "normalize=" not in t_src,
            )

    # __all__ exports both
    check(
        "video_join.py __all__ exports JoinVideosForm",
        "'JoinVideosForm'" in vj_src
        or '"JoinVideosForm"' in vj_src,
    )
    check(
        "video_join.py __all__ exports CrossfadeVideosForm",
        "'CrossfadeVideosForm'" in vj_src
        or '"CrossfadeVideosForm"' in vj_src,
    )

    # ==================================================================
    # 3. Legacy class removal + deletion marker
    # ==================================================================
    vp_path = (REPO_ROOT / "mufasa" / "ui" / "pop_ups"
               / "video_processing_pop_up.py")
    vp_src = vp_path.read_text()
    vp_tree = ast.parse(vp_src)
    defined = {
        n.name for n in vp_tree.body
        if isinstance(n, ast.ClassDef)
    }
    check(
        "CrossfadeVideosPopUp class no longer defined in "
        "video_processing_pop_up.py",
        "CrossfadeVideosPopUp" not in defined,
    )
    marker_re = re.compile(
        r"^# CrossfadeVideosPopUp: removed in patch 122t",
        re.MULTILINE,
    )
    check(
        "deletion-record marker for CrossfadeVideosPopUp present",
        bool(marker_re.search(vp_src)),
    )

    # ==================================================================
    # 4. SimBA.py — no code-level reference to CrossfadeVideosPopUp
    # ==================================================================
    simba_src = (REPO_ROOT / "mufasa" / "SimBA.py").read_text()
    leaked = any(
        "CrossfadeVideosPopUp" in line
        and not line.lstrip().startswith("#")
        for line in simba_src.splitlines()
    )
    check(
        "SimBA.py has no code-level reference to CrossfadeVideosPopUp",
        not leaked,
    )

    # ==================================================================
    # 5. Video Processing page registers CrossfadeVideosForm
    # ==================================================================
    vpp_src = (REPO_ROOT / "mufasa" / "ui_qt" / "pages"
               / "video_processing_page.py").read_text()
    check(
        "video_processing_page imports CrossfadeVideosForm",
        "CrossfadeVideosForm" in vpp_src,
    )
    # Multi-form section: both Join + Crossfade registered together
    check(
        "Join & transition section registers BOTH JoinVideosForm "
        "and CrossfadeVideosForm",
        "JoinVideosForm" in vpp_src
        and "CrossfadeVideosForm" in vpp_src,
    )

    print(
        f"smoke_join_transition_cluster: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
