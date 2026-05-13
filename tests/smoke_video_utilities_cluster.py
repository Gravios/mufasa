"""
tests/smoke_video_utilities_cluster.py
======================================

Patch 122u: regression guard for the three single-utility forms
in ``mufasa/ui_qt/forms/video_utilities.py``.

Coverage:

1. **ReverseVideoForm** — _ScopePicker + save_dir + quality +
   gpu; target() calls reverse_videos with the right kwargs.
2. **ChangeSpeedForm** — _ScopePicker + speed (QDoubleSpinBox)
   + quality + gpu; target() dispatches to
   change_playback_speed (single) or change_playback_speed_dir
   (directory); rejects speed = 1.0 as no-op.
3. **PixelsPerMMForm** — video file picker + known mm distance;
   target() constructs GetPixelsPerMillimeterInterface and
   surfaces the ppm result.
4. **Legacy removals**: ReverseVideoPopUp +
   CalculatePixelsPerMMInVideoPopUp gone from
   video_processing_pop_up.py with deletion-record markers;
   change_speed_popup.py file deleted entirely.
5. **SimBA.py** — no code-level references to any of the three
   legacy classes survive.
6. **Page** — Utilities section registers all three forms.

AST-only.
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
    vu_path = (REPO_ROOT / "mufasa" / "ui_qt" / "forms"
               / "video_utilities.py")
    check("video_utilities.py exists", vu_path.is_file())
    vu_src = vu_path.read_text()
    vu_tree = ast.parse(vu_src)

    # ------------------ ReverseVideoForm ------------------
    cls = _find_class(vu_tree, "ReverseVideoForm")
    check("ReverseVideoForm class defined", cls is not None)
    if cls is not None:
        class_src = ast.unparse(cls)
        for attr in (
            "self.scope", "self.save_dir_edit",
            "self.quality", "self.gpu",
        ):
            check(
                f"ReverseVideoForm sets {attr}",
                attr in class_src,
            )
        methods = _methods(cls)
        if "target" in methods:
            target_src = ast.unparse(methods["target"])
            check(
                "ReverseVideoForm.target calls reverse_videos",
                "reverse_videos(" in target_src,
            )
            check(
                "ReverseVideoForm.target passes path / save_dir / "
                "quality / gpu",
                all(f"{kw}=" in target_src
                    for kw in ("path", "save_dir",
                               "quality", "gpu")),
            )

    # ------------------ ChangeSpeedForm ------------------
    cls = _find_class(vu_tree, "ChangeSpeedForm")
    check("ChangeSpeedForm class defined", cls is not None)
    if cls is not None:
        class_src = ast.unparse(cls)
        for attr in (
            "self.scope", "self.speed", "self.save_dir_edit",
            "self.quality", "self.gpu",
        ):
            check(
                f"ChangeSpeedForm sets {attr}",
                attr in class_src,
            )
        # speed uses QDoubleSpinBox (not QSpinBox — 0.1× steps)
        check(
            "ChangeSpeedForm uses QDoubleSpinBox for speed",
            "QDoubleSpinBox" in class_src,
        )
        methods = _methods(cls)
        if "collect_args" in methods:
            ca_src = ast.unparse(methods["collect_args"])
            check(
                "ChangeSpeedForm.collect_args rejects speed = 1.0 "
                "as a no-op",
                "1.0" in ca_src and "no-op" in ca_src,
            )
        if "target" in methods:
            target_src = ast.unparse(methods["target"])
            check(
                "ChangeSpeedForm.target dispatches to "
                "change_playback_speed_dir for directory mode",
                "change_playback_speed_dir(" in target_src,
            )
            check(
                "ChangeSpeedForm.target dispatches to "
                "change_playback_speed for single-file mode",
                "change_playback_speed(" in target_src,
            )

    # ------------------ PixelsPerMMForm ------------------
    cls = _find_class(vu_tree, "PixelsPerMMForm")
    check("PixelsPerMMForm class defined", cls is not None)
    if cls is not None:
        class_src = ast.unparse(cls)
        for attr in ("self.video_path_edit", "self.known_mm"):
            check(
                f"PixelsPerMMForm sets {attr}",
                attr in class_src,
            )
        methods = _methods(cls)
        if "target" in methods:
            target_src = ast.unparse(methods["target"])
            check(
                "PixelsPerMMForm.target uses "
                "GetPixelsPerMillimeterInterface",
                "GetPixelsPerMillimeterInterface" in target_src,
            )
            check(
                "PixelsPerMMForm.target surfaces result via "
                "QMessageBox",
                "QMessageBox" in target_src
                or "QMessageBox" in class_src,
            )

    # ------------------ Legacy class removal ------------------
    vp_path = (REPO_ROOT / "mufasa" / "ui" / "pop_ups"
               / "video_processing_pop_up.py")
    vp_src = vp_path.read_text()
    vp_tree = ast.parse(vp_src)
    defined = {
        n.name for n in vp_tree.body
        if isinstance(n, ast.ClassDef)
    }
    for d in ("ReverseVideoPopUp",
              "CalculatePixelsPerMMInVideoPopUp"):
        check(
            f"legacy {d} class no longer defined in "
            "video_processing_pop_up.py",
            d not in defined,
        )
        marker_re = re.compile(
            rf"^# {re.escape(d)}: removed in patch 122u",
            re.MULTILINE,
        )
        check(
            f"deletion-record marker for {d} present",
            bool(marker_re.search(vp_src)),
        )

    # change_speed_popup.py — whole file gone
    cs_path = (REPO_ROOT / "mufasa" / "ui" / "pop_ups"
               / "change_speed_popup.py")
    check(
        "legacy change_speed_popup.py file deleted",
        not cs_path.exists(),
    )

    # ------------------ SimBA.py — no code-level refs ------------------
    simba_src = (REPO_ROOT / "mufasa" / "SimBA.py").read_text()
    for d in ("ReverseVideoPopUp", "ChangeSpeedPopup",
              "CalculatePixelsPerMMInVideoPopUp"):
        leaked = any(
            d in line and not line.lstrip().startswith("#")
            for line in simba_src.splitlines()
        )
        check(
            f"SimBA.py has no code-level reference to {d}",
            not leaked,
        )

    # ------------------ Page wiring ------------------
    vpp_src = (REPO_ROOT / "mufasa" / "ui_qt" / "pages"
               / "video_processing_page.py").read_text()
    check(
        "video_processing_page imports ReverseVideoForm",
        "ReverseVideoForm" in vpp_src,
    )
    check(
        "video_processing_page imports ChangeSpeedForm",
        "ChangeSpeedForm" in vpp_src,
    )
    check(
        "video_processing_page imports PixelsPerMMForm",
        "PixelsPerMMForm" in vpp_src,
    )
    check(
        "Utilities section registered on Video Processing page",
        "Utilities" in vpp_src,
    )

    print(
        f"smoke_video_utilities_cluster: "
        f"{CHECKS_PASSED}/{CHECKS_TOTAL} checks passed"
    )
    return 0 if CHECKS_PASSED == CHECKS_TOTAL else 1


if __name__ == "__main__":
    sys.exit(main())
