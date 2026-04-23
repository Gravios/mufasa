#!/usr/bin/env python3
"""
scripts/e2e_canary.py
=====================

End-to-end canary harness for the Mufasa Qt port.

Statically validates every form's ``target()`` would find its backend
module, class, and accept the kwargs the form passes. Does NOT run
backends — no pose CSV, no video encoding, no classifier inference.
Pure reflection.

Catches:
  - Missing backend modules / classes (lazy imports hide these)
  - Form passes kwargs backend does not accept (rename drift)
  - Backend requires kwargs form does not pass (missing defaults)

Exit codes:
  0 — all OK (no unexpected errors)
  1 — one or more unexpected errors
  2 — harness crashed
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

# Numba JIT is useless for signature introspection and its cold start
# is 30+ s. Disable before any mufasa import.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("NO_COLOR", "1")
os.environ.setdefault("ANSI_COLORS_DISABLED", "1")

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #
@dataclass
class BackendTarget:
    form: str
    modpath: str
    symbol: str
    kwargs_passed: tuple
    notes: str = ""
    expected_fail: str = ""


@dataclass
class Finding:
    form: str
    backend: str
    status: str   # "ok" | "warn" | "error"
    message: str


# --------------------------------------------------------------------------- #
# Known upstream / out-of-tree issues — auto-classified to WARN
# --------------------------------------------------------------------------- #
_KNOWN_IMPORT_ISSUES = [
    ("parallel_backend",
     "upstream: sklearn removed parallel_backend from sklearn.utils"),
    ("No module named 'mufasa.data_processors.keypoint_dropper'",
     "out-of-tree in this fork"),
    ("No module named 'mufasa.roi_tools.ROI_clf_calculator",
     "out-of-tree in this fork"),
]


def _classify_known_issue(message: str) -> str:
    for pattern, reason in _KNOWN_IMPORT_ISSUES:
        if pattern in message:
            return reason
    return ""


# --------------------------------------------------------------------------- #
# Introspection
# --------------------------------------------------------------------------- #
def _signature_of(obj) -> inspect.Signature:
    if inspect.isclass(obj):
        try:
            return inspect.signature(obj.__init__)
        except (TypeError, ValueError):
            return inspect.signature(obj)
    return inspect.signature(obj)


def analyze(target: BackendTarget) -> list[Finding]:
    backend_name = f"{target.modpath}.{target.symbol}"
    findings: list[Finding] = []

    try:
        mod = importlib.import_module(target.modpath)
    except Exception as exc:
        msg = (f"import failed: {type(exc).__name__}: "
               f"{str(exc).splitlines()[0][:160]}")
        known = _classify_known_issue(msg)
        if known:
            return [Finding(target.form, backend_name, "warn",
                            f"[expected: {known}] {msg}")]
        return [Finding(target.form, backend_name, "error", msg)]

    obj = getattr(mod, target.symbol, None)
    if obj is None:
        return [Finding(target.form, backend_name, "error",
                        f"symbol {target.symbol!r} not found in "
                        f"{target.modpath}")]

    try:
        sig = _signature_of(obj)
    except (TypeError, ValueError) as exc:
        return [Finding(target.form, backend_name, "warn",
                        f"could not get signature: {exc}")]

    params = {k: v for k, v in sig.parameters.items() if k != "self"}
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD
                     for p in params.values())
    accepted = set(params)
    passed = set(target.kwargs_passed)

    if not has_var_kw:
        extras = passed - accepted
        if extras:
            findings.append(Finding(
                target.form, backend_name, "warn",
                f"form passes kwargs backend does not accept: "
                f"{sorted(extras)}"))

    required = {n for n, p in params.items()
                if p.default is inspect.Parameter.empty
                and p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                   inspect.Parameter.VAR_KEYWORD)}
    missing = required - passed
    if missing:
        findings.append(Finding(
            target.form, backend_name, "error",
            f"backend requires kwargs form does not pass: "
            f"{sorted(missing)}"))

    if not findings:
        findings.append(Finding(
            target.form, backend_name, "ok",
            f"{len(accepted)} params accepted "
            f"({len(required)} required). Form passes "
            f"{len(passed)} kwargs."))

    if target.expected_fail:
        findings = [Finding(
            f.form, f.backend,
            "warn" if f.status == "error" else f.status,
            (f"[expected: {target.expected_fail}] {f.message}"
             if f.status == "error" else f.message))
            for f in findings]
    return findings


# --------------------------------------------------------------------------- #
# Target enumeration
# --------------------------------------------------------------------------- #
def collect_individual_targets() -> list[BackendTarget]:
    T = BackendTarget
    vp = "mufasa.video_processors.video_processing"
    out: list[BackendTarget] = []

    # Pose cleanup
    out += [
        T(form="pose_cleanup.SmoothingForm",
          modpath="mufasa.data_processors.smoothing", symbol="Smoothing",
          kwargs_passed=("config_path", "data_path", "time_window",
                         "method", "copy_originals")),
        T(form="pose_cleanup.InterpolateForm",
          modpath="mufasa.data_processors.interpolate", symbol="Interpolate",
          kwargs_passed=("config_path", "data_path", "type", "method",
                         "copy_originals")),
        T(form="pose_cleanup.DropBodypartsForm",
          modpath="mufasa.data_processors.keypoint_dropper",
          symbol="KeyPointRemover",
          kwargs_passed=("config_path", "body_parts", "copy_originals"),
          expected_fail="out-of-tree in this fork"),
    ]

    # ROI
    out += [
        T(form="roi.ROIAnalysisForm[simple]",
          modpath="mufasa.roi_tools.ROI_analyzer", symbol="ROIAnalyzer",
          kwargs_passed=("config_path", "data_path", "detailed_bout_data",
                         "calculate_distances", "threshold", "body_parts")),
        T(form="roi.ROIAnalysisForm[aggregate]",
          modpath="mufasa.roi_tools.roi_aggregate_statistics_analyzer",
          symbol="ROIAggregateStatisticsAnalyzer",
          kwargs_passed=("config_path", "data_path", "threshold", "body_parts",
                         "detailed_bout_data", "calculate_distances",
                         "total_time", "entry_counts", "first_entry_time",
                         "last_entry_time", "mean_bout_time", "outside_rois")),
        T(form="roi.ROIAnalysisForm[time_bins]",
          modpath="mufasa.roi_tools.roi_time_bins_analyzer",
          symbol="ROITimebinAnalyzer",
          kwargs_passed=("config_path", "bin_size", "data_path", "threshold",
                         "body_parts", "detailed_bout_data",
                         "calculate_distances")),
        T(form="roi.ROIFeaturesForm[append]",
          modpath="mufasa.roi_tools.ROI_feature_analyzer",
          symbol="ROIFeatureCreator",
          kwargs_passed=("config_path", "body_parts", "data_path",
                         "append_data")),
        T(form="roi.ROIManageForm[import]",
          modpath="mufasa.roi_tools.import_roi_csvs",
          symbol="ROIDefinitionsCSVImporter",
          kwargs_passed=("config_path", "rectangles_path", "circles_path",
                         "polygon_path", "append")),
        T(form="roi.ROIManageForm[standardize]",
          modpath="mufasa.roi_tools.ROI_size_standardizer",
          symbol="ROISizeStandardizer",
          kwargs_passed=("config_path", "reference_video")),
        T(form="roi.ROIVisualizeForm[tracking,sp]",
          modpath="mufasa.plotting.roi_plotter", symbol="ROIPlotter",
          kwargs_passed=("config_path", "video_path", "body_parts",
                         "threshold", "show_animal_name", "show_body_part",
                         "show_bbox")),
        T(form="roi.ROIVisualizeForm[tracking,mp]",
          modpath="mufasa.plotting.roi_plotter_mp",
          symbol="ROIPlotMultiprocess",
          kwargs_passed=("config_path", "video_path", "body_parts",
                         "threshold", "show_animal_name", "show_body_part",
                         "show_bbox")),
        T(form="roi.ROIVisualizeForm[features]",
          modpath="mufasa.plotting.ROI_feature_visualizer_mp",
          symbol="ROIfeatureVisualizerMultiprocess",
          kwargs_passed=("config_path", "video_path", "body_parts",
                         "show_animal_names", "core_cnt", "gpu")),
    ]

    # Classifier
    out.append(T(form="classifier.ClassifierManageForm[print]",
                 modpath="mufasa.utils.read_write", symbol="tabulate_clf_info",
                 kwargs_passed=("clf_path",)))

    # Annotation
    out += [
        T(form="annotation.ThirdPartyAppenderForm",
          modpath="mufasa.third_party_label_appenders.third_party_appender",
          symbol="ThirdPartyLabelAppender",
          kwargs_passed=("config_path", "data_dir", "app", "file_format",
                         "error_settings", "log")),
        T(form="annotation.AnnotationReportsForm[extract]",
          modpath="mufasa.labelling.extract_labelled_frames",
          symbol="AnnotationFrameExtractor",
          kwargs_passed=("config_path", "data_paths", "clfs",
                         "img_downsample_factor", "img_format",
                         "img_greyscale")),
        T(form="annotation.AnnotationReportsForm[counts]",
          modpath="mufasa.labelling.extract_labelling_meta",
          symbol="AnnotationMetaDataExtractor",
          kwargs_passed=("config_path", "split_by_video", "annotated_bouts")),
    ]

    # Features
    out.append(T(form="features.FeatureSubsetExtractorForm",
                 modpath="mufasa.feature_extractors.feature_subsets",
                 symbol="FeatureSubsetsCalculator",
                 kwargs_passed=("config_path", "feature_families",
                                "file_checks", "save_dir", "data_dir",
                                "append_to_features_extracted",
                                "append_to_targets_inserted")))

    # Project setup
    out.append(T(form="project_setup.ArchiveFilesForm",
                 modpath="mufasa.utils.read_write",
                 symbol="archive_processed_files",
                 kwargs_passed=("config_path", "archive_name")))

    # Add-ons
    out += [
        T(form="addons.CueLightDataForm",
          modpath="mufasa.data_processors.cue_light_analyzer",
          symbol="CueLightAnalyzer",
          kwargs_passed=("config_path", "data_dir", "cue_light_names",
                         "save_dir", "core_cnt", "detailed_data", "verbose")),
        T(form="addons.CueLightClfForm",
          modpath="mufasa.data_processors.cue_light_clf_statistics",
          symbol="CueLightClfAnalyzer",
          kwargs_passed=("config_path", "cue_light_names", "clf_names",
                         "data_dir", "pre_window", "post_window")),
        T(form="addons.CueLightMovementForm",
          modpath="mufasa.data_processors.cue_light_movement_statistics",
          symbol="CueLightMovementAnalyzer",
          kwargs_passed=("config_path", "cue_light_names", "bp_name",
                         "data_dir", "pre_window", "post_window", "verbose")),
        T(form="addons.CueLightVisualizerForm",
          modpath="mufasa.plotting.cue_light_visualizer",
          symbol="CueLightVisualizer",
          kwargs_passed=("config_path", "cue_light_names", "video_path",
                         "data_path", "frame_setting", "video_setting",
                         "core_cnt", "show_pose", "verbose")),
        T(form="addons.KleinbergForm",
          modpath="mufasa.data_processors.kleinberg_calculator",
          symbol="KleinbergCalculator",
          kwargs_passed=("config_path", "classifier_names", "sigma", "gamma",
                         "hierarchy", "verbose", "save_originals",
                         "hierarchical_search", "input_dir", "output_dir")),
        T(form="addons.MutualExclusivityForm",
          modpath="mufasa.data_processors.mutual_exclusivity_corrector",
          symbol="MutualExclusivityCorrector",
          kwargs_passed=("rules", "config_path")),
        T(form="addons.PupRetrievalForm",
          modpath="mufasa.data_processors.pup_retrieval_calculator",
          symbol="PupRetrieverCalculator",
          kwargs_passed=("config_path", "settings")),
        T(form="addons.SpontaneousAlternationForm[calc]",
          modpath="mufasa.data_processors.spontaneous_alternation_calculator",
          symbol="SpontaneousAlternationCalculator",
          kwargs_passed=("config_path", "arm_names", "center_name",
                         "animal_area", "threshold", "buffer", "verbose",
                         "detailed_data", "data_path")),
        T(form="addons.SpontaneousAlternationForm[plot]",
          modpath="mufasa.plotting.spontaneous_alternation_plotter",
          symbol="SpontaneousAlternationsPlotter",
          kwargs_passed=("config_path", "arm_names", "center_name",
                         "animal_area", "threshold", "buffer", "core_cnt",
                         "verbose", "data_path")),
    ]

    # Video Processing — post-fix kwargs (save_path/save_dir/output_path etc.)
    out += [
        T(form="video_processing.VideoFormatConverterForm[MP4]",
          modpath=vp, symbol="convert_to_mp4",
          kwargs_passed=("path", "codec", "quality", "keep_audio")),
        T(form="video_processing.VideoFormatConverterForm[AVI]",
          modpath=vp, symbol="convert_to_avi",
          kwargs_passed=("path", "codec", "quality")),
        T(form="video_processing.VideoFormatConverterForm[MOV]",
          modpath=vp, symbol="convert_to_mov",
          kwargs_passed=("path", "codec", "quality")),
        T(form="video_processing.VideoFormatConverterForm[WEBM]",
          modpath=vp, symbol="convert_to_webm",
          kwargs_passed=("path", "codec", "quality")),
        T(form="video_processing.ClipVideosForm",
          modpath=vp, symbol="clip_video_in_range",
          kwargs_passed=("file_path", "start_time", "end_time")),
        T(form="video_processing.CropVideosForm[rect,single]",
          modpath=vp, symbol="crop_single_video",
          kwargs_passed=("file_path",)),
        T(form="video_processing.CropVideosForm[rect,dir]",
          modpath=vp, symbol="crop_multiple_videos",
          kwargs_passed=("directory_path", "output_path")),
        T(form="video_processing.ResizeVideosForm[downsample]",
          modpath=vp, symbol="downsample_video",
          kwargs_passed=("file_path", "video_width", "video_height",
                         "scale_factor")),
        T(form="video_processing.ResizeVideosForm[change_fps]",
          modpath=vp, symbol="change_fps_of_multiple_videos",
          kwargs_passed=("path", "fps")),
        T(form="video_processing.RotateFlipForm[rotate]",
          modpath=vp, symbol="rotate_video",
          kwargs_passed=("video_path", "degrees", "interactive")),
        T(form="video_processing.RotateFlipForm[flip]",
          modpath=vp, symbol="flip_videos",
          kwargs_passed=("video_path", "flip_code")),
        T(form="video_processing.VideoFiltersForm[clahe]",
          modpath=vp, symbol="clahe_enhance_video_mp",
          kwargs_passed=("file_path", "clip_limit", "tile_grid")),
        T(form="video_processing.VideoFiltersForm[greyscale_single]",
          modpath=vp, symbol="video_to_greyscale",
          kwargs_passed=("file_path",)),
        T(form="video_processing.VideoFiltersForm[greyscale_batch]",
          modpath=vp, symbol="batch_video_to_greyscale",
          kwargs_passed=("path",)),
        T(form="video_processing.VideoFiltersForm[bg_subtract]",
          modpath=vp, symbol="video_bg_subtraction_mp",
          kwargs_passed=("video_path", "bg_method")),
        T(form="video_processing.ExtractFramesForm[all]",
          modpath=vp, symbol="extract_frames_single_video",
          kwargs_passed=("file_path", "save_dir")),
        T(form="video_processing.ExtractFramesForm[range]",
          modpath=vp, symbol="extract_frame_range",
          kwargs_passed=("file_path", "start_frame", "end_frame")),
        T(form="video_processing.JoinVideosForm[temporal]",
          modpath=vp, symbol="temporal_concatenation",
          kwargs_passed=("video_paths", "save_path")),
        T(form="video_processing.JoinVideosForm[horizontal]",
          modpath=vp, symbol="horizontal_video_concatenator",
          kwargs_passed=("video_paths", "save_path")),
        T(form="video_processing.JoinVideosForm[mosaic]",
          modpath=vp, symbol="mosaic_concatenator",
          kwargs_passed=("video_paths", "save_path",
                         "height_idx", "width_idx")),
        T(form="video_processing.ImageFormatConverterForm[png]",
          modpath=vp, symbol="convert_to_png", kwargs_passed=("path",)),
        T(form="video_processing.ImageFormatConverterForm[jpeg]",
          modpath=vp, symbol="change_img_format",
          kwargs_passed=("directory", "file_type_in", "file_type_out")),
        T(form="Tools.reverse_video", modpath=vp, symbol="reverse_videos",
          kwargs_passed=("path",)),
        T(form="Tools.crossfade", modpath=vp, symbol="crossfade_two_videos",
          kwargs_passed=("video_path_1", "video_path_2")),
        T(form="Tools.change_speed", modpath=vp, symbol="change_playback_speed",
          kwargs_passed=("video_path", "speed")),
    ]

    # Targeted clips dialog
    out.append(T(form="targeted_clips.TargetedClipsDialog",
                 modpath=vp, symbol="multi_split_video",
                 kwargs_passed=("file_path", "start_times", "end_times",
                                "out_dir", "include_clip_time_in_filename")))
    return out


def _kwargs_from_extras_widget(widget_cls, parent=None) -> set[str]:
    try:
        from PySide6.QtWidgets import QApplication
        if QApplication.instance() is None:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            QApplication([])
        inst = widget_cls(parent)
        return set(inst.to_kwargs().keys())
    except Exception:
        return set()


def collect_route_targets() -> list[BackendTarget]:
    out: list[BackendTarget] = []

    try:
        from mufasa.ui_qt.forms.data_import import (ROUTES as DI_ROUTES,
                                                     _EXTRAS_WIDGETS)
    except Exception as e:
        print(f"warning: data_import ROUTES import failed: {e}",
              file=sys.stderr)
        DI_ROUTES, _EXTRAS_WIDGETS = [], {}

    extras_by_key = {k: _kwargs_from_extras_widget(cls)
                     for k, cls in _EXTRAS_WIDGETS.items()}

    for r in DI_ROUTES:
        if r.backend is None:
            continue
        bname = getattr(r.backend, "__name__", "") or ""
        if "." not in bname or bname.startswith("<"):
            continue
        modpath, _, classname = bname.rpartition(".")
        km = r.kwargs_map
        kwargs = {km.get("source_path", "source_path"),
                  km.get("save_path", "save_path")}
        if r.needs_video:
            kwargs.add(km.get("video_path", "video_path"))
        for ekey in extras_by_key.get(r.extras_key, set()):
            kwargs.add(km.get(ekey, ekey))
        for flag in ("greyscale", "clahe", "verbose"):
            if flag in r.common_flags:
                kwargs.add(km.get(flag, flag))
        out.append(BackendTarget(
            form=f"data_import.ConverterForm[{r.source_label}→{r.target_label}]",
            modpath=modpath, symbol=classname,
            kwargs_passed=tuple(sorted(kwargs))))

    try:
        from mufasa.ui_qt.forms.visualizations import ROUTES as VIZ_ROUTES
    except Exception as e:
        print(f"warning: visualizations ROUTES import failed: {e}",
              file=sys.stderr)
        VIZ_ROUTES = []

    for r in VIZ_ROUTES:
        for kind, backend in (("sp", getattr(r, "backend_sp", None)),
                              ("mp", getattr(r, "backend_mp", None))):
            if backend is None:
                continue
            bname = getattr(backend, "__name__", "") or ""
            if "." not in bname or bname.startswith("<"):
                continue
            modpath, _, classname = bname.rpartition(".")
            km = getattr(r, "kwargs_map", {})
            kwargs = set()
            if getattr(r, "scope_kind", "project") == "project":
                kwargs.add("config_path")
            else:
                kwargs.add(km.get("source_path", "data_path"))
            if getattr(r, "needs_video", False):
                kwargs.add(km.get("video_path", "video_path"))
            if getattr(r, "needs_save_dir", False):
                kwargs.add(km.get("save_path", "save_dir"))
            if getattr(r, "data_paths_source", None):
                kwargs.add(km.get("data_paths", "data_paths"))
            if getattr(r, "data_path_source", None):
                kwargs.add(km.get("data_path", "data_path"))
            for k in getattr(r, "default_kwargs", {}).keys():
                kwargs.add(km.get(k, k))
            toggle_map = {"frame": "frame_setting", "video": "video_setting",
                          "last_frame": "last_frame", "gpu": "gpu"}
            for tog in getattr(r, "common_toggles", ()):
                kwargs.add(km.get(tog, toggle_map.get(tog, tog)))
            for ext in getattr(r, "extras", ()):
                kwargs.add(km.get(ext[0], ext[0]))
            if kind == "mp":
                kwargs.add("core_cnt")
            out.append(BackendTarget(
                form=f"visualizations.VisualizationForm[{r.label}] ({kind})",
                modpath=modpath, symbol=classname,
                kwargs_passed=tuple(sorted(kwargs))))

    try:
        from mufasa.ui_qt.forms.analysis import ROUTES as ANAL_ROUTES
    except Exception as e:
        print(f"warning: analysis ROUTES import failed: {e}",
              file=sys.stderr)
        ANAL_ROUTES = []

    for r in ANAL_ROUTES:
        for kind, backend in (("sp", getattr(r, "backend_sp", None)),
                              ("mp", getattr(r, "backend_mp", None))):
            if backend is None:
                continue
            bname = getattr(backend, "__name__", "") or ""
            if "." not in bname or bname.startswith("<"):
                continue
            modpath, _, classname = bname.rpartition(".")
            km = getattr(r, "kwargs_map", {})
            needs = getattr(r, "needs", ())
            if "severity_mode" in needs:
                kwargs = {"config_path", "settings"}
            else:
                kwargs = {"config_path"}
                if "classifiers" in needs:
                    kwargs.add(km.get("classifiers", "classifiers"))
                if "body_parts" in needs:
                    kwargs.add(km.get("body_parts", "body_parts"))
                if "bin_length" in needs:
                    kwargs.add(km.get("bin_length", "bin_length"))
                for ext in getattr(r, "extras", ()):
                    kwargs.add(km.get(ext[0], ext[0]))
                if kind == "mp":
                    kwargs.add("core_cnt")
            out.append(BackendTarget(
                form=f"analysis.AnalysisForm[{r.label}] ({kind})",
                modpath=modpath, symbol=classname,
                kwargs_passed=tuple(sorted(kwargs))))

    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def render_report(findings: list[Finding]) -> str:
    lines = []
    by_status = {"ok": [], "warn": [], "error": []}
    for f in findings:
        by_status.setdefault(f.status, []).append(f)

    lines.append("=" * 72)
    lines.append(" Mufasa Qt port — end-to-end canary report")
    lines.append("=" * 72)
    lines.append(f"  Total: {len(findings)}")
    lines.append(f"  OK:    {len(by_status['ok'])}")
    lines.append(f"  WARN:  {len(by_status['warn'])}")
    lines.append(f"  ERROR: {len(by_status['error'])}")
    lines.append("")

    for status in ("error", "warn"):
        group = by_status[status]
        if not group:
            continue
        lines.append(f"-- {status.upper()} ({len(group)}) " + "-" * 50)
        for f in group:
            lines.append(f"  [{status.upper():5}] {f.form}")
            lines.append(f"          → {f.backend}")
            lines.append(f"          {f.message}")
            lines.append("")

    if by_status["ok"]:
        lines.append(f"-- OK ({len(by_status['ok'])}) " + "-" * 50)
        for f in by_status["ok"]:
            lines.append(f"  [OK] {f.form:<70} {f.backend}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    try:
        targets = collect_individual_targets() + collect_route_targets()
    except Exception:
        traceback.print_exc()
        return 2

    if args.only:
        targets = [t for t in targets if args.only.lower() in t.form.lower()]

    findings: list[Finding] = []
    for t in targets:
        findings.extend(analyze(t))

    display_findings = findings
    if args.quiet:
        display_findings = [f for f in findings if f.status != "ok"]
    print(render_report(display_findings))

    if args.json:
        payload = {
            "findings": [asdict(f) for f in findings],
            "summary": {
                "total": len(findings),
                "ok": sum(1 for f in findings if f.status == "ok"),
                "warn": sum(1 for f in findings if f.status == "warn"),
                "error": sum(1 for f in findings if f.status == "error"),
            },
        }
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nJSON report → {args.json}")

    errors = sum(1 for f in findings if f.status == "error")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
