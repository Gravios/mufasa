"""
mufasa.ui_qt.pages.annotation_page
==================================

Annotation workbench page. Five sections covering:

* **Frame labelling** — :class:`FrameLabellingLauncher`. Verb-form
  button label ('Label'). Launches the per-frame scrubber with
  classifier checkboxes.
* **Targeted annotation clips** — :class:`TargetedAnnotationClipsLauncher`.
  Define clip ranges within a video for annotation.
* **Third-party annotation import** — :class:`ThirdPartyAppenderForm`.
  Inline form for merging BORIS / BENTO / Ethovision / Solomon /
  DeepEthogram annotations.
* **Review classifier predictions** — :class:`ClipReviewLauncher`.
  Verb-form button label ('Review'). Bout-by-bout validation.
* **Reports** — :class:`AnnotationReportsForm`. Extract labelled
  frames as images, or compute per-classifier annotation counts.

Patch 122aa
-----------
* Two long verb-phrase button labels ('Select video and launch
  labeller…', 'Select video and launch reviewer…') shortened to
  'Label' and 'Review' respectively. Matches the short verb-
  based action labels used elsewhere in the workbench.
* Form descriptions updated to document the
  features_extracted / targets_inserted / machine_results path
  conventions, which the underlying
  :mod:`mufasa.ui_qt.frame_labeller` still resolves through the
  legacy INI ``project_path`` key. v1 projects work but the
  labels and features land under the legacy subtree beneath the
  v1 project root rather than under
  ``derived/classifications/<run_id>/``. Plumbing the labeller
  through :func:`project_paths_from_config` is a separate
  deferred item.
* Project-load error messages updated to mention both v1
  (``project.toml``) and legacy (``project_config.ini``)
  project files.
"""
from __future__ import annotations

from mufasa.ui_qt.forms.annotation import (
    AnnotationReportsForm,
    ClipReviewLauncher,
    FrameLabellingLauncher,
    TargetedAnnotationClipsLauncher,
    ThirdPartyAppenderForm,
)
from mufasa.ui_qt.workbench import WorkflowPage


def build_annotation_page(workbench,
                          config_path: str | None = None) -> WorkflowPage:
    page = workbench.add_page("Annotation", icon_name="label")
    page.add_section("Frame labelling",          [(FrameLabellingLauncher, {})])
    page.add_section("Targeted annotation clips",
                     [(TargetedAnnotationClipsLauncher, {})])
    page.add_section("Third-party annotation import",
                     [(ThirdPartyAppenderForm, {})])
    page.add_section("Review classifier predictions",
                     [(ClipReviewLauncher, {})])
    page.add_section("Reports",                  [(AnnotationReportsForm, {})])
    return page


__all__ = ["build_annotation_page"]
