# Qt form registration audit

**Generated:** post-patch 122cf (May 2026).
**Purpose:** Confirm every Qt form defined under `mufasa/ui_qt/forms/` is reachable from at least one page's `add_section()` call. Companion to `qt_form_runtime_gaps.md` and `backend_audit.md`.

## 1. Finding

**60 of 60 OperationForm subclasses are registered.** Zero orphans.

Both `AverageFrameForm` (rewritten in 122cc) and `DropBodypartsForm` (rewritten in 122ce) are properly wired to their pages — the registration-uncertainty caveats noted in those patches were precautionary but unfounded.

Per-page section counts:

```
addons_page.py               10  forms
analysis_page.py              1  (AnalysisForm — also used by roi_page)
annotation_page.py            5  forms
classifier_page.py            4  forms (incl. ValidateClassifierForm, TrainClassifierForm, ClassifierManageForm, RunInferenceForm)
data_import_page.py           2  (PoseImportForm, VideoImportForm)
features_page.py              1  (FeatureSubsetExtractorForm)
pose_cleanup_page.py         10  forms (incl. AverageFrameForm? no — that's in video_processing)
roi_page.py                   5  forms (4 dedicated + AnalysisForm shared)
tools_page.py                 4  forms
video_processing_page.py     17  forms (largest page)
visualizations_page.py        1  (VisualizationForm)
```

Note: AnalysisForm appears in both `analysis_page.py` and `roi_page.py` — that's intentional re-use, not duplication. Same for ConverterForm (tools_page + video_processing_page).

## 2. Methodology

Reproducible AST script: for every `OperationForm` subclass in `mufasa/ui_qt/forms/*.py` (excluding private `_` -prefixed helpers), check whether the class name appears in any `mufasa/ui_qt/pages/*.py` file.

Limits:
* **Substring match.** A class name appearing in a comment, docstring, or unrelated context counts as registered. False-positive rate appears low (no forms are "discussed" in pages by name without also being imported), but a future code edit that names a form in a comment without registering it would slip through.
* **Doesn't verify the registration is reachable.** A form registered in a page that itself isn't routed by the workbench launcher would still count as "registered" here. The workbench page tree isn't audited.
* **Doesn't check section ordering / visibility.** A form registered as the 50th section on a page is "registered" but might be effectively hidden from a user who never scrolls.

## 3. Regression guard

The audit is now enforced by `tests/smoke_122cg_form_registration.py`. Any new `OperationForm` subclass added under `mufasa/ui_qt/forms/` MUST be referenced from at least one page file or the test fails. This is light-touch enforcement: it can't tell you which page to register the form on, but it WILL flag a new orphan at commit time.

Adding a form intentionally as a backend-only or programmatically-launched surface (no page registration) is a rare case; if needed, add the form to the audit script's exclusion list with a comment explaining why.

## 4. Caveats

* **One-shot finding.** The audit was run at the time of patch 122cg. Future patches that add new forms without registering them will be caught by the smoke test, but the count in §1 will go stale.
* **AnalysisForm + ConverterForm double-registration is intentional.** Some forms serve multiple navigation entry points (e.g., classifier analysis appears on both the Analysis page and the ROI page).
* **`AverageFrameForm` lives in `image_conversion.py` but is registered in `video_processing_page.py`.** Module location doesn't dictate page placement — historical artifact. Working fine.
