# CHANGELOG

Working notes on what landed in each session, what's deferred, and what
to pick up next time. Keep entries dated and grouped by patch series so
"where did this behavior come from" is answerable from git log alone.

---

## Session 2026-07-17 — marker names are case-sensitive; the model's layout wins

**122jn — Jitter near-discrete features in figures.**
The Landscape… / Animate… names table gets a Jitter tick per feature, saved
as `[jitter]` in `logs/measures/feature_names.toml`. Marked features are
drawn with uniform noise of ±0.4 × the median step between their distinct
values (`feature_names.jitter_values`; levels stay separable, NaN stays
NaN, reproducible): mufasa-landscape's joint PDFs (titled "x/y jittered")
and mufasa-landscape-animate's feature spaces. The dip test, its boundaries,
`tsne_samples.parquet` and the t-SNE keep the real values. Both tools take
`--jitter COL` (repeatable) and `--no-jitter`. Feature Visualization's joint
plot gets a Jitter checkbox and always jitters marked features.
Test: `tests/smoke_122jn_jitter_discrete_features.py` (19 checks);
smoke_122jl allows the extra table column.

**122jm — Animation: number of points, sampling method, session intro.**
mufasa-landscape-animate gets `--sampling random | balanced | even` next to
`--n-points`: random as before (dense regions dominate, as in the data),
balanced takes the same number from every session, even spreads the points
over the t-SNE map (a 40 × 40 lattice drawn from in turn) so sparse regions
stay visible. The GIF now opens with `--intro-frames` (default 40) frames of
the t-SNE map coloured by session ("coloured by session (N)" under the
title), then eases over `--intro-fade` (20) frames into the `--color`
colouring; skipped when colouring by session anyway. Session colours stay
distinct beyond 20 sessions. The Animate… popup has Points in animation,
Sampling and Intro frames. smoke_122iy runs its timing check with
`--intro-frames 0`.
Test: `tests/smoke_122jm_animate_sampling_and_session_intro.py` (19 checks).

**122jl — Names popup now opens; per-feature linear / log10 scale.**
The Landscape… / Animate… buttons connected clicked(bool) straight to
`generate_*(ask=True)`, so clicked's `False` skipped the names popup; they
now call with no arguments. The popup's table gains a Scale column
(linear / log10 per feature, "Set all scales" buttons), saved as `[scales]`
next to the names in `logs/measures/feature_names.toml`. mufasa-landscape
analyses log10 features as log10 (values ≤ 0 → missing), labels them
"log10 <name>", and lists them in the run's `log10_columns.txt`
(`--no-scales` to ignore the marks); mufasa-landscape-animate labels them the
same way, and a landscape made with other scales is not reused.
Test: `tests/smoke_122jl_feature_scales_and_names_popup.py` (19 checks).

**122jk — Landscape / animation from the selection; figure names; joint
styles; viewer grid and follow.**
Feature Visualization gets *Landscape…* and *Animate…*: the ticked features
go through mufasa-landscape (`--sources features --columns-file`, pooled over
all sessions; exactly the selection — no family filter, no near-duplicate
pruning, full-circle angles kept — and the best-ranked pairs plotted whether
or not they are multimodal) and mufasa-landscape-animate (`--through-tsne`
with explicit `--pair`s: t-SNE → pair 1 → t-SNE → pair 2 → … → t-SNE; an
earlier landscape of the same selection is reused). Both run as child
processes with the log streaming into a dialog that then previews the PNGs
and plays the GIF. Before either runs, a table of the selected features
(column | name in figures) is shown; edits are saved to
`logs/measures/feature_names.toml` (`mufasa.utils.feature_names`) and used by
both tools (`--names-file`, default the project's file) and the form's joint
plot. Joint plots take a style: heatmap (now with a colour bar), scatter or
scatter-heatmap. The video × feature viewer draws a light 5-line grid per
lane (0/25/50/75/100 %) and follows the selection: ticking / unticking
features updates open viewers (debounced; "Follow selection" to opt out).
Test: `tests/smoke_122jk_landscape_from_selection.py` (52 checks);
smoke_122je allows the heatmap colour bar axes.

**122jj — Feature Visualization: swap axes, linear / log10.**
New "Axes:" row under the plot buttons: *Swap X ↔ Y* and an X and a Y scale
(linear / log10) for the displayed axes, applied to the plot on screen at
once from cached data (no re-read; cache cleared on Refresh). Swapping also
exchanges the two scale settings, so each quantity keeps its scale.
Distributions: values on x, density on y (swap → sideways histograms); a
log value axis uses log-spaced bins and leaves out values ≤ 0, counted in
the panel; AUC unaffected. Joint: the features exchange axes; log axes use
log-spaced bins, frames ≤ 0 on a log axis are counted in the title. Over
time: time on x (swap → time running down y, panels side by side);
double-click reads time from whichever axis holds it.
`plot_distributions` / `plot_joint` / `plot_over_time` take `swap`, `xlog`,
`ylog`.
smoke_122jj_feature_view_axes: 19/19. 122je 38/38, 122jg 20/20.

**122ji — Compute Features: ROI families show "modified" when the ROIs change.**
ROI-family outputs (distances to ROI centres, body parts inside ROIs) now
carry in their parquet metadata a fingerprint of the ROIs each video was
computed from (`RoiStore.fingerprint`: hash of names, shapes, geometry,
order; `write_with_roi_fingerprint` / `read_roi_fingerprint`). New
`feature_io.roi_family_status` compares outputs with rois.json: *changed*
(a ROI added, removed, renamed, moved or reshaped), *missing* (a video
gained ROIs after the run), *removed* (a video lost its ROIs), *unknown*
(output from before 122ji and older than rois.json). The Compute Features
form badges such a family "● … — modified" in amber with a tooltip listing
the videos per reason; it re-checks when rois.json changes on disk (file
watcher on the file and its folder, 300 ms debounce), when shown, and after
a run. Recomputing clears it. Subject families unaffected.
smoke_122ji_roi_features_modified_status: 16/16 (real ROI-family run;
add / move / remove ROIs; untracked old outputs; watcher; show; recompute).
Badge, Compute Features, ROI and feature-pool suites as before.

**122jh — ROI "Apply to all" copies only the selected ROI; Manage classifiers above Features.**
ROI editor: the button is now "Apply '<roi>' to all…", enabled only with a
ROI selected. It copies that one ROI to every other video, rescaled to each
video's frame size; a ROI with the same name there is replaced in place,
their other ROIs are kept, the source video is unchanged. The confirmation
names the ROI and lists videos that already have one of that name. New
`RoiStore.copy_roi` and `roi_utils.copy_roi_to_all`. The subset button is
relabelled "Copy all ROIs to chosen videos…" (behaviour unchanged).
Workbench: the Manage classifiers page is built right after ROI, above
Features (Annotation / Train / Validate / Inference still follow Features).
smoke_122jh_roi_apply_one_and_page_order: 18/18 (copy_roi in-place replace,
keep, add, rescale, error; editor: disabled without selection, question
text, decline, apply across a mixed-resolution project, status; page order
by AST). 122ij and 122dl updated to the new button behaviour / label; ROI
and page-order suites otherwise as before.

**122jg — feature tree compacted by shared name parts.**
Inside each family the wording all its features share is dropped from the
leaves and shown once beside the family (`Angle (degrees) …`,
`Animal_1 … video edge distance (mm)`), and features are grouped by their
first differing component — the first body part — with a second level when
a group would still hold more than 24 (455 three-point angles of 15 markers
→ 13 body-part groups → sub-groups by the second body part; 1365 four-point
hulls likewise). Singletons and small families stay flat. Groups and
sub-groups have their own none / some / all checkbox with counts; a tick
selects their shown features. Search matches feature, group and family
names. "Group by name parts" switches to flat leaves (still without the
shared wording). Expansion is remembered by node path across rebuilds and
view toggles. Search is debounced (200 ms); building / ticking a 2455-leaf
tree takes < 1 s / ~0.1 s.
Qt-free grouping in `feature_plots.compact_family` / `Node`.
smoke_122jg_feature_tree_compact: 20/20 (real feature-name shapes of the
15-marker layout: templates, grouping levels, loose leaves, every column
once; tree states, counts, group / family ticks under search, flat toggle,
expansion memory, debounce). smoke_122je_feature_visualization 38/38.

**122jf — Feature Visualization: features as a tree by family.**
The family drop-down and flat list are replaced by a tree: one bold branch
per feature family, its features as checkable leaves. A family's checkbox
shows none / some / all selected and its title the counts ("1/3
selected", "2/455 shown" while searching); ticking a family takes all its
*shown* features, so search + family tick selects a subset. Search matches
every word against feature and family names, hides families with no
match and expands the rest. Expand all / Collapse all; expansion and
selection survive rebuilds (video change, Clear selection).
smoke_122je_feature_visualization: 38/38 (tree structure, counts,
partial / full family states, family tick under a search, search over
family names, rebuild persistence).

**122je — Features → Feature Visualization.**
New section on the Features page (`FeatureVisualizationForm`). Pick a video,
filter features by family and by search words, tick any number (the
selection survives filtering), then:
*Distributions* — histogram per feature (≤ 12), this video or all videos
pooled; with *Colour by* a behaviour, labelled frames vs the rest with the
feature's AUC for that behaviour in each panel and the best-separating
features summarised (videos with no labels for it are left out of the split
and reported). *Joint* — two features as a 2-D log density with behaviour
frames overlaid. *Over time* — whole session per feature (min/max decimated)
with behaviour bouts shaded; double-click opens the video there. *Open with
video* — `FeatureViewerWidget` in a floating dock: the scrubber with one
trace lane per feature under it (each scaled to its 0.5–99.5 % range over
the session, value at the cursor beside the name), a behaviour strip,
click / drag to seek, wheel or the ±window box to zoom.
Only the selected columns are read: `feature_io.feature_catalog` (schemas
only) and `feature_io.load_feature_columns` (parquet column projection).
Plot drawing is Qt-free in `mufasa/tools/feature_plots.py` (`auc`,
`plot_distributions`, `plot_joint`, `plot_over_time`).
smoke_122je_feature_visualization: 31/31 (offscreen on a generated project
with real videos, two feature families and labels: catalogue / projection,
AUC vs sklearn, filtering and selection, every plot, scope and the
unlabelled-video rule, guards, viewer seek / drag / wheel / spin, page
registration). Page / section / feature-I/O suites as before.

**122jd — Frame Labeling table: "Frames labelled" column.**
New column after Status: frames carrying at least one behaviour, out of the
video's frames, with the percentage (a frame with two behaviours counts
once). The per-behaviour breakdown moves to "Per behaviour". Sorts by count.
`VideoAnnotation.any_labelled` in `mufasa.utils.annotation_status`.
smoke_122jc_labeling_status_complete_plot_seek: 32/32 (column position,
union counting, sorting, live refresh).

**122jc — Frame Labeling: annotation status table, "Mark video complete", click-to-seek in the label plot.**
Frame Labeling page: a table of every project video with its status
(✓ complete / ● in progress / ○ saved, empty / — not started), labelled
frames per behaviour (count and %), and last save; a summary line; numeric
sorting; Refresh, plus live refresh whenever a labeller saves or changes a
complete mark (`annotation_events()` signal). **Label** (or double-click)
opens the selected video directly; with no selection it still asks for a
file. Choosing *New labeling* for a video that already has labels asks
first and offers Continue (New would start empty and overwrite on save).
Labeller: **Mark video complete** checkbox next to Save, initialised from
and written to `derived/labels/annotation_status.toml` (new
`mufasa.utils.annotation_status`); checking it saves unsaved labels first.
Label plot: left-click (or drag) inside the lanes moves the video to that
frame, using the same window maths as the drawing; clicks on the name
column are ignored; seeks clamp to the video.
Two fixes found on the way: the TOML writer emitted every key bare, so a
key with a dot or space (a dotted video name, a classifier called "head
dip" in `[classifiers.keys]`) produced a project.toml that could not be read
back — keys are now quoted when needed (`_toml_key`); and the label /
feature / classification / CSV-export helpers stripped "extensions" with
`Path.stem`, so labels for a video named `rec.v2` were saved as `rec` — they
now strip only real video / table extensions (`strip_media_ext`).
smoke_122jc_labeling_status_complete_plot_seek: 30/30 (offscreen form +
labeller on real video files: table, sorting, launch / guard / double-click,
complete mark incl. save-first and live refresh, plot click / name column /
edge / drag / clamp, TOML keys, extension stripping). Labelling, label /
feature / classification I/O and project-layout suites as before.

**122jb — mufasa-landscape: angles stop taking over; family filters and caps.**
User report (after 122iz): still only angle pairs. With 15 markers the
three-point-angle family has one column per marker triple (455), and much
of their structure is circular in ways rotating the 0/360° cut cannot fix:
angles covering the whole circle (no empty direction) and mirror images
(bent left / bent right). Now:
`--angles unwrap` (default) also drops full-circle angles (emptiest
direction > 25 % of the mean density); `--angles fold` maps a → min(a,
360 − a) so mirror images coincide; `--angles drop`.
`--include-families` / `--exclude-families` (substrings of the family folder
name; `kfv2` = pose) restrict the screen.
Plot selection caps pairs touching any one family (`--max-per-family`,
default 6; animate default 2) on top of the per-column cap; the run says
when the caps leave slots empty.
`pairs.csv` gains `family_a` / `family_b`; the log counts multimodal pairs
per family.
smoke_122jb_landscape_angles_and_families: 14/14 (unwrap / fold / drop on
mirror, full-circle and wrap-around angles; a project with a 40-column
circular angle family beside distance / movement / head-velocity families
sharing a two-state structure: families spread in the plots, head velocity
among them, fold cuts angle×angle multimodal pairs 231 → 0, include /
exclude / drop). 122ix / 122iy / 122iz pass.

**122ja — fix: HEAD VELOCITY IN BODY FRAME failed with NameError 'warnings'.**
User report: every video failed with "NameError: name 'warnings' is not
defined". `feature_subset_kernels` imported `warnings` only inside the
fallback branch taken when the compiled (Cython) kernels are missing; the
122iz kernel used it at module scope. Without compiled kernels (the
development sandbox) the fallback ran and the name existed, so 122iz's tests
passed; on an installed copy the compiled kernels load and the name was
never bound. `warnings` is now imported at top level.
smoke_122ja_kernels_module_imports: 4/4 — AST check that every module global
a kernel function uses is bound on both import paths (would have caught
this), plus a run-time import with a stub `mufasa._native` (the compiled
path) and a head-velocity call; fails 1/4 on the 122iz file. 122iz, kernel
and Compute Features suites as before.

**122iz — head velocity in the body frame; landscape: angles, -1 placeholders, per-family log.**
User report: mufasa-landscape "only used the angles". All families were
loaded (checked on a real Compute Features output), but angle columns
crowded every plot for two artefacts the screen took for modes: angles wrap
at 0/360° (an angle near 0° splits into lumps at 0 and 360), and the feature
writer stores missing values as -1. Now `clean_columns` reads -1 in
non-negative columns as missing and rotates each angle column so its wrap
falls at its emptiest direction (still degrees); both are logged, and the
log lists columns kept / loaded per feature family (from the per-family
parquet schemas), plus which families lost columns missing in some sessions.
New feature family **HEAD VELOCITY IN BODY FRAME (MM/S)**
(`compute_head_velocity_body_frame`): head point = mean of the head markers,
body point = mean of the body markers, axis = body → head; the head point's
velocity (central differences, mm/s from px/mm and fps) split into
*forward* (along the axis) and *lateral* (to the animal's left, image y
down). Markers from project.toml `[features] head_markers/body_markers`,
else `[[pose.segments]]` (segment `head` / `[features] head_segment`, root
segment / `body_segment`), else marker names; the run prints which and from
where, and an unresolvable set raises saying what to add. Signed, so its
parquet keeps NaN instead of the -1 placeholder (`SIGNED_FAMILIES`). Listed
in Compute Features automatically. `[features]` documented in
docs/v1_project_layout.md.
smoke_122iz_head_velocity_and_landscape_columns: 24/24 (kernel geometry:
forward/lateral sign and magnitude, |v|² = f² + l², rotation invariance,
missing markers; resolution order and errors; a real FeatureSubsetsCalculator
run on a [[pose.segments]] project; the form's family list; angle unwrap,
-1 handling, family log). smoke_feature_kernels' documented-kernel list
gains the new kernel. Feature/landscape suites as before.

**122iy — `mufasa-landscape-animate`: frames gliding between t-SNE and feature spaces.**
New `mufasa/tools/landscape_animate.py` (console script
`mufasa-landscape-animate`) renders a presentation GIF (`--mp4` adds an MP4)
from a `mufasa-landscape` run: the sampled frames move from the t-SNE map
into the joint scatter of each chosen pair and back (t-SNE → p1 → p2 → … →
t-SNE, or `--through-tsne` to return between pairs), with a hold on each
space and cubic ease-in-out moves. Colour stays fixed per point — by default
its position in the t-SNE map (hue = direction, saturation = distance from
the centre), so each map region can be followed into every feature space;
`--color session|<column>` otherwise. Each space is shown in its own units
(0.5–99.5 percentile range; t-SNE keeps its aspect); title, axis labels,
ticks and a faint density backdrop cross-fade with the motion. Pairs: `--top
N` from `pairs.csv` (same ranking, `--max-per-feature` default 1) or
`--pair A B` (repeatable). One shared palette for all GIF frames (no
flicker); `--theme dark|light`, `--size`, `--n-points`, `--fps`, `--hold`,
`--transition`. Run found from a run dir or the project (latest run).
`mufasa-landscape` now also writes `tsne_samples.parquet` (the embedded
frames' column values, row-aligned with `tsne.parquet`), which the
animation reads — no recomputation; runs from 122ix lack it and get a
clear "re-run mufasa-landscape" error.
smoke_122iy_landscape_animate: 20/20 (timeline and easing, GIF size / loop
/ duration, start = end, exact point positions at the end and middle of a
move, aspect, colour modes, --top / --through-tsne / project discovery,
MP4, error paths). 122ix, 122iw, 122gs pass.

**122ix — `mufasa-landscape`: dipsplit joint PDFs and t-SNE across sessions.**
New `mufasa/tools/feature_landscape.py` (console script `mufasa-landscape`,
needs the new `analysis` extra = `diptest`). Loads, for every session, the
latest Kalman v2 run's smoothed pose (x/y, prefixed `kfv2.`; `--pose-cols all`
adds p and variances) and all feature families, samples
`--frames-per-session` frames from each, and keeps the columns all sessions
share. Sessions lacking a requested source (`--sources`) are skipped and
named; pose/feature length mismatches are reported and truncated.
Screen: columns that are mostly NaN, near-discrete (flags such as inside-ROI
0/1, which are trivially "bimodal") or near-duplicates (|r| > 0.98) are
dropped; every remaining pair is robust-z-scored and projected on the two
axes, both diagonals and both principal axes. dipsplit counts modes per
projection: Hartigan's dip test (p ≤ `--alpha` and dip ≥ `--min-dip`, an
effect-size floor because at tens of thousands of frames the test alone
flags skewed/heavy-tailed unimodal data) decides, the deepest valley of the
smoothed density is the cut, both halves are re-tested. Pairs are ranked
emergent (both columns unimodal, a joint projection not) → adds modes (more
modes jointly than in either column) → by dip. Plots: joint log-density with
marginals and split boundaries (joint splits drawn as lines in raw units) for
the top `--n-plot` pairs, at most `--max-per-feature` per column
(`jpdf_top.png`, `jpdf_pairs.pdf`); t-SNE (PCA to 50, perplexity 30) on
`--tsne-sample` frames coloured by session and by the columns most frequent
among the plotted pairs (`tsne.png`, `tsne.parquet`). All screened pairs in
`pairs.csv`; provenance in `run.toml`; output under
`derived/plots/feature_landscape/<run_id>/`. About 3 ms per pair on 20k
frames (300 columns ≈ 2.5 CPU-minutes, parallel over `--n-jobs`).
smoke_122ix_feature_landscape: 24/24 (dipsplit mode counts on normal,
lognormal, t3, uniform, 2- and 3-component mixtures; a generated project
with a planted emergent pair, a bimodal column, a 0/1 flag, an orphan
session, a stale run and a length mismatch; every output). README command
and extras tables updated (122iw tripwire passes).

**122iw — README brought up to date with the project.**
The README described an older Mufasa. Corrected against the code:
license (said GPL v3; LICENSE is SimBA's BSD 3-Clause, modified, academic
use only — badge and section fixed); install (`pip install -e .` gave no Qt —
PySide6 is in the `qt` extra — now `.[qt]`, with the other extras listed and
the Cython build noted; clone URL Gravios/mufasa; TensorFlow mention
dropped); commands (`mufasa-preview` added, removed-command history cut to
one line); a table of the workbench pages; the standalone Kalman v2 command;
project tree (`derived/pose|interpolated|smoothed/kalman_v2/<run_id>`,
publishing into `outlier_corrected`, per-family features, flat labels /
classifications, `models/` for Kalman models + classifiers,
`logs/measures/rois.json`, `backups/`); importers (FreeDLC first); docs
table (Kalman design docs, testing workflow, CHANGELOG); "Relationship to
SimBA" now says how Mufasa diverged (own formats, no SimBA projects) instead
of "core algorithms are identical"; citation asks for SimBA for the
classifier pipeline. `docs/v1_project_layout.md` got the same tree, a
corrected `derived/` table and `models/` section, rois.json, and lost
"Both layouts remain supported".
smoke_122iw_readme_current: 22/22 — each README fact checked against its
source (pyproject scripts and extras, page titles, import form, layout code,
LICENSE, smoother flags, links). 122df's GPL check now checks the README
names LICENSE's license. README/doc tripwires pass; 122cn as before.

**122iv — banner tool icons head their doc sections.**
The banner's three tool tiles are standalone SVGs: `docs/images/icon_smoothing.svg`,
`icon_roi.svg`, `icon_classifiers.svg`. As in neurosuite-3's README, a new
"The tools" section has one collapsible `<details>` block per tool — icon and
bold name in the `<summary>`, four bullets, doc links. The icons also head
the matching sections of `docs/workflows.md` (Smooth pose; Stages 4–8) and
the Kalman design track in `docs/README.md`. Links into workflows.md point at
the file, not at heading anchors, since an image in a heading changes
GitHub's anchor.
smoke_122iv_docs_section_icons: 9/9. Doc tripwires unchanged (122cn fails
as before).

**122iu — README / docs banner (wide + mobile, dark + light).**
New `docs/images/`: the Mufasa banner in the neurosuite-3 layout and the lion
palette — `mufasa_banner_{dark,light}` (2560x800), a stacked narrow-screen
version `mufasa_banner_mobile_{dark,light}` (1200x820) and a GitHub social
preview (1280x640), each as PNG plus SVG source. The README and the docs
index open with a `<picture>` that picks mobile vs wide by viewport width
(`max-width: 700px`, listed first because the first matching source wins)
and light vs dark by `prefers-color-scheme`; the dark wide banner is the
fallback `<img>`. Tagline: "Kalman pose smoothing and animal behavior
analysis from video pose-tracking data".
smoke_122iu_readme_banner: 11/11. 122cn (docs index lists 5 unlisted docs)
and 122br (SimBA.py gone) fail as before.

**122it — Mufasa lion logo: app icon and startup splash.**
New `mufasa/assets/branding/`: the lion-head symbol (`mufasa_icon_<16…512>.png`,
`.svg`, `.ico`, transparent background) and the symbol + MUFASA wordmark
(`mufasa_splash.png`, `mufasa_wordmark.svg`). New `mufasa.ui_qt.branding`:
`app_icon()` (all sizes in one QIcon), `splash_pixmap()` (opaque dark card —
no reliance on window translucency, which remote X11 often lacks),
`MufasaSplash`, `show_splash()` (skipped with MUFASA_NO_SPLASH=1 and on
offscreen/minimal platforms). The workbench entry point sets the app icon
and desktop file name, shows the splash with "Opening <project>…" while the
workbench is built and closes it when the window appears; the main window
icon is the lion instead of the SimBA logo; `Paths.LOGO_ICON_*` point at the
new files.
smoke_122it_branding: 10/10 (assets + transparency, icon sizes, splash,
a real workbench window's icon, entry-point wiring). 122d5 module ceiling
432 → 433. Workbench suites unchanged.

**122is — parallel feature extraction: spawn workers, not fork.**
User report: Compute Features with 16 workers → every video "BrokenProcessPool:
A process in the process pool was terminated abruptly". Reproduced here: the
workers print "Terminating: fork() called from a process already using GNU
OpenMP, this is unsafe." `FeatureSubsetsCalculator._run_parallel` used a
plain ProcessPoolExecutor (Linux default start method: fork) from a process
that had already run numba-parallel kernels (OpenMP threading layer), and a
forked child of a process with live GNU OpenMP threads aborts; one dead
worker breaks the whole pool. It also forked the Qt GUI process from its
worker thread. The pool now uses `mp_context=get_context("spawn")`; the
console-script launchers guard `main()`, so spawned workers don't re-run the
app. Not a leftover pool from an earlier attempt — each run makes a new one.
smoke_122is_feature_pool_spawn: 4/4 — OpenMP threads started first, then a
2-worker run, output identical to sequential (~16 s). On the fork pool:
0/4 with the same "fork() … GNU OpenMP" abort. Feature suites unchanged.

**122ir — published Kalman runs appear under video names.**
User report (via 122iq's new message): ROIs defined for 39 videos
(`K7_cam_…-triplet`), tracking files named `K7_cam_…-triplet_smoothed_v2.
4d4273a837d0cc40`. Kalman v2 writes `<video>_smoothed_v2.<hash>.parquet`;
`publish_to_stage` linked the whole run directory into
derived/outlier_corrected/, and every downstream tool (features, ROI
analysis, classifiers, plots) takes the file name as the video name — so
nothing keyed by video (ROIs, video_info calibration) matched. Only the
egocentric aligner had a private suffix-stripping helper.
* `project_layout.video_name_from_pose_stem` (moved from the egocentric
  aligner, which now uses it).
* `publish_to_stage`: when a run's pose files carry the smoother suffix, the
  published run is a real directory of relative per-file links named
  `<video>.parquet` (+ a `.published_from` marker), built in a temp dir and
  swapped in; idempotent; other runs keep the plain directory link; real
  data directories are still never clobbered.
* `refresh_published_runs`, called by ConfigReader when a project is
  opened, upgrades runs published the old way (links only, never data).
smoke_122ir_published_video_names: 12/12 — the user's layout (directory
symlink to a run of `_smoothed_v2.<hash>` files), ROI + calibration features
computed for every video with correct values. Tripwire: without the refresh
on open the publish checks fail. Full sweep: no regression.

**122iq — ROI-feature "no coverage" error says why.**
User report: "NO ROI DATA ERROR: … none of the 39 videos have all required
ROIs defined." ROI features need, for every tracking file, a video with the
same name in rois.json carrying every ROI name used in the project (each ROI
is a feature column). The message didn't say which condition failed. Now
`_explain_no_roi_coverage` reports either that no ROI video name matches a
tracking file (showing both sides' names), or the project's ROI names and
what each video lacks (with the fix: rename in the ROI table, or Apply to
all / selected) — the latter is easy to hit with the editor's automatic
names.
smoke_122iq_roi_coverage_message: 5/5 (real projects, FeatureSubsetsCalculator
with the ROI families: name mismatch, inconsistent names, consistent names →
features written). On the previous code 1/5.

**122ip — the calibration is actually saved.**
User report: calibrated in Preprocessing → Video Calibration, yet Compute
Features said video_info.csv was missing. `VideoInfoForm` inherited the
standard Run button; Run called its no-op `target()` and reported "Done.",
while only a separate "Save to video_info.csv" button in the action row wrote
the file (the form's comments claimed there was no Run button and that
on_run was overridden — neither was true). Now:
* Run is labelled "Save calibration" and saves (`on_run` → `_save`).
* Edits and per-row calibrations flag "● Unsaved changes" with the target
  path until saved; a fresh load is not an edit.
* The missing-file error (read_video_info_csv / ConfigReader) names
  Preprocessing → Video Calibration instead of SimBA's "[Video parameters]
  tab"; stale logs/video_info.csv wording in the form fixed.
smoke_122ip_calibration_save: 12/12 (form offscreen on a generated project,
saved CSV read by read_video_info_csv and ConfigReader). Tripwire: removing
only the Run→save override reproduces the report (file never written).
smoke_video_info_form, 122ag, 122er unchanged.

**122io — Compute Features runs again (shap, numba cross2d, ROI shape types).**
User report: Compute Features → "preflight failed … No module named 'shap'".
* `train_model_mixin` imported `shap` at module level; shap was dropped from
  the dependencies, and feature computation (`FeatureSubsetsCalculator`)
  inherits the mixin. shap is now imported inside the SHAP helper only
  (annotations via `TYPE_CHECKING`). Feature computation, training,
  inference and result plotting all import without it.
* `perimeter_jit` imported `numba.np.extensions.cross2d`, gone in numba 0.68
  (pyproject allows any numba ≥ 0.60) — the convex-hull families failed to
  import. The 2-D cross product is written out; results match scipy's
  ConvexHull to 1e-11.
* ROI tables said `Shape_type` "Rectangle" / "Circle" / "Polygon" (inherited
  from the old ROILogic writer); the inside-ROI feature kernel and the ROI
  classifier code compare to SimBA's lower-case values, so "body-parts
  inside ROIs" silently wrote no file for Mufasa-drawn ROIs. `roi_store` now
  emits lower-case; `ROI_directing_analyzer` (the one consumer expecting
  "Circle") compares case-insensitively.
smoke_122io_compute_features: 8/8 — shap blocked via a meta-path finder,
a generated project with ROIs, preflight + every family, inside-ROI booleans
checked against the geometry. Tripwires: top-level `import shap` and
capitalised shape types each fail it (cross2d is absent from this numba, so
the old perimeter_jit fails too). Full sweep: no regression.

**122in — a click is not a drag (ROI cycling with a real mouse).**
User report after 122im: the ROI under another still couldn't be selected.
Cause: no drag threshold. A real click moves the cursor a pixel or two
between press and release; the Select tool treated that as a drag, nudged
the selected ROI by that amount and, since it "moved", never cycled to the
ROI beneath. 122im's test used QTest.mouseClick, which never moves, so it
passed. Moving / reshaping now starts only once the cursor has travelled
`QApplication.startDragDistance()` (default 10 px); below that the press is
a click.
smoke_122im_roi_overlap: +4 checks with realistic clicks (2 px jitter):
cycling works, ROIs and sides are not nudged. On the 122im canvas those
fail (12/17); now 17/17. 122ij, 122dm, 122fd, smoke_roi_canvas unchanged.

**122im — overlapping ROIs stay reachable.**
The Select tool resolved a click to the top-most (most recently drawn) ROI
whose area contained it, so an ROI under a newer one could not be selected,
and its sides / polygon vertices could not be dragged where the newer ROI
covered them. `ROICanvas._hit_test` priority is now:
1. handles of every ROI (polygon vertices, rectangle corners / side
   midpoints, circle rim handle), the selected ROI's first;
2. the nearest outline of any ROI — rectangle sides (resize that side),
   circle rims (radius), polygon edges (new: drag moves both vertices);
3. interiors, smallest area first; clicking the same spot again (without
   dragging) cycles through the overlapping ROIs.
smoke_122im_roi_overlap: 13/13 (editor offscreen via QTest: small rectangle,
polygon and circle under a newer large rectangle). Tripwires (top-most
interior wins, no outline priority, no cycling) each fail it. 122ij, 122dm,
122fd and smoke_roi_canvas unchanged.

**122il — ROIs stored in `logs/measures/rois.json`; SimBA H5 dropped.**
The project's ROIs move from SimBA's `ROI_definitions.h5` (three pickled HDF5
tables: Python/pandas-version fragile, needs pytables, opaque, stores
derived values) to one JSON file. No legacy compatibility, by request: the H5
is neither read nor written.
* New `mufasa/roi_tools/roi_store.py`: `Roi` / `VideoRois` / `RoiStore`;
  per video `frame_size` + ROIs in drawing order; geometry + style only
  (rectangle x/y/w/h, circle center/radius, polygon vertices — any number —,
  `#rrggbb` colour, line width, tag size). `load_rois` (invalid file →
  `RoiFileError`, never "empty"), `save_rois`, `update_rois` (locked
  read-modify-write, atomic temp+rename), `rois_to_frames` (the three
  analysis tables with every derived column computed on read, cm sizes from
  the CURRENT video_info pixels/mm), `frames_to_store` / `write_frames`.
* `project_paths_from_config()["roi_definitions_path"]` → `rois.json` (v1
  and legacy layouts); section provenance and `Paths.ROI_DEFINITIONS` too.
* Readers: `read_write.read_roi_data` and `ConfigReader.read_roi_data` read
  rois.json, so every analysis / plotting / feature module keeps working
  unchanged (verified: ROIAggregateStatisticsAnalyzer on a 30-vertex polygon,
  a circle and a rectangle gives the right times).
* Writers: `ROILogic.save` (editor) is a locked update of its own video —
  two editors open at once can't drop each other's work; `multiply_ROIs`
  now rescales copied ROIs to each target's frame size; `reset_video_ROIs`,
  the duplicate dialog, CSV import (append keeps other videos) and the size
  standardizer all write through `write_frames`.
* Removed the 122ik SimBA-row normalisation (and its test) — moot without H5.
smoke_122il_roi_store: 21/21 (real project with two resolutions, two
editors at once, ConfigReader, multiply with rescale, reset, write_frames,
corrupt file, a real ROI aggregate analysis). Tripwires (save without
re-read, no rescale, bad file read as empty) each fail it. Path pins updated
in 122ai/122eh/122en/122ep/layout_paths_extension/roi_definitions_description/
roi_video_table_helpers; 122ek/122er now pin the store-based multiply/reset;
smoke_roi_logic loads roi_store and replaces its H5-specific cases; 122d5
module ceiling 431 → 432. Full sweep: no other change.

**122ik — saving ROIs no longer loses other videos' ROIs.**
`ROILogic.save` rewrites the whole ROI_definitions.h5. It read the existing
file through `read_roi_data` (drops every rectangle / circle row with a NaN),
swallowed read errors and then wrote a file holding only the current video,
and wrote its own rows in a private format: `Color BGR` / `Tags` / `vertices`
as strings, circles with extra `Center_X` / `Center_Y` columns. Mixed with
SimBA-format rows that left NaN cells, so the next read dropped rows —
measured: saving one circle on video A deleted video B's SimBA circle on the
first save. String vertices / tags also broke ROI analysis
(`vertices.astype`) and visualisation (`Tags.items()`).
* Save reads the file raw; an unreadable file aborts the save
  (`RuntimeError`) instead of being replaced.
* Every row — new and existing — is written SimBA-native (tuple colour, dict
  tags incl. the full rectangle tag set and circle 'Border tag', ndarray
  vertices) with SimBA's column set; missing standard values are filled;
  an extra column is kept only if no row lacks it. Old-Mufasa rows are
  repaired on the next save.
* Load reads raw as well and normalises each row, so mixed files show every
  ROI; an unreadable file sets `ROILogic.load_error` and the panel warns.
smoke_122ik_roi_save_preserves: 15/15 (real H5 files in SimBA and old-Mufasa
format, ROILogic load/save, read_roi_data, multiply_ROIs, corrupt file).
Tripwires (swallowed read error, extra NaN columns, string vertices) each
fail it. smoke_roi_logic: added its missing `import os`; its cross-video save
case (19), which had only been skipping for lack of pytables, fails on the
old code and passes now (20/20).

**122ij — ROI editor redesign; polygons have no vertex limit.**
* Bug: `ROICanvas._commit_polygon` ran every polygon through
  `shapely.simplify(tolerance=20)`, silently deleting each vertex within 20
  frame px of the line through its neighbours. On a 640×360 frame that capped
  polygons at ~4–8 vertices however many were clicked (measured: a 30- or
  60-vertex ring of radius 80 px → 8). Vertices are now kept as placed; only
  exact consecutive duplicates are dropped.
* Logic: Esc now abandons a polygon in progress (it used to commit it with
  ≥3 vertices). Finish with a click on the first vertex, double-click or
  Enter; Backspace / right-click removes the last vertex.
* Workflow: no more "type a name → pick a tool → press Draw → draw one shape".
  Tools (Select V / Rectangle R / Circle C / Polygon P) are one exclusive group
  and drawing tools are sticky; new shapes are named from an optional Name box
  or automatically (`polygon_1`…). The separate Edit toggle that disabled Draw
  is replaced by the Select tool.
* Editing: rectangles get edge-midpoint handles too; polygon vertices can be
  dragged, inserted (double-click an edge) and removed (right-click); arrow
  keys nudge (Shift: 10 px). Name (double-click), colour and line width are
  editable per row in the table (`ROILogic.update_style`). Duplicate / Delete
  buttons; undo / redo (Ctrl+Z / Ctrl+Shift+Z, `ROILogic.snapshot/restore`).
* View: wheel zoom about the cursor, middle-drag pan, F to fit; ROI names drawn
  on the frame (toggle); live size / radius / vertex-count readouts; a hint
  line says what the current tool expects.
* Layout: three panes — videos (with filter) | tools + canvas + frame nav |
  ROI table, new-shape style, copy-to-videos, import, clear; unsaved-changes
  indicator, Save (Ctrl+S). `,` / `.` step frames.
* "Apply to all…" now confirms and names how many other videos' existing ROIs
  `multiply_ROIs` will replace (it rewrites the whole H5). Import /
  standardize now reloads the current video so imported ROIs appear.
smoke_122ij_roi_editor: 32/32 — real project, generated videos, the widget
offscreen driven by QTest mouse/key events, saved H5 read back by a fresh
ROILogic. Tripwires (simplify restored, Esc commits, no undo, no apply-all
confirmation, non-sticky tool) each fail it. smoke_122dm updated: the Edit
toggle checks now pin the Select tool; move/resize bodies moved into
`_move_selected_to` / `_drag_handle_to`. Other ROI suites unchanged.
Note: smoke_roi_logic case 19 had been skipping for lack of pytables in the
sandbox; with pytables installed it fails (missing `import os`, then
`other_video_rois` empty in its stub harness) on the old roi_logic as well —
pre-existing, not touched here.

**122ii — the joint-prior fit uses the worker pool.**
`fit_joint_priors_v2` accepted `pool` and ignored it: its pass (a filter +
smoother per session, the cost of one EM E-step) ran every session serially in
the main process on one core, while the perspective pass just before it used
all workers. EM also called it with `verbose=False`, so it sat silent. The
per-session work is now `_joint_prior_session_samples`, shared by the serial
path and a new `_pool_joint_prior_pass` worker; results are accumulated in
session order, so the pooled fit is identical to the serial one. Progress
prints per session (`[v2-jprior] n/N … ETA`).
smoke_122ii_joint_prior_pool: 5/5 (real multiprocessing pool via
`_pool_init`; pooled == serial exactly; nothing runs in the main process).
Tripwire: forcing the serial path fails it. smoke_122hf's source check now
reads the helper. Every Kalman v2 suite passes.

**122ih — no blocking progress dialog.**
Running an operation opened a window-modal QProgressDialog that froze the whole
workbench — including the console dock showing that run's own output (e.g. the
Kalman v2 EM iterations), which couldn't be scrolled until the run ended.
`run_with_progress` now shows progress inline in the main window's status bar
(label, bar, Cancel; busy until the first progress report) and opens no dialog.
Parents outside a QMainWindow get a NON-modal dialog. Since nothing is modal,
`OperationForm._lock_run_button()` disables Run while the form's operation is
in flight (base `on_run`, Features, Video filters); the heatmap popup disables
its two run buttons the same way. New `on_finished` callback on
`run_with_progress`.
smoke_122ih_inline_progress: 18/18 (real OperationForm in a QMainWindow,
offscreen, blocking worker). Tripwires: modal fallback, no status-bar path,
no Run lock, indicator left behind — each fails. Suites touching the
workbench/progress/forms: no change vs the previous sweep.

**122ig — root orientation from the trunk; root-noise ceiling as a setting.**
* Bug: the root's initial orientation and the initial q_root_ori estimate read
  the legacy rig's `back1`/`back3` by name. Any other rig started at identity
  (up to 180° off the frame its root offsets are fitted in) with q_root_ori at
  the default 1.0 — and the M-step ceiling is 10× the initial value, so the
  fitted q_root_ori pinned at exactly 10 (seen in every strohA run). Both now
  come from the root cluster's own body axis (`_root_axis_per_frame`, the same
  widest-chord axis the root offsets use); back1/back3 only as fallback.
* All whole-body motion runs through the root's constant-velocity model, so on
  bursty movement EM pins q_root_pos / q_root_ori at the 10× ceiling within a
  few iterations. New `BodyLayout.root_q_ceiling_factor` (default 10 = prior
  behaviour), `--root-q-ceiling`, and "Root noise ceiling" in the Kalman v2
  form; absolute hard caps unchanged.
Measured on 6,000 frames of the user's strohA-xw session (drift on, back,head
orientation drift, head high-angular, joint prior, threshold 0.5, 15 EM
iterations): 10× -> median offset 2.36 px, worst marker 6.7 px; 50× -> 2.09 px,
6.7 px, q_root_pos at the 50,000 hard cap, no instability. EM still does not
converge (q_root_ori keeps rising) — a model mismatch with bursty motion, not
fixed here.
smoke_122ig_root_axis_and_ceiling: 19/19 (real functions, real median M-step,
form offscreen). Tripwires: identity start, default q_root_ori, ceiling
ignoring the layout, hard cap removed, GUI default changed — each fails.
Canonical rig 152/152.
smoke_122hf_joint_prior: the near-uniform-tail check compared the full prior
set on vs off and passed at <2° only because both runs sat at chance (~105°).
With q_root_ori now data-estimated (~9 vs 1.0 in that rig) the ACTIVE
back_rear/neck priors steer the unobservable 20 s tail_3 walk onto another
chance-level path (57° vs 114°; hold-last scores 104°, circular mean 109°). The
check now tests what it meant — tail_3's own inert prior alone changes nothing
(bit-identical) — plus a sanity bound (<130°). 43/43.

**122if — remove sessions via the Data Import table.**
The "Already imported" table (Data Import -> Import video) gains a "Pose data"
column (the imported pose file(s) with the video's stem), multi-row selection,
and *Remove selected…* — button, Delete key, or right-click. Confirmation lists
the sessions, with "Also remove their imported pose data" (ticked when there is
any). New Qt-free `mufasa/model/session_removal.py` (`session_files`,
`remove_sessions`): nothing is deleted — the video (a symlink as the link; the
footage is never touched), the pose file(s) if chosen, and the session's
`video_info.csv` row go to `backups/removed_sessions-<run_id>/` (rename on the
same disk, so instant for large videos) with a full `video_info.csv` copy and
a `manifest.toml` of from -> to. Derived outputs are left alone.
smoke_122if_remove_sessions: 21/21 (real backend on a ProjectConfigCreator
project; the real form offscreen, message boxes stubbed). Tripwires: delete
instead of move; following the symlink; video_info not cleaned; pose box
ignored; single selection; placeholder row removable — each fails.
mufasa/**/*.py 430 -> 431 (smoke_122d5 ceiling bumped).

**122ie — FreeDLC parquet without `.fdlc` naming; pose-file autodetect says parquet.**
FreeDLC (the modified DeepLabCut) now writes `pose.parquet` per run + a run
`.toml`, skeleton embedded — not `<video>.fdlc.parquet` + `.fdlc.toml`.
Nothing requires the old naming any more (it is still read, as legacy):
* one parquet import route ("FreeDLC parquet (single animal)"; 122ib's
  separate "DLC parquet" entry folded back), hint describes run folders;
* Add-sessions file filter and ingest suffix list drop `*.fdlc.parquet`
  (any long parquet is accepted by schema); importer/preview docs and the
  "nothing found" message describe the run-folder layout.
* Create-project "Auto-detect from pose file…" (renamed from "…DLC file"):
  dialog lists `Pose files (*.parquet *.h5 *.csv)` with parquet first; the
  skeleton filter is the shared `pose_parquet.skeleton_edges_for`.
* File -> Reconfigure: its text and placeholder still said ".h5 or .csv";
  now parquet too, filter `Pose files (*.parquet *.h5 *.csv)`. (Its apply step
  is the legacy INI-only `project_reconfigure` — unchanged, pre-existing.)
* Edit-metadata autodetect error text mentions parquet.
smoke_122ie_freedlc_no_fdlc_names: 12/12, driving the real create-form and
reconfigure buttons (dialogs stubbed, Qt offscreen) with a run's pose.parquet,
with and without embedded metadata. smoke_122dj route count back to 12.

**122id — pair a run's pose.parquet with its video through the run sidecar.**
The DeepLabCut exporter writes a fixed-name `pose.parquet` per analysis run
with a run TOML beside it (named after the video): `kind = "analyze"`,
`inputs = ["…/<video>.mp4"]`, `outputs = ["pose.parquet"]`, model/snapshot
provenance. No skeleton in it. `pose_parquet.read_run_sidecar` finds the
record whose `outputs` names the file (other TOMLs ignored);
`pose_parquet_video_name` returns the recorded video's stem first.
* Discovery descends one level into run folders (only where a run sidecar
  names a long parquet) — importer and Add-sessions alike.
* Importer names each import after the run's video and refuses two files that
  map to one video (they would overwrite each other).
* Smoother names output `<video>_smoothed_v2.<hash>.parquet` for run files
  (`_session_output_stem`, shared by writer and incremental skip); other
  files keep their names.
* mufasa-preview pairs the run with the recorded video even outside the folder.
smoke_122id_run_sidecar_pairing: 17/17 (real importer, writer, preview).
Tripwires: name ignores sidecar, no run-folder descent, no clash check,
smoother stem from file name, preview ignores sidecar — each fails. Real data:
the user's run TOML beside a slice as `pose.parquet` -> named
`K7_cam_20251124_1510_A10-0-051_640x360-triplet`, found from the parent folder,
smoothed standalone to `K7_cam_…_640x360-triplet_smoothed_v2.<hash>.parquet`.

**122ic — processing and viewing use the data's own skeleton.**
`pose_parquet.resolve_skeleton(paths, config=None)`: explicit project.toml, then
the pose files' own skeleton (embedded / `.fdlc.toml`; a folder is probed
through its long parquets), then the project enclosing the files.
* Smoother: `layout_from_fdlc_sidecar` reads the embedded skeleton too (and a
  folder of agreeing files; nodes taken from the edges if `bodyparts` is
  absent), so `--beside-input` on a standalone DeepLabCut parquet builds its
  tree with no project. Imports the light module, not the importer.
* Smoothed output keeps the input's skeleton (`b"pose"` metadata, edges
  between its markers), so a standalone smoothed file stays self-describing.
* Overlay viewer: draws the resolved skeleton (new `--config`) instead of
  `DEFAULT_SKELETON_EDGES`, the legacy rat-rig names, which drew nothing for
  any other rig. pose_viewer auto-resolves when `--config` isn't given.
* Fix: pose_viewer's direct-read loader blanket-lower-cased columns (the
  122hb bug, flagged in 122hx), so every smoothed file's back_T4 became
  back_t4 — smoothed markers stopped matching the raw file and the skeleton
  (10 of 24 edges drawn on the real rig). Now suffix-only normalisation.
smoke_122ic_skeleton_downstream: 20/20 (real writer, real overlay main()
offscreen). Tripwires: layout ignores embedded, writer drops skeleton, loader
lower-cases, overlay ignores resolution, no project walk-up — each fails.
Real data: a 1,500-frame slice of the user's export smoothed standalone
(`--beside-input`, no project) — layout "from the pose files' own skeleton
(15 markers)", output carries 24 edges; overlay draws 24/24 raw and 24/24
smoothed lines (10/24 smoothed before the case fix).

**122ib — DeepLabCut long-format parquet with an embedded skeleton.**
The modified DeepLabCut exporter writes the same long table as FreeDLC
(frame | individual | bodypart | x | y | likelihood) but names it
`<video>.parquet` and carries the skeleton inside the file: parquet key/value
metadata `b"pose"` = JSON `{"version", "bodyparts", "skeleton"}` (DLC's own
config.yaml keys; edges as name or index pairs). New dependency-light module
`mufasa/pose_importers/pose_parquet.py` is the shared layer: schema-based
detection (`is_long_pose_parquet`, non-recursive `find_long_pose_parquets`),
`read_pose_skeleton` (embedded first, then the FreeDLC `.fdlc.toml` sidecar),
`pose_parquet_bodyparts`, `embed_pose_metadata` + a CLI
(`python -m mufasa.pose_importers.pose_parquet FILE... --skeleton-from
project.toml|config.yaml|x.fdlc.toml|x.parquet`) to stamp existing files.
`read_fdlc_skeleton` moved here (re-exported by the importer) so the smoother
and viewers no longer drag in the importer's h5py chain.

* Import: FDLCParquetImporter finds long parquets by schema, so `<video>.parquet`
  is picked up beside `*.fdlc.parquet` while wide/smoothed/unrelated parquets
  are ignored; the video stem works for both names; the embedded skeleton is
  written to project.toml [skeleton]; new `file_paths=` imports exactly the
  given files. "DLC parquet (single animal)" added to the pose-import list.
* Project creation: "Auto-detect from DLC file" accepts .parquet
  (`extract_bodyparts_from_parquet`), keeps the file's skeleton (edges between
  its own markers) and writes it to the new project's [skeleton].
* Add sessions: long parquet routed by format, not by the `.fdlc.parquet`
  suffix — a `<video>.parquet` used to be copied in raw as if already wide.
  The importer now receives the accepted files only, so a rejected file in the
  same folder is no longer imported with them (that aborted the whole ingest).

smoke_122ib_dlc_parquet: 37/37, run for real (pyarrow, the importer, the create
form offscreen, session ingest). Tripwires: suffix-only discovery, embedded
skeleton ignored, suffix-based ingest routing, whole-folder ingest import,
form not writing the skeleton, parquet missing from autodetect — each fails the
suite. Real data: the user's 108,030-frame × 15-marker export, with the
project's 24-edge skeleton embedded (0.3 s, data byte-identical), imports to a
108,030 × 45 wide file and a 24-edge project [skeleton]. Test harnesses of
smoke_122gm / smoke_122gn now take `_LONG_COLUMNS` / `read_fdlc_skeleton` from
the new module. mufasa/**/*.py 429 -> 430 (smoke_122d5 ceiling bumped).

**122ia — scale pose coordinates for reduced-resolution video.**
Pose estimated on a downscaled video (e.g. *_reduced_negate.mp4) is in
reduced-resolution coordinates; overlaid on the full-res video the markers
cluster in a corner. A --pose-scale FACTOR multiplies pose coordinates by the
factor — 2 for a half-resolution pose, 4 for quarter — so they line up. The
factor is also a live Scale selector in the viewer (presets 1×/2×/4×, editable
for any value), so which factor aligns can be eyeballed without a restart.

Scaling applies to marker positions, skeleton endpoints, and the
variance-ellipse *centre and radii* — all in pose-pixel space, so a coordinate
rescale scales the ellipse's std-dev linearly too. It does NOT scale the marker
dot size, which is a fixed on-screen size (the dot's own geometry, set once;
only setPos is scaled). Threaded through OverlayScene (pose_scale + set_pose_scale),
the overlay CLI, and mufasa-preview (--pose-scale passthrough, omitted when 1).

smoke_122ia_pose_scale: 23/23 — the scaling arithmetic (positions ×s, endpoints
×s, ellipse centre+radii ×s, s=1 a no-op) and the scale-label helper run in
isolation; AST confirms the dots/skeleton/ellipse scale by s while the dot
radius definitions carry no ×s, the set_pose_scale method, the CLI arg, the live
Scale selector (with a non-numeric entry reverting rather than crashing), and
the mufasa-preview passthrough. Tripwires: unscale the dot positions -> 22/23;
scale the dot radius -> 22/23 (the check pins the radius lines); unscale the
ellipse radii -> 22/23; drop the preview passthrough -> 22/23. 122hy 26/26 and
122hz 23/23 unchanged. Both sweeps vs 5671eed: 210 smoke_122 + 73 others, no
regressions. ruff 1 (overlay, pre-existing) / 0 (preview), F821 = 7, I001 = 1,
mufasa/**/*.py = 429.

**122hz — record button in the overlay viewer.**
Adds a Record button to pose_video_overlay: first press starts, second press
stops, and the saved clip contains the composited overlay frames (video + pose
markers, exactly as shown) for every frame displayed in between — whether
advanced by play, the scrubber, or stepping.

Because the overlay is Qt-composited (markers are QGraphicsItems over the video
pixmap, not painted into the frame pixels), the clip is captured by rendering
the QGraphicsScene to a QImage and converting to BGR for cv2.VideoWriter — not
from the raw cv2 frame. Capture is hooked in _set_frame, the single path every
displayed frame passes through, so any on-screen frame during recording is in
the clip. A _ClipRecorder (Qt-free, takes plain BGR arrays) owns the writer:
created lazily on the first frame so it sizes to the actual render, dimensions
forced even for codec compatibility, later frames conformed to the locked size.
The clip is saved beside the video by default (a Save-As dialog offers a
different path); an in-progress recording is finalised on window close so the
file isn't truncated. Degrades gracefully (status message) when cv2 is absent.

smoke_122hz_overlay_record: 23/23 — the _ClipRecorder state machine (lazy
writer, frame count, finish message, empty-recording failure, even-dimension
lock, size conforming) and the QImage->BGR reshape (row-padding stripped,
channels reversed, C-contiguous) run directly against a stubbed cv2 / simulated
buffer; AST confirms the button, the toggle_record start/stop, the capture hook
inside _set_frame after update_frame, the scene render, the close-finalisation,
and the no-cv2 fallback. Tripwires: drop the capture hook -> 21/23; break the
empty-recording check -> 22/23; break the even-dimension rounding -> 21/23.
Both sweeps vs 5671eed: 209 smoke_122 + 73 others, no regressions. ruff 1
(overlay — one better than the pre-existing baseline of 2, a nested-if collapse
cleared it), F821 = 7, I001 = 1, mufasa/**/*.py = 429 (edits only, no new
module). GUI feature; the Qt-free logic is tested directly, the rest AST-checked
(PySide6/cv2 absent in the sandbox).

**122hy — mufasa-preview: open the overlay viewer by paths or folder search.**
Adds a mufasa-preview console command (new module mufasa/tools/pose_preview.py,
entry point in pyproject.toml) that pairs a video with its pose files and opens
pose_video_overlay — so the long, hash-tagged pose paths don't have to be typed
by hand. Two modes: explicit (mufasa-preview --video rec.mp4 [--smoothed ...]
[--raw ...]; with only --video, pose files beside it are auto-found) and folder
search (mufasa-preview --folder .; videos paired with their pose files by name).

Pairing reduces a video and its pose files to a common session stem, so rec.mp4
matches rec.fdlc.parquet (raw) and rec.fdlc_smoothed_v2.<hash>.parquet
(smoothed) — the parameter-hash tag (122hm/hn) and the .fdlc/_smoothed_v2
suffixes are stripped; among several smoothed hashes the newest wins. The viewer
runs in-process via pose_video_overlay.main (same package, no subprocess);
--print shows the equivalent command instead of opening it; --all previews every
match in a multi-video folder (ambiguity otherwise errors with the match list).

smoke_122hy_mufasa_preview: 26/26 — stem reduction (incl. the reported real
filename), folder discovery (pairing, no-pose skipped, newest-hash wins), argv
construction (present/omitted opts), both --print modes, the error/ambiguity
paths (ambiguous folder, no video, missing pose, missing video, --video/--folder
mutual exclusion), the in-process overlay invocation with the built argv, and
the registered entry point. Tripwires: break the hash stripping (smoothed won't
pair); disable the ambiguity guard -> 25/26; skip the overlay call -> 24/26.
Backend logic, tested directly (real temp folders + a stubbed overlay main).

Adds a source module, so mufasa/**/*.py 428 -> 429; the smoke_122d5_stage_b
py-count ceiling is bumped 428 -> 429 to match (26/26). Both sweeps vs 5671eed:
208 smoke_122 + 73 others, no regressions. F821 = 7, I001 = 1, ruff 0 (module).

**122hx — shared pose loader handles all three FreeDLC formats (dedup).**
Fixes pose_video_overlay / pose_viewer._load_pose_file failing on a raw FreeDLC
long-format parquet — the same shape 122hw fixed, but in a *separate* loader
copy (the tools loader, distinct from the smoother's). Rather than copy the fix
a third time, the two non-flat FreeDLC parquet shapes are now handled in the one
place both loaders already fall back to: kalman_diagnostic.load_pose_file. Its
parquet branch now flattens a 3-level MultiIndex header (imported form, 122ht)
and pivots the long/tidy schema (raw form, 122hw) before marker detection.
pose_viewer inherits the fix by falling through to it; the smoother keeps its
inline copy for now.

The canonical _pivot_fdlc_long_to_wide lives here now (kalman_diagnostic is the
dependency-light shared module); the smoother's 122hw copy is an equivalent that
a later pass can collapse onto this one. That fuller consolidation — the
smoother and pose_viewer both delegating entirely to this loader instead of
carrying near-identical direct-read blocks (the smoother's own code comments
already note it duplicates pose_viewer._load_pose_file) — is a reasonable
follow-up; this patch fixes the immediate failure and removes the *format*
duplication without touching the smoother's hot path.

smoke_122hx_shared_loader_formats: 12/12 — the shared loader reads all three
shapes (raw long, imported MultiIndex, wide flat) with case preserved
(back_T4); pose_viewer's direct read finds no markers on a long file (so it
falls back) and the shared loader then loads it; and the pivot helper's
unit behaviour (sentinel clip, multi-animal rejection). Tripwires: drop the long
handling / drop the MultiIndex flatten -> the loader raises on that shape.
Canonical rig 152/152; 122ht 6/6, 122hw 14/14, 122hb 29/29 unchanged. Both
sweeps vs 5671eed: 207 smoke_122 + 73 others, no regressions (the two
pre-existing diagnostic-suite import failures are unchanged). ruff 1 (diagnostic,
baseline), F821 = 7, I001 = 1, mufasa/**/*.py = 428.

**122hw — smoother reads a raw FreeDLC long-format parquet.**
Fixes "Could not load .../K7_..._reduced_negate.fdlc.parquet. Tried direct read
and DLC multi-row header parsing." when smoothing a *raw* FreeDLC file
standalone (--load-model --beside-input). FreeDLC's native .fdlc.parquet is
LONG/tidy — (frame, individual, bodypart, x, y, likelihood), one row per
bodypart per frame — not the wide <bp>_x/_y/_p the loader expects. 122ht handled
the *imported* wide MultiIndex form; a raw file that never went through the
importer is long, has no _x columns, and fell through to the DLC parser and
failed.

Adds _pivot_fdlc_long_to_wide (mirrors the importer's long_to_wide but
layout-independent: every bodypart -> <bp>_x/_y/_p, no project order imposed,
-1 no-detection sentinel clipped to 0, frame index made contiguous, multi-animal
rejected with an actionable message). The loader detects the long schema
(non-MultiIndex columns containing frame/bodypart/x/y) and pivots before the
marker check; wide and imported-MultiIndex inputs are untouched.

smoke_122hw_fdlc_long_format: 14/14 (pivot recovers all bodyparts with case
preserved, emits _x/_y/_p, preserves or reindexes the frame count, clips the -1
sentinel, rejects multi-animal with the explicit message; AST confirms the long
detection is gated to non-MultiIndex and precedes the marker check). Tripwires:
remove the long detection -> 9/14; disable multi-animal rejection -> 13/14 (the
strengthened check pins the explicit message, not pandas' incidental reshape
error); break the sentinel clip -> 12/14. Canonical rig 152/152; 122ht 6/6,
122hu 18/18, 122gm 8/8 unchanged. Both sweeps vs 5671eed: 206 smoke_122 + 73
others, no regressions. F821 = 7, I001 = 1, ruff 22 (kalman), mufasa/**/*.py =
428.

**122hv — --load-model needs no project, --config, or skeleton.**
Answers "do I still need the inputs / a project to smooth an external file once
the model is built?" — no. A saved v2 model already stores its own layout
(save_model_v2 serializes every segment's name/parent/markers; load_model_v2
rebuilds the BodyLayout), and smooth_pose_v2 uses the loaded layout and ignores
any passed one. So with --load-model the only things still needed are the input
files and where to write.

main() had been running layout resolution (project search -> FreeDLC sidecar ->
rig fallback) unconditionally, before smooth_pose_v2, even with --load-model —
wasted work whose result the model overrides, and for a standalone file it would
emit the rig-fallback warning needlessly and the multi-project error could even
block a legitimate load-model run. This guards the whole layout-resolution block
behind `if args.load_model is None:`. With a model, config and layout are left
None (smooth_pose_v2 supplies the model's own layout) and any layout-shaping
flags passed anyway (--config, --with-drift, --*-segments, --no-*) are reported
as ignored. Without a model, resolution runs exactly as before (project /
sidecar / rig), so the 122hu standalone-sidecar path is unchanged.

Net: `... --load-model model.npz some_file.parquet --beside-input` smooths an
external file with no project and no skeleton, output beside it. Nothing needs
to track the smoother's inputs — the model is the complete spec.

smoke_122hv_load_model_skips_layout: 12/12 (the model serializes/rebuilds the
layout and smooth_pose_v2 prefers it; the guard exists and its else-branch holds
all of find_project_config / sidecar / rig; find_project_config runs only under
the no-model branch; config/layout default to None; the ignored-flag reporting).
Tripwires: disable the guard -> 8/12; drop the `config = None` default -> 11/12.
Canonical rig 152/152 and smoke_122hu 18/18 both survive the 217-line
re-indentation. Both sweeps vs 5671eed: 205 smoke_122 + 73 others, no
regressions. F821 = 7, I001 = 1, ruff 22 (kalman), mufasa/**/*.py = 428.

**122hu — smooth a standalone pose file in place, outside a project.**
Two CLI additions let the smoother process a parquet that isn't in a Mufasa
project, with output beside the input:

* ``layout_from_fdlc_sidecar(pose_path, root=None, **flags)`` builds a
  BodyLayout from a FreeDLC ``<stem>.fdlc.toml`` skeleton sidecar
  (read_fdlc_skeleton -> segments_from_skeleton -> BodyLayout), so the
  kinematic tree travels with the FreeDLC export and no project.toml is needed.
  main() tries this before the built-in-rig fallback when no project is found
  above the input and no --config is given — but only when every input resolves
  to a sidecar and they agree on the marker set; otherwise it falls through to
  the existing rig-fallback warning.
* ``--beside-input`` writes each smoothed file into its input's own directory
  instead of --output-dir. Inputs are grouped by parent directory and the
  smoother is run once per group with that directory as the output; a directory
  argument resolves to itself. Without the flag it is a single group -> a
  single --output-dir call, exactly as before.

smoke_122hu_standalone_beside_input: 18/18 — layout_from_fdlc_sidecar (defined;
builds a full spanning-tree layout from a sidecar; flags pass through; resolves
from either the .parquet or .toml path; None when absent so main() can fall
back); the CLI wiring by AST including a parsed check that the beside-branch
really builds batches from the per-directory groups (not a collapsed single
output); and the batching algorithm across single-file/multi-folder/directory/
no-flag cases. Tripwires: sidecar helper -> None breaks the standalone layout;
beside-branch collapsed to one output dir -> 17/18; main() not trying the
sidecar -> 14/15 (pre-strengthening) / caught. Canonical rig 152/152. Both
sweeps vs 5671eed: 204 smoke_122 + 73 others, no regressions. F821 = 7, I001 =
1, ruff 22 (kalman), mufasa/**/*.py = 428.

Note: this is the smoother step in place; extending --beside-input to the later
pipeline stages (outlier, egocentric) is a follow-up. The sidecar read uses the
importer's read_fdlc_skeleton (works where h5py is installed — the user's
environment); the sandbox can't exercise that import, so the layout half is
verified with a stubbed reader mirroring the real TOML parse.

**122ht — smoother reads the imported MultiIndex pose parquet.**
Fixes "Could not load .../sources/pose/<session>.parquet. Tried direct read and
DLC multi-row header parsing." hit when adding sessions with smoothing. The
FreeDLC/SimBA importer writes sources/pose parquets with a 3-level MultiIndex
column header (scorer/bodypart/coords = IMPORTED_POSE/IMPORTED_POSE/<bp>_x), via
write_df's multi_idx_header path. The smoother's inline loader read the parquet
then looked for columns ending in "_x"; on a MultiIndex the columns come back as
tuples, so no markers matched, the load fell through to the DLC-CSV diagnostic
parser, and raised.

Fix: flatten a MultiIndex column header to its last level — which holds the flat
<bp>_x/_y/_p names — immediately after the direct read and before the marker
detection. Plain (already-flat) columns are untouched by the isinstance guard,
so smoothed-flat parquets and pre-flattened CSVs load exactly as before; a
header whose last level isn't <bp>_x (e.g. a 2-level DLC header) still falls
through to the diagnostic loader as before.

smoke_122ht_imported_multiindex_parquet: 6/6 — against a parquet written exactly
as the importer writes it: all markers recovered, case preserved (back_T4 not
back_t4), plain-column parquets still load, _var columns not mistaken for
markers, and AST confirms the flatten sits in the loader ahead of the marker
check. Tripwire: remove the flatten -> 4/6. Canonical rig 152/152; FDLC importer
and pose-import suites unchanged (122gm 8/8, 122gn 9/9, 122dh 27/27, 122di
12/12, 122dj 14/14). Both sweeps vs 5671eed: 203 smoke_122 + 73 others, no
regressions. F821 = 7, I001 = 1, ruff 22 (kalman), mufasa/**/*.py = 428.

**122hs — matplotlib 3.11 compat: replace removed cm.get_cmap.**
Fixes "module 'matplotlib.cm' has no attribute 'get_cmap'" hit when adding
sessions with smoothing on a recent matplotlib. cm.get_cmap was deprecated in
3.7 and removed in 3.11; the colour-palette helpers still called it. Replaced
every cm.get_cmap(name, N) with the version-safe
matplotlib.colormaps[name].resampled(N) across plotting_mixin.py,
utils/data.py (create_color_palette / create_color_palettes) and
utils/lookups.py, updating imports (colormaps added; cm dropped where no longer
used). In lookups this also repaired a latent bug — the old branch tested an
int index for membership in the colormap dict (never matched, always fell to
"spring"); the new code tests the actual colormap name, so a requested map is
used when it exists and falls back to spring otherwise.

Verified the modern call is numerically identical to the old API (same N, same
sampled colours for viridis/jet/spring/tab20) and that the fixed helpers run
even with cm.get_cmap deleted (simulating 3.11), reproducing the documented
create_color_palette('jet',3,hex) output.

smoke_122hs_get_cmap_removal: 12/12 (no cm.get_cmap( call remains in mufasa/;
the three files use resampled(); old/new equivalence where the old API is still
present; resampled() palette shape; the spring fallback). Tripwires: reintro
cm.get_cmap in data.py -> 10/12; drop the colormaps import in plotting_mixin ->
11/12. Both sweeps vs 5671eed: 202 smoke_122 + 73 others, no regressions.
ruff at baseline (40 plotting_mixin / 35 data / 12 lookups), F821 = 7, I001 = 1,
mufasa/**/*.py = 428.

**122hr — hotfix: restore _launch_synced_viewer swallowed by 122hp's closeEvent.**
Patch 122hp added a closeEvent method to MufasaWorkbench by inserting it before
_launch_synced_viewer, but the edit consumed that method's `def` line — so
_launch_synced_viewer's body was absorbed into closeEvent and the method
vanished. The workbench then crashed on construction: _build_menus wires
"Synced video viewer…" to self._launch_synced_viewer, which no longer existed
(AttributeError, window won't open at all).

Fix restores the `def _launch_synced_viewer(self) -> None:` line and its
docstring; closeEvent and _launch_synced_viewer are again separate methods.

Also hardens smoke_122hp_console_dock so this class of edit error can't recur
silently: it now parses MufasaWorkbench and asserts closeEvent and
_launch_synced_viewer are both real methods and that every menu handler
(_launch*/_on*/_open*/_switch*/_show*) resolves to an existing method — i.e. no
def was swallowed. 33/33 (was 29). Confirmed the guard catches the bug:
reintroducing the swallowed def drops the suite to 31/33 on exactly the two new
checks. F821 = 7, I001 = 1, ruff 0 (workbench), mufasa/**/*.py = 428.

**122hq — egocentric alignment: strip smoother suffixes before the video lookup.**
Fixes "NO FILES FOUND ERROR: could not find a video file representing
Cacna_..._smoothed_v2.<hash>" from egocentric alignment. The aligner finds each
pose file's source video by name, but derived the name straight from the pose
stem — which carries _smoothed_v2 and (since the parameter hash, 122hm/hn) a
trailing .<hash> that the source video filename does not. The _smoothed_v2
suffix alone would already have broken this; the hash made it visible.

Adds _video_base_from_pose_name(), which strips a trailing hex .<hash> (only
when what remains still ends in _smoothed_v2, so a legitimately dotted video
name is never truncated) and then the _smoothed_v2 suffix, returning anything
unrecognised unchanged. Used only for the three find_video_of_file lookups (the
pre-flight check and the two in run()); the output data/video keep the full
pose-derived video_name, so a rotated output still traces to the specific
smoothed input (important now that different parameter hashes are distinct
inputs).

smoke_122hq_egocentric_video_base: 16/16 (the reported 16-char-hash case, the
8-char and no-hash cases, plain stems unchanged, and the false-positive guards:
a dotted video name without the marker, a hex extension without _smoothed_v2, a
non-hex suffix, and a hash-like token are all left alone; idempotent). Also
verifies the wiring — lookups use the stripped base, outputs keep video_name,
and no lookup still passes the raw name. Tripwires: don't strip _smoothed_v2 ->
12/16; drop the hex-guard so it over-strips -> 15/16; use the raw video_name in
the run() lookups -> 14/16. Existing egocentric suites pass:
smoke_122ex_egocentric_section_id 10/10, smoke_122ez_egocentric_detect_path
10/10. Both sweeps vs 5671eed: 201 smoke_122 + 73 others, no regressions.
F821 = 7, I001 = 1, ruff 2 (aligner, baseline), mufasa/**/*.py = 428.

**122hp — dockable console for verbose output.**
Adds a read-only console docked at the bottom of the workbench that mirrors
stdout/stderr live, so GUI users see the verbose progress (the smoother's
[smoother-v2] lines, per-EM-iteration logs, provenance notes) that previously
only reached the terminal.

New module mufasa/ui_qt/console_dock.py: a _StreamRedirector that stands in for
sys.stdout/stderr, tees every write to the original stream (terminal users lose
nothing) and emits it on a Qt signal; a ConsoleDockWidget (bounded, read-only,
with Clear / Copy / Autoscroll / Wrap); and attach/detach helpers. Because
operations print from a ProcessorRunner worker thread and Qt widgets are
GUI-thread-only, the signal->slot uses a queued connection so the append lands
on the GUI thread — this is the thread-safety crux. The redirector implements
enough of the file protocol (write/flush/isatty/writable/fileno) to be a safe
stand-in, and a broken original stream can't lose console output or crash.

Wired into MufasaWorkbench: attach_console_dock in __init__ (guarded so a
console failure can't stop the window opening), detach in closeEvent to restore
the streams on teardown. Project switches rebuild the workbench; attach unwraps
any existing mufasa redirector (_base_stream) so it tees to the real terminal
rather than chaining redirector->redirector, and detach only restores when
still the active stream, so the rebuild doesn't duplicate output or clobber the
new window.

smoke_122hp_console_dock: 29/29. AST-checks the module structure and workbench
wiring (queued connection, bottom dock, idempotent attach, read-only + bounded
view, attach/detach), and exercises the pure logic in isolation (a stub
signal): tee to both streams, empty write emits nothing, write returns length,
_base_stream unwraps a chain to the real stream and handles None, and a broken
original neither loses output nor crashes. Tripwires: break the tee -> 28/29;
drop the queued connection -> 28/29; break the unwrap -> 28/29; remove the
workbench attach -> 28/29. GUI-only, AST/pure-logic verified (PySide6 absent in
the sandbox); the Qt widget runtime is validated on hardware.

Note: this ADDS a source module, so mufasa/**/*.py goes 427 -> 428. The
smoke_122d5_stage_b py-count ceiling is bumped 427 -> 428 to match (it is an
explicit loose ceiling that already moved 425 -> 427 in 122gz). Sweeps vs
5671eed: 200 smoke_122 + 73 others = no regressions after the ceiling bump.
F821 = 7, I001 = 1, ruff 0 (new module) / 0 (workbench).

**122ho — project-aware segment fields in the Kalman v2 GUI.**
Fixes the confusing failure where the form's orientation-drift / const-accel /
high-angular-noise fields carried a stale 'body,head' example (and persisted
value), and submitting it died deep in the smoother with a raw dataclasses
ValueError stack trace: "orientation_drift_segments references unknown segment
'body'". The rig was renamed, so this project's segments are back / back_rear /
head / neck / tail_1.. — there is no 'body' (it became 'back').

Three parts: (1) a _project_segment_names(config_path) helper resolves the
project's actual segment names via layout_from_config, degrading to [] on any
failure so the form never crashes when no project is loaded; (2) the form
validates the three segment fields against those real names BEFORE
dataclasses.replace hits the layout's own validator, raising a clear message
that lists the available segments and notes the body->back rename — the
smoother's __post_init__ stays as the backstop; (3) the stale 'body,head'
examples are replaced with project-aware placeholders (the real segment names)
and tooltips, and the stale docstrings are corrected.

smoke_122ho_segment_validation: 16/16 (helper definition + safe runtime
degradation on bad/empty paths; validation present and ahead of _dc.replace;
correct accept/reject behaviour incl. the exact 'body' case, a typo, and empty
lists; no 'body,head' example remains; placeholders are project-aware).
Tripwires: remove the validation -> 13/16; hardcode the placeholders -> 15/16;
narrow the helper's except so it no longer degrades -> the runtime check
raises. GUI suites unchanged: smoke_pose_cleanup_v2_wiring 2/2,
smoke_122hk_gui_and_model_io 21/21, smoke_122hd_segment_flag_errors 27/27. Both
sweeps vs 5671eed: 271 suites, no regressions. F821 = 7, I001 = 1, ruff 12
(form), mufasa/**/*.py = 427.

**122hn — longer parameter hash.**
Widened the smoother parameter hash (122hm) from a 4-byte digest (8 hex chars)
to an 8-byte digest (16 hex chars), for stronger collision resistance across
parameter variants while still keeping the output filename readable. One-line
change (blake2b digest_size 4 -> 8); everything downstream is length-agnostic
(the writer and the incremental skip both build "." + param_hash). The 122hm
test's length assertion is updated 8 -> 16 accordingly.

smoke_122hn_hash_length: 8/8 (asserts the 16-char length, determinism,
parameter-sensitivity at the new length, filename propagation, and digest_size
== 8). Tripwire: revert digest_size to 4 -> 122hn 3/8 and 122hm 21/22.
smoke_122hm 22/22 with the updated assertion, 122hl 20/20, 122hk 21/21,
canonical 152/152. smoke_122 sweep vs 5671eed: 198 suites, no regressions.
(The 73 non-smoke_122 suites don't touch the hash helper or the output
filename, so a digest-width change can't affect them.) F821 = 7, I001 = 1,
ruff 22, mufasa/**/*.py = 427.

**122hm — parameter hash in the smoothed-output filename.**
Smoothed output is now written as ``<stem>_smoothed_v2.<hash>.parquet`` (and
the ``.csv`` fallback likewise), where ``<hash>`` is a short blake2b digest (8
hex chars) of the smoother parameters. Two effects: parameter variants coexist
in one output dir instead of clobbering each other, and the incremental skip
(122hl) becomes parameter-aware — a changed setting yields a different
filename, so the skip re-smooths rather than leaving the stale file (the exact
gap flagged when 122hl landed).

The digest is built from the effective smoothing configuration as it exists at
write time: the fitted noise params (all numeric NoiseParamsV2 fields), the
layout flags and tree, the joint prior (mu/kappa) and perspective presence, and
the smoothing-time knobs (likelihood_threshold, fps, apply_constraints).
Reading from the effective objects makes it mode-independent — a model trained
here and the same model reloaded later produce the same digest (their params
came off the same fit), so a file produced in a training run and one produced
in a later load run of that model share a digest and the skip treats them as
the same. Floats are rounded to 6 significant figures before hashing, so
meaningless last-bit differences don't perturb the digest while a genuinely
different fit (different data, EM settings, or config, all of which move the
fitted params) does.

Computed once per run (the config is identical across sessions), threaded into
both the incremental skip's existence check and both writer call sites
(parallel and serial). Legacy behaviour is preserved when no hash is supplied
(in-memory callers with output_dir=None get the un-suffixed name). Blast radius
checked: nothing globs or reconstructs the ``_smoothed_v2.parquet`` name;
consumers use the returned output_path.

smoke_122hm_param_hash_filename: 22/22. Tripwires flip: drop
likelihood_threshold from the payload -> 21/22; make the writer ignore
param_hash -> 20/22; remove the float rounding -> 21/22. 122hl still 20/20 (its
skip is now hash-aware), canonical 152/152. Both sweeps baseline-diffed against
5671eed: 270 suites, no regressions. F821 = 7, I001 = 1, ruff 22 (kalman),
mufasa/**/*.py = 427.

**122hl — incremental smoothing / overwrite option.**
smooth_pose_v2 gains an ``overwrite`` parameter (default True — unchanged
behaviour, re-smooth every input). With overwrite=False, any session whose
smoothed output already exists (``<stem>_smoothed_v2.parquet`` or the ``.csv``
fallback in the output dir) is skipped, so a re-run only processes files that
have no output yet — for resuming an interrupted batch or adding new sessions
without recomputing the finished ones.

The skip is computed once, before the smoothing dispatch, and applied in both
the parallel path (filtering task_args) and the serial path (continue in the
loop), so skipped sessions never enter the per-session smoother — it saves
compute, not just the write. The skip is gated on output_dir being set and
overwrite being False; with no output_dir (in-memory use) nothing is skipped.
skip_session indexes raw_sessions / sessions_arr, which are built in lockstep,
so the index is stable across both paths.

CLI: a mutually-exclusive ``--skip-existing`` (overwrite off) / ``--overwrite``
(the default), passed through main. GUI: an "Overwrite existing smoothed
output" checkbox in the Kalman v2 form's common options, threaded through both
collect_args returns into the three output-writing smooth_pose_v2 calls
(two-pass pass-2, single-pass, load); pass-1 writes no output so it is left
alone.

smoke_122hl_incremental_overwrite: 20/20. Tripwires flip: remove the parallel
task_args filter -> 19/20; drop the "not overwrite" gate (skip even when
overwriting) -> 19/20; drop overwrite from a GUI output-writing call -> 19/20.
Related suites pass: smoke_122hk 21/21, smoke_pose_cleanup_v2_wiring 2/2,
canonical 152/152. Both sweeps baseline-diffed against 5671eed: 269 suites, no
regressions. F821 = 7, I001 = 1, ruff 22 (kalman) / 12 (form),
mufasa/**/*.py = 427.

**122hk — GUI wiring for the new smoother options + model save/load fix.**
Exposes the three smoother options added this session in the Kalman v2 form,
and fixes a silent correctness bug in model persistence.

The save/load bug (the important half). Training a model with the joint prior
(122hf), saving it, and loading it to smooth other sessions silently DROPPED
the prior: the save_model_v2 call in smooth_pose_v2 passed perspective but not
joint_prior, so save_model_v2 took its default None and wrote
has_joint_prior=False. A loaded model then smoothed new sessions WITHOUT the
prior — reintroducing the covariance divergence (tail tip and rear back
markers flying off to thousands of px) that the prior is there to prevent. The
save/load path is otherwise correct: load_model_v2 already reads the prior back
and returns it, and the final smoothing pool already threads it to the filter;
only the save call omitted it. One line: pass joint_prior=joint_prior. Verified
by a round-trip (kappa/mu/n_obs preserved) and by reproducing the old loss.
tau_accel (122hg) already round-tripped; high_angular_noise_segments (122hj)
round-trips via the layout save/load added in that patch.

GUI wiring. KalmanV2SmoothingForm exposed with-drift / orientation-drift /
const-accel but not the joint prior, accel-tau, or high-angular-noise
segments. Added: a joint-prior checkbox, an accel-tau spinbox (0 shows
"off (random walk)" and maps to None, not an invalid 0.0), and a
high-angular-noise-segments field. Threaded through collect_args into both
train smooth_pose_v2 calls (single-pass and two-pass pass-1) and, for
high-angular-noise, into the layout replacements so it is saved with the model
and restored on load like the other segment flags. Load mode is unchanged —
those settings come from the saved model, which now includes them.

smoke_122hk_gui_and_model_io: 21/21. Tripwires flip: drop joint_prior from the
save call -> 20/21; drop the enable_joint_prior collect_args key -> 20/21; drop
accel_tau from a train smooth call -> 20/21. Directly-affected suites pass:
smoke_pose_cleanup_v2_wiring 2/2, smoke_122hf_joint_prior 42/42,
smoke_save_destination_validation 8/8 (smoke_model_dual_save's 22/23 is a
pre-existing legacy-INI failure, unrelated). Canonical rig 152/152. Both sweeps
baseline-diffed against 5671eed: 268 suites, no regressions. F821 = 7, I001 =
1, ruff 22 (kalman) / 12 (form), mufasa/**/*.py = 427.

**122hj — per-segment orientation-noise ceiling, for a fast head.**
The head impulse-tracking failure, root-caused from a real 67-session EM log.
The M-step clips each segment's fitted orientation process noise q_seg_ori to
q_initial * _M_STEP_Q_CEILING_FACTOR (10x). The log shows the head's q_seg_ori
pinned at exactly 20.0 (its init is 2.0, so 2.0*10) across every iteration
while the mean rises — a segment starved by its cap. The head turns ~3x faster
than the slowest segment (measured 6.2 vs 1.9 rad/s), so a single global
ceiling factor can't serve both: at the value that keeps the trunk stable, the
head is denied the process noise it needs, over-trusts its smooth prediction,
and rejects fast high-likelihood observations at acceleration impulses — the
head "not keeping up" with sharp movements, and exactly the user's report.

Confirmed the mechanism against the data: at high-acceleration impulses where
FreeDLC likelihood is high (p>=0.7, marker clearly visible), the smoothed head
still misses by ~9.5px, and the miss is not a lag (min error at time-shift -1)
and not damping (speed ratio ~1.0). The filter is rejecting good, visible, fast
observations because its dynamics prior is too tight.

The fix adds a layout field high_angular_noise_segments and a
--high-angular-noise-segments CLI flag (mirroring --const-accel-segments):
listed segments use a raised ceiling factor (_M_STEP_Q_CEILING_FACTOR_HIGH,
40x), all others stay at 10x. For the head that lifts the cap from 20 to 80.
Low-likelihood observations are still hard-gated out in the filter update
(markers below --likelihood-threshold are dropped, not down-weighted), so a
higher ceiling changes responsiveness only to observations that already pass
the gate — it does NOT make the head chase spurious detections (verified on a
real explosion frame: 11/15 markers below 0.5 there are correctly excluded).

Field-whitelist-rot guard: the model save/load path hand-lists layout fields,
so high_angular_noise_segments is persisted and restored symmetrically; pre-
122hj model files lack the key and load empty. Only q_seg_ori uses the per-
segment ceiling; q_length and the root ceilings keep the global factor.

smoke_122hj_per_segment_ceiling: 18/18. Tripwires flip: high factor == default
-> 16/18; selector ignores the flag -> 17/18; validation dropped -> 16/18.
Canonical rig unchanged (152/152, it sets no flag). Both sweeps baseline-diffed
against 5671eed: 267 suites, no regressions. F821 = 7, I001 = 1, ruff 22,
mufasa/**/*.py = 427.

**122hi — root-cluster frame from the widest chord, not PCA.**
122hh fixed the tail but not the trunk, and testing on the user's real 54k-frame
session showed why. 122hh derived the root cluster's body frame from per-frame
PCA and gated it on anisotropy, assuming a trunk is elongated like a spine.
Measured on real top-down mouse data it is not: the six dorsal markers project
into a roughly ROUND 2D patch — ~40px across but eigenvalue ratio ~1.75, no
dominant axis. PCA of a round patch returns a noise direction; the anisotropy
gate then rejected ~89% of frames; the cluster fell back to the placeholder
ring; the trunk kept bunching exactly as before.

A round patch still has a well-defined frame: the two markers farthest apart on
average define its widest chord, whose per-frame direction is stable even when
the cluster as a whole is round. 122hi uses that chord as the body axis — no
marker names, no elongation assumption, works for an elongated spine and a
round patch alike. The chord is inherently signed, so unlike a PCA eigenvector
there is no sign-flip ambiguity to repair. The anisotropy gate is removed. The
back1/back3 fallback stays for the canonical round cluster.

Verified on the real session: fitted trunk offsets match observed inter-marker
distances to within 1px (back_T4-back_L6 fitted 39.8 vs observed 38.6; the hip
pair 46.9 vs 47.4), and zero markers remain on the ring. The canonical rig is
unchanged (152/152). smoke_122hi_root_pair_axis (renamed from _pca_axis): 18/18,
including a controlled orthogonal-pair case that detects a wrong (closest-pair)
selection. Both sweeps baseline-diffed against 5671eed: 266 suites, no
regressions.

**122hh — root-cluster offsets from a PCA body axis; the bunched-back-markers fix.**
The reported bug: back markers bunched together even when FreeDLC saw them
clearly, and it started around 121. Traced to 122gx. Every non-root segment
gets its body frame from `parent_distal -> seg_distal`; the root has no parent,
so the code filled the gap with two hardcoded rig marker names (`back3 ->
back1`). 122gx generalized the tree to any skeleton but left that proxy
rig-specific — so `strohA-kj`, whose trunk markers are `back_T8..hip_right`,
matched nothing, `parent_distal` stayed None, and `fit_body_lengths` skipped
offset-fitting for the WHOLE root cluster. Every non-distal trunk marker kept
its placeholder ring offset `(1.0, angle)`: six markers pinned at unit radius
around a circle instead of along the spine. The forward model then placed them
on that ring no matter what the data said — hence "bunched even when plainly
visible." Not perspective, not convergence, not this session's patches; a
rig-name assumption left behind by a generalization three patches back, which
is exactly why it worked before 121.

Trunk markers lie along the body, so the first principal component of the
cluster's point cloud IS the body axis — for any rig, no names. Computed per
frame (the animal rotates). On strohA-kj this turns the ring into real
geometry: spine collinear (max |y| < 0.2px), lengths increasing down the back
(0 -> 8 -> 16px), hips straddling symmetrically (+-10px). PCA's sign is
arbitrary and can flip frame to frame, which would scatter each marker's angle
across two clusters 180 deg apart; a mean-axis fold fixes it. Only per-frame
consistency matters, not which end is "forward" — a globally tail-forward axis
just rotates the whole offset set 180 deg, which the filter absorbs into the
segment orientation. (An earlier attempt oriented toward a child marker; for
this tree that marker sits almost on the trunk line, making the sign test
degenerate — dropped.)

Gated on cluster anisotropy. PCA needs an elongated cluster; the canonical
rig's root markers are nearly coincident (eigenvalue ratio ~1.1, a round
blob), where the principal axis is just noise. Below a 4:1 ratio the axis is
refused and the code falls back to the old back3->back1 proxy, so the built-in
rig is bit-unchanged and 122hh strictly ADDS the elongated-cluster case. The
guard is load-bearing: disabling it reproduces the exact `r_drift` regression
(12.19 vs [0.8,6]) it was added to prevent.

**122hg — Singer acceleration: mean-reverting instead of a random walk.**
Answers "could we use the jerk if it was conditioned on velocity and/or
acceleration?" Conditioning Q on the state is mechanically fine — condition on
the *previous EM iteration's* smoothed state and Q_t is a known sequence, so
the filter stays linear-Gaussian. But conditioning on velocity is backwards
during a dropout: under const-accel with no data `v += a*dt` grows while `a`
holds, so q_jerk(|v|) rises as the extrapolation runs away. Positive feedback
on exactly the divergence we hit.

The Singer model gets both halves without state dependence. 121d's comment
says it plainly — "ax_new = ax (random walk)", the accel diagonal left at the
1.0 np.eye(D) put there. Setting it to phi = exp(-dt/tau_accel) makes
acceleration Ornstein-Uhlenbeck: still a tracked state, so fast motion is
captured exactly as before, but unobserved it relaxes to zero and the
extrapolation degrades CA -> CV -> CP.

Measured growth exponents: **t^5.01 -> t^3.17**. At a 1s dropout the two agree
within 1.5x (nothing given up where const-accel earns its keep); at 20s they
differ by 15x. `tau_accel=None` is the default and reproduces 121d bit-for-bit
— the OU noise `q*T/2*(1-e^(-2dt/T))` tends to `q*dt` as T -> inf, so this is
an identity, not an approximation. Fourth mean-reverting block in the file
after alpha_drift (120a), orientation drift (121a) and the joint prior (122hf).

**Two latent bugs surfaced doing it.**

`build_F_v2` takes a `params` argument to read `alpha_drift`. All four call
sites — forward_filter_v2, rts_smooth_v2, per_session_fit_from_stats,
finalize_m_step_v2 — omitted it, so it has *always* been None. `alpha_drift`
has therefore been inert since 120a: the block silently used build_F_v2's
hardcoded 0.05 fallback, which happens to equal the default, which is why no
one noticed. `--accel-tau` was silently ignored the same way until this was
fixed. rts_smooth_v2 mattered most: the RTS gain uses F, so a backward pass
building a different F than the forward pass is wrong, not merely inert.

And the whitelist, for the third time. `finalize_m_step_v2` and
`finalize_m_step_v2_from_per_session` each hand-listed 15 of NoiseParamsV2's
16 fields. 122hc fixed the warm-start site and flagged these two for audit;
adding `tau_accel` is what called it in. Both now `dataclasses.replace` from
prev_params. `fit_initial_params_v2` stays a constructor — it has no prev to
copy from. The next field cannot be forgotten because there is nothing left
to forget.

**122hf — von Mises joint prior on segment angles.** Answers "for longer
stretches could we constrain the markers with the prior given the current
inter-marker relationships?" Yes, and the mechanism was already in the file.

Five of the seven segments in this tree carry exactly one marker. Two-plus
markers means losing one is free — the survivors pin the segment and FK places
the missing marker. One marker means no redundancy: when it drops, nothing
observes the segment's angle, and nothing constrains it either.
`build_F_v2` makes orientation a random walk; `cos²+sin²=1` fixes the radius
and says nothing about the angle. At the fitted `q_seg_ori` ≈ 6, the (cos,sin)
sd passes 1.4 after one second on a circle of radius 1 — the norm constraint is
just projecting a wild vector back at an arbitrary angle.

The prior: two pseudo-observation rows per segment on the same channel that
already carries the norm constraint, pulling `(cos_s, sin_s)` toward
`(cos mu_s, sin mu_s)` with variance `1/kappa_s`. Linear, so the Jacobian is
constant. The vector form rather than the tidier one-row `sin(theta-mu)=0`,
because that is satisfied at both `mu` and `mu+pi` and would settle the tail
folded back on itself. `mu`/`kappa` from a von Mises MLE fitted once before EM
(same lifecycle as the perspective model) and **only from frames where the
segment's marker was observed**. Zero extra state dims.

Measured, tail tip blinded for 20 s: **115° RMSE (worst 180°) -> 18° (worst
46°)**.

The kappa floor turned out to matter as much as the prior. During a dropout
nothing competes with it, so even a weak prior fully determines the angle —
on a near-uniform synthetic the fitted kappas were 0.2-0.8 and applying them
made the worst case *worse* (151° -> 180°), because mu was meaningless. At
`kappa >= 1.0` (sigma_theta <= 57°) a diffuse joint is skipped entirely and
the output is identical to before. Failure mode is "no help", never "wrong
help".

### Found while building it: the (L, theta) sign gauge

A segment's length is a free state with process noise and **nothing holds it
positive**. `(L, theta)` and `(-L, theta+pi)` place the marker at exactly the
same point, so the filter is free to sit in either. Measured, tail tip blinded
20 s: L walks through zero during the dropout (47% of frames negative),
re-acquires the marker in the mirrored gauge, and **stays there — 100%
negative afterwards, theta reported 180° from truth**. The user's run has
`q_length` ≈ 105 against fitted lengths of 10-20 px, which is ample to cross
zero.

Pre-existing, not introduced here. Marker *positions* stay correct (the
degeneracy is exact), so output isn't directly corrupted — but the RTS
smoother can smear between the two modes through `L ≈ 0`, which puts the
marker at the parent joint, and any consumer reading segment angles gets
garbage. 122hf only folds the mirrored frames back before averaging, because a
circular mean across both gauges is meaningless (measured: mu off by 177°,
kappa collapsed 13.3 -> 1.1). **Holding L > 0 in the filter is the real fix
and is not done.**

### Correction to the record

122hc and 122he both assert that `--const-accel-segments` had "never run".
That is wrong. The broken whitelist sits inside `if enable_warm_start_sigma:`;
verified on `5671eed` with rig-named data, `enable_warm_start_sigma=False`
runs const-accel end-to-end through EM. The accurate claim is **unusable in
the default configuration** — the CLI has `--no-warm-start-sigma` and the GUI
a checkbox (`pose_cleanup.py:1259`, default checked). Every log the user
showed has warm-start on, so it never ran *for them*. 121d wired q_jerk into
four of the five reconstruction sites and missed only the warm-start one.

### And a hole in my own verification

The sweep glob is `tests/smoke_122*.py`, which misses **73 test files**.
122he (pushed as `2859314`) rephrased the divergence message and broke
`smoke_kalman_pose_smoother_v2.py` — 152/152 on `5671eed`, failing on
upstream. Fixed here, and both pinned-wording assertions relaxed to pin
behaviour. **Sweeps should cover `tests/*.py`, not `tests/smoke_122*.py`.**

### Next
* Hold `L > 0` in the filter — the sign gauge above.
* GUI checkbox for the joint prior (CLI only for now).
* Fit `q_length` against the gauge too; it is likely inflated by sign flips.



**122he — cap iteration 0.** With the layout finally correct, the user's run
reached the filter and the EKF diverged: "validation hook triggered at
iteration 0", 96% NaN.

Not reproduced in-sandbox (3 GB RAM caps synthetic sessions far below their
54k frames), so this ships a structural fix and a diagnostic, not a
confirmed repair. **The user should drop `--const-accel-segments` first.**

The structural finding: *every* q hard cap in this module was applied only
inside `finalize_m_step_v2`. They guarded iterations 1..N and left iteration 0
running on whatever `fit_initial_params_v2` returned. The caps encode a
numerical stability boundary for the EKF, and the EKF does not care which
iteration it is on. Patch 121c's own comment says the absolute caps exist to
backstop "cases where the initial q is already large (heavy-noise sessions
push initial fit upward)" — it saw the initial fit going large, capped the
M-step's *use* of it, and never capped the value itself.

It is not a corner case. On ordinary synthetic mouse data the initial fit
returns `q_root_pos` ≈ 78,000–86,000 against 121c's documented boundary of
50,000 — past the point where, in its words, "even a moderate dropout (1
second) produces prediction covariances ... where the EKF S = HPH^T + R loses
conditioning". `_cap_params_to_hard_limits` is now the single place that
enforces the caps, applied to the initial fit and reported under `--verbose`.

Why const-accel is the prime suspect: `q_jerk_* = q_root_* × 10` at init
(uncapped), and jerk drives prediction variance as τ⁵ rather than τ³ across a
dropout. A tail marker occluded for seconds costs quadratically more with
const-accel on. And per 122hc, `--const-accel-segments` had *never run* — the
121d whitelist bug killed it in `build_Q_v2` before any EM. 121d and 121e are
both untested code that 122hc unblocked yesterday; 121e's jerk caps were
sized as "roughly 10× the q_root caps", which is dimensionally naive (px²/s⁵
vs px²/s³) and, at 500,000, sit above what the initial fit produces anyway.

The divergence message also stopped speculating. It listed three candidate
causes and told the reader to "inspect the early frames of x_smooth" — an
array they have no access to. It now reports NaN fraction, first non-finite
frame and its timestamp, which parameters sit on their cap, and whether
const-accel is active.

### Next
* Ask the user for a `--verbose` run without `--const-accel-segments`. If it
  survives, 121d's const-accel needs a properly calibrated q_jerk cap
  (measured against the τ⁵ boundary, not scaled off the q_root cap).
* CLI `--em-aggregation` still defaults to `pooled`; the handoff's own lesson
  is that median is strongly preferred across many sessions. Changing a
  default needs a decision, so it's flagged rather than done.



**122hd** fixes the error message 122hc introduced. It was readable and wrong.
Against a project that declared its own `[[pose.segments]]` and typed
`--orient-drift-segments body,head`, it asserted the layout came from
`[skeleton]`, that there was "no 'body' or 'head' to point at" (`head` was in
the list the message itself had just printed), and that the fix was to add a
`[[pose.segments]]` block the user already had — while printing the flag as
`--orientation-drift-segments`, which is not the flag. The one real fault,
`body` should be `back`, was the one thing it never said.

The lesson is the session's lesson again, one level up: a canned narrative
that asserts a cause it hasn't checked. An error that guesses is worse than one
that only lists facts, because the whole value of an error is that it can be
trusted. 122hd states what is missing, lists what exists, names the root, and
explains only where there is a signal checkable against the object in hand:
the name belongs to the built-in rig (compare against `standard_rat_layout()`),
or every segment is named after a marker and therefore the tree really was
derived from `[skeleton]` (a property of the layout, not a guess about
provenance). Near-match suggestions use a strict cutoff, since a loose one
invents advice (`trunk` -> `neck`) that is worse than silence.



**122hc follows 122hb the same day**, because 122hb's fix was incomplete in a
way worth recording: it made `smooth_pose_v2` stop assuming the built-in rig
when `layout is None`, and added `--config` to the CLI so the layout could come
from the project — but `main()` still *eagerly built the rig* when `--config`
wasn't passed, so it never reached the new guard. The user's next run failed
identically. A flag you have to know about is still a default for everyone who
doesn't. `find_project_config` now walks up from the input path, and the CLI
looks for the project before assuming anything.

Two more bugs surfaced underneath, both previously masked by the layout failure:

* `--orient-drift-segments body,head` cannot work against a `[skeleton]`-derived
  tree, which names each segment after its distal marker (`head_mid`, `back_L2`)
  and therefore has no `body` or `head`. That is not a bug — it's the reason
  `[[pose.segments]]` exists — but the failure was a raw `__post_init__`
  traceback. It now names the available segments and says to declare the tree.
* The warm-start rebuild of `initial_params` in `fit_noise_params_em_v2` was a
  hand-written whitelist of fields to preserve. It rotted three times: patch
  120b added the drift trio after `with_drift` runs crashed having lost them,
  patch 121a added the orientation-drift trio for the same reason, and patch
  121d added const-accel and never came back — so `--const-accel-segments`
  dropped `q_jerk_*` to None and died in `build_Q_v2`. Now a copy-every-field
  `dataclasses.replace`, which cannot rot again.



One patch, **122hb**, closing two bugs that both presented as "the smoother
can't find my markers" and were each misdiagnosed once.

**Bug 1 — the loaders renamed the markers.** Every pose reader did
`[str(c).lower() for c in df.columns]`. The intent was case-insensitive
*suffix* matching (`_X` vs `_x`); the effect was that any marker with an
upper-case letter came back renamed. `back_T4` → `back_t4`, `tail_V6` →
`tail_v6`. A layout built from `project.toml` (which keeps the case the user
typed) then matched nothing for those markers.

The tell was the split: on the strohA-kj project, 7 of 15 markers matched and
8 did not, and the 8 were **exactly** the 8 whose names contain an upper-case
letter. This was previously read as evidence of a partial marker rename, and
cost a session hunting for a second generation of files under `backups/` and
`derived/` that never existed. The recursive-`rglob` discovery scope is still
worth tightening, but it was not this bug and never was. The user's own
report — "the csv files show the old labels and the parquet files show the new
labels" — was correct and should have been believed.

Marker names are identifiers. `normalize_pose_columns` now folds the suffix
only and returns the base verbatim. No-op for the all-lower-case marker sets
of legacy projects.

**Bug 2 — the layout was resolved before the model that carries it.**
`smooth_pose_v2` defaulted `layout` to `standard_rat_layout()`, validated and
packed the data against it, and *then* — fifty lines later — opened the `.npz`
whose layout it should have used. Both load-mode paths in the GUI (the "load
model" mode, and pass 2 of the train-on-a-subset two-pass workflow) pass no
layout, so both validated project data against the built-in 2019 rat rig and
died citing `nose`/`back1`/`tailbase`. Load now happens first; the model's
layout is authoritative because its parameters are dimensioned to it.

A default that names markers the project has never heard of is not a default.
`layout=None` with no model now resolves from `config_path`, and falls back to
the built-in rig only when the data genuinely is the built-in rig — otherwise
it raises and says what to add.

**The 122gv hole is closed.** That guard only fired on *zero* overlap; a 7/15
mismatch merely warned that the missing markers "will have default offsets",
which undersells "eight segments are unobservable, the covariances will blow
up, and this will look like a hang". Any missing layout marker is now a hard
error, and the error names case-only differences explicitly.

### Shipped

* `normalize_pose_columns` in `kalman_diagnostic.py`; all four loaders there
  plus the smoother's own direct reader routed through it.
* `_resolve_layout` / `_validate_layout_against_data` in
  `kalman_pose_smoother_v2.py`; `load_model_v2` hoisted above both.
* `config_path=` on `smooth_pose_v2`, threaded through all four GUI call sites
  and a new `--config` on the CLI (which also stopped assuming the rat rig).
* Pre-flight block: layout, `state_dim`, markers matched, sessions, workers,
  projected GiB/worker, and which stage is about to run — printed before the
  first frame is filtered.
* `find_project_config` in `project_layout.py`; the CLI uses it to locate the
  project from the input path, warns when it falls back to the rig, and refuses
  inputs spanning two projects rather than guessing between them.
* `fit_noise_params_em_v2`'s warm-start rebuild: whitelist -> copy-everything.
* `tests/smoke_122hb_marker_case_and_layout.py` (29 checks) and
  `tests/smoke_122hc_project_discovery.py` (32 checks). Real tests: actual
  parquet/CSV/DLC-header files through the actual loaders, and real subprocess
  CLI runs against real project trees — not AST-only.

### Next

* Discovery scope: `discover_pose_files` and the `smooth_pose_v2` fallback
  glob are still recursive and still sweep `backups/` and `derived/`. Now that
  the guard is strict this surfaces as a loud error rather than silent garbage,
  but it should be scoped or excluded outright.
* Audit the other `NoiseParamsV2` rebuild sites (`finalize_m_step_v2`,
  `finalize_m_step_v2_from_per_session`) for the same whitelist rot.
* Heartbeat inside warm-start / EM — long phases are still silent between
  iterations.
* `[pose.skeleton]` schema migration (drop `nodes`, move under `[pose]`).
* Joint model phase 1: `joint = "rigid"` → merge into parent.



Two patches: **122e** makes `ConfigReader` (959 lines, inherited
by 268+ files across the codebase) read v1 `project.toml` while
preserving its legacy attribute surface, so every backend
continues to work transparently against v1 projects without
per-file modification. **122f** removes direct `configparser`
usage from every UI form method, encapsulating the legacy INI
branch in a handful of layout-agnostic helpers.

Directional commitment: as of this session, Mufasa no longer
depends on `project_config.ini` for any production code path.
INI parsing is kept alive in three places only — the migration
tool (`legacy_layout.py` + `migrate_project.py`), the
`read_config_file` fallback for legacy projects opened post-
migration, and the encapsulated legacy branches of the new
read/write helpers.

### Shipped

#### TOML-aware ConfigReader (patch 122e)

New module **`mufasa/utils/toml_to_configparser.py`** translates
a parsed v1 `project.toml` into a synthetic
`configparser.ConfigParser` populated with the legacy section
and key names. Coverage:

- `[General settings]` — `project_path` (= v1 root, not a
  `project_folder/` subdir), `project_name`, `file_type`,
  `workflow_file_type`, `animal_no`, `os_system`
- `[SML settings]` — `model_dir`, `no_targets`, `target_name_N`,
  `model_path_N` for each classifier
- `[threshold_settings]`, `[Minimum_bout_lengths]` — per-
  classifier placeholders (`"NaN"`)
- `[create ensemble settings]` — pose preset code +
  ML-training placeholders (`"NaN"` for unset)
- `[Multi animal IDs]` — comma-joined `id_list`
- `[Outlier settings]` — movement / location criteria
- Plus `[Frame settings]`, `[Line plot settings]`,
  `[Path plot settings]`, `[ROI settings]`, `[Directionality
  settings]`, `[process movement settings]` (empty placeholders
  the legacy creator wrote)

Missing TOML values default to `"NaN"` (the legacy
`Dtypes.NONE` sentinel) so existing
`read_config_entry(..., default_value=...)` calls behave
identically.

**`mufasa/utils/read_write.py`** — `read_config_file` now
branches on the suffix: `.toml` → delegates to the shim;
anything else → the original INI parser. Case-insensitive
(`.TOML` works too). Legacy projects opened after a migration
keep their existing path verbatim.

**`mufasa/mixins/config_reader.py`** — `ConfigReader.__init__`
detects v1 by suffix, pre-loads the parsed TOML on
`self._v1_toml_data`, and routes the body-parts handling
through TOML rather than the `project_bp_names.csv` file. A
new method `_apply_v1_path_overrides` runs early in `__init__`
(before any filesystem read) and rewrites the 30+ path
attributes — `input_csv_dir`, `outlier_corrected_dir`,
`features_dir`, `targets_folder`, `machine_results_dir`,
`video_dir`, `video_info_path`, all the plot dirs, the SHAP /
directionality / ROI / clf-validation dirs, plus `logs_path`
and `roi_coordinates_path` — to v1 equivalents under
`sources/`, `derived/`, `models/`, `logs/`. Multi-run stages
(outlier_corrected, features, classifications) resolve to the
**latest run** subdir (`derived/<stage>/<run_id>/`) when any
exist, falling back to the stage parent. File lists are re-
globbed against the new paths.

#### configparser removal from ui_qt forms (patch 122f)

Two new helpers in **`mufasa/project_layout.py`**:

- `project_paths_from_config(config_path)` → dict mapping
  `project_root`, `video_dir`, `input_pose_dir`, `logs_dir`,
  `video_info_path`, `models_dir` to v1 or legacy equivalents
  depending on the config suffix.
- `project_metadata_from_config(config_path)` → dict with
  `animal_count`, `file_type`, `body_parts`, `animal_ids`,
  `classifier_targets`, `pose_config_code`. v1 reads from
  `project.toml` directly; legacy parses the INI plus the
  `project_bp_names.csv` file.

Forms migrated (every direct `configparser.ConfigParser()` call
in form methods replaced):

- **pose_cleanup.py** — `SmoothingForm.target`,
  `InterpolateForm.target`, `OutlierSettingsForm.build/target`,
  `EgocentricAlignmentForm.build/collect_args`,
  `KalmanV2SmoothingForm.build` (legacy fallback),
  `RunOutlierCorrectionForm.build/target`. The
  read-modify-write of outlier settings is encapsulated in two
  module-level helpers `_read_outlier_settings` and
  `_write_outlier_settings`. v1's TOML schema gains a nested
  `[outlier_settings.references]` table
  (`Animal_X = ["bp1", "bp2"]`) mirroring the legacy
  per-animal reference keys.
- **classifier.py** — `_refresh_remove_options`,
  `_add_classifier`, `_remove_classifier` now use two
  module-level helpers `_read_classifiers` and
  `_write_classifiers`. v1 read-modify-writes `project.toml`'s
  `[classifiers].targets`.
- **analysis.py** — `_load_classifier_names` delegates to
  `project_metadata_from_config`. Top-level `import
  configparser` removed.
- **visualizations.py** — the data-paths auto-population path
  uses `project_paths_from_config` + `project_metadata_from_config`.
  Inline `import configparser` inside `target` removed.
- **addons.py** — `_load_cue_light_names` branches on suffix.
  v1 returns `[]` (cue-light ROI names aren't in the v1 schema
  yet; form falls back to free-text entry).
- **annotation.py**, **roi.py** — dead `import configparser`
  lines removed (no other use).

Net effect: every form *method* is `configparser`-free.
`configparser` references that remain are in three places, all
encapsulated:

1. `pose_cleanup.py` — legacy branches of `_read_outlier_settings`
   and `_write_outlier_settings`
2. `classifier.py` — legacy branch of `_write_classifiers`
3. `addons.py` — legacy branch of `_load_cue_light_names`

### Test counts at session end

| suite                                    | count    |
|------------------------------------------|----------|
| smoke_project_layout                     |   9/9    |
| smoke_migrate_project                    |   5/5    |
| smoke_recent_project                     |   6/6    |
| smoke_pose_cleanup_v2_wiring             |   2/2    |
| smoke_model_dual_save                    |  23/23   |
| smoke_input_source_picker                |  29/29   |
| smoke_outlier_forms_wiring               |  40/40   |
| smoke_empty_classifier                   |   1/1    |
| smoke_config_creator_v1                  |  38/38   |
| smoke_v1_configreader (NEW, 122e)        |  69/69   |
| smoke_v1_form_configparser_removal       |  37/37   |
| (NEW, 122f)                              |          |

### Sandbox testing limitation

ConfigReader itself can't be instantiated in this sandbox —
its import chain pulls cv2, h5py, trafaret, and tkinter, none
of which are available. The 122e test exercises the three
layers it can reach (the TOML shim behaviorally, the
`read_config_file` routing via AST, the ConfigReader v1 logic
via AST). Full behavioral verification — actually constructing
a `ConfigReader(config_path=project.toml)` and inspecting its
attribute surface against the expected v1 paths — is the
user's to confirm on a real install. The wiring is testable
even if the runtime isn't.

### Deliberately deferred

**Backend writes leak provenance.** ConfigReader's
`_apply_v1_path_overrides` resolves multi-run stage attributes
to the **latest run** subdir. That's correct for reads but wrong
for writes — a backend that does `os.makedirs(self.outlier_corrected_dir,
exist_ok=True)` and then writes files there will mutate the
prior run's directory in place, corrupting provenance. Each
writing backend needs `ProjectPaths.stage_run_dir(...)`
allocation at run time instead. Track as the "v1-aware backend
writes" thread.

**Configs directory.** `configs_meta_dir` falls back to
`<root>/configs/` for v1; the v1 layout doesn't model a configs
dir yet.

**Body-parts path in v1.** `body_parts_path` is set to the
project.toml path itself (informational only — v1 reads body
parts from TOML data directly). Any legacy code that tries to
parse this file as a CSV will fail loudly.

**reconfigure_project_user_defined.** Still INI-only.
Currently unreachable from CreateProjectDialog. Kept for the
File → Reconfigure menu action; v1 rewrite is a small patch
when needed.

**Cue-light ROI metadata in v1.** `addons.py._load_cue_light_names`
returns `[]` for v1. If cue-lights matter going forward, the v1
schema needs a `[roi.cue_lights]` table plus form-side writers.

### Pickup checklist for next session

1. Confirm tests green:
   ```bash
   for t in tests/smoke_*.py; do PYTHONPATH=. python "$t"; done
   ```
2. **Full ConfigReader behavioral verification** on a real
   install — instantiate against a v1 project.toml and walk
   the attribute surface. Anything still under `csv/` instead
   of `sources/`/`derived/` is a missed override.
3. **v1-aware backend writes** — the highest-priority deferred
   thread. Each writing backend (outlier correction, feature
   extraction, classifier training, visualization) needs
   `ProjectPaths.stage_run_dir(...)` allocation at run time
   instead of reading a "latest" path from ConfigReader.
4. **Batch pipeline through inference** — still the original
   ask from session 2026-05-11; never started.
5. **Migration prompt on legacy projects** — when
   `detect_layout(project) == 'legacy'`, dialog at workbench
   launch offering migration. Small patch.

**Also fixed here: a regression this session shipped.** 122he rephrased the
divergence message and broke `tests/smoke_kalman_pose_smoother_v2.py`, which
pinned the string "EKF has diverged". It went unnoticed because the sweep glob
is `smoke_122*.py` and **73 test files sit outside it**, that one included. The
suite was 152/152 before this session and 122he is already pushed. Both sweeps
are now baseline-diffed against 5671eed (pre-session): 264 suites, no
regressions.

### Next after 122hf
* The sweep glob is a verification hole. `smoke_122*.py` is what the handoff
  prescribes, but it misses 73 files; the loop should cover `tests/*.py`.
* `--joint-prior` is opt-in and unproven on real data. On strohA-kj the fit
  would learn mu/kappa for `tail_V32` from a state its sigma=225px marker
  barely influences; kappa may come back under the floor and disable itself.
  Honest, but it means the sigma=225 problem has to be solved, not routed
  around.
* A 20 s reconstruction at 18 deg RMSE is a prior talking, not a measurement.
  Anything scoring tail kinematics needs to know which it is looking at.

---

## Session 2026-05-11 — Dual-save model store + InputSourcePicker + pose-cleanup redesign + v1-only new projects

Three patch series landed: **122b** factors model provenance and
input-source selection into reusable primitives; **122c** uses
them to redesign the pose-cleanup page with explicit Run / Skip
outlier-correction forms and the legacy forms folded into an
"Advanced / legacy" section; **122d** cuts new-project creation
over to v1-only and makes the Kalman v2 form v1-aware so the
fresh-project pose-cleanup workflow runs end-to-end without
touching legacy paths.

### Shipped

#### Dual-save model store (patch 122b)

Trained models now live in two places. The **global cache** at
`~/.config/mufasa/models/<name>.npz` is a cross-project library
— a model trained once is discoverable from any project on the
same machine. The **project store** at `<project>/models/<name>/`
holds `model.npz` plus a `card.toml` carrying provenance
(`source_path`, `sha256`, `copied_at`, `mufasa_version`). Both
get a copy on every save / load crossing.

- New `project_layout.py` helpers: `import_model_into_project`,
  `mirror_model_to_global_cache`, `file_sha256`,
  `global_model_cache_dir`, `resolve_v1_project_root`.
  Content-hash idempotence: re-importing/mirroring the same
  model is a no-op. Different-content collision raises
  `FileExistsError` rather than silently overwriting (training
  output passes `overwrite=True`; load-side imports use the
  default no-overwrite to protect known-good models).
- `KalmanV2SmoothingForm.target`: a `_post_train_dual_save()`
  closure runs after every `smooth_pose_v2(save_model=...)`
  call (both single-pass and two-pass training paths); the
  load branch imports the chosen model into the project's
  `models/` before passing it to the smoother. Failures in
  either are soft — logged, but the original `.npz` is left
  intact.

#### InputSourcePicker (patch 122b)

New module `mufasa/ui_qt/input_source_picker.py`.

- `discover_input_sources(config_path, project_root)` — pure
  function. Walks both layouts: legacy `csv/{input_csv,
  outlier_corrected_movement, outlier_corrected_movement_location,
  smoothed_v2}/` and v1 `sources/pose/` + `derived/smoothed/
  <flavor>/<run>/` + `derived/outlier_corrected/<run>/`. v1
  candidates come first when both are present (post-migration
  transient). Multi-run stages list one entry per run, newest
  first. Empty / nonexistent dirs are skipped silently.
- `_DEFAULT_PREFER_ORDER`: kalman_v2 → outlier_corrected →
  savitzky → raw. Exactly one source is marked `is_default`.
  Overridable per call (the new `RunOutlierCorrectionForm`
  prefers raw, since outlier correction usually precedes
  smoothing).
- `InputSourcePicker` Qt widget: combobox + path field + browse
  + refresh (`↻`). Custom-path sentinel un-greys the path field.
  PySide6 imported lazily so the pure discovery function tests
  headless.
- **First consumer**: `EgocentricAlignmentForm`. The previous
  auto-detect of `csv/outlier_corrected_movement_location/`-vs-
  `csv/input_csv/` is gone; users can now feed egocentric
  alignment any prior stage's output, including a specific
  Kalman v2 run.

#### Pose cleanup redesign (patch 122c)

- New `RunOutlierCorrectionForm`. Chains
  `OutlierCorrecterMovement` → `OutlierCorrecterLocation`. Input
  via `InputSourcePicker` (preference RAW). Per-stage checkboxes
  let users disable movement or location independently. Output
  dir auto-generates `derived/outlier_corrected/<run_id>/` on v1
  projects (with `run.toml` provenance written after) and falls
  back to legacy `csv/outlier_corrected_movement_location/`
  otherwise. When both stages run, movement writes to a
  `_movement_intermediate/` sibling that location reads from.
- New `SkipOutlierCorrectionForm`. Thin wrapper around
  `OutlierCorrectionSkipper` — no fields, just a Run button.
  Writes `run.toml` with `params.skipped = true` and a
  `SKIPPED` marker file when run inside a v1 project so future
  layout-aware tooling can distinguish "skipped" from "missing"
  runs.
- Pose cleanup page sections reordered to match the conceptual
  pipeline:

  ```
  1. Interpolate missing frames
  2. Kalman v2 smoothing
  3. Run outlier correction         ← NEW
  4. Skip outlier correction        ← NEW
  5. Egocentric alignment           (now picker-driven)
  6. Advanced / legacy              (stacks SmoothingForm,
                                     OutlierSettingsForm,
                                     DropBodypartsForm)
  ```

  Legacy `Smoothing` / `Outlier correction settings` /
  `Drop body-parts` top-level sections are gone; their forms
  still exist (settings is still consumed by the new outlier
  forms — it writes the thresholds they read) but they're
  folded into one Advanced section using
  `WorkflowPage._instantiate`'s existing form-stacking.

#### v1-only new projects + Kalman v2 v1-awareness (patch 122d)

`ProjectConfigCreator` was rewritten end-to-end. The legacy
SimBA tree generator (the `__create_directories` /
`__create_configparser_config` pair that built `project_folder/
csv/{input_csv, outlier_corrected_*, features_extracted, ...}`
+ `project_config.ini`) is gone. New projects land as v1:

```
<parent>/<project_name>/
├── project.toml                       (new)
├── sources/{videos,pose,annotations}/  (via ProjectPaths.ensure_skeleton)
├── derived/
├── models/
└── logs/
```

The new `project.toml` schema is small and explicit:

- top-level: `project_layout_version`, `project_name`,
  `created`, `mufasa_version`, `os_platform`
- `[pose]`: `animal_count`, `file_type`, `body_parts` (list,
  in order), `pose_config_code`, `pose_config_idx`, `animal_ids`
- `[classifiers]`: `targets` (may be empty)
- `[outlier_settings]`: `movement_criterion`, `location_criterion`
  (initial `"NaN"` matches the legacy `Dtypes.NONE` sentinel
  so existing settings code can read it verbatim once it's TOML-
  aware)

The Qt create-project dialog (`CreateProjectDialog`) was
simplified accordingly: the previous "create with placeholder
preset, then call `reconfigure_project_user_defined` to patch
in the autodetected body parts" two-step is gone. Both the
preset and autodetect paths now funnel body parts directly into
`ProjectConfigCreator` via the new `body_parts=` parameter, so
the project is fully configured on first write. The dialog's
docstring and the project-setup-page banner were updated to
say `project.toml` instead of `project_config.ini` (with the
banner still acknowledging legacy `project_config.ini` for
projects opened from before this cutover).

`KalmanV2SmoothingForm` was made v1-aware so the fresh-project
pose-cleanup workflow runs end-to-end on v1 without any legacy
path resolution:

- **Input dir** defaults to `<project>/sources/pose/`.
- **Output dir** defaults to
  `<project>/derived/smoothed/kalman_v2/<run_id>/`. A fresh
  `run_id` is allocated at form-build time (cached on
  `self._v1_run_id`) so the path the user sees in the field is
  exactly what gets written on Run. Closing and reopening the
  form gives a fresh `run_id`; the form never overwrites a
  prior run.
- **Save-model path** defaults to `<run_dir>/model.npz`,
  co-located with the smoothed pose data. The 122b dual-save
  flow still mirrors to `~/.config/mufasa/models/model.npz` and
  imports into `<project>/models/model/` on top.
- After smoothing completes, the form writes a
  `<run_dir>/run.toml` provenance file tagged
  `stage = "smoothed.kalman_v2"`, with `params` capturing every
  scalar/list kwarg passed to `smooth_pose_v2` plus the mode
  (`"train"` or `"load"`). Soft-fails (logged) if anything goes
  wrong — pose data already on disk is left intact.

Legacy projects still work via the existing `configparser`
fallback in `build()`. `resolve_v1_project_root` distinguishes
the two cases.

Net effect: the data flow for the v1 pose-cleanup pipeline is
now self-consistent — raw pose lives at `sources/pose/`, the
Kalman v2 smoother writes to
`derived/smoothed/kalman_v2/<run_id>/`, and the egocentric
alignment form's `InputSourcePicker` surfaces both raw and any
prior smoothed run as candidates (with the most-processed run
as the default). Subsequent pose-cleanup operations read from
raw or smoothed exactly as intended.

### Test counts at session end

| suite                          | count    |
|--------------------------------|----------|
| smoke_kalman_pose_smoother_v2  | 152/152  |
| smoke_project_layout           |   9/9    |
| smoke_migrate_project          |   5/5    |
| smoke_recent_project           |   6/6    |
| smoke_pose_cleanup_v2_wiring   |   2/2    |
| smoke_model_dual_save          |  23/23   |
| smoke_input_source_picker      |  29/29   |
| smoke_outlier_forms_wiring     |  40/40   |
| smoke_empty_classifier         |   1/1    |
| smoke_config_creator_v1        |  38/38   |

`smoke_model_dual_save` redirects `HOME` so it doesn't pollute
the real `~/.config/mufasa/models/`. `smoke_input_source_picker`
covers the pure discovery function behaviorally and the Qt
widget at AST level (PySide6-free sandbox). The wiring tests
are AST-only. `smoke_empty_classifier` was rewritten from its
legacy INI shape into a v1-`project.toml` shape; the original
contract (empty `target_list` is accepted) is preserved.

### Deliberately deferred

**Model store**

- Trained-model auto-naming. The form's default save path is
  now `<run_dir>/model.npz` for v1 projects (one model per run,
  collision-free by construction); the global cache still flat-
  names them as `model.npz` which IS collision-prone. Probably
  want to use the run_id as the global-cache name too.
- Card.toml schema is minimal — `model_name`, `source_path`,
  `sha256`, `copied_at`, `mufasa_version`. No training params,
  no eval metrics, no fingerprint of the training data. The
  changelog's earlier note on `card.toml` for classifiers still
  applies; this is the v0.

**InputSourcePicker**

- Picker doesn't show file counts or last-modified time per
  source. For a project with many runs it'd be useful to know
  which has data and how fresh it is. Two lines per item in the
  combobox would do it.
- No source preview / inspect button. Picking a run and seeing
  its `run.toml` parameters before committing would help.

**Pose cleanup**

- "Drop body-parts" still lives in Advanced / legacy. It's
  really a project-setup decision, not a per-run cleanup step.
  Move it to the project-setup page once that's the right
  surface.
- `RunOutlierCorrectionForm` writes `run.toml` for v1 but
  doesn't yet enumerate the actual input file list (params has
  the `data_dir`, not the per-file paths). The `write_run_toml`
  `inputs` field accepts a list; populating it would make
  full reproducibility checkable.
- `SkipOutlierCorrectionForm` writes a v1 `run.toml` next to a
  `SKIPPED` marker file but doesn't actually populate the v1
  `derived/outlier_corrected/<run>/` with the skip-copied
  files. That's because `OutlierCorrectionSkipper` writes to
  the legacy `csv/outlier_corrected_movement_location/` and
  isn't yet v1-aware. Fixing this is part of the broader
  "make each backend v1-aware" thread.
- The legacy `SmoothingForm` (Savitzky-Golay) is still mounted
  in Advanced. Once we're confident Kalman v2 covers every
  pre-Kalman use case the dataset hit, this form can be
  removed entirely.

**v1 cutover (122d) — the iceberg below the waterline**

The 122d patch makes new projects v1, but the downstream forms
that consume those projects are still mostly INI-driven. The
**only** form that fully works in a fresh v1 project right now
is `KalmanV2SmoothingForm` (because its target reads pose
directly from `input_dir` without ever touching
`config_path`). Every other form — `InterpolateForm`,
`RunOutlierCorrectionForm`, `SkipOutlierCorrectionForm`,
`EgocentricAlignmentForm`, the legacy three, plus every form
on every other page — calls into a backend that uses
`ConfigReader` to parse `config_path` as an INI. In a v1
project `config_path` points to `project.toml`, and those
backends will fail to read it.

The migration thread that fixes this:

1. Add a `ConfigReader` subclass (or a `read_project_toml`
   helper) that reads `project.toml` and exposes the same
   attributes the legacy ConfigReader does (`animal_cnt`,
   `body_parts`, `file_type`, `input_csv_dir`, etc.). For v1,
   `input_csv_dir` resolves to `sources/pose/`,
   `outlier_corrected_dir` to a latest-run path under
   `derived/outlier_corrected/`, and so on.
2. Route every `ConfigReader`-using backend through the
   subclass.
3. Make each form's `target()` pass the right v1 paths down
   so its backend never has to guess.

That's a multi-session thread; the priority order matches the
pose-cleanup page's section order (Interpolate, Outlier-Run,
Outlier-Skip, Egocentric backend, Advanced/legacy).

Until that's done, the practical v1 workflow is:

- Drop raw pose into `<project>/sources/pose/`.
- Run **Kalman v2 smoothing** — produces
  `derived/smoothed/kalman_v2/<run>/`.
- (Other pose-cleanup forms will fail with a config error.)
- Egocentric alignment partially works — the form constructs
  fine (no INI needed in the form itself, since the picker
  drives the data path), but its backend
  (`EgocentricalAligner`) reads `config_path` and may fall
  over depending on how deeply it relies on the INI.

In short: **122d unlocks the smoother in v1 mode; the rest of
the pipeline catches up patch by patch.**

### Open questions

- Should `RunOutlierCorrectionForm`'s save_dir picker accept a
  v1 run directory the user already created (rather than always
  auto-generating)? Useful for re-running with tweaked
  thresholds in the same provenance dir. Currently auto-gen
  only.
- The InputSourcePicker shows v1 runs newest-first. For a
  project with dozens of runs this could get unwieldy. A
  "latest only" toggle (showing just the most recent run per
  flavor) would help; the full list could be reachable via an
  "All runs…" entry. Not urgent.
- The legacy `reconfigure_project_user_defined` helper still
  exists but isn't called by `CreateProjectDialog` anymore.
  Keep it for the File → Reconfigure project from DLC file…
  menu action, or retire it entirely? Probably keep until the
  v1-aware reconfigure equivalent exists.

### Pickup checklist for next session

1. Confirm tests still green:
   ```bash
   PYTHONPATH=. python tests/smoke_project_layout.py && \
   PYTHONPATH=. python tests/smoke_migrate_project.py && \
   PYTHONPATH=. python tests/smoke_recent_project.py && \
   PYTHONPATH=. python tests/smoke_pose_cleanup_v2_wiring.py && \
   PYTHONPATH=. python tests/smoke_model_dual_save.py && \
   PYTHONPATH=. python tests/smoke_input_source_picker.py && \
   PYTHONPATH=. python tests/smoke_outlier_forms_wiring.py && \
   PYTHONPATH=. python tests/smoke_empty_classifier.py && \
   PYTHONPATH=. python tests/smoke_config_creator_v1.py
   ```
2. Three threads ready to pick up:
   - **v1-aware ConfigReader.** Highest priority — every form
     beyond Kalman v2 fails in a fresh v1 project until this
     exists. See "iceberg below the waterline" section above.
   - **Batch pipeline through inference** (still the original
     ask from this session — scoped out, never started).
     Schema for `pipeline.toml`, CLI scaffold, ephemeral-project
     synthesizer, four stage adapters. Estimated 1500–2500 LOC
     across one or two sessions.
   - **Migration prompt on legacy projects.** Original
     deferred item from 122a: when `detect_layout(project) ==
     'legacy'`, surface a dialog at workbench launch offering
     migration. Small patch.

---

## Session 2026-05-08 to 2026-05-11 — Kalman v2 smoother feature completion + workbench wiring + project-layout redesign

### Shipped

#### Smoother backend (patch 121d–e)

- **121d — constant-acceleration extension** for selected segments.
  Listed segments gain an acceleration block at the end of the state
  vector (+4 dims for root translation + orientation, +2 dims per
  non-root segment). F gets integrator cross-terms (`v_new = v + a·dt`,
  `x_new = x + v·dt + a·dt²/2`); Q gets diagonal jerk noise on accel
  slots. FK and Jacobian unchanged. New CLI flag
  `--const-accel-segments`. Backward compatible (empty list = no-op).
  Tests 145-148 (4 new on top of 121b/c's 144).

- **121e — M-step learning of q_jerk_***. EM now adapts `q_jerk_root_pos`,
  `q_jerk_root_ori`, `q_jerk_seg_ori` from Q_hat's accel-block
  diagonals instead of holding them at the initial 10×-q-root default.
  Floor/ceiling/hard-cap pattern from 121c
  (`_M_STEP_Q_JERK_*_HARD_CAP = 500000 / 500 / 500`). Per-session
  aggregator still passes through prev values — deferred. Tests 149-152.

#### Workbench wiring (patch 121f–i)

- **121f — Kalman v2 smoother in Pose Cleanup page**. New
  `KalmanV2SmoothingForm` section sitting below the legacy Savitzky-
  Golay smoother. Calls `smooth_pose_v2` via Python API (not
  subprocess) so errors surface in the progress dialog. Fields:
  input/output dirs, fps, likelihood threshold, EM iters, workers,
  per-marker drift, orientation drift segments, const-accel segments,
  save-model checkbox. PySide6-independent AST-level wiring smoke
  test.

- **121g — train/load mode + training subset + EM parameters**. Mode
  selector at top of form (radio buttons: train new / load saved).
  Training-mode group adds a file-list picker for subset selection
  (auto-refreshes on input-dir change, with select-all/clear/refresh
  buttons), full EM parameter surface (tolerance, damping, pooled-vs-
  per-session aggregation, warm-start σ, perspective, validation),
  save-model path with browse. Load-mode group surfaces the model
  file picker. Two-pass workflow runs when training on a subset: fit
  on subset → load → smooth all.

- **121h — standard model directory at `~/.config/mufasa/models/`**.
  Module-level helper `_default_model_dir()`. Five touchpoints
  (placeholder, save-browse, load-browse, collect_args, two-pass
  workflow) route through it. Models trained once are discoverable
  from any project on the same machine.

- **121i — persistent recent-project at `~/.config/mufasa/recent`**.
  New `mufasa/ui_qt/recent_project.py` module (PySide6-free, testable
  headless). Launch priority: `--project` → CWD auto-discover →
  recent file → none. `--no-recent` flag disables fallback.
  `_switch_to_project` saves on every project switch. Stale/empty/
  unwriteable entries fail silently.

#### Project layout redesign (patch 122a)

Major restructure — patch number stepped to 122 to signal the break.

- **122a — full project-layout v1 + migration tool**. New layout:

  ```
  <project_root>/
  ├── project.toml                     # was project_config.ini
  ├── sources/{videos,pose,annotations}/   # read-only inputs
  ├── derived/                         # generated; safe to delete
  │   ├── smoothed/<flavor>/<run_id>/
  │   ├── outlier_corrected/<run_id>/
  │   ├── features/<run_id>/
  │   ├── classifications/<run_id>/
  │   └── frames/{extracted,annotated}/
  ├── models/<model_name>/
  └── logs/<run_id>/
  ```

  Three new modules: `mufasa/project_layout.py` (`ProjectPaths`,
  `Stages`, `SmoothingFlavors`, `generate_run_id`, TOML I/O, version
  validation), `mufasa/legacy_layout.py` (SimBA-to-v1 mapping table,
  INI parser preserving unknown sections under `[legacy.*]`,
  body-part-name file reader), and `mufasa/cli/migrate_project.py`
  (`MigrationPlan` separates planning from execution, dry-run default,
  `--commit` to actually move, idempotent on already-migrated
  projects, `MIGRATION.toml` manifest written for auditability).

  Run-id format: `YYYYMMDD-HHMMSS-<6hex>` — sortable lexically =
  sortable chronologically. Per-run `run.toml` carries stage,
  parameters, input file list, mufasa version, results.

  Tests: `tests/smoke_project_layout.py` (9 checks: constants, run-id
  format, ProjectPaths skeleton, stage_run_dir, list_runs/latest_run,
  TOML roundtrip, version guard, run.toml roundtrip, detect_layout)
  and `tests/smoke_migrate_project.py` (5 checks: synthetic SimBA
  project → migration → verify all files at expected v1 locations +
  project.toml content + MIGRATION.toml manifest + idempotence + dry-
  run + unknown-path exit code + project_folder-pointed call shape).

### Test counts at session end

| suite                              | count    |
|------------------------------------|----------|
| smoke_kalman_pose_smoother_v2      | 152/152  |
| smoke_workbench_auto_discover      |  10/10   |
| smoke_workbench_launcher           |  10/10   |
| smoke_pose_cleanup_v2_wiring       |   2/2    |
| smoke_recent_project               |   6/6    |
| smoke_project_layout               |   9/9    |
| smoke_migrate_project              |   5/5    |

### Real-data observations worth preserving

Confirmed during 121bc real-data runs (67-session dataset, 30 fps):

- `--orient-drift-segments body` made the tail look better and gave
  honest larger variance circles (the over-confidence from before was
  hiding real uncertainty), but head markers became too constrained —
  root angular residual no longer absorbing.
- Adding `head` to drift loosened it. `body,head` worked.
- With drift wired and `--const-accel-segments body,head`, motion was
  slightly over-smoothed during fast turns. Hypothesis: the constant-
  velocity predictor lags on accelerating motion; const-accel should
  help. 121d/e land that.
- Useful knobs found during tuning: `--likelihood-threshold` (lower
  includes more obs, less prior reliance), q-scaling via save/load
  model.
- Recommended config for this dataset:
  `--with-drift --orient-drift-segments body,head --const-accel-segments body,head --fps 30 --em-max-iter 20 --workers 12`.

### Deliberately deferred

Notes on shape, not commitment to implementation — pick whichever
matters next.

**Smoother backend**

- M-step learning of q_jerk in the per-session aggregator. Currently
  only the pooled path (the default) learns; the per-session path
  passes through prev_params. Pattern would mirror what's in
  `finalize_m_step_v2` already — same sufficient stats, same caps,
  per-session aggregation via the existing scalar/median machinery.

**Workbench**

- Live progress reporting from `smooth_pose_v2`. The form currently
  shows an indeterminate progress dialog; the `[smoother-v2] ...`
  lines still go to stdout. A Qt-friendly progress-callback parameter
  on `smooth_pose_v2` plus a signal-based bridge into the dialog
  would surface them. Worth doing but not blocking.
- Cancel button. Dialog has one but `smooth_pose_v2`'s worker pool
  doesn't poll for cancellation between sessions.
- Multi-segment pickers. Orientation-drift and const-accel segments
  are typed as comma-separated strings. Checkbox grid against the
  layout's actual segment list would be a UX upgrade.
- Model inspection. Loading a model just feeds it to the smoother;
  the form doesn't surface the layout extensions or fitted q values
  from the loaded file. An "Inspect" button could read the npz and
  show a summary.

**Project layout / consumer migration**

The 122a patch lands the layout primitives but NOT the form
integration. Recommended order for the next session:

1. **Launcher UX**: when `detect_layout(project_root) == 'legacy'`,
   pop a dialog offering to run `migrate_project`. Small patch
   touching `workbench_app.py`'s startup path.
2. **`KalmanV2SmoothingForm` v1-aware**: when `config_path` resolves
   to a v1 project, default output_dir to
   `ProjectPaths.smoothed_run_dir(SmoothingFlavors.KALMAN_V2)` (auto-
   generated run id) and default save_model to
   `<run_dir>/model.npz`. Falls back to legacy defaults for legacy
   projects. Write `run.toml` after each successful run.
3. **Each other form** (outlier correction, features, classification)
   gets the same one-form-at-a-time treatment. Stop once the user's
   workflow doesn't hit a legacy path anymore.
4. **`ConfigReader` subclass** that reads `project.toml`. Existing
   consumers route through it transparently.
5. **Run-id selector** in the v2 form: optional "Run name" field
   (default: auto), and the "Load saved model" picker enumerates
   `derived/smoothed/kalman_v2/*/model.npz` from the current project
   when one is loaded.
6. Once all forms migrated: deprecate the legacy INI reader. The
   legacy-layout reader stays as long as we want migrate-project to
   work.

**Schemas to design later**

- `models/<model_name>/card.toml` for trained classifiers — what they
  were trained on, when, what params. Migration just moves
  `models/` as-is right now; the format for new training runs is open.
- Anything that wants a stable schema in `project.toml` beyond what
  `parse_legacy_config` produces (animal_count, body_parts, stage
  defaults). Worth letting it grow organically as consumer code
  migrates.

### Open questions

- Is `~/.config/mufasa/models/` the right place for the global model
  cache once projects have their own `<project>/models/`? Two options:
  keep both (global cache for cross-project reuse, project-local for
  reproducibility) or drop the global one and require explicit copy.
  No urgency — current 121h behavior is fine for the immediate need.
- Migration of `logs/` is bucketed under `logs/imported_<date>/`
  alongside per-run dirs. If existing log-parsing code expects flat
  `logs/`, will need a small compat shim — but nothing observed yet
  to suggest the migration would break anything.

### Pickup checklist for next session

1. `cd` into the project, confirm tests still green:
   `python tests/smoke_kalman_pose_smoother_v2.py && \
    PYTHONPATH=. python tests/smoke_project_layout.py && \
    PYTHONPATH=. python tests/smoke_migrate_project.py`
2. Decide between the layout-consumer work (item 1 above) vs. the
   smoother polish work (live progress, cancel) vs. continuing
   real-data tuning of the 121d/e behavior.
3. If continuing real-data tuning: run with the recommended config
   above, compare against the 121bc baseline you have. q_jerk values
   are now in the saved model; check whether EM is adapting them
   reasonably or hitting caps. If hitting caps, the cap values may
   need rethinking.
