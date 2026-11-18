"""Smoke test for patch 122ko — 3-D features in Compute Features.

User: "I would like 3d features added to the feature list."

A generated project with a 2-D pose file and a synthetic 3-D pose file
(rat skeleton markers, a missing stretch of frames, a rearing second half):

* the four 3-D families are in FEATURE_FAMILIES and have their own group
  on the form, with a choice of 3-D source (newest run, each run, motion
  capture); collect_args carries the choice;
* values: heights, 3-D distances, movements; posture signs — head pitch up,
  head roll right side up, body pitch when rearing, spine bend (concave
  back positive, hunched negative), pelvis / lumbar columns passed through;
* rows follow the video's frames: frames missing from the 3-D file are
  NaN (heights, angles) or -1 (distances, movements) in the written parquet;
* the 3-D file is found in the newest derived/pose3d run, else
  sources/pose3d; an explicit folder wins;
* only 3-D families and no 3-D file: a clear error; with 2-D families as
  well: the 3-D ones are skipped, the 2-D ones written.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


T = 60
MISSING = range(10, 15)            # frames the 3-D file lacks
SPEED = 2.0                        # mm per frame along +x
SPINE = ["tail_V6", "back_V2", "back_L2", "back_T4", "head_back"]


def synthetic_3d() -> pd.DataFrame:
    """Rat walking along +x; from frame 30 rearing (head end raised 40°),
    head pitched up 20°, right side of the head 10° up; the spine is
    concave (bending up toward the head) in the first half, hunched in the
    second."""
    rows = []
    for k in range(T):
        if k in MISSING:
            continue
        x0 = SPEED * k
        rear = k >= 30
        body = np.radians(40.0 if rear else 0.0)
        # spine segment elevations relative to the body line: concave (+8°
        # per joint) or hunched (−8° per joint)
        step = -8.0 if rear else 8.0
        pts = {"tail_V6": np.array([x0, 0.0, 20.0])}
        el = np.degrees(body) - 1.5 * step
        p = pts["tail_V6"]
        for m in SPINE[1:]:
            e = np.radians(el)
            p = p + 30.0 * np.array([np.cos(e), 0.0, np.sin(e)])
            pts[m] = p
            el += step
        hb = pts["head_back"]
        hp = np.radians(20.0)
        pts["head_nose"] = hb + 25.0 * np.array([np.cos(hp), 0.0, np.sin(hp)])
        hr = np.radians(10.0)
        mid = hb + 10.0 * np.array([1.0, 0, 0])
        pts["head_left"] = mid + 8.0 * np.array([0.0, np.cos(hr), -np.sin(hr)])
        pts["head_right"] = mid + 8.0 * np.array([0.0, -np.cos(hr), np.sin(hr)])
        row = {"frame": k, "pelvis_pitch_deg": 5.0 + k, "lumbar_angle_deg": 150.0}
        for m, X in pts.items():
            row.update({f"{m}_x": X[0], f"{m}_y": X[1], f"{m}_z": X[2], f"{m}_likelihood": 1.0})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    import mufasa.feature_extractors.feature_subsets as FS
    from mufasa.feature_extractors import features_3d as F3
    from mufasa.mixins.config_reader import ConfigReader
    from mufasa.project_layout import project_paths_from_config
    from mufasa.utils.config_creator import ProjectConfigCreator
    from mufasa.utils.errors import InvalidInputError
    from mufasa.utils.feature_io import family_slug

    check("the four 3-D families are in the feature list",
          all(f in FS.FEATURE_FAMILIES for f in F3.FAMILIES_3D) and len(F3.FAMILIES_3D) == 4
          and FS.FeatureFamily.POSTURE_3D.value == F3.POSTURE_3D)

    tmp = Path(tempfile.mkdtemp())
    bps = ["nose", "ear_l", "ear_r", "center", "tail"]
    cfg = ProjectConfigCreator(
        project_path=str(tmp), project_name="f", target_list=[],
        pose_estimation_bp_cnt="user_defined", body_part_config_idx=0,
        animal_cnt=1, file_type="parquet", body_parts=bps,
    ).config_path
    root = Path(cfg).parent
    P = project_paths_from_config(cfg)
    vd = Path(P["video_dir"])
    vd.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(vd / "v1.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 360))
    for _ in range(3):
        w.write(np.zeros((360, 640, 3), np.uint8))
    w.release()
    pd.DataFrame({"Video": ["v1"], "fps": [30], "Resolution_width": [640], "Resolution_height": [360],
                  "Distance_in_mm": [0], "pixels/mm": [2.0]}).to_csv(P["video_info_path"], index=False)
    cols = {}
    for i, b in enumerate(bps):
        cols[f"{b}_x"] = np.linspace(100, 300, T) + i * 5
        cols[f"{b}_y"] = np.full(T, 180.0)
        cols[f"{b}_p"] = np.full(T, 0.99)
    cr = ConfigReader(config_path=cfg, read_video_info=False)
    os.makedirs(cr.outlier_corrected_dir, exist_ok=True)
    pd.DataFrame(cols).to_parquet(os.path.join(cr.outlier_corrected_dir, "v1.parquet"))

    def calc(fams, **kw):
        return FS.FeatureSubsetsCalculator(
            config_path=cfg, feature_families=fams, file_checks=True, save_dir=None, data_dir=None,
            derived_features_dir=P["derived_features_dir"], n_workers=1, overwrite_existing=True, **kw)

    # no 3-D file yet
    try:
        calc([F3.HEIGHTS_3D]).run()
        raised = False
    except InvalidInputError as e:
        raised = "3-D pose file" in str(e)
    check("only 3-D families and no 3-D file: a clear error", raised)
    two_d = "TWO-POINT BODY-PART DISTANCES (MM)"
    calc([two_d, F3.HEIGHTS_3D]).run()
    feat = Path(P["derived_features_dir"])
    check("with a 2-D family too: 2-D written, 3-D skipped",
          (feat / family_slug(two_d) / "v1.parquet").is_file()
          and not (feat / family_slug(F3.HEIGHTS_3D)).exists())

    # 3-D files: mocap, an older run (decoy, wrong values), the newest run
    df3 = synthetic_3d()
    decoy = df3.copy()
    for c in decoy.columns:
        if c.endswith("_z"):
            decoy[c] = 999.0
    (root / "sources" / "pose3d").mkdir(parents=True, exist_ok=True)
    decoy.to_parquet(root / "sources" / "pose3d" / "v1.parquet")
    old = root / "derived" / "pose3d" / "20260101-000000"
    new = root / "derived" / "pose3d" / "20260301-000000"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    decoy.to_parquet(old / "v1_3d.parquet")
    df3.to_parquet(new / "v1_3d.parquet")
    src = F3.pose3d_sources(root)
    check("3-D sources: newest run first, motion capture last",
          [d.name for d in src] == [new.name, old.name, "pose3d"])
    check("find_pose3d_file: <video>_3d.parquet, else <video>*.parquet",
          F3.find_pose3d_file("v1", src) == new / "v1_3d.parquet"
          and F3.find_pose3d_file("v1", [src[-1]]) == root / "sources" / "pose3d" / "v1.parquet"
          and F3.find_pose3d_file("v2", src) is None)

    calc(list(F3.FAMILIES_3D)).run()
    def read(f):     # columns carry the feature-subset suffix, like the 2-D families
        d = pd.read_parquet(feat / family_slug(f) / "v1.parquet")
        return d.rename(columns=lambda c: c.removesuffix("_FEATURE_SUBSET"))
    out = {f: read(f) for f in F3.FAMILIES_3D}
    check("all four 3-D families written, one row per video frame",
          all(len(d) == T for d in out.values()))
    h = out[F3.HEIGHTS_3D]
    present = [k for k in range(T) if k not in MISSING]
    check("heights from the newest run (not the decoys)",
          np.allclose(h["Height (mm) tail_V6"].to_numpy()[present], 20.0))
    check("frames missing from the 3-D file: NaN heights",
          h["Height (mm) tail_V6"].iloc[list(MISSING)].isna().all())
    d = out[F3.DISTANCES_3D]
    want = np.linalg.norm(df3[["head_nose_x", "head_nose_y", "head_nose_z"]].to_numpy()
                          - df3[["head_back_x", "head_back_y", "head_back_z"]].to_numpy(), axis=1)
    check("3-D distance head_back-head_nose = 25 mm; missing frames -1",
          np.allclose(d["3-D distance (mm) head_back-head_nose"].to_numpy()[present], want)
          and np.allclose(want, 25.0)
          and (d["3-D distance (mm) head_back-head_nose"].iloc[list(MISSING)] == -1).all())
    check(f"{len(d.columns)} distance columns = every marker pair",
          len(d.columns) == 8 * 7 // 2)
    mv = out[F3.MOVEMENT_3D]["3-D movement tail_V6 (mm)"].to_numpy()
    walk = [k for k in range(1, T) if k not in MISSING and k - 1 not in MISSING]
    check("movement: 2 mm per frame walking; 0 at frame 0; -1 next to missing frames",
          np.allclose(mv[walk], SPEED) and mv[0] == 0 and mv[15] == -1 and mv[10] == -1)
    po = out[F3.POSTURE_3D]
    flat = [k for k in present if k < 30]
    rear = [k for k in present if k >= 30]
    check("head pitch +20° (nose above head_back)", np.allclose(po["Head pitch (degrees)"].to_numpy()[present], 20))
    check("head roll +10° (right side up)", np.allclose(po["Head roll (degrees)"].to_numpy()[present], 10))
    bp = po["Body pitch (degrees)"].to_numpy()
    check(f"body pitch: about 0° walking, about 40° rearing ({bp[flat].mean():.1f}, {bp[rear].mean():.1f})",
          abs(bp[flat].mean()) < 3 and abs(bp[rear].mean() - 40) < 3)
    check("head pitch relative to body = head pitch − body pitch",
          np.allclose(po["Head pitch relative to body (degrees)"].to_numpy()[present], 20 - bp[present]))
    bend_cols = [f"Spine bend at {m} (degrees)" for m in SPINE[1:-1]]
    b = po[bend_cols].to_numpy()
    check(f"spine bend at {len(bend_cols)} joints: +8° concave, −8° hunched",
          [c for c in po.columns if c.startswith("Spine bend at")] == sorted(bend_cols)
          and np.allclose(b[flat], 8) and np.allclose(b[rear], -8))
    check("pelvis pitch and lumbar angle passed through; NaN where missing",
          np.allclose(po["Pelvis pitch (degrees)"].to_numpy()[present], 5.0 + np.array(present))
          and po["Lumbar angle (degrees)"].iloc[list(MISSING)].isna().all())

    # explicit folder: the motion-capture decoy
    calc([F3.HEIGHTS_3D], pose3d_dir=str(root / "sources" / "pose3d")).run()
    h2 = read(F3.HEIGHTS_3D)
    check("an explicit 3-D folder wins", np.allclose(h2["Height (mm) tail_V6"].to_numpy()[present], 999))

    # the form
    from PySide6.QtCore import Qt

    from mufasa.ui_qt.forms.features import FeatureSubsetExtractorForm
    form = FeatureSubsetExtractorForm(config_path=cfg)
    fams3 = [form.families_3d.item(r).data(Qt.UserRole) for r in range(form.families_3d.count())]
    subj = [form.subject_families.item(r).data(Qt.UserRole) for r in range(form.subject_families.count())]
    check("form: '3-D features' group lists the 3-D families, not the subject list",
          fams3 == list(F3.FAMILIES_3D) and not set(fams3) & set(subj))
    labels = [form.pose3d_source.itemText(i) for i in range(form.pose3d_source.count())]
    datas = [form.pose3d_source.itemData(i) for i in range(form.pose3d_source.count())]
    check("form: 3-D source — newest, each run, motion capture",
          len(labels) == 4 and datas[0] is None and datas[1] == str(new)
          and "Motion capture" in labels[3])
    check("form: computed 3-D families badged",
          all("✓" in form.families_3d.item(r).text() for r in range(form.families_3d.count())))
    form.families_3d.item(0).setSelected(True)
    form.pose3d_source.setCurrentIndex(2)
    args = form.collect_args()
    check("form: collect_args carries the 3-D families and the source",
          args["feature_families"] == [F3.HEIGHTS_3D] and args["pose3d_dir"] == str(old))
    import inspect
    check("form: target() passes pose3d_dir on",
          "pose3d_dir" in inspect.signature(form.target).parameters)

    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122ko_features_3d: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
