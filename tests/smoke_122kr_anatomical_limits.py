"""Smoke test for patch 122kr — anatomical limits for single-camera 3-D pose
from DeepMind's virtual rodent.

User: "Only have one camera with deeplabcut markers and the three dimensional
estimates. Trying to keep this simple. Can we constrain the height estimates
with the blender and the biomechanical models?" … "Can you find the same for
a rat?" … "Incorporate and start the plan".

* the virtual rodent (dm_control rodent.xml + skin, Apache 2.0) is packaged
  with its licence and notice; the model loads (65 bodies, 67 hinge joints
  with their default-class ranges), the skin follows the skeleton (rest
  pose reproduces it), bending a joint moves the bodies behind it;
* the project's labels sit where they should on the model;
* the packaged limits: a height range per head/trunk label, the lumbar and
  neck bend ranges, the model's sizes;
* the bends are measured exactly as the spine chain measures them;
* mono: the head is moved along its rays into its range (still
  reprojecting), trunk labels are kept in theirs, the chain keeps the
  lumbar/neck bends in range (SciPy and PyTorch alike), the report says
  what changed; switching it off restores the previous result;
* the inference form and the CLI pass the option.
"""
from __future__ import annotations

import json
import os
import sys
import types
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_tk = types.ModuleType("tkinter")
_tk.messagebox = types.ModuleType("tkinter.messagebox")
sys.modules.setdefault("tkinter", _tk)
sys.modules.setdefault("tkinter.messagebox", _tk.messagebox)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


from mufasa.pose3d import rodent_model as rm  # noqa: E402
from mufasa.pose3d.camera import Camera  # noqa: E402
from mufasa.pose3d.mono import Anatomy, MonoConfig, load_pose2d, reconstruct_mono  # noqa: E402

D = rm.MODEL_DIR
check("virtual rodent packaged: model, skin, Apache licence, notice with the dm_control commit",
      (D / "rodent.xml").is_file() and (D / "rodent_walker_skin.skn").is_file()
      and "Apache License" in (D / "LICENSE").read_text()
      and "dm_control" in (D / "NOTICE.md").read_text() and (D / "virtual_rodent_limits.json").is_file())

m = rm.RodentModel()
check("model: 65 bodies, 67 hinge joints, 6880-vertex skin",
      len(m.body_names) == 65 and len(m.joints) == 67 and m.verts.shape == (6880, 3))
j = m.joints[m.joint_index["vertebra_1_extend"]]
check("default classes inherited: lumbar extend −30…45°, its pivot from the lumbar class",
      np.allclose(np.degrees([j.lo, j.hi]), [-30, 45]) and np.allclose(j.pos, [3, 0, -3]))
check("atlas: its own class axis (0, −1, 0), −5…45°",
      np.allclose(m.joints[m.joint_index["atlas"]].axis, [0, -1, 0])
      and np.allclose(np.degrees(m.joints[m.joint_index["atlas"]].hi), 45))
q0 = m.qpos()
V0 = m.skin_points(q0)[0]
trunk_v = np.isin(m.weights.argmax(1), [m.body_index[b] for b in ("torso", "pelvis", "vertebra_3", "skull")])
check(f"skin scaled to the skeleton (×{m.skin_scale:.3f}); the rest pose reproduces the trunk and head",
      abs(m.skin_scale - 0.82) < 0.01 and np.abs(V0[trunk_v] - m.verts[trunk_v]).max() < 3.0)
ext = rm.RodentModel.fk(m, m.qpos(vertebra_1_extend=np.deg2rad(30)))[0]
rest = m.fk(q0)[0]
pz = ext[m.body_index["pelvis"], 2, 3] - rest[m.body_index["pelvis"], 2, 3]
check(f"extending the lumbar spine raises the pelvis ({pz:.1f} mm) and leaves the thorax",
      pz > 5 and np.allclose(ext[m.body_index["torso"]], rest[m.body_index["torso"]]))

L0 = m.labels(q0)
z = {k: v[0, 2] for k, v in L0.items()}
x = {k: v[0, 0] for k, v in L0.items()}
check("labels: nose in front, back labels in order along the back, ears either side",
      x["head_nose"] > x["head_mid"] > x["head_back"] > x["back_T4"] > x["back_T8"] > x["back_L2"]
      > x["back_L6"] > x["back_V2"] > x["tail_V6"] > x["tail_V18"] > x["tail_V32"]
      and L0["head_left"][0, 1] > 10 > -10 > L0["head_right"][0, 1])
check("labels: the back labels on top of the back (above the hips), the tail tip on the floor",
      min(z[k] for k in ("back_T4", "back_T8", "back_L2", "back_L6", "back_V2")) > z["hip_left"] + 10
      and abs(z["tail_V32"]) < 3)

lim = rm.load_limits()
h = lim["label_heights"]
labs = [k for k in rm.LABELS if not k.startswith("tail_")]
check("limits: a height range for every head and trunk label, the rest pose inside it",
      set(labs) <= set(h) and all(h[k]["min_mm"] <= h[k]["rest_mm"] <= h[k]["max_mm"] for k in labs))
check(f"limits: rearing on the hind paws, the head reaches {h['head_back']['max_mm']:.0f} mm "
      f"(standing {h['head_back']['rest_mm']:.0f} mm)",
      180 < h["head_back"]["max_mm"] < 300 and h["head_back"]["max_mm"] > 1.8 * h["head_back"]["rest_mm"])
sb = lim["spine_bend"]
check(f"limits: lumbar bend {sb['lumbar']['min_deg']}…{sb['lumbar']['max_deg']}°, neck "
      f"{sb['neck']['min_deg']}…{sb['neck']['max_deg']}°, the rest pose inside both",
      all(sb[r]["min_deg"] < sb[r]["rest_deg"] < sb[r]["max_deg"] for r in ("lumbar", "neck"))
      and sb["lumbar"]["max_deg"] - sb["lumbar"]["min_deg"] > 60)
check("limits: the model's sizes (back length, ear width)",
      100 < lim["sizes_mm"]["back_path_horizontal"] < 200 and 25 < lim["sizes_mm"]["ear_width"] < 45)

# the bends as the spine chain measures them (mono._chain_chunk's elevations)
rng = np.random.default_rng(1)
P = {k: rng.normal(0, 30, (50, 3)) for k in rm.CHAIN}
bends, el = rm.chain_bends(P)
hd = P["head_back"][:, :2] - P["back_V2"][:, :2]
hd /= np.linalg.norm(hd, axis=1, keepdims=True)
nodes = [P[k] for k in rm.CHAIN]
a = [np.arctan2(nodes[i + 1][:, 2] - nodes[i][:, 2], np.einsum("ij,ij->i", nodes[i + 1][:, :2] - nodes[i][:, :2], hd))
     for i in range(len(nodes) - 1)]
check("bends measured as the spine chain measures them",
      np.allclose(el, np.stack(a, 1)) and np.allclose(bends[:, 0], a[1] - a[0]))

# ------------------------------------------------------------------ mono
sys.path.insert(0, str(REPO / "tests"))
cam = Camera.straight_down((330, 190), 243.0, 430.0, (640, 360),
                           box={"x": [-340, 340], "y": [-260, 260], "z": [0, 300]})


def synth(T=600, seed=0):
    """A walking rat whose head, half the time, is placed far too high
    (a label error: the ears too close together)."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) / 30
    cx, cy = 110 * np.sin(t * .31), 35 * np.sin(t * .47 + 1)
    yaw = np.unwrap(np.arctan2(np.gradient(cy), np.gradient(cx)))
    fwd = np.c_[np.cos(yaw), np.sin(yaw), 0 * t]
    lat = np.c_[-np.sin(yaw), np.cos(yaw), 0 * t]
    up = np.c_[0 * t, 0 * t, 1 + 0 * t]
    X = {}
    hips = np.c_[cx, cy, 40 + 0 * t]
    X["hip_left"], X["hip_right"] = hips + 22 * lat, hips - 22 * lat
    X["back_V2"] = hips - 25 * fwd + 30 * up
    for k, f, zz in (("back_L6", 10, 62), ("back_L2", 40, 70), ("back_T8", 75, 68), ("back_T4", 105, 62)):
        X[k] = hips + f * fwd + np.c_[0 * t, 0 * t, zz - 40 + 0 * t]
    hb = X["back_T4"] + 30 * fwd + 5 * up
    for k, o in {"head_back": (0, 0, 0), "head_mid": (20, 0, -4), "head_nose": (45, 0, -18),
                 "head_left": (3, 18, 0), "head_right": (3, -18, 0)}.items():
        X[k] = hb + o[0] * fwd + o[1] * lat + o[2] * up
    X["tail_V6"] = X["back_V2"] - 35 * fwd
    X["tail_V6"][:, 2] = 8
    X["tail_V18"] = X["tail_V6"] - 70 * fwd
    X["tail_V18"][:, 2] = 5
    X["tail_V32"] = X["tail_V18"] - 70 * fwd
    X["tail_V32"][:, 2] = 5
    cols = {}
    bad = (np.arange(T) // 60) % 2 == 1                    # every other 2 s
    for k, Pk in X.items():
        uv = cam.project(Pk) + rng.normal(0, .7, (T, 2))
        if k in ("head_left", "head_right"):                # ears drawn together: the head looks far away …
            c = (cam.project(X["head_left"]) + cam.project(X["head_right"])) / 2
            uv = np.where(bad[:, None], c + (uv - c) * 4.0, uv)  # … or here, much closer (too high)
        cols[k + "_x"], cols[k + "_y"], cols[k + "_p"] = uv[:, 0], uv[:, 1], np.full(T, .99)
    import pandas as pd
    return X, pd.DataFrame(cols), bad


X, df, bad = synth()
pose = load_pose2d(df)
SYN_Z = {"head_back": 0.0, "head_mid": -4.0, "head_nose": -18.0}
base = MonoConfig(chunk=600, head_template_z=SYN_Z, device="cpu", anatomy_scale=1.0)
off, rep_off = reconstruct_mono(pose, cam, replace(base, anatomy=None))
on, rep = reconstruct_mono(pose, cam, base)
an = rep.get("anatomy", {})
auto = reconstruct_mono(pose, cam, replace(base, anatomy_scale=None))[1]["anatomy"]
check(f"report: virtual rodent, size estimated {auto.get('scale')} ({auto.get('scale_from')})",
      auto.get("model") == "virtual_rodent" and 0.5 < auto.get("scale", 0) < 2
      and "back length" in auto.get("scale_from", ""))
hz_off, hz_on = off["head_back_z"].to_numpy(), on["head_back_z"].to_numpy()
anat = Anatomy(lim, an["scale"], {"head_left": "head_left", "head_right": "head_right", "back_T4": "back_T4"},
               margin=base.anatomy_margin)
hi_ear = anat.zrange("head_left")[1]
ez_on = (on["head_left_z"] + on["head_right_z"]).to_numpy() / 2
check(f"without limits the bad frames' head is too high ({np.median(hz_off[bad]):.0f} mm); with them the "
      f"ears stay within {hi_ear:.0f} mm ({an['head_frames_lowered']} frames lowered)",
      np.median(hz_off[bad]) > hi_ear and np.nanmax(ez_on) <= hi_ear + 1.0 and an["head_frames_lowered"] >= bad.sum() * .8)
check("frames within range are left alone (good frames' head unchanged)",
      np.allclose(hz_on[~bad], hz_off[~bad], atol=1e-6))
check("the lowered head still reprojects onto its labels (< 2 px median)",
      np.nanmedian(on["head_back_reproj_px"][bad]) < 2 and np.nanmedian(on["head_nose_reproj_px"][bad]) < 2)
trunk = ["back_T4", "back_T8", "back_L2", "back_L6", "back_V2", "hip_left", "hip_right"]
inside = all(((on[k + "_z"] >= Anatomy(lim, an["scale"], {k: k}, margin=base.anatomy_margin).zrange(k)[0] - 1e-6)
              & (on[k + "_z"] <= Anatomy(lim, an["scale"], {k: k}, margin=base.anatomy_margin).zrange(k)[1] + 1e-6)).all()
             for k in trunk)
check("every trunk label within its height range, on its camera ray (< 0.5 px)",
      inside and all(np.nanmedian(on[k + "_reproj_px"]) < .5 for k in trunk))
chain = list(reversed(["back_T4", "back_T8", "back_L2", "back_L6", "back_V2"])) + ["head_back"]
Pon = {k: on[[k + "_x", k + "_y", k + "_z"]].to_numpy() for k in chain}
rb = rm.region_bends(Pon)
lo_l, hi_l = np.deg2rad(sb["lumbar"]["min_deg"] - 6), np.deg2rad(sb["lumbar"]["max_deg"] + 6)
check(f"lumbar bend within range in every frame ({np.degrees(rb['lumbar']).min():.0f}…"
      f"{np.degrees(rb['lumbar']).max():.0f}°); report counts frames outside: {an.get('bend_outside_range')}",
      (rb["lumbar"] >= lo_l).all() and (rb["lumbar"] <= hi_l).all() and "bend_outside_range" in an)
check("steps: the head step is listed", "head within its height range (virtual rodent)" in rep["steps"]
      and not any("virtual rodent" in s for s in rep_off["steps"]))
check("scale can be set instead of estimated", an["scale"] == 1.0 and an["scale_from"] == "set")

# the chain's bend limits bite: a pelvis forced into a deep hunch
from mufasa.pose3d.mono import _chain_chunk  # noqa: E402

n = 40
C = cam.center
fixedP = {"back_V2": np.tile([-58.0, 0, 5], (n, 1)), "back_L6": np.tile([-54.0, 0, 80], (n, 1)),
          "head_back": np.tile([90.0, 0, 60], (n, 1))}
free = ["back_L2", "back_T8", "back_T4"]
ptsF = {"back_L2": [-15.0, 0, 60], "back_T8": [20.0, 0, 62], "back_T4": [50.0, 0, 5]}
Dd = {k: np.tile(np.array(v) - C, (n, 1)) for k, v in ptsF.items()}
z0 = np.tile([60.0, 62, 5], (n, 1))
lim4 = {"M50": np.full(n, -10.0), "M95": np.full(n, 10.0), "F5": np.full(n, -40.0), "F95": np.full(n, 40.0)}
anat1 = Anatomy(lim, 1.0, {k: k for k in rm.CHAIN}, margin=0.1)
pay = {"C": C, "chain": rm.CHAIN, "free": free, "n_mid": 2, "fixed": fixedP, "D": Dd, "z0": z0,
       "hd": np.tile([1.0, 0], (n, 1)), "lim": lim4, "device": "cpu",
       "rw": np.ones(n)}
z_no = _chain_chunk({**pay, "anat": None})
z_an = _chain_chunk({**pay, "anat": anat1.chain_terms(rm.CHAIN, free)})


def lumbar(zf):
    P = {k: fixedP[k] for k in fixedP}
    for i, k in enumerate(free):
        d = Dd[k]
        P[k] = C + ((zf[:, i] - C[2]) / d[:, 2])[:, None] * d
    return np.degrees(rm.region_bends(P)["lumbar"])


lb_no, lb_an = lumbar(z_no), lumbar(z_an)
check(f"chain: pelvis pitched up, thorax down: a {lb_no.mean():.0f}° lumbar bend; with the limits "
      f"{lb_an.mean():.0f}° (≥ {sb['lumbar']['min_deg'] - 5:.0f}°)",
      lb_no.mean() < sb["lumbar"]["min_deg"] - 5 and lb_an.min() >= sb["lumbar"]["min_deg"] - 5 - 1.0)
try:
    from mufasa.pose3d import gpu
    has_torch = gpu.available()
except Exception:
    has_torch = False
z_off = _chain_chunk({**pay, "anat": anat1.chain_terms(rm.CHAIN, free), "rw": np.zeros(n)})
check("chain: where the bend cannot be measured from overhead (weight 0) it is left to Rat7M",
      np.allclose(z_off, z_no, atol=1e-3))
Xs = {k: np.tile(v, (3, 1)).astype(float) for k, v in {**{k: v[0] for k, v in fixedP.items()}, **ptsF}.items()}
Xs["back_T8"][1, 0] = -30.0                             # T8 behind L2 from overhead
check("measurable: frames with a backward segment are flagged",
      Anatomy.measurable(Xs, rm.CHAIN).tolist() == [True, False, True])
if has_torch:
    z_t = _chain_chunk({**pay, "anat": anat1.chain_terms(rm.CHAIN, free), "device": "torch-cpu"})
    check(f"chain on PyTorch: the same ({np.abs(z_t - z_an).max():.2f} mm)", np.abs(z_t - z_an).max() < 1.0)

# ------------------------------------------------------------------ GUI / CLI
from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication([])
import tempfile  # noqa: E402

from mufasa.project_layout import (  # noqa: E402
    PROJECT_CONFIG_FILENAME,
    ProjectPaths,
    write_project_toml,
)
from mufasa.ui_qt.forms.pose3d import Pose3DInferenceForm  # noqa: E402

root = Path(tempfile.mkdtemp()) / "proj"
root.mkdir()
ProjectPaths(root).ensure_skeleton()
write_project_toml(root / PROJECT_CONFIG_FILENAME, {"project_layout_version": 1, "project_name": "p3d"})
cfgp = root / PROJECT_CONFIG_FILENAME
f = Pose3DInferenceForm(config_path=str(cfgp))
check("inference form: 'anatomical limits (virtual rodent)' on by default, body size auto",
      f.anat.isChecked() and f.anat_scale.value() == 0 and f.anat_scale.specialValueText() == "auto")
f.videos.clear()
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QListWidgetItem  # noqa: E402

it = QListWidgetItem("v1")
it.setData(Qt.UserRole, "v1")
it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
it.setCheckState(Qt.Checked)
f.videos.addItem(it)
args = f.collect_args()
f.anat.setChecked(False)
f.anat_scale.setValue(1.2)
args2 = f.collect_args()
check("inference form: the option and the size reach the settings",
      args["settings"].mono.anatomy == "virtual_rodent" and args["settings"].mono.anatomy_scale is None
      and args2["settings"].mono.anatomy is None and args2["settings"].mono.anatomy_scale == 1.2)
from mufasa.tools import pose3d_cli  # noqa: E402

src = Path(pose3d_cli.__file__).read_text()
check("CLI: --no-anatomy and --anatomy-scale", "--no-anatomy" in src and "--anatomy-scale" in src)
check("docs: pose3d.md describes the limits; the model README credits the virtual rodent",
      "virtual rodent" in (REPO / "docs" / "pose3d.md").read_text().lower()
      and "virtual_rodent" in (REPO / "mufasa" / "assets" / "models" / "pose3d" / "README.md").read_text())
check("limits script ships", (REPO / "scripts" / "build_virtual_rodent_limits.py").is_file())
check("limits JSON valid", isinstance(json.loads((D / "virtual_rodent_limits.json").read_text()), dict))

bad_ = [n for n, ok in checks if not ok]
for n, ok in checks:
    print(("PASS " if ok else "FAIL ") + n)
print(f"{len(checks) - len(bad_)}/{len(checks)} checks passed")
sys.exit(1 if bad_ else 0)
