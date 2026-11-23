"""
mufasa.biomech.animal
=====================

Patch 122ln: an animal calibrated once, its file reused (``mufasa-pose3d animal``).

The biomechanical model needs two things about the animal that do not change between sessions:
its size against the virtual rodent, and where each label sits on its body (the labeller's
convention, calibrated as in STAC, :func:`.fit.calibrate`). Until now every run recalibrated on
400 frames of its own recording. An animal file holds them instead:

* calibrated once, on frames drawn from one or several sessions of the same animal (more
  behaviour, a better calibration);
* reused by ``dynamics`` and ``smooth --model`` (``--animal``): the same body and labels in every
  session, and no calibration to wait for.

The file is JSON: the size, the weighed mass if given, each label's body and its place on it
(at size 1, so it scales with the model), the joint limits used, the sessions it came from and
the calibration's fit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

VERSION = 1


def calibrate_animal(files, out, *, frames: int = 600, mass_g: float | None = None, name: str | None = None,
                     limits: str = "model", size: float | None = None, progress=None) -> dict:
    """Calibrate the virtual rodent to an animal from its 3-D pose files; writes ``out``.
    ``size``: the animal against the virtual rodent (default: nose to tail base, the median over
    the sessions)."""
    from mufasa.biomech import fit as F
    from mufasa.biomech.mjmodel import build
    from mufasa.pose3d.gait import BodyPhysics, animal_size
    from mufasa.pose3d.io import marker_names, read_pose3d
    say = progress or (lambda *_: None)
    data, names = [], None
    for f in files:
        df, _ = read_pose3d(f)
        n = marker_names(df)
        names = n if names is None else [k for k in names if k in n]
        data.append((n, df))
    stacks = []
    for n, df in data:
        stacks.append(np.stack([df[[f"{k}_x", f"{k}_y", f"{k}_z"]].to_numpy(float) for k in names], 1))
    xyz = np.concatenate(stacks)
    sizes = [animal_size(BodyPhysics.virtual_rodent(), names, s) for s in stacks]
    size = float(size or np.median(sizes))
    M = build(size, mass_kg=None if not mass_g else mass_g / 1000.0, limits=limits)
    use = [k for k in names if k in M.labels]
    iu = [names.index(k) for k in use]
    good = np.flatnonzero(np.isfinite(xyz[:, iu]).all((1, 2)))
    if len(good) < 20:
        raise ValueError("under 20 frames with every label: nothing to calibrate on")
    # frames spread over every session: an equal share each
    rng = np.random.default_rng(0)
    bounds = np.cumsum([0] + [len(s) for s in stacks])
    pick = []
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        g = good[(good >= a) & (good < b)]
        if len(g):
            pick.extend(rng.choice(g, min(len(g), max(1, frames // len(stacks))), replace=False).tolist())
    pick = np.array(sorted(pick))
    say(f"calibrating on {len(pick)} frames from {len(stacks)} session(s), size {size:.3f}")
    cal = F.calibrate(M, use, xyz[pick][:, iu])
    m = M.model
    sites = {k: {"body": m.body(m.site_bodyid[M.labels[k]]).name,
                 "pos": (m.site_pos[M.labels[k]] / size).round(6).tolist()} for k in use}
    doc = {"mufasa_animal": VERSION, "name": name or Path(out).stem.split(".")[0], "size": round(size, 4),
           "session_sizes": [round(float(s), 4) for s in sizes], "mass_g": mass_g, "limits": limits,
           "labels": sites, "sessions": [str(f) for f in files], "frames": int(len(pick)),
           "calibration_rms_mm": cal["rms_mm"], "label_shift_mm": cal["shift_mm"]}
    Path(out).write_text(json.dumps(doc, indent=1))
    return doc


def load_animal(path) -> dict:
    d = json.loads(Path(path).read_text())
    if "mufasa_animal" not in d:
        raise ValueError(f"{path}: not a mufasa animal file")
    return d


def build_for(animal, **kw):
    """The virtual rodent built for an animal file (its size, mass, limits, label places)."""
    from mufasa.biomech.mjmodel import build
    a = animal if isinstance(animal, dict) else load_animal(animal)
    mass = kw.pop("mass_kg", None) or (a["mass_g"] / 1000.0 if a.get("mass_g") else None)
    M = build(a["size"], mass_kg=mass, limits=kw.pop("limits", a.get("limits", "model")), **kw)
    apply_labels(M, a)
    return M


def apply_labels(M, animal) -> int:
    """Put the animal file's label places on the model; returns how many labels were placed."""
    a = animal if isinstance(animal, dict) else load_animal(animal)
    m = M.model
    n = 0
    for k, s in a["labels"].items():
        if k not in M.labels:
            continue
        sid = M.labels[k]
        if m.body(m.site_bodyid[sid]).name != s["body"]:
            raise ValueError(f"label {k}: on {s['body']} in the animal file, on "
                             f"{m.body(m.site_bodyid[sid]).name} in the model")
        m.site_pos[sid] = np.asarray(s["pos"], float) * M.size
        n += 1
    return n


__all__ = ["VERSION", "apply_labels", "build_for", "calibrate_animal", "load_animal"]
