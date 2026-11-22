"""Smoke test for patch 122lf — gait from the momentum of the lower, mid and upper body, and paw
placement scanning forward and backward.

User: "Does the gait analysis use the body vector to decompose forward and lateral momentum?"
(it did not: one speed, any direction) → "Do it for the momentum of the lower, mid and upper
body to better estimate efficient, reasonable paw placement. Scan forward and backward to make it
accurate as possible."

* the body in three sections (lower: pelvis, hind legs, tail; mid: thorax, lumbar spine; upper:
  shoulders, forelegs, neck, head), each with its mass, centre of mass, heading (level frames
  only), velocity and momentum along / across its heading, and turning rate;
* the hind paws follow the lower body and half the trunk, the forepaws the thorax; stance, swing
  and stride follow forward speed, not speed in any direction;
* forward: a paw lands where its girdle's place will be at mid-stance, read from the body's
  actual path; backward: once the stances are known, each paw stands where it stays closest to
  its place over the whole stance, within reach throughout;
* turning, stepping sideways and a back too steep for a heading are flagged, and kept out of
  the walking strides; the body is computed for all frames at once.
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


def rest_markers():
    """The standing virtual rodent's labels, heading along +x, on the floor."""
    from mufasa.bodymodel.rodent_rig import stance_qpos
    from mufasa.pose3d import gait as G
    from mufasa.pose3d.rodent_model import LABELS, RodentModel
    m = RodentModel()
    q0 = stance_qpos(m)
    T0 = m.fk(q0)
    L = {k: v[0] for k, v in m.labels(q0, T0).items()}
    floor = min(T0[0, m.body_index[b], 2, 3] for b in ("toe_L", "toe_R"))
    names = list(LABELS)
    X0 = np.array([L[k] for k in names]) - np.array([0, 0, floor])
    h = G.body_frame(L)[1][:2, 0]
    a = -math.atan2(h[1], h[0])
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    return names, X0 @ Rz.T


def move(X0, path, heading, T, fps, rear=None):
    """Markers moved rigidly: position ``path(t)`` (2,), heading ``heading(t)`` (rad); rearing
    ``rear`` degrees about the hind end for the middle third."""
    xyz = np.zeros((T, len(X0), 3))
    back = X0[:, 0].min()
    for i in range(T):
        t = i / fps
        X = X0.copy()
        if rear is not None and T // 3 <= i < 2 * T // 3:
            a = math.radians(rear)
            d = X - np.array([back, 0, 0])
            X = np.array([back, 0, 0]) + np.c_[math.cos(a) * d[:, 0] - math.sin(a) * d[:, 2], d[:, 1],
                                                math.sin(a) * d[:, 0] + math.cos(a) * d[:, 2]]
        c, s = math.cos(heading(t)), math.sin(heading(t))
        X = X @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T
        X[:, :2] += path(t)
        xyz[i] = X
    return xyz


def main() -> int:
    from mufasa.pose3d import gait as G
    phys = G.BodyPhysics.virtual_rodent()
    sm = phys.section_mass
    check(f"three sections: lower {sm['lower'] * 1000:.0f} g, mid {sm['mid'] * 1000:.0f} g, upper "
          f"{sm['upper'] * 1000:.0f} g = {phys.mass_kg * 1000:.0f} g",
          abs(sum(sm.values()) - phys.mass_kg) < 1e-9 and sm["lower"] > sm["mid"] > sm["upper"] > 0.03
          and G.body_section("pelvis") == "lower" and G.body_section("vertebra_C12") == "lower"
          and G.body_section("torso") == "mid" and G.body_section("vertebra_3") == "mid"
          and G.body_section("vertebra_cervical_2") == "upper" and G.body_section("scapula_L") == "upper")

    names, X0 = rest_markers()
    fps, T = 100.0, 300
    rng = np.random.default_rng(1)

    # all frames at once = frame by frame
    xyz = move(X0, lambda t: np.array([200.0 * t, 0.0]), lambda t: 0.3 * math.sin(2 * t), T, fps)
    xyz += rng.normal(0, 0.5, xyz.shape)
    xyz[40:50, names.index("back_T4")] = np.nan
    xyz[60:64] = np.nan
    Y, S = phys.predict_all(names, xyz, 1.1)
    Y1 = np.array([(lambda y: np.full((5, 3), np.nan) if y is None else y)(
        phys.predict({n: xyz[i, j] for j, n in enumerate(names)}, 1.1)) for i in range(T)])
    check(f"the body for all frames at once = frame by frame ({np.nanmax(np.abs(Y - Y1)):.1e} mm; missing frames "
          "alike)", np.nanmax(np.abs(Y - Y1)) < 1e-9 and (np.isnan(Y) == np.isnan(Y1)).all()
          and np.isfinite(S[:, :, 0]).sum() > 0.9 * 3 * T)

    mid = slice(40, 260)

    # straight ahead
    xyz = move(X0, lambda t: np.array([200.0 * t, 0.0]), lambda t: 0.0, T, fps)
    f = G.estimate_gait(names, xyz, np.arange(T), fps).frames
    vf, vl = f["mid_v_fwd_mm_s"][mid], f["mid_v_lat_mm_s"][mid]
    p = f["lower_p_fwd_mNs"][mid].median()
    mass_lower = sm["lower"] * (G.animal_size(phys, names, xyz) ** 3)
    check(f"straight at 200 mm/s: forward {vf.median():.1f}, sideways {vl.abs().max():.1f} mm/s; lower body's "
          f"momentum {p:.1f} mN·s (= {mass_lower * 200:.1f}); not turning, not sideways",
          abs(vf.median() - 200) < 3 and vl.abs().max() < 2 and abs(p - mass_lower * 200) < 0.02 * p
          and not f.turning[mid].any() and not f.sidestep[mid].any() and f.heading_ok[mid].all())

    # sideways
    xyz = move(X0, lambda t: np.array([0.0, 150.0 * t]), lambda t: 0.0, T, fps)
    f = G.estimate_gait(names, xyz, np.arange(T), fps).frames
    check(f"stepping sideways (left) at 150 mm/s: forward {f.v_fwd_mm_s[mid].abs().max():.1f}, sideways "
          f"{f.v_lat_mm_s[mid].median():.1f} mm/s; flagged {f.sidestep[mid].mean():.0%}",
          f.v_fwd_mm_s[mid].abs().max() < 3 and abs(f.v_lat_mm_s[mid].median() - 150) < 3 and f.sidestep[mid].all())

    # a turn: an arc of radius 120 mm at 200 mm/s (≈ 95°/s)
    r, v = 120.0, 200.0
    w = v / r
    arc = (lambda t: np.array([r * math.sin(w * t), r * (1 - math.cos(w * t))]), lambda t: w * t)
    xyz = move(X0, *arc, T, fps)
    res = G.estimate_gait(names, xyz, np.arange(T), fps)
    f, s = res.frames, res.steps
    yr = f.mid_yaw_rate_deg_s[mid].median()
    check(f"turning: {yr:.0f}°/s (true {math.degrees(w):.0f}°/s), forward {f.v_fwd_mm_s[mid].median():.0f} mm/s, "
          f"flagged {f.turning[mid].mean():.0%}", abs(yr - math.degrees(w)) < 3
          and abs(f.v_fwd_mm_s[mid].median() - v) < 5 and f.turning[mid].all())

    # placement in the turn: each paw beside its girdle, on its own side, within reach throughout
    def placement(res):
        f, s = res.frames, res.steps
        lat, fwd, reach_ok = [], [], True
        for st in s.itertuples():
            a, z = st.on_frame, st.off_frame
            if z - a < 3:
                continue
            m = (a + z) // 2
            g = lambda q: f.loc[m, [f"{q}_girdle_x", f"{q}_girdle_y"]].to_numpy(float)  # noqa: E731
            pm = (g(st.paw) + g(G.PARTNER[st.paw])) / 2
            h = (g("LF") + g("RF")) / 2 - (g("LH") + g("RH")) / 2
            h /= np.linalg.norm(h)
            d = np.array([st.x, st.y]) - pm
            lat.append(float(np.array([-h[1], h[0]]) @ d) * (1 if st.paw[0] == "L" else -1))
            fwd.append(float(h @ d))
            gi = f.loc[a:z, [f"{st.paw}_girdle_x", f"{st.paw}_girdle_y", f"{st.paw}_girdle_z"]].to_numpy(float)
            dist = np.linalg.norm(np.c_[np.array([st.x, st.y]) - gi[:, :2], gi[:, 2]], axis=1)
            reach_ok &= bool((dist <= phys.reach_max[st.paw] * G.animal_size(phys, names, xyz) + 0.6).all())
        return np.array(lat), np.array(fwd), reach_ok
    lat, fwd, reach_ok = placement(res)
    check(f"turning, {len(lat)} stances: every paw on its own side (closest {lat.min():.1f} mm out), centred under "
          f"its girdle at mid-stance (median {np.median(fwd):+.1f} mm, spread {np.median(np.abs(fwd)):.1f} mm), "
          "within reach throughout", len(lat) >= 12 and lat.min() > 0 and abs(np.median(fwd)) < 6 and reach_ok)

    # backward: where speed and turning change within a stance (a rat speeding up, slowing,
    # weaving), the stances re-placed over their whole length beat the forward pass's places
    Tw = 600
    tt = np.arange(Tw) / fps
    spd = 160.0 + 110.0 * np.sin(2.1 * tt) * np.sin(0.7 * tt)
    hd = 0.9 * np.sin(1.3 * tt) + 0.4 * np.sin(3.1 * tt)
    pos = np.cumsum(np.c_[spd * np.cos(hd), spd * np.sin(hd)], 0) / fps
    xyz = move(X0, lambda t: pos[min(int(round(t * fps)), Tw - 1)], lambda t: hd[min(int(round(t * fps)), Tw - 1)],
               Tw, fps)
    res2 = G.estimate_gait(names, xyz, np.arange(Tw), fps)
    lat1, fwd1, reach1 = placement(res2)
    real = G._place_stance
    try:
        G._place_stance = lambda *a, **k: None
        lat0, fwd0, _ = placement(G.estimate_gait(names, xyz, np.arange(Tw), fps))
    finally:
        G._place_stance = real
    e1 = np.median(np.abs(fwd1)) + np.median(np.abs(lat1 - np.median(lat1)))
    e0 = np.median(np.abs(fwd0)) + np.median(np.abs(lat0 - np.median(lat0)))
    check(f"scanning backward (speeding up, slowing, weaving; {len(lat1)} stances): off the girdles' places by "
          f"{e1:.1f} mm (forward pass alone {e0:.1f} mm); moved {res2.summary['replaced_mm']} mm (median); within "
          "reach throughout", e1 < e0 and reach1 and res2.summary["replaced_mm"] is not None)

    # one stance, by hand: the mean of its girdle's places, within reach
    nb = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    gd = np.array([[0.0, -10.0, 30.0], [10.0, -10.0, 30.0], [20.0, -10.0, 30.0]])
    p1 = G._place_stance(nb, gd, 60.0)
    p2 = G._place_stance(nb, gd + [[-60, 0, 0], [0, 0, 0], [60, 0, 0]], 60.0)
    check(f"one stance: the mean place {np.round(p1, 1)}; none within reach throughout → the forward pass's "
          f"({p2})", np.allclose(p1, [10, 0]) and p2 is None)

    # rearing: the back too steep for a heading; held from the frames either side
    xyz = move(X0, lambda t: np.array([100.0 * t, 0.0]), lambda t: 0.0, T, fps, rear=75.0)
    f = G.estimate_gait(names, xyz, np.arange(T), fps).frames
    up = slice(T // 3 + 5, 2 * T // 3 - 5)
    check(f"rearing 75°: no heading from the back in {1 - f.heading_ok[up].mean():.0%} of the rearing frames; "
          f"the heading held ({f.mid_heading_deg[up].abs().max():.1f}° off)",
          not f.heading_ok[up].any() and f.heading_ok[:T // 3 - 5].all() and f.mid_heading_deg[up].abs().max() < 2)

    # walking strides: straight ones only; flat summary; the file
    s2 = s.assign(stride_s=0.4, stride_mm=80.0)
    check(f"walking strides leave out turning ones ({len(G._walking(s2))} of {len(s2)} kept while turning)",
          len(G._walking(s2)) == 0 and len(G._walking(s2.assign(straight=True))) == len(s2))
    new_keys = [k for k in res.summary if k.startswith(G.SECTIONS) or k in ("drive", "turning", "sidestep",
                                                                              "replaced_mm")]
    nested = [k for k in new_keys if isinstance(res.summary[k], dict)]
    with tempfile.TemporaryDirectory() as td:
        out, _ = G.write_gait(res, Path(td) / "v_gait.parquet")
        back, meta = G.read_gait(out)
    cols = [f"{s_}_{c}" for s_ in G.SECTIONS for c in ("v_fwd_mm_s", "v_lat_mm_s", "p_fwd_mNs", "p_lat_mNs",
                                                       "p_z_mNs", "yaw_rate_deg_s", "heading_deg", "com_x")]
    check(f"the gait file: per-section velocity, momentum, heading; the run record takes the summary (no tables "
          f"in tables: {len(new_keys)} new entries, nested {nested})", all(c in back for c in cols) and "turning" in back
          and not nested and len(new_keys) >= 16
          and meta["summary"]["drive"] == "hind: 1 lower + 0.5 mid; fore: 1 mid")

    doc = (REPO / "docs" / "pose3d.md").read_text()
    check("docs: the sections, their momentum, scanning both ways", "lower, mid and upper" in doc
          and "scanning backward" in doc.lower() and "v_lat_mm_s" in doc)

    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122lf_gait_sections: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
