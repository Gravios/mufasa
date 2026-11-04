"""Smoke test for patch 122ii — the joint-prior fit uses the worker pool.

fit_joint_priors_v2 took a ``pool`` argument and ignored it: the pass (one
filter + smoother per session, the same cost as an EM E-step) ran every
session serially in the main process on one core, while the perspective pass
just before it used all workers. It was also called with verbose=False, so
it sat silent. Now sessions go to the pool via _pool_joint_prior_pass, are
accumulated in session order (pooled == serial, exactly), and progress is
printed per session.

Runs for real: a multiprocessing pool initialised with the smoother's own
_pool_init, on synthetic sessions with a trailing tail.
"""
from __future__ import annotations

import contextlib
import io
import math
import multiprocessing as mp
import sys
import types
from pathlib import Path

_tk = types.ModuleType("tkinter")
_tk.messagebox = types.ModuleType("tkinter.messagebox")
_tk.messagebox.showerror = lambda *a, **k: None
sys.modules.setdefault("tkinter", _tk)
sys.modules.setdefault("tkinter.messagebox", _tk.messagebox)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from mufasa.data_processors import kalman_pose_smoother_v2 as K  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


SPEC = [
    {"name": "back", "markers": ["b1", "b2", "b3", "b4"]},
    {"name": "neck", "parent": "back", "markers": ["n"]},
    {"name": "tail", "parent": "back", "markers": ["t"]},
]


def make(seed, T=500):
    rg = np.random.default_rng(seed)
    th = np.cumsum(rg.normal(0, 0.02, T))
    c = np.stack([300 + np.cumsum(rg.normal(0, 0.5, T)),
                  200 + np.cumsum(rg.normal(0, 0.5, T))], 1)
    u = np.stack([np.cos(th), np.sin(th)], 1)
    a_n = 0.3 * np.sin(np.arange(T) / 15.0)
    a_t = rg.normal(0, 0.15, T)
    un = np.stack([np.cos(th + a_n), np.sin(th + a_n)], 1)
    ut = np.stack([np.cos(th + np.pi + a_t), np.sin(th + np.pi + a_t)], 1)
    pts = [c + 18 * u, c + 6 * u, c - 6 * u, c - 18 * u,
           c + 18 * u + 10 * un, c - 18 * u + 14 * ut]
    pos = np.stack(pts, 1) + rg.normal(0, 0.5, (T, 6, 2))
    lik = np.full((T, 6), 0.95)
    lik[200:260, 5] = 0.05                      # a tail dropout
    return pos, lik


if __name__ == "__main__":
    layout = K.layout_from_segments(SPEC)
    names = list(layout.marker_names)
    sessions = [make(s) for s in range(4)]
    fl = K.fit_body_lengths(sessions[0][0], sessions[0][1], layout, names,
                            0.5, dt=1 / 30)
    p0 = K.fit_initial_params_v2(sessions[0][0], sessions[0][1], layout,
                                 names, fl, 30.0, likelihood_threshold=0.5)

    serial = K.fit_joint_priors_v2(sessions, layout, names, fl, p0, 30.0,
                                   likelihood_threshold=0.5)

    calls: list[int] = []
    orig = K._joint_prior_session_samples

    def spy(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    K._joint_prior_session_samples = spy
    buf = io.StringIO()
    with mp.Pool(2, initializer=K._pool_init,
                 initargs=(sessions, layout, fl, names, 30.0, 0.5, True)
                 ) as pool, contextlib.redirect_stdout(buf):
        pooled = K.fit_joint_priors_v2(
            sessions, layout, names, fl, p0, 30.0,
            likelihood_threshold=0.5, verbose=True,
            session_names=[f"s{i}" for i in range(4)], pool=pool)
    K._joint_prior_session_samples = orig

    check("with a pool, no session is processed in the main process",
          calls == [])
    check("pooled fit is identical to the serial fit",
          pooled.mu == serial.mu and pooled.kappa == serial.kappa
          and pooled.n_obs == serial.n_obs)
    check("the fit is non-trivial (an active prior was found)",
          any(k > 0 for k in serial.kappa.values())
          and all(math.isfinite(m) for m in serial.mu.values()))
    out = buf.getvalue()
    check("progress is printed per session",
          out.count("[v2-jprior]") == 4 and "4/4" in out and "ETA" in out)

    src = (REPO / "mufasa/data_processors/kalman_pose_smoother_v2.py").read_text()
    call = src[src.index("joint_prior = fit_joint_priors_v2("):]
    call = call[:call.index(")\n")]
    check("EM passes the pool and verbose through",
          "pool=pool" in call and "verbose=verbose" in call)

    n_pass = sum(1 for _, ok in checks if ok)
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"smoke_122ii_joint_prior_pool: {n_pass}/{len(checks)} checks passed")
    sys.exit(0 if n_pass == len(checks) else 1)
