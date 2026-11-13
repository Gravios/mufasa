"""
mufasa.pose3d.gpu
=================

Patch 122kj: the single-camera head and spine-chain fits in PyTorch, on a
GPU (CUDA) or on the CPU.

Both fits are long frame sequences: each frame has a few unknowns (6
head-pose values; one height per free spine label) and its own residuals,
and neighbouring frames are tied by first-difference smoothness terms.

* residuals and their per-frame Jacobians come from ``torch.func``
  (``vmap`` over frames of ``jacfwd`` of one frame's residuals) — exact,
  one pass per chunk;
* the solver (``solve_frames``) is a port of the one the CPU path uses,
  SciPy's ``least_squares(method="trf", x_scale="jac", tr_solver="lsmr")``,
  with the Jacobian applied as batched per-frame blocks plus the
  smoothness rows — so the results are the CPU results within rounding;
* the same 6000-frame chunks as the CPU path.

Float64 throughout: the work is many tiny batched operations, so GPU time
is launch-bound rather than arithmetic-bound and consumer-GPU FP64 rates
do not matter.

``python -m mufasa.pose3d.gpu`` (or ``mufasa-pose3d gpu-check``) checks
that CUDA works and compares it with the CPU on a synthetic sequence.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

try:                                            # optional dependency
    import torch
    from torch.func import jacfwd, vmap
except ImportError:                             # pragma: no cover
    torch = None


def available() -> bool:
    return torch is not None


def cuda_available() -> bool:
    return torch is not None and torch.cuda.is_available()


def resolve_device(spec: str) -> str:
    """``"cuda"``, ``"torch-cpu"`` or ``"auto"`` → a torch device string."""
    if torch is None:
        raise RuntimeError("the GPU path needs PyTorch (pip install torch; the CUDA build for a GPU)")
    if spec in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available to PyTorch on this machine")
        return "cuda"
    if spec == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def _torch_device(device: str) -> str:
    """Mufasa device names → torch ("torch-cpu" → "cpu")."""
    return "cpu" if device in ("torch-cpu", "cpu") else device


class _one_thread:
    """PyTorch on the CPU solves with one thread: results then do not depend
    on the thread count (batched sums round differently with more threads),
    so a video gives the same 3-D pose whatever the number of workers; the
    workers fill the cores instead."""

    def __init__(self, device: str) -> None:
        self.on = device == "cpu"

    def __enter__(self):
        if self.on:
            self.n = torch.get_num_threads()
            torch.set_num_threads(1)

    def __exit__(self, *exc):
        if self.on:
            torch.set_num_threads(self.n)


# --------------------------------------------------------------------------- #
# Trust-region Gauss–Newton for frame sequences
# --------------------------------------------------------------------------- #
@dataclass
class SequenceResult:
    x: np.ndarray
    cost: float
    iterations: int
    nfev: int
    status: str
    seconds: float


def _sym_ortho(a: float, b: float):
    """Stable Givens rotation (as SciPy's LSQR/LSMR)."""
    if b == 0:
        return math.copysign(1.0, a) if a != 0 else 0.0, 0.0, abs(a)
    if a == 0:
        return 0.0, math.copysign(1.0, b), abs(b)
    if abs(b) > abs(a):
        tau = a / b
        s_ = math.copysign(1.0, b) / math.sqrt(1 + tau * tau)
        return s_ * tau, s_, b / s_
    tau = b / a
    c = math.copysign(1.0, a) / math.sqrt(1 + tau * tau)
    return c, c * tau, a / c


def _lsmr(matvec, rmatvec, b, n_x: int, *, damp: float = 0.0, atol: float = 1e-6,
          btol: float = 1e-6, conlim: float = 1e8, maxiter: int | None = None):
    """SciPy's ``lsmr`` (Fong & Saunders), on torch vectors: minimise
    ``‖A x − b‖² + damp² ‖x‖²`` with ``A`` given by ``matvec``/``rmatvec``.
    Same recurrences and stopping tests."""
    def nrm(v):
        return float(torch.linalg.norm(v))
    m_b = b.numel()
    maxiter = min(m_b, n_x) if maxiter is None else maxiter
    u = b.clone()
    normb = nrm(b)
    x = torch.zeros(n_x, dtype=b.dtype, device=b.device)
    beta = normb
    if beta > 0:
        u = u / beta
        v = rmatvec(u)
        alpha = nrm(v)
    else:
        v = torch.zeros_like(x)
        alpha = 0.0
    if alpha > 0:
        v = v / alpha
    zetabar, alphabar, rho, rhobar, cbar, sbar = alpha * beta, alpha, 1.0, 1.0, 1.0, 0.0
    h, hbar = v.clone(), torch.zeros_like(x)
    betadd, betad, rhodold, tautildeold, thetatilde, zeta, d = beta, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0
    normA2 = alpha * alpha
    maxrbar, minrbar = 0.0, 1e100
    ctol = 1 / conlim if conlim > 0 else 0.0
    if alpha * beta == 0 or normb == 0:
        return x
    itn = 0
    while itn < maxiter:
        itn += 1
        u = matvec(v) - alpha * u
        beta = nrm(u)
        if beta > 0:
            u = u / beta
            v = rmatvec(u) - beta * v
            alpha = nrm(v)
            if alpha > 0:
                v = v / alpha
        chat, shat, alphahat = _sym_ortho(alphabar, damp)
        rhoold = rho
        c, s_, rho = _sym_ortho(alphahat, beta)
        thetanew = s_ * alpha
        alphabar = c * alpha
        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = _sym_ortho(cbar * rho, thetanew)
        zeta = cbar * zetabar
        zetabar = -sbar * zetabar
        hbar = h - (thetabar * rho / (rhoold * rhobarold)) * hbar
        x = x + (zeta / (rho * rhobar)) * hbar
        h = v - (thetanew / rho) * h
        betaacute = chat * betadd
        betacheck = -shat * betadd
        betahat = c * betaacute
        betadd = -s_ * betaacute
        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = _sym_ortho(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = -stildeold * betad + ctildeold * betahat
        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud = (zeta - thetatilde * tautildeold) / rhodold
        d = d + betacheck * betacheck
        normr = math.sqrt(d + (betad - taud) ** 2 + betadd * betadd)
        normA2 += beta * beta
        normA = math.sqrt(normA2)
        normA2 += alpha * alpha
        maxrbar = max(maxrbar, rhobarold)
        if itn > 1:
            minrbar = min(minrbar, rhobarold)
        condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp)
        normar = abs(zetabar)
        normx = nrm(x)
        test1 = normr / normb
        test2 = normar / (normA * normr) if normA * normr != 0 else math.inf
        test3 = 1 / condA
        t1 = test1 / (1 + normA * normx / normb)
        rtol = btol + atol * normA * normx / normb
        if (itn >= maxiter or 1 + test3 <= 1 or 1 + test2 <= 1 or 1 + t1 <= 1
                or test3 <= ctol or test2 <= atol or test1 <= rtol):
            break
    return x



def _tr_2d(B: np.ndarray, g: np.ndarray, Delta: float) -> np.ndarray:
    """SciPy's ``solve_trust_region_2d``."""
    try:
        np.linalg.cholesky(B)                          # positive definite?
        p = -np.linalg.solve(B, g)
        if p @ p <= Delta ** 2:
            return p
    except np.linalg.LinAlgError:
        pass
    a, b, c = B[0, 0] * Delta ** 2, B[0, 1] * Delta ** 2, B[1, 1] * Delta ** 2
    d, f = g[0] * Delta, g[1] * Delta
    t = np.roots([-b + d, 2 * (a - c + f), 6 * b, 2 * (-a + c + f), -b - d])
    t = np.real(t[np.isreal(t)])
    p = Delta * np.vstack((2 * t / (1 + t ** 2), (1 - t ** 2) / (1 + t ** 2)))
    value = 0.5 * np.sum(p * B.dot(p), axis=0) + g @ p
    return p[:, int(np.argmin(value))]


def solve_frames(frame_res, x0, data: tuple, smooth_w, *, device="cpu", max_nfev: int = 150,
                 ftol: float = 1e-8, xtol: float = 1e-8, gtol: float = 1e-8) -> SequenceResult:
    """Minimise ``½ Σ_k ‖r(x_k, data_k)‖² + ½ Σ_k Σ_c w_c (x_{k+1,c} − x_{k,c})²``.

    ``frame_res(x_k, *data_k)`` returns one frame's residuals (torch);
    ``x0`` (n, k); ``data`` arrays with leading dimension n; ``smooth_w``
    (k,) first-difference weights.

    A port of SciPy's ``least_squares(method="trf", x_scale="jac",
    tr_solver="lsmr")`` — the solver the CPU path uses — for this
    structure: the same trust-region loop, regularised LSMR for the
    Gauss–Newton direction, step from the 2-D subspace of gradient and
    Gauss–Newton direction, radius update and stopping tests. The Jacobian
    is exact (autograd, per frame) and applied as batched per-frame
    blocks plus the smoothness rows. Solvers that step differently (exact
    Levenberg–Marquardt, with one or per-frame damping) were tried and
    settled in clearly worse local minima on K7 (33–37k against 19k)."""
    device = _torch_device(device)
    with _one_thread(device):
        return _solve_frames(frame_res, x0, data, smooth_w, device, max_nfev, ftol, xtol, gtol)


def _solve_frames(frame_res, x0, data, smooth_w, device, max_nfev, ftol, xtol, gtol) -> SequenceResult:
    t0 = time.time()
    dt = torch.float64
    x = torch.as_tensor(np.asarray(x0, float), dtype=dt, device=device)
    data = tuple(torch.as_tensor(np.asarray(d, float), dtype=dt, device=device) for d in data)
    w = torch.as_tensor(np.asarray(smooth_w, float), dtype=dt, device=device)
    sw = torch.sqrt(w)
    n, k = x.shape
    res_all = vmap(frame_res)
    jac_all = vmap(jacfwd(frame_res, argnums=0))

    def full_res(x):
        r = res_all(x, *data)
        return torch.cat([r.reshape(-1), (sw * (x[1:] - x[:-1])).reshape(-1)]), r.shape[1]

    f, m = full_res(x)
    nfev = 1
    cost = 0.5 * float(f @ f)
    nr = n * m

    def make_ops(Jb, dsc):
        def matvec(v):                                   # J_h v, v (n*k,)
            dv = (dsc * v.reshape(n, k))
            a = (Jb @ dv[..., None])[..., 0].reshape(-1)
            return torch.cat([a, (sw * (dv[1:] - dv[:-1])).reshape(-1)])

        def rmatvec(u):                                  # J_hᵀ u
            uf = u[:nr].reshape(n, m)
            us = u[nr:].reshape(n - 1, k) * sw
            out = (Jb.transpose(1, 2) @ uf[..., None])[..., 0]
            out[1:] += us
            out[:-1] -= us
            return (dsc * out).reshape(-1)
        return matvec, rmatvec

    def jacobian(x):
        return jac_all(x, *data)                         # (n, m, k)

    def col_norms(Jb):
        cn = (Jb * Jb).sum(1)
        deg = torch.full((n, 1), 2.0, dtype=dt, device=device)
        deg[0] = deg[-1] = 1.0 if n > 1 else 0.0
        return torch.sqrt(cn + deg * w)

    Jb = jacobian(x)
    mv0, rmv0 = make_ops(Jb, torch.ones_like(x))
    g = rmv0(f)
    scale_inv = col_norms(Jb)
    scale_inv = torch.where(scale_inv == 0, torch.ones_like(scale_inv), scale_inv)
    Delta = float(torch.linalg.norm(x.reshape(-1) * scale_inv.reshape(-1))) or 1.0
    status, it = "max_nfev", 0
    while True:
        if float(torch.linalg.norm(g, ord=math.inf)) < gtol:
            status = "gtol"
            break
        if nfev >= max_nfev:
            break
        dsc = 1.0 / scale_inv
        g_h = dsc.reshape(-1) * g
        mv, rmv = make_ops(Jb, dsc)
        # regularisation of LSMR (SciPy: regularize=True)
        s_ = -g_h
        vv = mv(s_)
        a_q, b_q = 0.5 * float(vv @ vv), float(g_h @ s_)
        to_tr = Delta / float(torch.linalg.norm(g_h))
        ts = [0.0, to_tr]
        if a_q != 0:
            ext = -0.5 * b_q / a_q
            if 0 < ext < to_tr:
                ts.append(ext)
        ag_value = min(t * (a_q * t + b_q) for t in ts)
        damp = math.sqrt(max(-ag_value / Delta ** 2, 0.0))
        gn_h = _lsmr(mv, rmv, f, n * k, damp=damp)
        S, _ = torch.linalg.qr(torch.stack([g_h, gn_h], 1))
        JS = torch.stack([mv(S[:, 0]), mv(S[:, 1])], 1)
        B_S = (JS.T @ JS).cpu().numpy()
        g_S = (S.T @ g_h).cpu().numpy()
        actual = -1.0
        while actual <= 0 and nfev < max_nfev:
            p_S = torch.as_tensor(_tr_2d(B_S, g_S, Delta), dtype=dt, device=device)
            step_h = S @ p_S
            Js = mv(step_h)
            pred = -(0.5 * float(Js @ Js) + float(step_h @ g_h))
            step = dsc.reshape(-1) * step_h
            x_new = x + step.reshape(n, k)
            f_new, _ = full_res(x_new)
            nfev += 1
            step_h_norm = float(torch.linalg.norm(step_h))
            if not bool(torch.isfinite(f_new).all()):
                Delta = 0.25 * step_h_norm
                continue
            cost_new = 0.5 * float(f_new @ f_new)
            actual = cost - cost_new
            if pred > 0:
                ratio = actual / pred
            elif pred == actual == 0:
                ratio = 1.0
            else:
                ratio = 0.0
            if ratio < 0.25:
                Delta_new = 0.25 * step_h_norm
            elif ratio > 0.75 and step_h_norm > 0.95 * Delta:
                Delta_new = 2.0 * Delta
            else:
                Delta_new = Delta
            step_norm = float(torch.linalg.norm(step))
            ftol_ok = actual < ftol * cost and ratio > 0.25
            xtol_ok = step_norm < xtol * (xtol + float(torch.linalg.norm(x)))
            if ftol_ok or xtol_ok:
                status = "ftol" if ftol_ok else "xtol"
                break
            Delta = Delta_new
        if actual > 0:
            x, f, cost = x_new, f_new, cost_new
            Jb = jacobian(x)
            _, rmv0 = make_ops(Jb, torch.ones_like(x))
            g = rmv0(f)
            cn = col_norms(Jb)
            scale_inv = torch.maximum(scale_inv, torch.where(cn == 0, torch.ones_like(cn), cn))
        it += 1
        if status in ("ftol", "xtol"):
            break
    if device == "cuda":
        torch.cuda.synchronize()
    return SequenceResult(x.cpu().numpy(), cost, it, nfev, status, time.time() - t0)


# --------------------------------------------------------------------------- #
# Residuals
# --------------------------------------------------------------------------- #
def _rotm_zyx(e):
    """Euler ZYX (yaw, pitch, roll), intrinsic — as SciPy's from_euler("ZYX")."""
    a, b, c = e[0], e[1], e[2]
    ca, sa, cb, sb, cc, sc = torch.cos(a), torch.sin(a), torch.cos(b), torch.sin(b), torch.cos(c), torch.sin(c)
    return torch.stack([
        torch.stack([ca * cb, ca * sb * sc - sa * cc, ca * sb * cc + sa * sc]),
        torch.stack([sa * cb, sa * sb * sc + ca * cc, sa * sb * cc - ca * sc]),
        torch.stack([-sb, cb * sc, cb * cc]),
    ])


def head_residual_fn(cam, T, box, *, sigma: float, roll_sd_deg: float, wall_tolerance_mm: float, device):
    """One frame's head residuals, as ``RigidHead.fit_sequence``: reprojection
    (weighted), box hinges (/2) and the roll prior. Frame data: uv (m, 2),
    wt (m,)."""
    device = _torch_device(device)
    dt = torch.float64
    R = torch.as_tensor(cam.R, dtype=dt, device=device)
    t = torch.as_tensor(cam.t, dtype=dt, device=device)
    K = torch.as_tensor(cam.K, dtype=dt, device=device)
    Tt = torch.as_tensor(T, dtype=dt, device=device)
    roll_sd = math.radians(roll_sd_deg)
    tol = float(wall_tolerance_mm)
    bx = None
    if box:
        (x0, x1), (y0, y1), (_, z1) = box["x"], box["y"], box["z"]
        bx = (float(x0), float(x1), float(y0), float(y1), float(z1))

    def f(p, uv, wt):
        Rm = _rotm_zyx(p[3:])
        Xw = p[:3] + Tt @ Rm.T                                  # (m, 3)
        Xc = Xw @ R.T + t
        xn = Xc[:, :2] / Xc[:, 2:3]
        pr = xn @ K[:2, :2].T + K[:2, 2]
        r = [((pr - uv) / sigma * wt[:, None]).reshape(-1)]
        h = [torch.relu(-Xw[:, 2])]
        if bx is not None:
            x0, x1, y0, y1, z1 = bx
            h += [torch.relu(Xw[:, 0] - x1 - tol), torch.relu(x0 - tol - Xw[:, 0]),
                  torch.relu(Xw[:, 1] - y1 - tol), torch.relu(y0 - tol - Xw[:, 1]),
                  torch.relu(Xw[:, 2] - z1)]
        r.append(torch.cat(h) / 2.0)
        r.append((p[5] / roll_sd)[None])
        return torch.cat(r)
    return f


HEAD_SMOOTH_W = np.r_[[1 / 8.0 ** 2] * 3, [1 / math.radians(8) ** 2] * 3]


def chain_residual_fn(C, chain: list, free: list, n_mid: int, device):
    """One frame's spine-chain residuals, as ``mono._chain_chunk``. Frame
    data: D (nf, 3) ray directions of the free labels, fixed (n_fixed, 3)
    positions of the other chain nodes in chain order, z0 (nf,), hd (2,),
    lim (4,) = M50, M95, F5, F95 (deg)."""
    device = _torch_device(device)
    dt = torch.float64
    Ct = torch.as_tensor(np.asarray(C, float), dtype=dt, device=device)
    rad = math.pi / 180
    free_idx = {m: i for i, m in enumerate(free)}
    fixed_idx = {m: i for i, m in enumerate([c for c in chain if c not in free])}
    nb = len(chain) - 2

    def f(z, D, fixed, z0, hd, lim):
        nodes = []
        for m in chain:
            if m in free_idx:
                i = free_idx[m]
                d = D[i]
                nodes.append(Ct + ((z[i] - Ct[2]) / d[2]) * d)
            else:
                nodes.append(fixed[fixed_idx[m]])
        el = [torch.atan2(nodes[i + 1][2] - nodes[i][2], (nodes[i + 1][:2] - nodes[i][:2]) @ hd)
              for i in range(len(nodes) - 1)]
        bends = [el[i + 1] - el[i] for i in range(len(el) - 1)]
        M50, M95, F5, F95 = lim[0], lim[1], lim[2], lim[3]
        r = [(z - z0) / 20.0]
        for j, bd in enumerate(bends):
            node_i = j + 1
            if node_i == 1:
                r.append((torch.relu(bd - 10 * rad) / (2 * rad))[None])
            elif node_i == nb:
                r.append((torch.relu(bd - F95 * rad) / (3 * rad))[None])
                r.append((torch.relu(F5 * rad - bd) / (3 * rad))[None])
            else:
                r.append(((bd - M50 * rad / n_mid) / (15 * rad))[None])
                r.append((torch.relu(bd - M95 * rad / n_mid - 3 * rad) / (2 * rad))[None])
        r.append(torch.relu(3 - z))
        return torch.cat(r)
    return f


# --------------------------------------------------------------------------- #
# Self-check
# --------------------------------------------------------------------------- #
def gpu_check(n: int = 20000, verbose=print) -> dict:
    """Run a synthetic head sequence on the CPU (PyTorch) and, when
    available, on CUDA; report times and agreement."""
    from mufasa.pose3d.camera import Camera
    rng = np.random.default_rng(0)
    cam = Camera.straight_down((330, 190), 243.0, 430.0, (640, 360),
                               box={"x": [-340, 340], "y": [-260, 260], "z": [0, 300]})
    T = np.array([[-3.6, 0, -0.4], [10, 0, -4.7], [45, 0, -18.4], [0, 18, 0], [0, -18, 0]])
    tt = np.arange(n) / 30
    P = np.c_[120 * np.sin(tt * .3), 80 * np.sin(tt * .45), 60 + 40 * np.sin(tt * .2) ** 2,
              tt * .4, .3 * np.sin(tt), .1 * np.sin(tt * .7)]
    Xw = P[:, None, :3] + np.einsum("nij,mj->nmi", _np_rotm(P[:, 3:]), T)
    uv = cam.project(Xw, distort=False) + rng.normal(0, 1, (n, 5, 2))
    wt = np.ones((n, 5))
    x0 = P + rng.normal(0, [5, 5, 15, .1, .1, .1], P.shape)
    out = {}
    for dev in (["cpu"] + (["cuda"] if cuda_available() else [])):
        f = head_residual_fn(cam, T, cam.box, sigma=2.5, roll_sd_deg=20, wall_tolerance_mm=20, device=dev)
        res = solve_frames(f, x0, (uv, wt), HEAD_SMOOTH_W, device=dev, max_nfev=150)
        err = float(np.median(np.linalg.norm(res.x[:, :3] - P[:, :3], axis=1)))
        out[dev] = {"seconds": round(res.seconds, 2), "cost": res.cost, "head_error_mm": round(err, 2),
                    "iterations": res.iterations, "x": res.x}
        verbose(f"{dev:5s}: {n} frames in {res.seconds:.2f} s ({res.iterations} iterations, "
                f"{res.status}); head position error {err:.2f} mm median")
    if "cuda" in out:
        d = np.abs(out["cuda"]["x"] - out["cpu"]["x"]).max()
        verbose(f"CUDA vs CPU: max |difference| {d:.2e}; speed-up ×{out['cpu']['seconds'] / out['cuda']['seconds']:.1f}")
        out["max_difference"] = float(d)
    else:
        verbose("CUDA: not available to PyTorch here" + ("" if available() else " (PyTorch missing)"))
    for v in out.values():
        if isinstance(v, dict):
            v.pop("x", None)
    return out


def _np_rotm(e):
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler("ZYX", e).as_matrix()


__all__ = ["HEAD_SMOOTH_W", "SequenceResult", "available", "chain_residual_fn",
           "cuda_available", "gpu_check", "head_residual_fn", "resolve_device", "solve_frames"]


if __name__ == "__main__":                                       # pragma: no cover
    gpu_check()
