"""
mufasa.ui_qt.linux_env
=====================

Linux-specific runtime helpers. SimBA targets **Linux-only** from this
point forward — Windows and macOS accommodations are no longer part of
the GUI layer.

What this module provides:

* :func:`cpu_count` — cgroup-/affinity-aware CPU count. Uses
  ``os.sched_getaffinity(0)`` which respects container limits (LXC,
  Docker, systemd slices) and ``taskset`` bindings. ``os.cpu_count()``
  returns the host total and misleads worker-pool sizing under any of
  those.
* :func:`setup_multiprocessing` — sets start method to ``"fork"`` once.
  ``fork`` is the Linux default but various SimBA modules force
  ``"spawn"`` defensively, which doubles import cost per worker and
  breaks shared-memory trivial-copy semantics. Qt code should call this
  once at application start.
* :func:`xdg_config_dir` / :func:`xdg_cache_dir` / :func:`xdg_data_dir`
  — proper XDG Base Directory Specification paths for SimBA's user
  state. Replaces the ``os.path.expanduser("~/.simba/…")`` scatter.
* :func:`cuda_available` / :func:`nvenc_available` — compile-time-free
  GPU capability probes. ``cuda_available`` short-circuits on
  ``CUDA_VISIBLE_DEVICES=""`` so tests can disable it explicitly.
* :func:`detect_display_server` — ``"wayland"`` / ``"x11"`` /
  ``"thinlinc"`` / ``"unknown"``. The ThinLinc case matters because
  some OpenGL / multimedia Qt widgets need fallback software rendering
  under Xvnc-backed sessions.

Design notes:

* All probes cache with ``@lru_cache`` — calling them is cheap.
* No Windows/macOS branches. ``import platform`` is deliberately absent.
* Module import is pure-stdlib + ``shutil``; no heavy deps.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------- #
# CPU / process
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def cpu_count() -> int:
    """Return the number of CPUs actually available to this process.

    Uses ``os.sched_getaffinity`` which on Linux reports the cgroup-
    restricted / ``taskset``-restricted set — not the host total. This
    matches what ``numba``, ``joblib``, and ``concurrent.futures`` will
    actually use as upper bounds.
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        # Kernel without sched_getaffinity (very old); fall through.
        return max(1, os.cpu_count() or 1)


def setup_multiprocessing() -> None:
    """Set multiprocessing start method to ``fork`` (idempotent).

    Linux default is already ``fork``, but SimBA's backend sets ``spawn``
    in several places defensively for Windows/Mac compat. Call this
    once from the Qt entry-point before any ``multiprocessing`` usage
    to undo any module-top forcing.
    """
    import multiprocessing
    try:
        current = multiprocessing.get_start_method(allow_none=True)
        if current != "fork":
            multiprocessing.set_start_method("fork", force=True)
    except (RuntimeError, ValueError):
        # Already started — too late to change. Harmless.
        pass


# --------------------------------------------------------------------------- #
# XDG Base Directory Specification
# --------------------------------------------------------------------------- #
def _xdg(env: str, default_rel: str) -> Path:
    base = os.environ.get(env)
    if base:
        p = Path(base)
    else:
        p = Path.home() / default_rel
    d = p / "mufasa"
    d.mkdir(parents=True, exist_ok=True)
    return d


@lru_cache(maxsize=1)
def xdg_config_dir() -> Path:
    """``~/.config/mufasa`` (honours ``$XDG_CONFIG_HOME``)."""
    return _xdg("XDG_CONFIG_HOME", ".config")


@lru_cache(maxsize=1)
def xdg_cache_dir() -> Path:
    """``~/.cache/mufasa`` (honours ``$XDG_CACHE_HOME``)."""
    return _xdg("XDG_CACHE_HOME", ".cache")


@lru_cache(maxsize=1)
def xdg_data_dir() -> Path:
    """``~/.local/share/mufasa`` (honours ``$XDG_DATA_HOME``)."""
    return _xdg("XDG_DATA_HOME", ".local/share")


def recent_projects_file() -> Path:
    """Where to persist the MRU list of project_config.ini paths."""
    return xdg_data_dir() / "recent_projects.txt"


# --------------------------------------------------------------------------- #
# GPU
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def cuda_available() -> bool:
    """True iff a usable CUDA device is present.

    Short-circuits to ``False`` when ``CUDA_VISIBLE_DEVICES`` is set to
    empty string (test convention).
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES", "__unset__") == "":
        return False
    try:
        # Cheapest probe: nvidia-smi. Avoids importing numba/cupy just
        # to answer the question.
        r = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True, text=True, timeout=2,
        )
        return r.returncode == 0 and bool(r.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@lru_cache(maxsize=1)
def nvenc_available() -> bool:
    """True iff FFmpeg has a usable NVENC encoder.

    Useful for routing video-write operations through the RTX
    hardware encoder instead of software x264 — roughly a 10× throughput
    increase for heatmap / path-plot / classifier-validation clips.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    try:
        r = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=3,
        )
        return "h264_nvenc" in r.stdout or "hevc_nvenc" in r.stdout
    except subprocess.TimeoutExpired:
        return False


@lru_cache(maxsize=1)
def cuda_capability() -> Optional[tuple[int, int]]:
    """Compute capability of device 0, or ``None`` if no CUDA.

    For RTX 5070 Ti (Blackwell) this returns ``(12, 0)``. Used to pick
    cupy kernels vs. numba.cuda kernels that need specific SM targets.
    """
    if not cuda_available():
        return None
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=2,
        )
        first = r.stdout.strip().splitlines()[0]
        major, minor = first.split(".")
        return int(major), int(minor)
    except (subprocess.TimeoutExpired, IndexError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Display server
# --------------------------------------------------------------------------- #
def detect_display_server() -> str:
    """``'wayland'`` | ``'x11'`` | ``'thinlinc'`` | ``'headless'``."""
    # ThinLinc sets TLSESSIONDATA and wraps an Xvnc server. Detect it
    # first since DISPLAY will also be set.
    if os.environ.get("TLSESSIONDATA"):
        return "thinlinc"
    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    if os.environ.get("DISPLAY"):
        return "x11"
    return "headless"


def recommended_qpa_platform() -> str:
    """Qt platform plugin to prefer. Set via ``QT_QPA_PLATFORM``.

    ThinLinc (Xvnc-backed) renders best with plain ``xcb`` — wayland
    doesn't work under it, and ``xcb`` with software GL is most
    compatible. Native Wayland sessions get ``wayland`` first with
    ``xcb`` fallback.
    """
    ds = detect_display_server()
    if ds == "wayland":
        return "wayland;xcb"
    return "xcb"


__all__ = [
    "cpu_count", "setup_multiprocessing",
    "xdg_config_dir", "xdg_cache_dir", "xdg_data_dir", "recent_projects_file",
    "cuda_available", "nvenc_available", "cuda_capability",
    "detect_display_server", "recommended_qpa_platform",
]
