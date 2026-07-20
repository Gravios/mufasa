"""``mufasa-preview`` — open the pose-overlay viewer on a video + its pose files.

Patch 122hy. A thin front-end over :mod:`mufasa.tools.pose_video_overlay` that
saves typing the long, hash-tagged pose paths by hand. Two ways to use it:

* **Explicit paths** — name the video and, optionally, the smoothed / raw pose
  files::

      mufasa-preview --video rec.mp4 \\
          --smoothed rec.fdlc_smoothed_v2.<hash>.parquet \\
          --raw rec.fdlc.parquet

  If only ``--video`` is given, its folder is scanned and any pose files that
  share the video's stem are picked up automatically.

* **Folder search** — point at a directory and let it pair videos with their
  pose files by name::

      mufasa-preview --folder .

The pairing reduces a video and its pose files to a common *session stem*, so
``rec.mp4`` matches ``rec.fdlc.parquet`` (raw FreeDLC) and
``rec.fdlc_smoothed_v2.<hash>.parquet`` (smoothed, parameter-hash tagged) alike.
When several smoothed files share a stem (different parameter hashes), the most
recent is used.

The viewer is invoked in-process via ``pose_video_overlay.main`` — this is the
same package, so there's no subprocess. ``--print`` shows the equivalent
``python -m mufasa.tools.pose_video_overlay`` command without opening anything.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

# Recognised media / pose extensions.
_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg")
_POSE_EXTS = (".parquet", ".csv")


# --------------------------------------------------------------------------- #
# Filename -> canonical "session stem" (so a video and its pose files pair up)
# --------------------------------------------------------------------------- #
def _video_stem(name: str) -> str:
    """Strip a known video extension; return the name otherwise unchanged."""
    low = name.lower()
    for ext in _VIDEO_EXTS:
        if low.endswith(ext):
            return name[: -len(ext)]
    return name


def _pose_stem(name: str) -> str:
    """Reduce a pose filename to the base session identity.

    Peels, in order: the ``.parquet`` / ``.csv`` extension; a trailing
    ``.<hexhash>`` parameter tag after ``_smoothed_v2`` (the smoother's
    filename hash, patch 122hm/hn); the ``_smoothed_v2`` suffix; and a trailing
    ``.fdlc`` marker (raw FreeDLC). So ``rec.fdlc_smoothed_v2.deadbeef.parquet``,
    ``rec.fdlc.parquet`` and ``rec.parquet`` all reduce to ``rec`` — matching a
    ``rec.mp4`` video.
    """
    stem = name
    low = stem.lower()
    for ext in _POSE_EXTS:
        if low.endswith(ext):
            stem = stem[: -len(ext)]
            break
    match = re.match(r"^(.*_smoothed_v2)\.[0-9a-fA-F]+$", stem)
    if match:
        stem = match.group(1)
    if stem.endswith("_smoothed_v2"):
        stem = stem[: -len("_smoothed_v2")]
    if stem.endswith(".fdlc"):
        stem = stem[: -len(".fdlc")]
    return stem


def _is_smoothed(name: str) -> bool:
    return "_smoothed_v2" in name


# --------------------------------------------------------------------------- #
# Folder discovery
# --------------------------------------------------------------------------- #
def _discover(folder: str) -> list[dict]:
    """Group a folder's videos with their raw / smoothed pose files by stem.

    Returns a list of ``{stem, video, raw, smoothed}`` dicts, one per video
    that has at least one pose file, sorted by stem. When several smoothed
    files share a stem (e.g. different parameter hashes), the most recently
    modified is chosen.
    """
    videos: dict[str, str] = {}
    raws: dict[str, str] = {}
    smoothed: dict[str, list[str]] = {}

    for entry in sorted(os.listdir(folder)):
        path = os.path.join(folder, entry)
        if not os.path.isfile(path):
            continue
        low = entry.lower()
        if low.endswith(_VIDEO_EXTS):
            videos.setdefault(_video_stem(entry), path)
        elif low.endswith(_POSE_EXTS):
            stem = _pose_stem(entry)
            if _is_smoothed(entry):
                smoothed.setdefault(stem, []).append(path)
            else:
                raws.setdefault(stem, path)

    groups: list[dict] = []
    for stem in sorted(videos):
        raw = raws.get(stem)
        sm_list = smoothed.get(stem, [])
        chosen = max(sm_list, key=os.path.getmtime) if sm_list else None
        if raw or chosen:
            groups.append(
                {"stem": stem, "video": videos[stem],
                 "raw": raw, "smoothed": chosen}
            )
    return groups


# --------------------------------------------------------------------------- #
# Overlay argv construction
# --------------------------------------------------------------------------- #
def _build_overlay_argv(
    video: str,
    smoothed: str | None,
    raw: str | None,
    likelihood_threshold: float,
    pose_offset: int,
    start_frame: int,
) -> list[str]:
    """Build the argv list for ``pose_video_overlay.main``."""
    argv = [video]
    if smoothed:
        argv += ["--smoothed", smoothed]
    if raw:
        argv += ["--raw", raw]
    if likelihood_threshold:
        argv += ["--likelihood-threshold", str(likelihood_threshold)]
    if pose_offset:
        argv += ["--pose-offset", str(pose_offset)]
    if start_frame:
        argv += ["--start-frame", str(start_frame)]
    return argv


def _shell_quote(token: str) -> str:
    if token and all(ch.isalnum() or ch in "-_./=" for ch in token):
        return token
    return "'" + token.replace("'", "'\\''") + "'"


def _printable(argv: list[str]) -> str:
    parts = ["python", "-m", "mufasa.tools.pose_video_overlay"]
    return " ".join(_shell_quote(t) for t in parts + argv)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mufasa-preview",
        description=(
            "Open the pose-overlay viewer on a video and its pose file(s), "
            "from explicit paths or by searching a folder."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--video", help="Path to the source video (explicit-paths mode).",
    )
    source.add_argument(
        "--folder", "-d",
        help="Folder to search for matching video + pose files.",
    )
    parser.add_argument(
        "--smoothed", help="Smoothed pose file (explicit-paths mode).",
    )
    parser.add_argument(
        "--raw", help="Raw pose file to overlay (explicit-paths mode).",
    )
    parser.add_argument(
        "--likelihood-threshold", type=float, default=0.0,
        help="Hide raw markers below this likelihood (default 0).",
    )
    parser.add_argument(
        "--pose-offset", type=int, default=0,
        help="Frame offset between video and pose data (default 0).",
    )
    parser.add_argument(
        "--start-frame", type=int, default=0,
        help="Open the viewer at frame N (default 0).",
    )
    parser.add_argument(
        "--print", dest="print_only", action="store_true",
        help="Print the equivalent overlay command(s) without opening the "
             "viewer.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="In --folder mode, preview every matched video in turn "
             "(default: require a single match; error if ambiguous).",
    )
    return parser


def _resolve_jobs(args: argparse.Namespace) -> tuple[list[list[str]], int]:
    """Turn parsed args into a list of overlay argvs.

    Returns ``(jobs, error_code)``; on success ``error_code`` is 0 and ``jobs``
    is non-empty, otherwise ``jobs`` is empty and ``error_code`` is the exit
    code (messages already printed to stderr).
    """
    # ---- explicit-paths mode ----
    if args.video:
        if not os.path.isfile(args.video):
            print(f"mufasa-preview: no such video: {args.video}",
                  file=sys.stderr)
            return [], 2
        if args.smoothed or args.raw:
            smoothed, raw = args.smoothed, args.raw
        else:
            # look beside the video for its pose files
            folder = os.path.dirname(os.path.abspath(args.video)) or "."
            stem = _video_stem(os.path.basename(args.video))
            match = next(
                (g for g in _discover(folder) if g["stem"] == stem), None
            )
            if match is None:
                print(
                    "mufasa-preview: no pose file given and none found beside "
                    f"the video for stem {stem!r}. Pass --smoothed / --raw, "
                    "or use --folder.",
                    file=sys.stderr,
                )
                return [], 2
            smoothed, raw = match["smoothed"], match["raw"]
        argv = _build_overlay_argv(
            args.video, smoothed, raw,
            args.likelihood_threshold, args.pose_offset, args.start_frame,
        )
        return [argv], 0

    # ---- folder mode ----
    folder = args.folder
    if not os.path.isdir(folder):
        print(f"mufasa-preview: not a folder: {folder}", file=sys.stderr)
        return [], 2
    groups = _discover(folder)
    if not groups:
        print(
            f"mufasa-preview: no video with a matching pose file found in "
            f"{folder}.",
            file=sys.stderr,
        )
        return [], 2
    if len(groups) > 1 and not args.all:
        print(
            f"mufasa-preview: {len(groups)} videos matched in {folder}; pass "
            "--all to preview each, or use --video to pick one. Matches:",
            file=sys.stderr,
        )
        for g in groups:
            tags = [t for t, present in
                    (("smoothed", g["smoothed"]), ("raw", g["raw"])) if present]
            print(f"  - {g['stem']}  ({', '.join(tags)})", file=sys.stderr)
        return [], 2

    jobs = [
        _build_overlay_argv(
            g["video"], g["smoothed"], g["raw"],
            args.likelihood_threshold, args.pose_offset, args.start_frame,
        )
        for g in groups
    ]
    return jobs, 0


def main(argv: list[str] | None = None) -> int:
    """``mufasa-preview`` entry point."""
    args = _build_parser().parse_args(argv)
    jobs, err = _resolve_jobs(args)
    if err:
        return err

    if args.print_only:
        for job in jobs:
            print(_printable(job))
        return 0

    # Invoke the overlay viewer in-process (same package, no subprocess).
    from mufasa.tools.pose_video_overlay import main as overlay_main
    for job in jobs:
        video = job[0]
        print(f"[preview] {os.path.basename(video)}")
        rc = overlay_main(job)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
