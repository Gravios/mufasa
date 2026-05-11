"""Persist the location of the most-recently-loaded project.

Patch 121i. Convention mirrors patch 121h (~/.config/mufasa/
models/): a single file at ~/.config/mufasa/recent holds the
absolute path to the last opened project_config.ini. One line,
plain text. Silent failures on read/write (read-only home,
network share quirks) — the launcher just degrades back to its
prior auto-discover behavior.

Kept in its own module (no PySide6 dependency) so it's
importable and testable from headless environments. Both
workbench_app.py (launch path) and workbench.py (File→Open
project path) consume from here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


_RECENT_PROJECT_PATH = (
    Path.home() / ".config" / "mufasa" / "recent"
)


def save_recent_project(config_path: str) -> None:
    """Write ``config_path`` to ~/.config/mufasa/recent.

    Called whenever a project is loaded (auto-discover,
    explicit --project flag, File→Open project, File→New
    project) so the next launch can resume. Failures are
    silent — a missing parent dir is created, but read-only
    homes or permission errors are swallowed (the launcher
    falls back to its non-recent path).
    """
    try:
        _RECENT_PROJECT_PATH.parent.mkdir(
            parents=True, exist_ok=True,
        )
        _RECENT_PROJECT_PATH.write_text(
            str(Path(config_path).resolve()) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def load_recent_project() -> Optional[Path]:
    """Read ~/.config/mufasa/recent and return its path if
    valid.

    Returns ``None`` when the file doesn't exist, is unreadable,
    is empty, or points to a path that no longer exists or is
    not a regular file. Stale entries (e.g. moved/deleted
    projects) cleanly fall through to the launcher's normal
    behavior.
    """
    try:
        text = _RECENT_PROJECT_PATH.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    line = text.strip()
    if not line:
        return None
    candidate = Path(line)
    if not candidate.is_file():
        return None
    return candidate
