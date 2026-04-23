"""
mufasa.ui_qt.operations
======================

Small abstraction for the "apply a transformation to one file or a
whole directory" pattern used by ~12 popups in the Tk codebase
(Interpolate, Smoothing, Outlier Correction, ...).

**Why this exists**

The legacy popups each reimplement:

1. Single-file ``FileSelect`` + directory ``FolderSelect`` + two RUN
   buttons + a ``run(multiple: bool)`` dispatcher.
2. A ``multi_index_df_headers`` flag derived from "is the data in the
   raw pose-estimation input directory?" — implemented twice, two
   different ways, one buggy (string equality vs resolved path).

By centralising this into :class:`DatasetOp`, ports can stop carrying
the duplicated + divergent path-comparison logic. Each specific op
(Interpolate, Smoothing, ...) subclasses once and the popup becomes
trivial widget wiring.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Iterator, Optional


class DatasetOp:
    """Unified wrapper for single-file-or-directory data transforms.

    Subclasses must override :meth:`_run_one` (do the work on a single
    file) and should not override :meth:`run` (which handles dispatch).

    The ``multi_index_df_headers`` detection — "is my target inside the
    project's *raw* pose-estimation input directory?" — is done here,
    once, using resolved absolute paths. This replaces the duplicated
    and divergent logic in the Tk popups.
    """

    def __init__(
        self,
        config_path: os.PathLike | str,
        data_path: os.PathLike | str,
        *,
        raw_input_dir: Optional[os.PathLike | str] = None,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self.data_path = Path(data_path).resolve()
        # ``raw_input_dir`` is the project's raw pose-estimation CSV
        # directory (``project_folder/csv/input_csv`` by SimBA convention).
        # Where DLC/SLEAP native multi-row headers are still present.
        self._raw_input_dir = (
            Path(raw_input_dir).resolve() if raw_input_dir else None
        )

    # ---- public API ------------------------------------------------- #
    @property
    def is_directory(self) -> bool:
        return self.data_path.is_dir()

    @property
    def has_multi_index_headers(self) -> bool:
        """True iff the target is the raw pose-estimation directory.

        Uses fully resolved paths — replaces the Tk version's fragile
        mix of ``str ==`` (Smoothing) and ``Path.resolve().absolute()
        ==`` (Interpolate).
        """
        if self._raw_input_dir is None:
            return False
        effective_dir = (
            self.data_path if self.data_path.is_dir() else self.data_path.parent
        )
        return effective_dir == self._raw_input_dir

    def iter_targets(self, exts: Iterable[str] = (".csv", ".parquet")) -> Iterator[Path]:
        """Yield every file the op should process."""
        if self.data_path.is_file():
            yield self.data_path
            return
        if self.data_path.is_dir():
            for ext in exts:
                yield from sorted(self.data_path.glob(f"*{ext}"))

    def run(self) -> None:
        """Dispatcher — calls :meth:`_run_one` for each target."""
        for target in self.iter_targets():
            self._run_one(target)

    # ---- to override ------------------------------------------------ #
    def _run_one(self, target: Path) -> None:
        raise NotImplementedError


__all__ = ["DatasetOp"]
