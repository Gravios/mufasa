"""
SimBA (Simple Behavioral Analysis)
Toolkit for computer classification and analysis of behaviors in experimental animals
"""
import multiprocessing
import os


def _is_wsl() -> bool:
    """Inline copy of ``mufasa.utils.checks.is_wsl``.

    Kept here to avoid triggering the heavyweight ``mufasa.utils.checks``
    import chain (numba, cv2, shapely, pandas) at package load time.
    Previously, any ``import mufasa.*`` paid the numba startup cost because
    ``mufasa/__init__.py`` pulled ``is_wsl`` from ``checks``, whose module
    top imports ``mufasa.data_processors.cuda.utils`` → ``numba.cuda``.
    """
    try:
        with open("/proc/version", "r", encoding="utf-8") as fh:
            return "microsoft" in fh.read().lower()
    except (OSError, FileNotFoundError):
        return False


if _is_wsl():
    multiprocessing.set_start_method("spawn", force=True)

__author__ = "Simon Nilsson"
__author_email__ = "sronilsson@gmail.com"
__maintainer__ = "Simon Nilsson"
__maintainer_email__ = "sronilsson@gmail.com"
__copyright__ = "Copyright 2024, Simon Nilsson"
__license__ = "Modified BSD 3-Clause License"
__url__ = "https://github.com/sgoldenlab/mufasa"
__description__ = "Toolkit for computer classification and analysis of behaviors in experimental animals"