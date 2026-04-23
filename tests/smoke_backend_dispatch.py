"""Smoke-test for mufasa.ui_qt.forms._backend_dispatch.filter_kwargs.

Run headless (no PySide6 needed):

    python tests/smoke_backend_dispatch.py
"""
from __future__ import annotations

import sys
import types


def main() -> int:
    from mufasa.ui_qt.forms._backend_dispatch import filter_kwargs

    # ------------------------------------------------------------------ #
    # Case 1 — class with explicit signature drops unknown kwargs
    # ------------------------------------------------------------------ #
    class Backend1:
        def __init__(self, a, b=1, c=2):
            pass
    out = filter_kwargs(Backend1, {"a": 1, "b": 2, "z": 99})
    assert out == {"a": 1, "b": 2}, f"case 1: {out}"

    # ------------------------------------------------------------------ #
    # Case 2 — function with **kwargs bypasses filter
    # ------------------------------------------------------------------ #
    def backend2(**kw):
        pass
    out = filter_kwargs(backend2, {"whatever": 1, "anything": 2})
    assert out == {"whatever": 1, "anything": 2}, f"case 2: {out}"

    # ------------------------------------------------------------------ #
    # Case 3 — unresolvable modpath falls back to pass-through. This is
    # the safety net — if anything about introspection fails we'd
    # rather the backend raise naturally than eat the user's Run click.
    # ------------------------------------------------------------------ #
    def f3(**kw):
        pass
    f3.__name__ = "nonexistent.module.Class"
    out = filter_kwargs(f3, {"x": 1, "y": 2, "z": 3})
    assert out == {"x": 1, "y": 2, "z": 3}, f"case 3: {out}"

    # ------------------------------------------------------------------ #
    # Case 4 — canonical Mufasa lazy factory pattern. A factory's
    # __name__ is set to "modpath.ClassName" so the filter can locate
    # the real class without eagerly importing it.
    # ------------------------------------------------------------------ #
    class RealClass:
        def __init__(self, a, b=1):
            pass
    mod4 = types.ModuleType("_fake_mod_ok")
    mod4.RealClass = RealClass
    sys.modules["_fake_mod_ok"] = mod4

    def lazy(**kw):
        return RealClass(**kw)
    lazy.__name__ = "_fake_mod_ok.RealClass"
    out = filter_kwargs(lazy, {"a": 1, "b": 2, "z": 99})
    assert out == {"a": 1, "b": 2}, f"case 4: {out}"

    # ------------------------------------------------------------------ #
    # Case 5 — regression test for the DLC2Yolo crash. The ConverterForm
    # for "DLC→YOLO keypoints" surfaces a 'sample_size' control that
    # DLC2Yolo's signature doesn't accept. Before the fix this crashed
    # with TypeError the moment the user clicked Run. After: dropped
    # silently, backend receives only what it accepts.
    # ------------------------------------------------------------------ #
    class FakeDLC2Yolo:
        def __init__(self, dlc_dir, save_dir, padding=0,
                     train_size=0.8, verbose=True):
            pass
    mod5 = types.ModuleType("_fake_dlc")
    mod5.FakeDLC2Yolo = FakeDLC2Yolo
    sys.modules["_fake_dlc"] = mod5

    def lazy_dlc(**kw):
        return FakeDLC2Yolo(**kw)
    lazy_dlc.__name__ = "_fake_dlc.FakeDLC2Yolo"
    out = filter_kwargs(lazy_dlc, {
        "dlc_dir": "/in", "save_dir": "/out", "padding": 5,
        "sample_size": 100,     # <- would have crashed
        "verbose": True,
    })
    assert "sample_size" not in out, f"case 5 kept sample_size: {out}"
    assert out["dlc_dir"] == "/in"

    # ------------------------------------------------------------------ #
    # Case 6 — input dict is not mutated
    # ------------------------------------------------------------------ #
    orig = {"a": 1, "z": 2}
    _ = filter_kwargs(Backend1, orig)
    assert orig == {"a": 1, "z": 2}, "input dict was mutated"

    print("smoke_backend_dispatch: 6/6 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
