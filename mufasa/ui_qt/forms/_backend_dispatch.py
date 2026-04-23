"""
mufasa.ui_qt.forms._backend_dispatch
====================================

Shared helpers for forms that call out to a backend class/function
with a ``**kwargs`` dict the form collected. Two forms-scale problems
this module addresses:

1. **Over-collection.** Forms surface UI controls to the user and
   collect their values into an extras dict. Some of those controls
   exist for UX (visual grouping, sensible defaults) but don't map
   onto backend parameters — e.g. the heatmap form has a
   ``heatmap_opacity`` slider that several backends silently don't
   accept. Without a filter, passing these through produces
   ``TypeError: <Class>.__init__() got an unexpected keyword
   argument 'heatmap_opacity'`` the moment the user clicks Run.

2. **Backend drift.** Upstream refactors occasionally rename,
   remove, or add parameters. The form can survive signature drift
   (at the cost of silently ignoring a widget) rather than break
   outright.

The filter is best-effort: if the backend can't be introspected
(closures, C extensions, decorators that eat the signature) we pass
everything through and let the backend raise naturally.

Usage::

    from mufasa.ui_qt.forms._backend_dispatch import filter_kwargs
    kwargs = filter_kwargs(backend, kwargs)
    runner = backend(**kwargs)
"""
from __future__ import annotations

import inspect
from typing import Any, Callable, Dict


def filter_kwargs(backend: Callable[..., Any],
                  kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop kwargs the backend's ``__init__`` (or callable) signature
    doesn't accept. Returns a new dict; does not mutate the input.

    Backends registered via lazy factories should set the factory's
    ``__name__`` to ``"{modpath}.{classname}"`` so this helper can
    dereference the underlying class without importing it eagerly.

    If the backend accepts ``**kwargs`` (VAR_KEYWORD) no filtering is
    applied — those backends explicitly opt into accepting anything.
    """
    try:
        f_name = getattr(backend, "__name__", "")
        # Closures set their own __name__ to "modpath.ClassName"; direct
        # references have "ClassName" with no dot. For direct references
        # fall through to inspect.signature on the backend itself.
        cls: Any = backend
        if "." in f_name:
            mod_path, _, cls_name = f_name.rpartition(".")
            mod = __import__(mod_path, fromlist=[cls_name])
            cls = getattr(mod, cls_name, backend)
        # For classes, inspect __init__; for functions, inspect the
        # function itself.
        target = cls.__init__ if inspect.isclass(cls) else cls
        sig = inspect.signature(target)
        params = sig.parameters
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD
                        for p in params.values())
        if has_varkw:
            return dict(kwargs)
        accepts = set(params) - {"self"}
        return {k: v for k, v in kwargs.items() if k in accepts}
    except Exception:
        # Can't introspect — let the backend fail naturally if kwargs
        # really are bad.
        return dict(kwargs)


__all__ = ["filter_kwargs"]
