"""Smoke test for patch 122hp — dockable console for verbose output.

The workbench runs operations in a ProcessorRunner (QThread); backend
functions report progress by printing, which only reached the terminal. This
patch adds a console docked at the bottom of the main window that mirrors
stdout/stderr live, via a thread-safe tee-and-emit redirector.

The console module can't be imported here (PySide6 is absent in the sandbox),
so this test verifies (1) the module's structure and wiring via AST, and (2)
the thread-safety-critical pure logic — the _StreamRedirector file protocol and
the _base_stream unwrap — by compiling those classes in isolation with a stub
signal. The runtime behaviour of the Qt widget itself is validated on real
hardware.
"""
from __future__ import annotations

import ast
import contextlib
import io
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


# ------------------------------------------------------------------ #
# 1. module structure (AST — no import, PySide6 absent)
# ------------------------------------------------------------------ #
mod_path = REPO / "mufasa/ui_qt/console_dock.py"
check("console_dock module exists", mod_path.exists())
src = mod_path.read_text()
tree = ast.parse(src)
classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

check("_StreamRedirector class defined", "_StreamRedirector" in classes)
check("ConsoleDockWidget class defined", "ConsoleDockWidget" in classes)
check("attach_console_dock defined", "attach_console_dock" in funcs)
check("detach_console_dock defined", "detach_console_dock" in funcs)
check("_base_stream unwrap helper defined", "_base_stream" in funcs)

# redirector implements the file protocol
redir = next(n for n in ast.walk(tree)
             if isinstance(n, ast.ClassDef) and n.name == "_StreamRedirector")
redir_methods = {n.name for n in redir.body if isinstance(n, ast.FunctionDef)}
for m in ("write", "flush", "isatty", "writable", "fileno"):
    check(f"_StreamRedirector.{m} present", m in redir_methods)
check("_StreamRedirector has a textWritten signal",
      "textWritten = Signal(str)" in src)

# attach: uses a queued connection (thread-safety) and a bottom dock area
check("attach uses a queued connection (worker->GUI thread safety)",
      "Qt.QueuedConnection" in src)
check("attach docks at the bottom",
      "Qt.BottomDockWidgetArea" in src)
check("attach tees to the unwrapped base stream",
      "_base_stream(sys.stdout)" in src
      and "_base_stream(sys.stderr)" in src)
check("attach is idempotent (returns existing dock)",
      "_console_dock" in src and 'getattr(main, "_console_dock", None)' in src)

# console view is bounded and read-only
check("console view is read-only",
      "setReadOnly(True)" in src)
check("console view is block-count bounded",
      "setMaximumBlockCount" in src)

# ------------------------------------------------------------------ #
# 2. workbench wiring (AST)
# ------------------------------------------------------------------ #
wb = (REPO / "mufasa/ui_qt/workbench.py").read_text()
check("workbench attaches the console dock",
      "attach_console_dock(self)" in wb)
check("workbench detaches on close (closeEvent)",
      "def closeEvent" in wb and "detach_console_dock(self)" in wb)
check("console attach failure can't block the workbench opening",
      "console dock unavailable" in wb)

# Regression guard (the closeEvent insertion once swallowed the def line of the
# method right after it): closeEvent must be added WITHOUT damaging adjacent
# methods. Verify every menu-referenced handler still exists as a real method
# on MufasaWorkbench, and closeEvent itself is a sibling method (not something
# that absorbed the next method's body).
_wb_tree = ast.parse(wb)
_wb_cls = next((n for n in ast.walk(_wb_tree)
                if isinstance(n, ast.ClassDef)
                and n.name == "MufasaWorkbench"), None)
check("MufasaWorkbench class is parseable", _wb_cls is not None)
_methods = {n.name for n in (_wb_cls.body if _wb_cls else [])
            if isinstance(n, ast.FunctionDef)}
check("closeEvent is a method of MufasaWorkbench", "closeEvent" in _methods)
check("_launch_synced_viewer survived the closeEvent insertion",
      "_launch_synced_viewer" in _methods)
_bm = next((n for n in (_wb_cls.body if _wb_cls else [])
            if isinstance(n, ast.FunctionDef) and n.name == "_build_menus"),
           None)
_menu_refs = set()
if _bm is not None:
    for node in ast.walk(_bm):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and node.attr.startswith("_")):
            _menu_refs.add(node.attr)
# handler-style refs (methods invoked from the menu) must all resolve
_handlers = {r for r in _menu_refs
             if r.startswith(("_launch", "_on", "_open", "_switch", "_show"))}
check("every menu handler resolves to a real method (no swallowed defs)",
      _handlers.issubset(_methods))

# ------------------------------------------------------------------ #
# 3. pure logic — _StreamRedirector protocol + _base_stream unwrap
#    (compiled in isolation with a stub Signal)
# ------------------------------------------------------------------ #
redir_ast = next(n for n in ast.walk(tree)
                 if isinstance(n, ast.ClassDef)
                 and n.name == "_StreamRedirector")
base_ast = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_base_stream")


class _StubSignal:
    def __init__(self, *a):
        self._slots = []

    def connect(self, slot, *a):
        self._slots.append(slot)

    def emit(self, *args):
        for s in self._slots:
            s(*args)


ns: dict = {
    "QObject": object,
    "Signal": lambda *a: _StubSignal(),
    "_StubSignal": _StubSignal,
    "contextlib": contextlib,
    "TextIO": io.TextIOBase,
    "__name__": "_console_pure",
}
# strip the base-class QObject so `object.__init__` is happy with the super call
redir_code = ast.get_source_segment(src, redir_ast)
redir_code = redir_code.replace("class _StreamRedirector(QObject):",
                                "class _StreamRedirector:")
redir_code = redir_code.replace("super().__init__(parent)", "pass")
# textWritten is a class attribute assigned via Signal(str); make it per-instance
redir_code = redir_code.replace("    textWritten = Signal(str)\n", "")
redir_code = redir_code.replace(
    "        self._original = original",
    "        self._original = original\n"
    "        self.textWritten = _StubSignal()")
exec(compile(redir_code, "<redir>", "exec"), ns)
exec(compile(ast.get_source_segment(src, base_ast), "<base>", "exec"), ns)
Redir = ns["_StreamRedirector"]
base_stream = ns["_base_stream"]

# tee: writes reach both original and the signal
real = io.StringIO()
got: list[str] = []
r = Redir(real)
r.textWritten.connect(lambda t: got.append(t))
r.write("[smoother-v2] iter 1\n")
r.write("hello")
check("write tees to the original stream",
      real.getvalue() == "[smoother-v2] iter 1\nhello")
check("write emits the same text to the console",
      "".join(got) == "[smoother-v2] iter 1\nhello")

# empty write emits nothing (no stray blank line)
got.clear()
r.write("")
check("empty write emits nothing", got == [])

# write returns the length (file protocol)
check("write returns the character count", r.write("abc") == 3)

# unwrap: chained redirectors collapse to the real stream
r1 = Redir(real)
r2 = Redir(r1)
r3 = Redir(r2)
check("_base_stream unwraps a chain to the real stream",
      base_stream(r3) is real)
check("_base_stream leaves a plain stream alone",
      base_stream(real) is real)
check("_base_stream handles None", base_stream(None) is None)

# broken original: console still gets text, no crash
class _Broken:
    def write(self, t):
        raise ValueError("boom")

    def flush(self):
        raise ValueError("boom")


got.clear()
r = Redir(_Broken())
r.textWritten.connect(lambda t: got.append(t))
r.write("still captured")
r.flush()
check("a broken original doesn't lose console output or crash",
      got == ["still captured"])

n_pass = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"smoke_122hp_console_dock: {n_pass}/{len(checks)} checks passed")
sys.exit(0 if n_pass == len(checks) else 1)
