"""Test for pyglet removal from mufasa.utils.lookups.

User asked: "can you rewrite load_simba_fonts, or remove it? Is
it necessary or can system fonts be used?" Investigation showed
load_simba_fonts was a no-op for everything that actually
renders text (Tk uses platform native fonts via Xft/GDI/Cocoa,
not pyglet's font cache). So the function is now a stub and
pyglet is removed from project dependencies.

Verifies:
- pyglet is no longer imported at module level in lookups.py
- The dead Windows COINIT_MULTITHREADED block is gone
- load_simba_fonts is preserved (legacy SimBA.py still calls
  it) but is a no-op
- pyproject.toml no longer lists pyglet

    PYTHONPATH=. python tests/smoke_pyglet_removal.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path


def main() -> int:
    # ------------------------------------------------------------------ #
    # Case 1: lookups.py no longer imports pyglet at module level
    # ------------------------------------------------------------------ #
    src = Path("mufasa/utils/lookups.py").read_text()
    tree = ast.parse(src)
    pyglet_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pyglet" or alias.name.startswith("pyglet."):
                    pyglet_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and (
                node.module == "pyglet" or node.module.startswith("pyglet.")
            ):
                pyglet_imports.append(node.module)
    assert not pyglet_imports, (
        f"pyglet still imported in lookups.py: {pyglet_imports}"
    )

    # ------------------------------------------------------------------ #
    # Case 2: dead Windows COINIT_MULTITHREADED assignment is gone
    # (the comment explaining why it was removed may still mention
    # the constant name; we check for the actual mutation)
    # ------------------------------------------------------------------ #
    # Look for any *assignment* node that touches COINIT_MULTITHREADED
    coinit_assignments = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (
                    isinstance(tgt, ast.Attribute)
                    and tgt.attr == "COINIT_MULTITHREADED"
                ):
                    coinit_assignments.append(ast.unparse(node))
    assert not coinit_assignments, (
        f"Dead Windows COINIT_MULTITHREADED assignment(s) found: "
        f"{coinit_assignments}"
    )

    # ------------------------------------------------------------------ #
    # Case 3: load_simba_fonts function still exists (legacy SimBA.py
    # callers must keep working) but is a no-op. The body's executable
    # code must not reference pyglet — a docstring explaining the
    # historical pyglet usage is fine and expected.
    # ------------------------------------------------------------------ #
    fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "load_simba_fonts"
    )
    # Get the body's executable statements excluding the docstring
    body_no_docstring = [
        stmt for stmt in fn.body
        if not (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        )
    ]
    code_src = "\n".join(ast.unparse(s) for s in body_no_docstring)
    # No pyglet calls in actual code
    assert "pyglet" not in code_src.lower(), (
        f"load_simba_fonts body code must not reference pyglet; got: "
        f"{code_src!r}"
    )
    assert "add_file" not in code_src
    # Body should be effectively a no-op (just `return None` or nothing)
    assert len(body_no_docstring) <= 1, (
        f"load_simba_fonts should be a no-op stub; got "
        f"{len(body_no_docstring)} non-docstring statements"
    )

    # ------------------------------------------------------------------ #
    # Case 4: pyproject.toml no longer lists pyglet
    # ------------------------------------------------------------------ #
    pyproject = Path("pyproject.toml").read_text()
    # Look for the dependency line specifically. There might be
    # a comment or string that mentions pyglet in passing for
    # documentation purposes — we care about the actual deps.
    dep_section_start = pyproject.index("dependencies = [")
    dep_section_end = pyproject.index("]", dep_section_start)
    dep_section = pyproject[dep_section_start:dep_section_end]
    assert '"pyglet' not in dep_section, (
        "pyglet must not appear in dependencies array"
    )
    assert "'pyglet" not in dep_section, (
        "pyglet must not appear in dependencies array"
    )

    # ------------------------------------------------------------------ #
    # Case 5: SimBA.py callers (in legacy Tk launcher) still call
    # load_simba_fonts — function is preserved for them. We're not
    # touching SimBA.py; just verify the calls still resolve.
    # ------------------------------------------------------------------ #
    simba_src = Path("mufasa/SimBA.py").read_text()
    # The function is imported and called
    assert "load_simba_fonts" in simba_src, (
        "SimBA.py should still import/call load_simba_fonts "
        "(now a no-op stub)"
    )

    # ------------------------------------------------------------------ #
    # Case 6: docstring on load_simba_fonts explains why it's a
    # no-op (so future maintainers don't 're-fix' by re-adding pyglet)
    # ------------------------------------------------------------------ #
    docstring = ast.get_docstring(fn)
    assert docstring is not None, "load_simba_fonts must have a docstring"
    # Must mention something about no-op / compat / pyglet absence
    keywords_present = any(
        kw in docstring.lower()
        for kw in ("no-op", "compat", "stub", "pyglet")
    )
    assert keywords_present, (
        f"Docstring should explain why load_simba_fonts is a stub. "
        f"Got: {docstring[:200]!r}"
    )

    # ------------------------------------------------------------------ #
    # Case 7: matplotlib.font_manager IS still imported (it's used by
    # other functions in lookups.py — get_fonts() reads system TTFs
    # via it). This is a regression guard: my removal patch shouldn't
    # have over-eagerly removed matplotlib font code.
    # ------------------------------------------------------------------ #
    assert "matplotlib.font_manager" in src, (
        "matplotlib.font_manager is used elsewhere in lookups.py "
        "(get_fonts) and must remain imported"
    )

    print("smoke_pyglet_removal: 7/7 cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
