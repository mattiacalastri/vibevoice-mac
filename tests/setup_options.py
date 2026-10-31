#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read packaging/setup_py2app.py's OPTIONS dict without executing it.

Importing that module calls `setup()` at module level and starts a real build,
so the packaging tests parse it instead — the same trick build_release.sh uses
to read SILERO_ONSET/OFFSET out of engine.py.

Plain `ast.literal_eval` is not enough: OPTIONS references module-level
constants (`VERSION`), and a bare literal_eval raises `malformed node or string`
on the Name node. Two test modules were parsing this file with their own copy of
the logic, and introducing VERSION broke the copy that had not been updated
(sess.9767) — hence one reader, imported by both.
"""
from __future__ import annotations

import ast
from pathlib import Path

SETUP = Path(__file__).resolve().parent.parent / "packaging" / "setup_py2app.py"


def py2app_options(setup_path: Path | None = None) -> dict:
    """The OPTIONS dict, with module-level constant references resolved."""
    tree = ast.parse((setup_path or SETUP).read_text())
    consts: dict[str, object] = {}
    options: ast.expr | None = None
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name == "OPTIONS":
            options = node.value
        else:
            try:
                consts[name] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                pass  # not a literal (e.g. a call) — no key of ours can need it
    if options is None:
        raise AssertionError("OPTIONS dict not found in setup_py2app.py")

    def resolve(node: ast.expr):
        if isinstance(node, ast.Name):
            return consts[node.id]
        if isinstance(node, ast.Dict):
            return {ast.literal_eval(k): resolve(v)
                    for k, v in zip(node.keys, node.values) if k is not None}
        if isinstance(node, (ast.List, ast.Tuple)):
            return [resolve(e) for e in node.elts]
        return ast.literal_eval(node)

    return resolve(options)
