# SPDX-License-Identifier: MIT
"""Grafted into the ALIAS bundle as Contents/Resources/sitecustomize.py.

The alias bundle boots py2app's own `site.py`, which at line 182 does

    try: import sitecustomize
    except ImportError: pass

and only defines `PREFIXES` twenty lines LATER. Homebrew's sitecustomize opens
with `site.PREFIXES[:] = …`, so on this machine the app died before drawing
anything:

    AttributeError: partially initialized module 'site' has no attribute 'PREFIXES'
    Fatal Python error: init_import_site: Failed to import the site module

The `except ImportError` does not catch an AttributeError, which is why it was
fatal rather than skipped. Note this is an ALIAS-mode fault only: a release
bundle carries its own stdlib, so Homebrew's sitecustomize is never on its path.

Seed the attribute with the same value site.py assigns later, then hand over to
the real Homebrew file — it still gets to shorten the Cellar paths and pin
sys.executable, which is its actual job. Shadowing it outright would work too,
and would silently change which site-packages the app resolves. Fixing the
ordering keeps the behaviour and removes only the crash.
"""
import os
import site
import sys

if not hasattr(site, "PREFIXES"):
    site.PREFIXES = [sys.prefix, sys.exec_prefix]

_here = os.path.dirname(os.path.abspath(__file__))
for _dir in sys.path:
    try:
        if not _dir or os.path.abspath(_dir) == _here:
            continue
        _real = os.path.join(_dir, "sitecustomize.py")
        if os.path.isfile(_real):
            with open(_real) as _fh:
                _src = _fh.read()
            exec(compile(_src, _real, "exec"))  # noqa: S102 — the host's own file
            break
    except Exception:
        # Cosmetic path tuning. Never worth taking the app down for.
        break
