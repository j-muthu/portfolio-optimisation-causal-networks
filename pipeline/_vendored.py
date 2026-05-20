"""Import bridge to the vendored ``causalnex`` and ``lingam`` source trees.

Both libraries live in this repo as full git checkouts (``thesis/causalnex`` and
``thesis/lingam``) rather than as pip-installed packages.  Importing them the
normal way is awkward for two reasons:

* ``causalnex.structure.__init__`` eagerly imports the PyTorch-backed
  ``DAGRegressor``/``DAGClassifier``.  We only need DYNOTEARS, so we do not want
  to drag in ``torch``.
* ``lingam.__init__`` imports every algorithm in the package (LiNA, CAMUV, ...),
  several of which need heavy optional dependencies.  We only need VARLiNGAM.

To sidestep both, we register lightweight stand-in package objects in
``sys.modules`` whose ``__path__`` points at the real source directories.  The
genuine submodules we care about (``dynotears``, ``var_lingam``, ...) are then
imported normally and resolved against that path, *without* running the
packages' ``__init__.py`` files.

This module also applies small forward-compatibility shims so the (older)
vendored code runs on the modern pandas installed in the venv.

Importing this module is the only supported entry point: ::

    from pipeline._vendored import from_pandas_dynamic, VARLiNGAM
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
THESIS_ROOT = Path(__file__).resolve().parent.parent
_CAUSALNEX_PKG = THESIS_ROOT / "causalnex" / "causalnex"
_LINGAM_PKG = THESIS_ROOT / "lingam" / "lingam"


# ----------------------------------------------------------------------------
# Compatibility shims
# ----------------------------------------------------------------------------
def _patch_pandas_compat() -> None:
    """Restore APIs the vendored code expects but newer pandas removed.

    ``causalnex.structure.transformers.DynamicDataTransformer`` calls
    ``df.index.is_integer()``.  ``pd.Index.is_integer`` was deprecated in pandas
    2.0 and removed in 2.1.  We re-add it with its documented semantics so the
    transformer keeps working on pandas >= 2.1 (the venv ships pandas 3.x).
    """
    if not hasattr(pd.Index, "is_integer"):
        pd.Index.is_integer = lambda self: pd.api.types.is_integer_dtype(self)


# ----------------------------------------------------------------------------
# Package stand-ins
# ----------------------------------------------------------------------------
def _register_namespace(name: str, path: Path) -> None:
    """Register a stub package in ``sys.modules`` without running its __init__.

    The stub behaves like a package (it carries a ``__path__``) so that genuine
    submodules are importable, but its own ``__init__.py`` is never executed.
    """
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    module.__package__ = name
    sys.modules[name] = module


def _load_causalnex() -> None:
    if "causalnex.structure.dynotears" in sys.modules:
        return
    if not _CAUSALNEX_PKG.is_dir():
        raise ImportError(f"causalnex source not found at {_CAUSALNEX_PKG}")
    _register_namespace("causalnex", _CAUSALNEX_PKG)
    _register_namespace("causalnex.structure", _CAUSALNEX_PKG / "structure")
    # dynotears.py does ``from causalnex.structure import StructureModel`` -- the
    # name must be an attribute of our stub package before that import runs.
    from causalnex.structure.structuremodel import StructureModel  # noqa: E402

    sys.modules["causalnex.structure"].StructureModel = StructureModel  # type: ignore[attr-defined]


def _load_lingam() -> None:
    if "lingam.var_lingam" in sys.modules:
        return
    if not _LINGAM_PKG.is_dir():
        raise ImportError(f"lingam source not found at {_LINGAM_PKG}")
    _register_namespace("lingam", _LINGAM_PKG)


_patch_pandas_compat()
_load_causalnex()
_load_lingam()

# These imports resolve against the stub packages registered above.
from causalnex.structure.dynotears import (  # noqa: E402
    from_numpy_dynamic,
    from_pandas_dynamic,
)
from causalnex.structure.structuremodel import StructureModel  # noqa: E402
from lingam.var_lingam import VARBootstrapResult, VARLiNGAM  # noqa: E402

__all__ = [
    "from_pandas_dynamic",
    "from_numpy_dynamic",
    "StructureModel",
    "VARLiNGAM",
    "VARBootstrapResult",
    "THESIS_ROOT",
]
