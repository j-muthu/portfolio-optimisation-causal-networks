"""Import bridge to the vendored ``causalnex`` and ``lingam`` source trees.

Registers stub packages in ``sys.modules`` so the submodules we need import
without running the packages' heavy ``__init__.py`` files (torch, LiNA, ...).
Import from here: ``from pipeline._vendored import from_pandas_dynamic``.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd

# Paths
THESIS_ROOT = Path(__file__).resolve().parent.parent
_CAUSALNEX_PKG = THESIS_ROOT / "causalnex" / "causalnex"
_LINGAM_PKG = THESIS_ROOT / "lingam" / "lingam"


# Compatibility shims
def _patch_pandas_compat() -> None:
    """Re-add ``pd.Index.is_integer`` (removed in pandas 2.1) for the vendored
    DynamicDataTransformer."""
    if not hasattr(pd.Index, "is_integer"):
        pd.Index.is_integer = lambda self: pd.api.types.is_integer_dtype(self)


# Package stand-ins
def _register_namespace(name: str, path: Path) -> None:
    """Register a stub package in ``sys.modules`` without running its __init__."""
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
    # dynotears.py imports StructureModel from causalnex.structure, so the
    # name must exist on the stub before that import runs.
    # source code available at: https://github.com/quantumblacklabs/causalnex
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
# source code available at: https://github.com/quantumblacklabs/causalnex
from causalnex.structure.dynotears import (  # noqa: E402
    from_numpy_dynamic,
    from_pandas_dynamic,
)
# source code available at: https://github.com/quantumblacklabs/causalnex
from causalnex.structure.structuremodel import StructureModel  # noqa: E402
# source code available at: https://github.com/cdt15/lingam
from lingam.var_lingam import VARBootstrapResult, VARLiNGAM  # noqa: E402

__all__ = [
    "from_pandas_dynamic",
    "from_numpy_dynamic",
    "StructureModel",
    "VARLiNGAM",
    "VARBootstrapResult",
    "THESIS_ROOT",
]
