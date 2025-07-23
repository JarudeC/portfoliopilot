# this file is made for modularization
"""Auto-switch to the modern NumPy / Pandas interpreter if needed."""

# pylance: reportMissingImports=false
import os, sys, importlib.util as _iu                            # noqa: E402
import pathlib as _pl                                            # noqa: E402

def _hop_into_np2():
    new_py = "/usr/local/bin/py311"          # ← updated location
    if sys.executable != new_py and _pl.Path(new_py).exists():
        os.execv(new_py, [new_py] + sys.argv)

spec = _iu.find_spec("numpy")
if spec:                          # running inside some Python already
    import numpy as _np           # noqa: E402
    if _np.__version__.startswith("1."):
        _hop_into_np2()           # re-exec with NumPy ≥ 2
