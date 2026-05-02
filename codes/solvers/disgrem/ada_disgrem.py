"""
ada_disgrem.py - AdaDisGrem (wrapper: no compression, Klazy=1)
Ported from MATLAB: AdaDisGrem.m
"""

from __future__ import annotations
import numpy as np
from solvers.disgrem.ce_ada_disgrem import ce_ada_disgrem


def ada_disgrem(x0: np.ndarray, prm: dict):
    """
    AdaDisGrem = CeAdaDisGrem with compression disabled and Klazy=1.
    """
    prm = dict(prm)
    prm["compressH"] = False
    prm["compressParam"] = None
    prm["compressor"] = "vector"
    prm["Klazy"] = 1
    prm["algName"] = "AdaDisGrem"
    return ce_ada_disgrem(x0, prm)
