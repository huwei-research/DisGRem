"""
disgrem.py – DisGrem (wrapper: no compression, Klazy=1)
Ported from MATLAB: DisGrem.m
"""

from __future__ import annotations
import numpy as np
from solvers.disgrem.ce_disgrem import ce_disgrem


def disgrem(x0: np.ndarray, prm: dict):
    """
    DisGrem = CeDisGrem with compression disabled and Klazy=1.
    """
    prm = dict(prm)
    prm["compressH"] = False
    prm["compressParam"] = None
    prm["compressor"] = "vector"
    prm["Klazy"] = 1
    prm["algName"] = "DisGrem"
    return ce_disgrem(x0, prm)
