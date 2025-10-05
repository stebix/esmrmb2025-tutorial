"""
Tom Griesler, 05/2024
tomgr@umich.edu

Jannik Stebani, 10/2025

Added export functionality for optimization results.
"""
import torch
import numpy as np
import attrs

from collections.abc import Sequence, Callable
from typing import Any
from numpy.typing import ArrayLike, NDArray

from plotting.sequence import SequenceParameters

def to_tensor(x, dtype):
    return x if torch.is_tensor(x) else torch.tensor(x, dtype=dtype)



def to_list(
    arraylike: ArrayLike | Any,
    cast_fn: Callable | None = None
) -> list[float]:
    """
    Recursively convert an array-like structure into pure Python lists of floats.
    Propagates data types of numpy arrays.
    Typical usage for serialization or JSON export.
    """
    cast_fn = cast_fn or (lambda x: x)
    lst = []
    for item in arraylike:
        if isinstance(item, np.ndarray):
            lst.append(to_list(item, cast_fn=cast_fn))
        elif isinstance(item, (list, tuple, Sequence)):
            lst.append(to_list(item, cast_fn=None))
        else:
            cast_fn = float if isinstance(item, (np.floating, float)) else int
            lst.append(cast_fn(item))
    return lst



@attrs.define
class OptimizationPackage:
    """
    Compound data structure to hold all relevant information about an
    optimization run.

    Attributes should respect JSON-serializability for easy saving/loading.
    """
    costfunction: str
    T1: list[float] | float
    T2: list[float] | float
    M0: float
    beats: int
    shots: int
    tr: list[float]
    fa_initial: list[float]
    fa_history: list[list[float]]
    ph: list[float]
    prep: list[int]
    ti: list[float]
    t2te: list[float]
    te: float
    fa_min: float
    fa_max: float
    fa_maxdiff: float
    n_iter_max: int
    ratio: float | None = None
    weighting: Sequence[float] | None = None

    @classmethod
    def from_preassembled(
        cls,
        costfunction: str,
        T1: NDArray,
        T2: NDArray,
        M0: float,
        sequence_params: SequenceParameters,
        fa_initial: NDArray,
        fa_history: list[NDArray],
        tr: NDArray,
        n_iter_max: int,
        fa_min: float,
        fa_max: float,
        fa_maxdiff: float,
        ratio: float | None = None,
        weigthing: Sequence[float] | None = None,
        ) -> "OptimizationPackage":
        """
        Convenience constructor that deduces attributes from preassembled sequence
        parameters.
        """
        return cls(
            costfunction=costfunction,
            T1=to_list(T1) if isinstance(T1, Sequence) else float(T1),
            T2=to_list(T2) if isinstance(T2, Sequence) else float(T2),
            M0=float(M0),
            beats=int(sequence_params.beats),
            shots=int(sequence_params.shots),
            tr=to_list(tr),
            fa_initial=to_list(fa_initial),
            fa_history=to_list(fa_history),
            ph=to_list(sequence_params.ph),
            prep=to_list(sequence_params.prep),
            ti=to_list(sequence_params.ti),
            t2te=to_list(sequence_params.t2te),
            te=float(sequence_params.te),
            fa_min=float(fa_min),
            fa_max=float(fa_max),
            fa_maxdiff=float(fa_maxdiff),
            n_iter_max=int(n_iter_max),
            ratio=ratio,
            weighting=to_list(weigthing) if weigthing is not None else None,
        )
        
