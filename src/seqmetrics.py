"""
Light tooling to evaluate designed MR sequences (i.e. flip angle trains and TR patterns)
with respect to SAR and sequence length.
"""
import numpy as np

from typing import Literal
from numpy.typing import ArrayLike

def compute_relative_sar(
    fa: ArrayLike,
    unit: Literal['grad', 'rad'] = 'rad'
) -> float:
    """Compute the relative SAR of a flip angle train.
    Parameters
    ----------
    fa : ArrayLike
        The flip angle train to compute the SAR for.
    unit : Literal['grad', 'rad'], optional
        The unit of the flip angles, by default 'rad'.

    Returns
    -------
    float
        The relative SAR of the flip angle train.
    """
    if unit not in {'rad', 'grad'}:
        raise ValueError(f'Invalid unit \'{unit}\'. Must be \'grad\' or \'rad\'.')
    fa = np.asarray(fa)
    if unit == 'rad':
        fa = np.rad2deg(fa)
    return np.sum((fa / 180.0)**2) / len(fa)


def compute_sar_ratio(
    fa: ArrayLike,
    reference_fa: ArrayLike,
    unit: Literal['grad', 'rad'] = 'rad'
) -> float:
    """Compute the SAR ratio of a flip angle train to a reference flip angle train.
    Parameters
    ----------
    fa : ArrayLike
        The flip angle train to compute the SAR for.
    reference_fa : ArrayLike
        The reference flip angle train to compute the SAR ratio against.
    unit : Literal['grad', 'rad'], optional
        The unit of the flip angles, by default 'rad'.

    Returns
    -------
    float
        The SAR ratio of the flip angle train to the reference flip angle train.
    """
    fa = np.asarray(fa)
    reference_fa = np.asarray(reference_fa)
    if len(fa) != len(reference_fa):
        msg = (f'Flip angle trains must have the same length, '
               f'but got {len(fa)} fa train and {len(reference_fa)}.')
        raise ValueError(msg)
    sar = compute_relative_sar(fa, unit=unit)
    reference_sar = compute_relative_sar(reference_fa, unit=unit)
    return sar / reference_sar


def compute_sequence_length(tr: ArrayLike) -> float:
    """
    Compute the total sequence length given a repetition time (TR) array.

    Parameters
    ----------
    tr : array-like
        An array of repetition times.

    Returns
    -------
    float
        The total sequence length.
    """
    tr = np.asarray(tr)
    total_length = np.sum(tr)
    return total_length


def compute_sequence_length_ratio(
    tr: ArrayLike,
    tr_reference: ArrayLike
) -> float:
    """
    Compute the ratio of the total sequence length to a reference sequence length.

    Parameters
    ----------
    tr : array-like
        An array of repetition times.
    tr_reference : array-like
        An array of reference repetition times.

    Returns
    -------
    float
        The ratio of the total sequence length to the reference sequence length.
    """
    tr = np.asarray(tr)
    tr_reference = np.asarray(tr_reference)
    total_length = np.sum(tr)
    total_length_reference = np.sum(tr_reference)
    return total_length / total_length_reference