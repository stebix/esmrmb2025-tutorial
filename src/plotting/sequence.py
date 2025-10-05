import attrs

from numpy.typing import ArrayLike
from copy import deepcopy


@attrs.define
class SequenceParameters:
    """
    Container for remaining sequence parameters needed for simulation.
    Other information (FA, TR, T1, T2) is provided by data providers.

    Attributes
    ----------
    ph : ArrayLike
        Phase values (in radians) for each shot.

    shots : int
        Number of shots in the sequence.

    prep : list[int]
        List of preparation pulse signifier for each beat.

    t2te : list[float]
        List of T2-preparation echo times (in ms) for each beat.

    ti : float
        Inversion time (in ms).

    te : float
        Echo time (in ms).
    """
    ph: ArrayLike
    shots: int
    prep: list[int]
    t2te: list[float]
    ti: float
    te: float
    beats: int = 1
    M0: float = 1.0

    def copy(self) -> 'SequenceParameters':
        """Create a deep copy of the parameters."""
        return deepcopy(self)