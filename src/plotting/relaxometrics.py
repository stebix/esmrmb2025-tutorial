import attrs


@attrs.define(frozen=True)
class RelaxometricSpecies:
    """
    Compound data object for relaxometric species parameters.
    """
    name: str
    M0: float
    T1: float
    T2: float

    @classmethod
    def make_anonymous(cls, M0: float, T1: float, T2: float) -> 'RelaxometricSpecies':
        """
        Create an anonymous relaxometric species with default name.

        Parameters
        ----------
        M0 : float
            Equilibrium magnetization.
        T1 : float
            Longitudinal relaxation time (ms).
        T2 : float
            Transverse relaxation time (ms).

        Returns
        -------
        RelaxometricSpecies
            Instance of RelaxometricSpecies with default name.
        """
        return cls(name='anonymous', M0=M0, T1=T1, T2=T2)
    

# Standard relaxometric species templates based on literature values.
DEFAULT_SPECIES_TEMPLATES: list[dict[str, str | float]] = [
    {
        'name': 'gm',
        'M0': 0.95,   # Relative to CSF (normalized)
        'T1': 1193,   # ms
        'T2': 109     # ms
    },
    {
        'name': 'wm',
        'M0': 0.95,   # Relative to CSF (normalized)
        'T1': 781,    # ms
        'T2': 65      # ms
    },
    {
        'name': 'csf',
        'M0': 1.0,    # Reference (normalized)
        'T1': 4160,   # ms
        'T2': 2000    # ms (can vary 1500-2200)
    },
    {
        'name': 'blood',
        'M0': 0.87,   # Relative to CSF (normalized)
        'T1': 1932,   # ms (arterial blood)
        'T2': 275     # ms (oxygenated)
    },
    {
        'name': 'fat',
        'M0': 1.0,    # High signal (normalized)
        'T1': 253,    # ms
        'T2': 68      # ms
    },
    {
        'name': 'muscle',
        'M0': 0.9,    # Relative to CSF (normalized)
        'T1': 1120,   # ms
        'T2': 44      # ms
    },
    {
        'name': 'marrow',
        'M0': 0.95,   # Relative to CSF (normalized)
        'T1': 365,    # ms
        'T2': 125     # ms
    },
    {
        'name': 'meninges',
        'M0': 0.85,   # Relative to CSF (normalized)
        'T1': 1200,   # ms (approximate)
        'T2': 90      # ms (approximate)
    }
]

# set of standard relaxometric species for easy access
relaxometric_species: set[RelaxometricSpecies] = {
    RelaxometricSpecies(**spec)
    for spec in DEFAULT_SPECIES_TEMPLATES
}