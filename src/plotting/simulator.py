from numbers import Real

import numpy as np
import ipywidgets as wgt

from numpy.typing import ArrayLike, NDArray
from typing import Any, Protocol
from collections.abc import Callable

import attrs

from signalmodel_epg import calculate_signal_epg
from plotting.relaxometrics import RelaxometricSpecies
from plotting.sequence import SequenceParameters


class CurveValues(Protocol):
    x: NDArray[np.floating]
    y: NDArray[np.floating]

    def copy(self) -> 'CurveValues':
        ...


class SequenceDataProvider(Protocol):
    def get_current_curve(self) -> CurveValues:
        ...

class Point(Protocol):
    """
    Compound relaxometric data from 2D canvas: T1 (x), T2 (y) coordinates.
    """
    x: Real
    y: Real


class RelaxometricDataProvider(Protocol):
    def get_points(self) -> list[Point]:
        ...


@attrs.define
class SimulatedSpecies:
    """
    Container for simulated relaxometric species data.
    """
    species: RelaxometricSpecies
    signal: NDArray[np.inexact]


@attrs.define
class SimulationOutputPackage:
    """
    Compound data encapsulating the results of a simulation run
    together with the input parameters and curves used.
    """
    result: list[SimulatedSpecies]
    sequence_parameters: SequenceParameters
    fa_curve: CurveValues
    tr_curve: CurveValues



def make_run_button(
    button_kwargs: dict | None = None,
    layout_kwargs: dict | None = None,
    style_kwargs: dict | None = None,
) -> wgt.Button:
    button_defaults = {
        'description': 'Run simulation',
        'tooltip': 'Press to run the simulation witht the current settings.',
        'button_style': '',  # 'success', 'info', 'warning', 'danger' or ''
        'icon': 'microchip',  # (FontAwesome names without the fa prefix)
    }
    layout_defaults = {
        'width': '250px',
        'height': '65px',
        'font_size': '20px',
        'font_weight': 'bold',
        'border': '2px solid #4CAF50',
        'border_radius': '1px'
    }
    style_defaults = {
        'font_size': '20px',
        'font_weight': 'bold'
    }
    button_kwargs = button_defaults | (button_kwargs or {})
    layout_kwargs = layout_defaults | (layout_kwargs or {})
    style_kwargs = style_defaults | (style_kwargs or {})
    layout = wgt.Layout(**layout_kwargs)
    style = wgt.ButtonStyle(**style_kwargs)
    button = wgt.Button(**button_kwargs, layout=layout, style=style)
    return button


def make_progress_bar(**kwargs) -> wgt.IntProgress:
    defaults = {
        'description' : 'Species Progress',
        'value' : 0,
        'min' : 0,
        'max' : 10,
        'bar_style' : '', # 'success', 'info', 'warning', 'danger' or ''
        'style' : {'bar_color': 'maroon', 'description_width': 'initial'},
        'orientation' : 'horizontal'
    }
    kwargs = defaults | kwargs
    return wgt.IntProgress(**kwargs)



class SimulationControllerDashboard:

    def __init__(
        self,
        run_button: wgt.Button | None = None,
        progressbar_kwargs: dict | None = None,
        ) -> None:

        self.run_button = run_button or make_run_button()

        self.progressbar_text = wgt.HTML(
            value='<b>Progress:</b> 0 / 0 species',
            placeholder='',
            description=''
        )
        self.progressbar = make_progress_bar(**(progressbar_kwargs or {}))
        # make composite with button height to align nicely
        composite = wgt.VBox(
            [self.progressbar_text, self.progressbar],
            layout=wgt.Layout(height=self.run_button.layout.height)
        )
        self.ui: wgt.HBox = wgt.HBox(
            [self.run_button, composite],
        )

    def setup_progressbar(self, max_value: int) -> None:
        """Setup progress bar for requested simulation run."""
        self.progressbar.max = max_value
        self.progressbar.value = 0
        self.progressbar_text.value = f'<b>Progress:</b> 0 / {max_value} species'

    def update_progressbar(self) -> None:
        """Increment progress bar."""
        self.progressbar.value += 1
        self.progressbar_text.value = (
            f'<b>Progress:</b> {self.progressbar.value} / {self.progressbar.max} species'
        )




class SimulationController:

    def __init__(
        self,
        dashboard: SimulationControllerDashboard,
        fa_provider: SequenceDataProvider,
        tr_provider: SequenceDataProvider,
        relax_provider: RelaxometricDataProvider,
        parameters: SequenceParameters,
    ) -> None:
        
        self.dashboard = dashboard
        self.fa_provider: SequenceDataProvider = fa_provider
        self.tr_provider: SequenceDataProvider = tr_provider
        self.relax_provider: RelaxometricDataProvider = relax_provider
        self.parameters: SequenceParameters = parameters

        self._signals: NDArray[np.inexact] = np.empty(
            self.parameters.shots, dtype=np.complex64
        )
        self._simulation_package: SimulationOutputPackage | None = None
        self._connect_run_button()

        self._on_run_complete_callbacks: list[Callable[[SimulationOutputPackage], Any]] = []


    @property
    def signals(self) -> NDArray[np.inexact]:
        return self._signals
    

    def _connect_run_button(self) -> None:
        """Connect run button to run method."""
        self.dashboard.run_button.on_click(lambda btn: self.run())

    def add_on_run_complete_callback(
        self,
        callback: Callable[[SimulationOutputPackage], Any]
    ) -> None:
        """Register a callback to be called when a simulation run completes."""
        self._on_run_complete_callbacks.append(callback)

    def run(self) -> None:
        """
        Perform simulation with current parameters and data provider values.
        """
        fa_values = self.fa_provider.get_current_curve().y
        tr_values = self.tr_provider.get_current_curve().y
        points = self.relax_provider.get_points()
        t1_values = np.array([p.x for p in points], dtype=np.float32)
        t2_values = np.array([p.y for p in points], dtype=np.float32)
        # simplification: constant M0 == 1.0 for all species
        M0_values = np.full_like(t1_values, self.parameters.M0, dtype=np.float32)
        self.dashboard.setup_progressbar(len(points))

        if not len(fa_values) == len(tr_values) == self.parameters.shots:
            msg: str = (
                f'FA and TR curve points must match number of shots. Got '
                f'{len(fa_values)} FA points, {len(tr_values)} TR points, '
                f'and {self.parameters.shots} shots.'
            )
            raise ValueError(msg)
        
        simulated_species_cache: list[SimulatedSpecies] = []
        
        for idx, (M0, T1, T2) in enumerate(zip(M0_values, t1_values, t2_values)):
            kwargs = {
                'fa': fa_values,
                'tr': tr_values,
                'ph': self.parameters.ph,
                'beats': self.parameters.beats,
                'shots' : self.parameters.shots,
                'prep': self.parameters.prep,
                't2te': self.parameters.t2te,
                'ti': self.parameters.ti,
                'te': self.parameters.te,
                't1': T1,
                't2': T2,
                'm0': M0
            }
            signal = calculate_signal_epg(**kwargs)
            # create output record
            simulated_species = SimulatedSpecies(
                species=RelaxometricSpecies.make_anonymous(M0=M0, T1=T1, T2=T2),
                signal=signal.numpy()
            )

            simulated_species_cache.append(simulated_species)
            self.dashboard.update_progressbar()


        simulation_package = SimulationOutputPackage(
            result=simulated_species_cache,
            sequence_parameters=self.parameters.copy(),
            fa_curve=self.fa_provider.get_current_curve().copy(),
            tr_curve=self.tr_provider.get_current_curve().copy()
        )
        self._simulation_package = simulation_package
        for callback in self._on_run_complete_callbacks:
            callback(simulation_package)


    def fetch_simulation_package(self) -> SimulationOutputPackage:
        """Fetch the last completed simulation package, if available."""
        if self._simulation_package is None:
            msg: str = ('Cannot fetch simulation package: '
                        'no simulation has been run yet.')
            raise RuntimeError(msg)
        return self._simulation_package


