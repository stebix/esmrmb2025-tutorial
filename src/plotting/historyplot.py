import json
import plotly.graph_objects as go
import ipywidgets as wgt

from numpy.typing import ArrayLike
from typing import Sequence, Literal
from pathlib import Path

from tools import OptimizationPackage


class OpacityScaler:
    def __init__(
        self,
        max_rank: int | None
        ):
        self.max_rank: int | None = max_rank

    def __call__(self, rank: int) -> float:
        raise NotImplementedError
    

class LinearOpacityScaler(OpacityScaler):
    def __call__(self, rank: int) -> float:
        if rank >= self.max_rank:
            return 0.1
        else:
            return 1.0 - (rank / self.max_rank) * 0.9

class ExponentialOpacityScaler(OpacityScaler):
    def __init__(
        self,
        base: float = 0.95,
        *args,
        **kwargs
    ) -> None:
        super().__init__(max_rank=None)
        self.base = base

    def __call__(self, rank: int) -> float:
        return self.base ** rank



class HistoryPlot:
    color: str = 'blue'
    width: int = 2

    def __init__(
        self,
        max_TTL: int = 5,
        opacity_scaling: Literal['linear', 'exponential'] = 'exponential',
        height: int = 400,
        width: int = 1200,
        showlegend: bool = True,
        xlabel: str = 'TR index',
        ylabel: str = 'Flip Angle (degrees)',
        ) -> None:

        self.fig = go.FigureWidget()
        self.max_TTL: int = max_TTL
        self._opacity_scaling: Literal['linear', 'exponential'] = opacity_scaling
        self._opacity_scaler = self._get_opacity_scaler()

        self.fig.update_layout(
            height=height,
            width=width,
            showlegend=showlegend,
            xaxis=dict(title=xlabel),
            yaxis=dict(title=ylabel)
        )

    def add_trace(self, y: ArrayLike) -> None:
        self._increment_TTL()
        self._prune_traces()
        self._update_opacities()
        meta: dict[str, int] = {'TTL' : 0, 'is_immortal': False}
        trace = go.Scatter(
            y=y,
            mode='lines',
            meta=meta,
            line=dict(
                width=self.width,
                color=self.color
            ),
            opacity=self._opacity_scaler(0)
        )
        self.fig.add_trace(trace)
        self._update_names()

    def add_immortal_trace(
        self,
        y: ArrayLike,
        name: str = 'immortal',
        width: int = 2,
        color: str = 'red',
        dash: str = 'dot',
        opacity: float = 1.0
        ) -> None:
        meta: dict[str, int] = {'TTL' : -999, 'is_immortal': True}
        trace = go.Scatter(
            y=y,
            mode='lines',
            meta=meta,
            name=name,
            line=dict(
                width=width,
                color=color,
                dash=dash
            ),
            opacity=opacity
        )
        self.fig.add_trace(trace)

    def _increment_TTL(self) -> None:
        for trace in self.fig.data:
            if trace.meta['is_immortal']:
                continue
            ttl = trace.meta['TTL']
            trace.meta['TTL'] = ttl + 1

    def _update_opacities(self) -> None:
        for trace in self.fig.data:
            if trace.meta['is_immortal']:
                continue
            ttl = trace.meta['TTL']
            trace.opacity = self._opacity_scaler(ttl)

    def _update_names(self) -> None:
        for trace in self.fig.data:
            if trace.meta['is_immortal']:
                continue
            ttl = trace.meta['TTL']
            if ttl == 0:
                trace.name = 'current'
            else:
                trace.name = f'iter - {ttl}'

    def _prune_traces(self) -> None:
        self.fig.data = [
            trace for trace in self.fig.data if trace.meta['TTL'] <= self.max_TTL or trace.meta['is_immortal']
        ]

    @property
    def opacity_scaling(self) -> Literal['linear', 'exponential']:
        return self._opacity_scaling

    @opacity_scaling.setter
    def opacity_scaling(
        self,
        new_scaling: Literal['linear', 'exponential']
    ) -> None:
        self._opacity_scaling = new_scaling
        self._opacity_scaler = self._get_opacity_scaler()
        self._update_opacities()


    def _get_opacity_scaler(self) -> OpacityScaler:
        mapping = {
            'linear' : LinearOpacityScaler,
            'exponential' : ExponentialOpacityScaler
        }
        return mapping[self._opacity_scaling](max_rank=self.max_TTL)



class SelectableHistoryPlot:
    initial_range: tuple[int, int] = (4, 5)

    def __init__(
        self,
        data: Sequence[ArrayLike],
        color: str = '#1e8ad8',
        linewidth: int = 2,
        opacity_scaling: Literal['linear', 'exponential'] = 'exponential',

    ) -> None:
        self.data = data
        self.fig = go.FigureWidget()

        center = len(data) // 2
        lr, rr = self.initial_range

        self.rangeselector = wgt.IntRangeSlider(
            value=[max(center-lr, 0), min(center+rr, len(data)-1)],
            min=0,
            max=len(data)-1,
            step=1,
            description='Select range:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
            continuous_updates=False,
        )

        self._color: str = color
        self._linewidth: int = linewidth
        self._opacity_scaling: Literal['linear', 'exponential'] = opacity_scaling
        self._opacity_scaler = self._get_opacity_scaler(opacity_scaling)
        self._init_figure()

    def _init_figure(self) -> None:
        self.fig.update_layout(
            height=400,
            width=1200,
            showlegend=False,
            xaxis=dict(title='TR index'),
            yaxis=dict(title='Flip Angle (degrees)')
        )
        self.rangeselector.observe(self._on_rangeselector_change, names='value')
        self._add_traces(self.data[slice(*self.rangeselector.value)])

    def set_data(self, data: Sequence[ArrayLike]) -> None:
        """
        Wipe old data and set new data (plot new data and repurpose plot).
        """
        self.data = data
        self.rangeselector.max = len(data) - 1
        center = len(data) // 2
        lr, rr = self.initial_range
        self.rangeselector.value = [max(center-lr, 0), min(center+rr, len(data)-1)]
        self._add_traces(self.data[slice(*self.rangeselector.value)])

    @property
    def color(self) -> str:
        return self._color

    @color.setter
    def color(self, new_color: str) -> None:
        self._color = new_color
        for trace in self.fig.data:
            trace.line.color = new_color

    def _add_traces(self, data: ArrayLike) -> None:
        # step through data in reverse order such that 'later' datum
        # has higher opacity
        for idx, datum in enumerate(reversed(data)):
            trace = go.Scatter(
                y=datum,
                mode='lines',
                line=dict(
                    width=self._linewidth,
                    color=self.color
                ),
                opacity=self._opacity_scaler(idx)
            )
            self.fig.add_trace(trace)

    def _on_rangeselector_change(self, change) -> None:
        slc = slice(*change['new'])
        with self.fig.batch_update():
            self.fig.data = []
            self._add_traces(self.data[slc])

    def _get_opacity_scaler(self, scaling: Literal['linear', 'exponential']) -> OpacityScaler:
        mapping = {
            'linear' : LinearOpacityScaler,
            'exponential' : ExponentialOpacityScaler
        }
        lower, upper = self._get_current_range()
        max_rank = upper - lower
        return mapping[scaling](max_rank=max_rank)
    
    def _get_current_range(self) -> tuple[int, int]:
        return self.rangeselector.value

    @property
    def opacity_scaling(self) -> Literal['linear', 'exponential']:
        return self._opacity_scaling
    
    @opacity_scaling.setter
    def opacity_scaling(self, new_scaling: Literal['linear', 'exponential']) -> None:
        self._opacity_scaling = new_scaling
        self._opacity_scaler = self._get_opacity_scaler(new_scaling)
        self._update_opacities()


    def _update_opacities(self) -> None:
        for rank, trace in enumerate(self.fig.data):
            trace.opacity = self._opacity_scaler(rank)     



class PrecomputedOptimizationPlot:
    options: list[dict[str, str]] = [
        {'label' : 'CRLB MC EPG',
         'value': 'crlb-mc-epg',
         'tooltip': 'Cramér-Rao Lower Bound for Multi-Compartment model using EPG signal model'},
        {'label' : 'CRLB SC EPG',
         'value': 'crlb-sc-epg',
         'tooltip': 'Cramér-Rao Lower Bound for Single-Compartment model using EPG signal model'},
        {'label' : 'Orthogonality EPG',
         'value': 'orth-epg',
         'tooltip': 'Orthogonality between signal vectors of relaxometric species for EPG signal model'},
    ]
    label2value: dict[str, str] = {option['label']: option['value'] for option in options}
    staticdir = Path('../src/static')

    def __init__(
        self,
        pkg: OptimizationPackage | None,
    ) -> None :
        # set up full UI state first since we deduce
        # some visualization settings from it
        self.runselector = wgt.ToggleButtons(
            options=[option['label'] for option in self.options],
            description='Precomputed optimization:',
            style={'description_width': 'initial'},
            tooltips=[option['tooltip'] for option in self.options],
        )
        self.runselector.observe(self._on_runselector_change, names='value')
        
        self.opacityselector = wgt.Dropdown(
            options=['linear', 'exponential'],
            value='linear',
            description='Opacity scaling:',
            style={'description_width': 'initial'}
        )
        self.opacityselector.observe(self._on_opacityselector_change, names='value')

        # if user did not provide package, load default from UI state
        self._pkg = self._init_pkg(pkg)
        self.plot: SelectableHistoryPlot = self._initialize_plot()

    def _init_pkg(self, candidate: OptimizationPackage | None) -> None:
        if candidate is not None:
            return candidate
        path = self._make_path(self.label2value[self.runselector.value])
        return self._load_pkg(path)

    def _on_runselector_change(self, change) -> None:
        label = change['new']
        value = self.label2value[label]
        path = self._make_path(value)
        pkg = self._load_pkg(path)
        self.set_pkg(pkg)

    def _on_opacityselector_change(self, change) -> None:
        new_opacity_scaling = change['new']
        self.plot.opacity_scaling = new_opacity_scaling
    
    def _make_path(self, option: str) -> Path:
        """Make path to static JSON file for given option."""
        filename = f'{option}-500.json'
        return self.staticdir / filename
    
    @staticmethod
    def _load_pkg(path: Path) -> OptimizationPackage:
        """Load optimization package from JSON file."""
        with open(path, 'r') as f:
            pkg_dict = json.load(f)

        return OptimizationPackage(**pkg_dict)

    def _initialize_plot(self) -> SelectableHistoryPlot:
        """Initialize the history plot with the data from the optimization package."""
        data = self._pkg.fa_history if self._pkg else []
        opacity_scaling = self.opacityselector.value
        plot = SelectableHistoryPlot(
            data=data,
            opacity_scaling=opacity_scaling
        )
        return plot

    def set_pkg(self, pkg: OptimizationPackage):
        """Set new optimization package and fully reinitialize the plot."""
        self._pkg = pkg
        self.plot.set_data(pkg.fa_history) 