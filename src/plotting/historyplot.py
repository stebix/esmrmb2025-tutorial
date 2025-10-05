import plotly.graph_objects as go
import ipywidgets as wgt

from numpy.typing import ArrayLike
from typing import Sequence, Literal

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

        self.selector = wgt.IntRangeSlider(
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
        self.selector.observe(self._on_slider_change, names='value')
        self._add_traces(self.data[slice(*self.selector.value)])

    @property
    def color(self) -> str:
        return self._color

    @color.setter
    def color(self, new_color: str) -> None:
        self._color = new_color
        for trace in self.fig.data:
            trace.line.color = new_color

    def _add_traces(self, data: ArrayLike) -> None:
        for idx, datum in enumerate(data):
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

    def _on_slider_change(self, change) -> None:
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
        return self.selector.value

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

