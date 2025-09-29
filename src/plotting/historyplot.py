import plotly.graph_objects as go

from numpy.typing import ArrayLike


class HistoryPlot:
    color: str = 'blue'
    width: int = 2

    def __init__(
        self,
        max_TTL: int = 5,
        height: int = 400,
        width: int = 1200,
        showlegend: bool = True,
        xlabel: str = 'TR index',
        ylabel: str = 'Flip Angle (degrees)',
        ) -> None:
        self.fig = go.FigureWidget()
        self.max_TTL: int = max_TTL

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
            opacity=self.exponential_ttl_to_opacity(0)
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
            trace.opacity = self.exponential_ttl_to_opacity(ttl)

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

    def exponential_ttl_to_opacity(self, ttl: int) -> float:
        base: float = 0.95
        return base ** ttl

    def linear_ttl_to_opacity(self, ttl: int) -> float:
        if ttl >= self.max_TTL:
            return 0.1
        else:
            return 1.0 - (ttl / self.max_TTL) * 0.9
