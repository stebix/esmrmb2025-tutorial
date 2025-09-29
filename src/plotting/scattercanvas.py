import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as wgt

from collections.abc import Mapping, Iterable
from numbers import Real
from typing import Any, NamedTuple

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection

from plotting.relaxometrics import relaxometric_species


class Point(NamedTuple):
    x: Real
    y: Real

    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Point') -> 'Point':
        return Point(self.x - other.x, self.y - other.y)
    
    def length(self) -> float:
        """Euclidean length of the vector from origin to this point"""
        return (self.x**2 + self.y**2)**0.5
    
    @classmethod
    def from_iterables(cls, xs: Iterable[Real], ys: Iterable[Real]) -> list['Point']:
        """Create a list of Points from two iterables of coordinates"""
        return [cls(x, y) for x, y in zip(xs, ys, strict=True)]


def get_figure_and_ax(
    figure_candidate: Figure | None,
    ax_candidate: Axes | None,
    creation_kwargs: Mapping[str, Any] | None = None,
) -> tuple[Figure, Axes]:
    """
    Utility to get or create a figure and axis.
    If figure and axes are provided, they are used.
    If only one is given, the other is derived from it.
    If neither is given, a new figure and axes
    are created.
    """
    if figure_candidate is not None and ax_candidate is not None:
        return (figure_candidate, ax_candidate)
    elif figure_candidate is not None and ax_candidate is None:
        return (figure_candidate, figure_candidate.gca())
    elif figure_candidate is None and ax_candidate is not None:
        return (ax_candidate.get_figure(), ax_candidate)
    else:
        creation_kwargs = creation_kwargs or {}
        fig, ax = plt.subplots(**creation_kwargs)
        return (fig, ax)


def make_clear_button(**kwargs) -> wgt.Button:
    """
    Create a standard 'Clear' button for clearing points.
    """
    default_kwargs = {
        'description': 'Clear Points',
        'button_style': '',
        'tooltip': 'Clear all points from the canvas',
        'icon': 'rotate-left'
    }
    default_kwargs.update(kwargs)
    return wgt.Button(**default_kwargs)


class InteractiveScatterCanvasDashboard:
    """
    Dashboard container for interactive scatter canvas.
    """
    def __init__(
        self,
        clear_button_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Initialize the dashboard with buttons and controls.
        """
        self.clear_button = make_clear_button(**(clear_button_kwargs or {}))
        self.ui = wgt.HBox([self.clear_button])


class InteractiveScatterCanvas:
    """
    Interactive scatter canvas for user-based setting of points.
    """
    figsize: tuple[float, float] = (6.0, 5.0)
    grid: bool = True
    grid_alpha: float = 0.3
    point_color: str = 'red'
    point_size: int = 50

    def __init__(
        self,
        xrange: tuple[float, float],
        yrange: tuple[float, float],
        xlabel: str = '',
        ylabel: str = '',
        title: str = '',
        fig: Figure | None = None,
        ax: Axes | None = None,
        fig_creation_kwargs: Mapping[str, Any] | None = None,
        ) -> None:

        self.fig: Figure
        self.ax: Axes
        # actual value is set in `setup_canvas`
        self.scatter: PathCollection
        
        self.points: list[Point] = []
        creation_kwargs = self.make_creation_defaults() | (fig_creation_kwargs or {})
        self.fig, self.ax = get_figure_and_ax(fig, ax, creation_kwargs)
        self.setup_canvas(
            xrange, yrange, xlabel, ylabel, title
        )
        self.connect_events()

    
    def make_creation_defaults(self) -> dict[str, Any]:
        """
        Default kwargs for figure creation.
        """
        return {
            'figsize': self.figsize,
        }


    def setup_canvas(
        self,
        xrange: tuple[float, float],
        yrange: tuple[float, float],
        xlabel: str,
        ylabel: str,
        title: str,
        ) -> None:
        """Set up the canvas with axes, labels, and grid"""
        self.ax.set_xlim(*xrange)
        self.ax.set_ylim(*yrange)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title)
        self.ax.grid(self.grid, alpha=self.grid_alpha)
        self.scatter = self.ax.scatter(
            [], [],
            c=self.point_color,
            s=self.point_size,
            picker=True
        )


    def connect_events(self):
        """Connect matplotlib events to handlers"""
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)


    def on_click(self, event):
        """Handle mouse click events"""
        if event.inaxes != self.ax:
            return
        
        point = Point(x=event.xdata, y=event.ydata)
        
        if event.button == 1:  # Left click - add point
            self.add_point(point)
        elif event.button == 3:  # Right click - remove nearest point
            self.remove_nearest_point(point)


    def add_point(self, point: Point) -> None:
        """Add a new point to the canvas"""
        self.points.append(point)
        self.update_plot()
    
    def remove_nearest_point(self, point: Point) -> Point | None:
        """Remove the point nearest to the click location"""
        if not self.points:
            return None
        
        # Find the nearest point
        distances = [
            (point - other).length()
            for other in self.points
        ]
        nearest_index = distances.index(min(distances))
        removed_point = self.points.pop(nearest_index)
        self.update_plot()
        return removed_point


    def update_plot(self) -> None:
        """Update the scatter plot with current points"""
        if self.points:
            x_coords, y_coords = zip(*self.points)
            self.scatter.set_offsets(np.column_stack([x_coords, y_coords]))
        else:
            self.scatter.set_offsets(np.empty((0, 2)))
        
        self.fig.canvas.draw_idle()
    
    def get_points(self) -> list[Point]:
        """Return the current list of points"""
        return self.points.copy()
    
    def clear_all_points(self) -> None:
        """Clear all points from the canvas"""
        self.points.clear()
        self.update_plot()
    

    def import_points(
        self,
        *,
        points: Iterable[Point] | None = None,
        coordinates: Iterable[tuple[float, float]] | None = None,
        xcoords: Iterable[Real] | None = None,
        ycoords: Iterable[Real] | None = None,
        ) -> None:
        """
        Import points programmatically from various formats.
        Only one of the points-specifying parameters should be provided:
            - points : Iterable of already constructed `Point` instances
            - coordinates : Iterable of (x, y) tuples
            - (xcoords, ycoords) : Two iterables of x and y coordinates
        """
        if points is not None:
            self.points.extend(points)
        elif coordinates is not None:
            self.points.extend(Point(x, y) for x, y in coordinates)
        elif xcoords is not None and ycoords is not None:
            points = Point.from_iterables(xcoords, ycoords)
            self.points.extend(points)
        else:
            msg: str = ('At least one of \'points\', \'coordinates\', '
                        'or jointly \'xcoords\' and \'ycoords\' must be provided.')
            raise ValueError(msg)
        self.update_plot()


class InteractiveRelaxometricParameterCanvas(InteractiveScatterCanvas):
    """
    Scatter canvas with sane presets for (T1, T2) relaxometric parameters.
    """
    def __init__(
        self,
        xrange: tuple[float, float] = (25, 5000),
        yrange: tuple[float, float] = (15, 3500),
        xlabel: str = 'T1 (ms)',
        ylabel: str = 'T2 (ms)',
        title: str = 'Relaxometric Parameter Canvas',
        fig: Figure | None = None,
        ax: Axes | None = None,
        fig_creation_kwargs: dict | None = None):
        super().__init__(xrange, yrange, xlabel, ylabel, title, fig, ax, fig_creation_kwargs)

    @classmethod
    def prepopulated(
        cls,
        species: set[str] | None = None,
        ax: Axes | None = None,
        fig: Figure | None = None,
    ) -> 'InteractiveRelaxometricParameterCanvas':
        """
        Create a canvas prepopulated with standard relaxometric species.
        If `species` is provided, only those species are added.
        """
        canvas = cls(ax=ax, fig=fig)
        if species is None:
            selected_species = relaxometric_species
        else:
            # make user-requested species lowercase to match case-insensitive
            species = {s.lower() for s in species}
            selected_species = {
                spec for spec in relaxometric_species
                if spec.name in species
            }
        
        canvas.import_points(
            points=[
                Point(x=spec.T1, y=spec.T2)
                for spec in selected_species
            ]
        )
        return canvas