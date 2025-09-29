"""
Tooling to interactively edit spline curves inside a matplotlib figure.

@author: jsteb 2025
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as wgt

from copy import deepcopy
from attrs import define
from typing import Any, NamedTuple
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import minimize
from numpy.typing import ArrayLike, NDArray

from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import Formatter, ScalarFormatter

from .splinetools import SplineSettingsDashboard, compute_spline

def is_monotonic(x: ArrayLike) -> bool:
    """Check if array is strictly monotonic increasing"""
    x = np.asarray(x)
    return np.all(np.diff(x) > 0)


class CurveValues(NamedTuple):
    x: ArrayLike
    y: ArrayLike

    def copy(self) -> 'CurveValues':
        return deepcopy(self)


@define
class Labels:
    title: str
    original_curve: str
    spline_curve: str
    control_points: str
    x_axis: str
    y_axis: str

    @classmethod
    def from_defaults(cls) -> 'Labels':
        return cls(
            title='Interactive Monotonic Curve Editor',
            original_curve='Original',
            spline_curve='Spline',
            control_points='Control Points',
            x_axis='X',
            y_axis='Y'
        )


class MonotonicCurveEditor:
    global_y_axlim: tuple[float, float] = (-10, 190)
        
    def __init__(
        self,
        initial_curve_x: ArrayLike,
        initial_curve_y: ArrayLike,
        dashboard: SplineSettingsDashboard,
        labels: Labels | None = None,
        fig: Figure | None = None,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        formatter: Formatter | None = None,
        initial_yaxis_range: tuple[float, float] | None = None,
    ) -> None:
        """
        Initialize the interactive curve editor.
        
        Parameters:
        -----------
        initial_curve_x : array-like
            X coordinates of the initial smooth curve
        initial_curve_y : array-like
            Y coordinates of the initial smooth curve
        dashboard : SplineSettingsDashboard
            Dashboard widget for spline settings
        labels : Labels, optional
            Labels for the plot (if None, defaults are used)
            See the Labels attrs class for details.
        fig : matplotlib.figure.Figure, optional
            Figure to use for plotting (if None, a new figure is created
            or deduced from ax)
        ax : matplotlib.axes.Axes, optional
            Axis to use for plotting (if None, a new axis is created
            or deduced from fig as the current active axis)
        figsize : tuple(float, float), optional
            Figure size if a new figure is created (ignored if preinstantiated
            fig or axes is provided)
        formatter : matplotlib.ticker.Formatter, optional
            Formatter for the y-axis (if None, a ScalarFormatter is used)
        initial_yaxis_range : tuple(float, float), optional
            Initial y-axis range (if None, the matplotlib autoscaling is used)

        Attributes
        ----------
        control_scatter : PathCollection
            Scatter plot of control points
        control_vertical_lines : list of Line2D
            Vertical lines indicating control point x-positions
        current_spline : CurveValues
            Current spline curve values as tuple of x and y arrays
        spline_function : CubicSpline
            Current spline function for evaluation
        spline_line : Line2D
            Line plot of the current spline curve
        initial_curve : Line2D
            Line plot of the initial curve for reference
        initial_x : array-like
            X coordinates of the initial curve
        initial_y : array-like
            Y coordinates of the initial curve
        dashboard : SplineSettingsDashboard
            Dashboard widget for spline settings
        x_min : float
            Minimum x value of the initial curve (for constraints)
        x_max : float
            Maximum x value of the initial curve (for constraints)
        initial_point_count : int
            Number of points in the initial curve (for spline evaluation)
        control_points : array-like, shape (n_control_points, 2)
            Current control points as array of shape (n_points, 2)
        fig : matplotlib.figure.Figure
            Figure for plotting
        ax : matplotlib.axes.Axes
            Axis for plotting
        dragging_point : int or None
            Index of the currently dragged control point (None if not dragging)
        epsilon : float
            Pixel distance threshold for detecting clicks on control points
        """
        if not is_monotonic(initial_curve_x):
            raise ValueError('initial_curve_x must be strictly monotonic increasing')
        
        self.control_scatter: PathCollection | None = None
        self.control_vertical_lines: list[Line2D] = []
        self.current_spline: CurveValues | None = None
        self.spline_function: CubicSpline | None = None
        self.spline_line: Line2D | None = None
        self.initial_curve: Line2D | None = None
        self.initial_x = np.array(initial_curve_x)
        self.initial_y = np.array(initial_curve_y)
        self.dashboard: SplineSettingsDashboard = dashboard
        self.labels = labels or Labels.from_defaults()

        if formatter is None:
            formatter = ScalarFormatter()
            formatter.set_scientific(False)
            formatter.set_useOffset(False)
        self.formatter: Formatter = formatter

        # Store x-range for constraints
        self.x_min: float = np.min(self.initial_x)
        self.x_max: float = np.max(self.initial_x)

        self.initial_point_count: int = len(self.initial_x)

        # Fit control points to initial curve
        self.control_points: NDArray = self._fit_control_points()

        plot_kwargs: dict[str, Any] = {}
        if figsize is not None:
            plot_kwargs['figsize'] = figsize
        
        # Setup the plot
        self.fig: Figure
        self.ax: Axes
        self.fig, self.ax = self.get_figure_and_axis(fig, ax, plot_kwargs)

        elements: list[wgt.Widget] = self.dashboard.elements()
        elements[0].observe(self.reset_control_points, names='value')

        for element in elements[1:-2]:
            element.observe(self.on_change, names='value')

        self.dashboard.autoscale_axis_button.on_click(self.autoscale_axis)
        self.dashboard.reset_axis_button.on_click(self.reset_control_points)

        self.setup_plot()
        self.connect_events()

        self.ax.set_ylim(initial_yaxis_range)
        
        # Interaction state
        self.dragging_point: int | None = None
        self.epsilon: float = 10  # pixels for click detection


    def get_figure_and_axis(
        self,
        fig_candidate: Figure | None,
        ax_candidate: Axes | None,
        plot_kwargs: dict[str, Any] | None = None
    ) -> tuple[Figure, Axes]:
        if fig_candidate is not None and ax_candidate is not None:
            # we get both elements specified
            fig = fig_candidate
            ax = ax_candidate
        elif ax_candidate is not None:
            # we only get an axis and get the figure from the reference
            fig = ax_candidate.get_figure()
            ax = ax_candidate
        elif fig_candidate is not None:
            # we only get a figure and take the current active axis
            msg: str = 'Figure specified without axis - using current active axis'
            warnings.warn(msg)
            ax = fig_candidate.gca()
            fig = fig_candidate
        else:
            # we get neither and create a new figure/axis pair
            plot_kwargs = plot_kwargs or {}
            fig, ax = plt.subplots(**plot_kwargs)
        return (fig, ax)
        

    def _fit_control_points(self) -> NDArray[np.floating]:
        """Fit control points to the initial curve with monotonic x constraint."""
        # Start with evenly spaced x-coordinates
        settings = self.dashboard.query_settings()

        # first get the control point x positions
        n_control_points = settings.num_control_points
        x_control = np.linspace(self.x_min, self.x_max, n_control_points)
        # y position initial estimate from cubic interpolation of data
        interpolator = interp1d(
            self.initial_x, self.initial_y, kind='cubic', 
            fill_value='extrapolate'
        )
        y_control = interpolator(x_control)
        
        # Optimize only y-coordinates (x stays fixed initially)
        def objective(y_params: NDArray) -> float:
            """Minimize difference between spline and original curve."""
            assert np.all(np.diff(self.initial_x) > 0), 'initial x must be monotonic'
            # Create cubic spline with fixed x-coordinates
            cs = CubicSpline(x_control, y_params, bc_type='natural')
            # Evaluate at original x positions
            spline_y = cs(self.initial_x)
            error = np.sum((spline_y - self.initial_y)**2)
            return error
        
        result = minimize(objective, y_control, method='BFGS')
        optimal_y = result.x

        return np.column_stack([x_control, optimal_y])
    

    def setup_plot(self):
        """Setup the interactive plot."""
        # Plot the spline curve
        self.update_spline()
        # Plot control points
        self.draw_control_points()
        # Plot initial curve for reference
        self.draw_initial_curve()
        # Add vertical lines to show x-constraints
        self.draw_control_points_vertical_lines()
        # Apply styling
        self.apply_canvas_styling()

    def apply_canvas_styling(self):
        self.ax.yaxis.set_major_formatter(self.formatter)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        self.ax.set_title(self.labels.title)

    def draw_control_points(self):
        """Draw (current) control points in the plot."""
        if self.control_scatter is not None:
            try:
                self.control_scatter.remove()
            except ValueError:
                msg: str = 'Control scatter removal failed - continuing anyway'
                warnings.warn(msg)
            self.control_scatter = None

        kwargs = {'c': 'red', 's': 100, 'zorder': 5, 'picker': True}
        self.control_scatter = self.ax.scatter(
            self.control_points[:, 0], 
            self.control_points[:, 1],
            **kwargs
        )

    def draw_control_points_vertical_lines(self):
        # destructively retrieve existing lines and remove them from the plot
        # list removal necessary to avoid bug: removed lines cannot be removed again
        while True:
            try:
                line = self.control_vertical_lines.pop()
                line.remove()
            except IndexError:
                break
        kwargs = {'color': 'gray', 'alpha': 0.2, 'linestyle': ':'}
        for x in self.control_points[:, 0]:
            line = self.ax.axvline(x, **kwargs)
            self.control_vertical_lines.append(line)


    def draw_initial_curve(self):
        """Draw the initial curve in the plot."""
        if self.initial_curve is not None:
            self.initial_curve.remove()
            self.initial_curve = None
        kwargs = {
            'ls': 'dashed',
            'color': 'black',
            'alpha': 0.3,
            'label': self.labels.original_curve
        }
        line = self.ax.plot(self.initial_x, self.initial_y, **kwargs)[0]
        self.initial_curve = line


    def reset_control_points(self, change: Any = None):
        """Reset control points to initial fit."""
        self.control_points = self._fit_control_points()
        self.draw_control_points()
        self.draw_control_points_vertical_lines()
        self.update_spline()
        self.autoscale_axis()


    def connect_events(self):
        """Setup the interactive plot."""
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)


    def update_spline_backend(self) -> None:
        """
        Update the spline curve (values only, not the plot).
        """
        settings = self.dashboard.query_settings()
        xfine, yfine, splinefunc = compute_spline(
            x=self.control_points[:, 0],
            y=self.control_points[:, 1],
            settings=settings,
            point_count=self.initial_point_count
        )
        self.current_spline = CurveValues(x=xfine, y=yfine)
        self.spline_function = splinefunc

    
    def update_spline_plot(self):
        """Update the spline curve in the plot based on current values."""
        if self.spline_line is not None:
            self.spline_line.remove()

        kwargs = {
            'linewidth': 2,
            'label': self.labels.spline_curve,
            'color': 'blue'
        }
        line = self.ax.plot(
            self.current_spline.x,
            self.current_spline.y, **kwargs)[0]
        self.spline_line = line


    def update_spline(self):
        """
        Full joint update of the spline curve values (backend)
        and the plot (frontend) based on current control points.
        """
        self.update_spline_backend()
        self.update_spline_plot()

    
    def on_change(self, change):
        """Handle changes in the dashboard settings."""
        # Update the spline with new settings
        self.update_spline()
        self.fig.canvas.draw_idle()


    def on_press(self, event):
        """Handle mouse press event."""
        if event.inaxes != self.ax:
            return
        
        # Find if we clicked near a control point
        display_points = self.ax.transData.transform(self.control_points)
        display_click = self.ax.transData.transform([[event.xdata, event.ydata]])[0]
        display_distances = np.sqrt(np.sum((display_points - display_click)**2, axis=1))
        
        if np.min(display_distances) < self.epsilon:
            self.dragging_point = np.argmin(display_distances)
    

    def on_release(self, event):
        """Handle mouse release event."""
        self.dragging_point = None
    

    def on_motion(self, event):
        """Handle mouse motion - allow both x and y movement but maintain order."""
        if self.dragging_point is None or event.inaxes != self.ax:
            return
        
        new_x = event.xdata
        new_y = event.ydata

        # Constrain x at edge points to original x (only vertical movement)
        if self.dragging_point in {0, len(self.control_points) - 1}:
            new_x = self.control_points[self.dragging_point, 0]
        else:
            # constrain to adjacent x values to maintain monotonicity
            min_x = self.control_points[self.dragging_point - 1, 0] + 0.01
            max_x = self.control_points[self.dragging_point + 1, 0] - 0.01
            # Clamp x to valid range
            new_x = np.clip(new_x, min_x, max_x)
        
        # Update control point
        self.control_points[self.dragging_point] = [new_x, new_y]
        
        # Update plot
        self.control_scatter.set_offsets(self.control_points)
        self.update_spline()
        
        # Redraw vertical constraint lines
        self.draw_control_points_vertical_lines()
        self.fig.canvas.draw_idle()


    def autoscale_axis(self, *args, **kwargs):
        """Isolatedly autoscale the axis to fit the current data."""
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw_idle()


    def get_current_curve(self) -> CurveValues:
        """Return the current spline curve coordinates."""
        return self.current_spline
    

    def get_control_points(self):
        """Return the current control points."""
        return self.control_points.copy()
    

    def evaluate(self, x):
        """Evaluate the spline function at given x values."""
        return self.spline_function(x)

