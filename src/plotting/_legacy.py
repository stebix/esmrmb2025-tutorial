import numpy as np
import matplotlib.pyplot as plt

from typing import Any
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import minimize
from numpy.typing import ArrayLike



class MonotonicCurveEditor:

    global_y_max: float = 180.0
    global_y_min: float = 0.0

    global_y_axlim: tuple[float, float] = (-10, 190)
        
    def __init__(
        self,
        initial_curve_x: ArrayLike,
        initial_curve_y: ArrayLike,
        n_control_points: int = 5,
        figsize: tuple[float, float] | None = None
    ) -> None:
        """
        Initialize the interactive curve editor.
        
        Parameters:
        -----------
        initial_curve_x : array-like
            X coordinates of the initial smooth curve
        initial_curve_y : array-like
            Y coordinates of the initial smooth curve
        n_control_points : int
            Number of control points to fit to the initial curve
        """
        self.initial_x = np.array(initial_curve_x)
        self.initial_y = np.array(initial_curve_y)
        self.n_control_points = n_control_points

        # Store x-range for constraints
        self.x_min = np.min(self.initial_x)
        self.x_max = np.max(self.initial_x)

        self.initial_point_count = len(self.initial_x)
        
        # Fit control points to initial curve
        self.control_points = self._fit_control_points()

        plot_kwargs: dict[str, Any] = {}
        if figsize is not None:
            plot_kwargs['figsize'] = figsize
        
        # Setup the plot
        self.fig, self.ax = plt.subplots(**plot_kwargs)
        self.setup_plot()
        
        # Interaction state
        self.dragging_point = None
        self.epsilon = 10  # pixels for click detection


    def _fit_control_points(self):
        """Fit control points to the initial curve with monotonic x constraint."""
        # Start with evenly spaced x-coordinates
        x_control = np.linspace(self.x_min, self.x_max, self.n_control_points)
        
        # Interpolate y-values at these x positions
        if np.all(np.diff(self.initial_x) > 0):  # If initial x is monotonic
            interp = interp1d(self.initial_x, self.initial_y, kind='cubic', 
                              fill_value='extrapolate')
            y_control = interp(x_control)
        else:
            # If not monotonic, use indices
            indices = np.linspace(0, len(self.initial_x)-1, self.n_control_points).astype(int)
            x_control = self.initial_x[indices]
            y_control = self.initial_y[indices]
            # Sort by x
            sort_idx = np.argsort(x_control)
            x_control = x_control[sort_idx]
            y_control = y_control[sort_idx]
        
        # Optimize only y-coordinates (x stays fixed initially)
        def objective(y_params):
            """Minimize difference between spline and original curve."""
            try:
                # Create cubic spline with fixed x-coordinates
                cs = CubicSpline(x_control, y_params, bc_type='natural')
                # Evaluate at original x positions
                if np.all(np.diff(self.initial_x) > 0):
                    spline_y = cs(self.initial_x)
                    error = np.sum((spline_y - self.initial_y)**2)
                else:
                    # Sample at regular intervals
                    x_eval = np.linspace(self.x_min, self.x_max, len(self.initial_x))
                    spline_y = cs(x_eval)
                    # Interpolate original curve to same x points
                    orig_interp = interp1d(self.initial_x, self.initial_y, 
                                           fill_value='extrapolate', bounds_error=False)
                    orig_y = orig_interp(x_eval)
                    error = np.sum((spline_y - orig_y)**2)
                return error
            except:
                return 1e10
        
        result = minimize(objective, y_control, method='BFGS')
        optimal_y = result.x
        
        return np.column_stack([x_control, optimal_y])
    

    def setup_plot(self):
        """Setup the interactive plot."""
        # Plot the spline curve
        self.update_spline()
        
        # Plot control points
        self.control_scatter = self.ax.scatter(
            self.control_points[:, 0], 
            self.control_points[:, 1],
            c='red', s=100, zorder=5, picker=True
        )
        
        # Plot initial curve for reference
        self.ax.plot(self.initial_x, self.initial_y, 'k--', alpha=0.3, label='Original')
        
        # Add vertical lines to show x-constraints
        for x in self.control_points[:, 0]:
            self.ax.axvline(x, color='gray', alpha=0.2, linestyle=':')
        
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        self.ax.set_title('Drag control points vertically (x-coordinates fixed for monotonicity)')
        
        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
    def update_spline(self):
        """Update the spline curve based on current control points."""
        if hasattr(self, 'spline_line'):
            self.spline_line.remove()
        
        # Create cubic spline (guaranteed to be a function since x is monotonic)
        cs = CubicSpline(self.control_points[:, 0], self.control_points[:, 1], 
                        bc_type='natural')
        
        # Evaluate spline
        x_fine = np.linspace(self.x_min, self.x_max, self.initial_point_count)
        y_fine = np.clip(
            cs(x_fine), self.global_y_min, self.global_y_max
        )

        self.spline_line, = self.ax.plot(x_fine, y_fine, 'b-', linewidth=2, label='Spline')
        self.current_spline = (x_fine, y_fine)
        self.spline_function = cs
    

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
        """Handle mouse motion event - only allow vertical movement."""
        if self.dragging_point is None or event.inaxes != self.ax:
            return
        
        # Update only y-coordinate (keep x fixed for monotonicity)
        self.control_points[self.dragging_point, 1] = event.ydata
        
        # Update plot
        self.control_scatter.set_offsets(self.control_points)
        self.update_spline()
        self.fig.canvas.draw_idle()
    
    def get_current_curve(self):
        """Return the current spline curve coordinates."""
        return self.current_spline
    
    def get_control_points(self):
        """Return the current control points."""
        return self.control_points.copy()
    
    def evaluate(self, x):
        """Evaluate the spline function at given x values."""
        return self.spline_function(x)


# Alternative: Allow horizontal dragging with automatic reordering
class FlexibleMonotonicCurveEditor(MonotonicCurveEditor):

    def on_motion(self, event):
        """Handle mouse motion - allow both x and y movement but maintain order."""
        if self.dragging_point is None or event.inaxes != self.ax:
            return
        
        new_x = event.xdata
        new_y = event.ydata
        
        # Constrain x to maintain monotonicity
        if self.dragging_point > 0:
            min_x = self.control_points[self.dragging_point - 1, 0] + 0.01
        else:
            min_x = self.x_min
            
        if self.dragging_point < len(self.control_points) - 1:
            max_x = self.control_points[self.dragging_point + 1, 0] - 0.01
        else:
            max_x = self.x_max
        
        # Clamp x to valid range
        new_x = np.clip(new_x, min_x, max_x)
        
        # Update control point
        self.control_points[self.dragging_point] = [new_x, new_y]
        
        # Update plot
        self.control_scatter.set_offsets(self.control_points)
        self.update_spline()
        
        # Redraw vertical constraint lines
        for line in self.ax.lines[2:]:  # Skip spline and original
            if line.get_linestyle() == ':':
                line.remove()
        for x in self.control_points[:, 0]:
            self.ax.axvline(x, color='gray', alpha=0.2, linestyle=':')
        
        self.fig.canvas.draw_idle()
