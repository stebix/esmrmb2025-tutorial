import numpy as np

from attrs import define
from enum import Enum
from typing import Literal, NamedTuple
from scipy.interpolate import CubicSpline
from numpy.typing import ArrayLike, NDArray

import ipywidgets as wgt


class BoundaryMode(Enum):
    CLIP: str = 'clip'
    REFLECT: str = 'reflect'


@define
class SplineSettings:
    num_control_points: int
    global_y_min: float
    global_y_max: float
    lower_boundary_mode: BoundaryMode
    upper_boundary_mode: BoundaryMode



DEFAULTS: dict[str, dict] = {
    'num_control_points_slider_kwargs': {
        'value': 25,
        'min': 5,
        'max': 100,
        'step': 1,
        'description': 'Control Points:',
        'continuous_update': False,
        'tooltip': ('Number of control points for the '
                    'spline. Warning: Resets manually positioned control points!'),
        'style' : {'description_width': 'initial'}
    },
    'global_y_min_slider_kwargs': {
        'value': 0.0,
        'min': -90.0,
        'max': 90.0,
        'step': 1.0,
        'description': 'Global Y Min:',
        'continuous_update': False,
        'style' : {'description_width': 'initial'},
    },
    'global_y_max_slider_kwargs': {
        'value': 90.0,
        'min': -90.0,
        'max': 90.0,
        'step': 1.0,
        'description': 'Global Y Max:',
        'continuous_update': False,
        'style' : {'description_width': 'initial'},
    },
    'lower_boundary_mode_dropdown_kwargs': {
        'options': [(mode.value, mode) for mode in BoundaryMode],
        'value': BoundaryMode.CLIP,
        'description': 'Lower Boundary:',
        'style' : {'description_width': 'initial'},
    },
    'upper_boundary_mode_dropdown_kwargs': {
        'options': [(mode.value, mode) for mode in BoundaryMode],
        'value': BoundaryMode.CLIP,
        'description': 'Upper Boundary:',
        'style' : {'description_width': 'initial'},
    },
    'autscale_axis_button_kwargs': {
        'description': 'Autoscale Axis',
        'tooltip': 'Autoscale the axis to fit the current data',
        'button_style': '',  # 'success', 'info', 'warning', 'danger' or ''
        'icon': 'recycle'  # (FontAwesome names without the `fa-` prefix)
    },
    'reset_axis_button_kwargs': {
        'description': 'Reset Axis',
        'tooltip': 'Reset the axis to the original view',
        'button_style': '',  # 'success', 'info', 'warning', 'danger' or ''
        'icon': 'rotate-left'  # (FontAwesome names without the `fa-` prefix)
    },
}


def make_control_points_slider(**kwargs) -> wgt.IntSlider:
    params = DEFAULTS['num_control_points_slider_kwargs'].copy()
    params.update(kwargs)
    return wgt.IntSlider(**params)

def make_global_y_min_slider(**kwargs) -> wgt.FloatSlider:
    params = DEFAULTS['global_y_min_slider_kwargs'].copy()
    params.update(kwargs)
    return wgt.FloatSlider(**params)

def make_global_y_max_slider(**kwargs) -> wgt.FloatSlider:
    params = DEFAULTS['global_y_max_slider_kwargs'].copy()
    params.update(kwargs)
    return wgt.FloatSlider(**params)

def make_lower_boundary_mode_dropdown(**kwargs) -> wgt.Dropdown:
    params = DEFAULTS['lower_boundary_mode_dropdown_kwargs'].copy()
    params.update(kwargs)
    return wgt.Dropdown(**params)

def make_upper_boundary_mode_dropdown(**kwargs) -> wgt.Dropdown:
    params = DEFAULTS['upper_boundary_mode_dropdown_kwargs'].copy()
    params.update(kwargs)
    return wgt.Dropdown(**params)

def make_autoscale_axis_button(**kwargs) -> wgt.Button:
    params = DEFAULTS['autscale_axis_button_kwargs'].copy()
    params.update(kwargs)
    return wgt.Button(**params)

def make_reset_axis_button(**kwargs) -> wgt.Button:
    params = DEFAULTS['reset_axis_button_kwargs'].copy()
    params.update(kwargs)
    return wgt.Button(**params)

class SplineSettingsDashboard:
    def __init__(
        self,
        control_points_slider_kwargs: dict | None = None,
        global_y_min_slider_kwargs: dict | None = None,
        global_y_max_slider_kwargs: dict | None = None,
        lower_boundary_mode_dropdown_kwargs: dict | None = None,
        upper_boundary_mode_dropdown_kwargs: dict | None = None,
        autoscale_axis_button_kwargs: dict | None = None,
        reset_axis_button_kwargs: dict | None = None,
        ) -> None:
        self.num_control_points_slider = make_control_points_slider(
            **(control_points_slider_kwargs or {})
        )
        self.global_y_min_slider = make_global_y_min_slider(
            **(global_y_min_slider_kwargs or {})
        )
        self.global_y_max_slider = make_global_y_max_slider(
            **(global_y_max_slider_kwargs or {})
        )
        self.lower_boundary_mode_dropdown = make_lower_boundary_mode_dropdown(
            **(lower_boundary_mode_dropdown_kwargs or {})
        )
        self.upper_boundary_mode_dropdown = make_upper_boundary_mode_dropdown(
            **(upper_boundary_mode_dropdown_kwargs or {})
        )
        self.autoscale_axis_button = make_autoscale_axis_button(
            **(autoscale_axis_button_kwargs or {})
        )
        self.reset_axis_button = make_reset_axis_button(
            **(reset_axis_button_kwargs or {})
        )
        self.ui = self.make_ui()

    def make_ui(self) -> wgt.VBox:
        """
        Create the dashboard UI via VBox and HBox composites.
        Currently hardcoded (but efficient).
        """
        column_1 = wgt.VBox([
            self.num_control_points_slider,
            self.global_y_min_slider,
            self.global_y_max_slider,
        ])
        column_2 = wgt.VBox([
            self.lower_boundary_mode_dropdown,
            self.upper_boundary_mode_dropdown,
        ])
        column_3 = wgt.VBox([
            self.autoscale_axis_button,
            self.reset_axis_button,
        ])
        return wgt.HBox([column_1, column_2, column_3])

    def elements(self) -> list[wgt.Widget]:
        """
        Return all individual elements of the dashboard as a flat list.
        """
        return [
            self.num_control_points_slider,
            self.global_y_min_slider,
            self.global_y_max_slider,
            self.lower_boundary_mode_dropdown,
            self.upper_boundary_mode_dropdown,
            self.autoscale_axis_button,
            self.reset_axis_button,
        ]
    
    def query_settings(self) -> SplineSettings:
        """
        Query the current settings of the dashboard
        that are relevant for spline computation.
        """
        return SplineSettings(
            num_control_points=self.num_control_points_slider.value,
            global_y_min=self.global_y_min_slider.value,
            global_y_max=self.global_y_max_slider.value,
            lower_boundary_mode=self.lower_boundary_mode_dropdown.value,
            upper_boundary_mode=self.upper_boundary_mode_dropdown.value
        )
    
    @classmethod
    def from_TR_defaults(cls) -> 'SplineSettingsDashboard':
        control_point_slider_kwargs = {
            'value': 10,
            'min': 3,
            'max': 30,
        }
        global_y_min_slider_kwargs = {
            'value': 1.0,
            'min': 1.0,
            'max': 50.0,
            'description': 'TR lower bound',
        }
        global_y_max_slider_kwargs = {
            'value': 75.0,
            'min': 2.0,
            'max': 100.0,
            'description': 'TR upper bound',
        }
        return cls(
            control_points_slider_kwargs=control_point_slider_kwargs,
            global_y_min_slider_kwargs=global_y_min_slider_kwargs,
            global_y_max_slider_kwargs=global_y_max_slider_kwargs,
        )
    
    @classmethod
    def from_FA_defaults(cls) -> 'SplineSettingsDashboard':
        control_point_slider_kwargs = {
            'value': 25,
            'min': 5,
            'max': 100,
        }
        global_y_min_slider_kwargs = {
            'value': 5.0,
            'min': 0.0,
            'max': 180.0,
            'description': 'FA lower bound',
        }
        global_y_max_slider_kwargs = {
            'value': 120.0,
            'min': 2.0,
            'max': 180.0,
            'description': 'FA upper bound',
        }
        return cls(
            control_points_slider_kwargs=control_point_slider_kwargs,
            global_y_min_slider_kwargs=global_y_min_slider_kwargs,
            global_y_max_slider_kwargs=global_y_max_slider_kwargs,
        )

def reflect_lower(arr: NDArray, boundary: float) -> NDArray:
    """Reflect values at the given lower boundary"""
    arr = arr.copy()
    below_boundary = arr < boundary
    arr[below_boundary] = 2 * boundary - arr[below_boundary]
    return arr

def reflect_upper(arr: NDArray, boundary: float) -> NDArray:
    """Reflect values at the given upper boundary"""
    arr = arr.copy()
    above_boundary = arr > boundary
    arr[above_boundary] = 2 * boundary - arr[above_boundary]
    return arr

def reflect(
    arr: NDArray,
    boundary: float,
    where: Literal['lower', 'upper']
) -> NDArray:
    """Enforce boundary on arr by reflecting at lower or upper value."""
    if where == 'lower':
        return reflect_lower(arr, boundary)
    elif where == 'upper':
        return reflect_upper(arr, boundary)
    else:
        raise ValueError(f'Invalid value for "where": {where}. Must be "lower" or "upper".')


def process_upper_boundary(
    arr: ArrayLike,
    mode: BoundaryMode,
    boundary: float
) -> NDArray:
    """Enforce upper boundary on arr according to mode"""
    if mode == BoundaryMode.CLIP:
        arr = np.clip(arr, None, boundary)
    elif mode == BoundaryMode.REFLECT:
        arr = reflect_upper(arr, boundary)
    else:
        raise ValueError(f'Invalid mode: {mode}')
    return arr


def process_lower_boundary(
    arr: ArrayLike,
    mode: BoundaryMode,
    boundary: float
) -> NDArray:
    """Enforce lower boundary on arr according to mode"""
    if mode == BoundaryMode.CLIP:
        arr = np.clip(arr, boundary, None)
    elif mode == BoundaryMode.REFLECT:
        arr = reflect_lower(arr, boundary)
    else:
        raise ValueError(f'Invalid mode: {mode}')
    return arr



class SplineResult(NamedTuple):
    x_fine: NDArray
    y_fine: NDArray
    splinefunc: CubicSpline


def compute_spline(
    x: ArrayLike,
    y: ArrayLike,
    settings: SplineSettings,
    point_count: int
) -> NDArray:
    """
    Compute a cubic spline interpolation of the given points (x, y)
    with the compound settings provided.
    The spline function is evaluated at `point_count` evenly spaced points
    between min(x) and max(x) of the control points defined by the
    settings.
    """
    bc_type: str = 'natural'
    splinefunc = CubicSpline(x, y, bc_type=bc_type)
    x_fine = np.linspace(
        np.min(x), np.max(x),
        num=point_count)
    y_fine = splinefunc(x_fine)
    y_fine = process_lower_boundary(
        y_fine, settings.lower_boundary_mode, settings.global_y_min
    )
    y_fine = process_upper_boundary(
        y_fine, settings.upper_boundary_mode, settings.global_y_max
    )
    return SplineResult(x_fine, y_fine, splinefunc)
