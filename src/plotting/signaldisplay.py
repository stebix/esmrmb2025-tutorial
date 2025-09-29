import attrs
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import ipywidgets as wgt

from collections.abc import MutableMapping, Callable
from numpy.typing import NDArray

from .simulator import SimulationController, SimulationOutputPackage


@attrs.define
class LineTraceState:
    """State and styling of a single line trace."""
    visible: bool
    linestyle: str
    opacity: float


@attrs.define
class SignalDisplayState:
    """
    Encapsulates the necessary state to stylize
    display of complex signal vectors.
    """
    normalize: bool
    legend_visible: bool
    imag_state: LineTraceState
    real_state: LineTraceState
    abs_state: LineTraceState


def make_linestyle_selector(
    description: str = '',
    value: str = 'solid',
) -> wgt.Dropdown:
    """Conveniently create a dropdown for a plotly line style selection.
    """
    options = ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
    if value not in options:
        raise ValueError(f'Value {value} not in options {options}')
    return wgt.Dropdown(
        options=options,
        value=value,
        description=description,
        style={'description_width': 'initial'},
    )


def make_normalization_toggle_buttons(
    description: str = '',
) -> wgt.ToggleButtons:
    """Conveniently create toggle buttons for signal normalization toggling."""
    buttons = wgt.ToggleButtons(
        options=['Nonnormalized', 'Normalized'],
        description=description,
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltips=['Display raw signal vector',
                  'Display L2-normalized signal vector'],
        icons=['scale-unbalanced', 'scale-balanced'],
    )
    return buttons


def make_legend_checkbox(
    description: str = 'Show legend',
    value: bool = False
) -> wgt.Checkbox:
    """Conveniently create a checkbox for legend display toggling."""
    return wgt.Checkbox(
        value=value,
        description=description,
        style={'description_width': 'initial'},
    )



class RepresentationControlBox:
    """
    Controls for the representation of a single trace.

    In the usual usage of this class in this context, the trace
    represents either the real, imaginary, or absolute part of a
    complex signal vector.
    """
    def __init__(
        self,
        description_prefix: str = '',
        initial_visibility: bool = True,
        initial_linestyle: str = 'solid',
        initial_opacity: float = 1.0
    ) -> None:
        self.visibility_checkbox: wgt.Checkbox = wgt.Checkbox(
            value=initial_visibility,
            description='Show',
            style={'description_width': 'initial'},
        )
        self.linstyle_selector: wgt.Dropdown = make_linestyle_selector(
            'Linestyle:', value=initial_linestyle
        )
        self.opacity_slider: wgt.FloatSlider = wgt.FloatSlider(
            value=initial_opacity,
            min=0.0,
            max=1.0,
            step=0.1,
            description='Opacity:',
            readout_format='.1f',
            style={'description_width': 'initial'},
        )
        border_color = 'gray'
        title = wgt.HTML(
            value=f"<h3 style='margin: 0; padding: 10px; font-weight: bold; background-color: #f5f5f5; border-bottom: 1px solid {border_color};'>{description_prefix} Part</h3>"
        )
        content_box: wgt.VBox = wgt.VBox(
            [self.visibility_checkbox, self.linstyle_selector, self.opacity_slider],
            layout=wgt.Layout(
                padding='5px'
            )
        )
        self.ui = wgt.VBox(
            children=[title, content_box],
            layout=wgt.Layout(
                border='solid 1px gray',
                padding='5px',
                margin='2px'
            )
        )



class SignalDisplayDashboard:
    def __init__(
        self,
        normalization_buttons: wgt.ToggleButtons,
        legend_checkbox: wgt.Checkbox,
        imag_control_box: RepresentationControlBox,
        real_control_box: RepresentationControlBox,
        abs_control_box: RepresentationControlBox,
    ) -> None:
        
        # normalization buttons
        self.normalization_buttons = normalization_buttons
        self.legend_checkbox = legend_checkbox
        # control boxes
        self.imag_control_box = imag_control_box
        self.real_control_box = real_control_box
        self.abs_control_box = abs_control_box

        self.ui = self.setup_ui()

    def setup_ui(self) -> wgt.Box:
        subbox_layout: wgt.Layout = wgt.Layout(
            display='flex',
            flex_flow='column',
            border='solid 1px gray',
            padding='5px',
            margin='2px'
        )
        norm_title = wgt.HTML(
            value="<h3 style='margin-bottom: 5px; color: #2E86AB; font-weight: bold;'>Signal Vector Normalization</h3>",
            layout=wgt.Layout(margin='0px 0px 5px 0px')
        )
        legend_title = wgt.HTML(
            value="<h3 style='margin-bottom: 5px; color: #2E86AB; font-weight: bold;'>Legend Display</h3>",
            layout=wgt.Layout(margin='0px 0px 5px 0px')
        )
        button_box = wgt.VBox(
            [norm_title, self.normalization_buttons, legend_title, self.legend_checkbox],
            layout=subbox_layout
        )
        box = wgt.HBox([
            button_box,
            self.imag_control_box.ui,
            self.real_control_box.ui,
            self.abs_control_box.ui
        ])
        return box
    

    def query_state(self) -> SignalDisplayState:
        """
        Query the current state of the dashboard and return as a structured object.
        """
        state = SignalDisplayState(
            normalize=(self.normalization_buttons.value == 'Normalized'),
            legend_visible=self.legend_checkbox.value,
            imag_state=LineTraceState(
                visible=self.imag_control_box.visibility_checkbox.value,
                linestyle=self.imag_control_box.linstyle_selector.value,
                opacity=self.imag_control_box.opacity_slider.value
            ),
            real_state=LineTraceState(
                visible=self.real_control_box.visibility_checkbox.value,
                linestyle=self.real_control_box.linstyle_selector.value,
                opacity=self.real_control_box.opacity_slider.value
            ),
            abs_state=LineTraceState(
                visible=self.abs_control_box.visibility_checkbox.value,
                opacity=self.abs_control_box.opacity_slider.value,
                linestyle=self.abs_control_box.linstyle_selector.value
            )
        )
        return state


    @classmethod
    def create(cls) -> 'SignalDisplayDashboard':
        """
        Create a dashboard with sensible default widgets.
        """

        normalization_buttons = make_normalization_toggle_buttons('Signal Vector Display:')
        legend_checkbox = make_legend_checkbox()
        imag_box = RepresentationControlBox(
            description_prefix='Imaginary',
            initial_visibility=True,
            initial_linestyle='solid',
            initial_opacity=1.0
        )
        real_box = RepresentationControlBox(
            description_prefix='Real',
            initial_visibility=False,
            initial_linestyle='dash',
            initial_opacity=1.0
        )
        abs_box = RepresentationControlBox(
            description_prefix='Absolute',
            initial_visibility=False,
            initial_linestyle='dot',
            initial_opacity=0.5
        )
        return cls(
            normalization_buttons=normalization_buttons,
            legend_checkbox=legend_checkbox,
            imag_control_box=imag_box,
            real_control_box=real_box,
            abs_control_box=abs_box
        )



def create_trace(
    x: NDArray[np.floating],
    data: NDArray[np.floating],
    meta: MutableMapping[str, str | float],
    name: str,
    color: str,
    linestyle: str,
    opacity: float,
    normalize: bool,
    visible: bool,
    *,
    epsilon: float = 1e-6
) -> go.Scatter:
    """
    Create a plotly scatter trace for the given data.
    Specialized for signal vector display via hovertemplate,
    metadata, and normalization.
    """
    if normalize:
        norm = np.linalg.norm(data)
        data = data / (norm + epsilon)
        extra_hovertemplate = ' L2 normalized<br>'
        extra_name = '(normalized)'
    else:
        extra_hovertemplate = ''
        extra_name = ''
    # build information strings and metadata
    hovertemplate = f'<b>{name}</b><br>{extra_hovertemplate}' + 'signal: %{y}<br>TR index: %{x}<extra></extra>'
    meta = meta | {'normalized' : normalize}
    trace = go.Scatter(
        x=x,
        y=data,
        meta=meta,
        visible=visible,
        name=' '.join((name, extra_name)),
        hovertemplate=hovertemplate,
        opacity=opacity,
        line=dict(color=color, dash=linestyle),
    )
    return trace


def add_traces(
    fig: go.FigureWidget | go.Figure,
    output: SimulationOutputPackage,
    state: SignalDisplayState
) -> None:
    """
    Add signal vector traces to the given figure widget.
    Visualization (visibility, styling, normalization) is controlled
    via the given state object.

    All representation (real, imag, abs) and normalization variants
    of a signal are always added - we dynamically toggle visibility.
    """
    representations: dict[str, tuple[Callable, LineTraceState]] = {
        'abs' : (np.abs, state.abs_state),
        'real': (np.real, state.real_state),
        'imag': (np.imag, state.imag_state),
    }

    colors = px.colors.qualitative.Plotly
    # Note: output.result is a list of SimulatedSpecies. This in turn
    # holds the relaxometric parameters and the simulated signal vector.
    for ID, simspec in enumerate(output.result):
        signal = simspec.signal
        T1 = simspec.species.T1
        T2 = simspec.species.T2
        M0 = simspec.species.M0 # noqa: F841
        color = colors[ID % len(colors)]
        
        for key, (repr_func, repr_state) in representations.items():
            # compute representation of the signal
            data = repr_func(signal)
            meta: dict[str, str | float] = {
                'ID' : ID,
                'T1' : T1,
                'T2' : T2,
                'representation' : key,
            }
            name: str = f'Species {ID} (T1={T1:.0f}ms, T2={T2:.0f}ms) [{key}]'
            x = np.arange(len(data))

            normalized_visible = state.normalize and repr_state.visible
            nonnormalized_visible = (not state.normalize) and repr_state.visible

            trace_normed = create_trace(
                x, data, meta, name, color,
                linestyle=repr_state.linestyle,
                opacity=repr_state.opacity,
                normalize=True,
                visible=normalized_visible
            )
            trace_nonnormed = create_trace(
                x, data, meta, name, color,
                linestyle=repr_state.linestyle,
                opacity=repr_state.opacity,
                normalize=False,
                visible=nonnormalized_visible
            )
            fig.add_trace(trace_normed)
            fig.add_trace(trace_nonnormed)



class SignalDisplay:
    """
    Plotly figure widget to display complex signal vectors interactively.
    """

    def __init__(
        self
    ) -> None:
        self.fig = go.FigureWidget()

    def add_traces(
        self,
        output: SimulationOutputPackage,
        state: SignalDisplayState
    ) -> None:
        add_traces(self.fig, output, state)

    def reinitialize_traces(
        self,
        output: SimulationOutputPackage,
        state: SignalDisplayState
    ) -> None:
        self.fig.data = []
        self.add_traces(output, state)

    def set_representation_visibility(
        self,
        representation: str,
        visible: bool
    ) -> None:
        # BUG: this does not fully the right thing when normalization
        # is toggled and both normalized and non-normalized traces
        # exist for the same representation.
        for trace in self.fig.data:
            if trace.meta['representation'] == representation:
                trace.visible = visible

    def set_imag_visibility(
        self,
        visible: bool
    ) -> None:
        self.set_representation_visibility('imag', visible)

    def set_real_visibility(
        self,
        visible: bool
    ) -> None:
        self.set_representation_visibility('real', visible)

    def set_abs_visibility(
        self,
        visible: bool
    ) -> None:
        self.set_representation_visibility('abs', visible)

    def set_representation_opacity(
        self,
        representation: str,
        opacity: float
    ) -> None:
        for trace in self.fig.data:
            if trace.meta['representation'] == representation:
                trace.opacity = opacity

    def set_imag_opacity(
        self,
        opacity: float
    ) -> None:
        self.set_representation_opacity('imag', opacity)

    def set_real_opacity(
        self,
        opacity: float
    ) -> None:
        self.set_representation_opacity('real', opacity)

    def set_abs_opacity(
        self,
        opacity: float
    ) -> None:
        self.set_representation_opacity('abs', opacity)

    def set_normalized_visibility(
        self,
        normalized: bool
    ) -> None:
        normalized_traces = [t for t in self.fig.data if t.meta['normalized'] is True]
        non_normalized_traces = [t for t in self.fig.data if t.meta['normalized'] is False]

        for nt, nnt in zip(normalized_traces, non_normalized_traces):
            # Setting the trace visibility is conditional on whether
            # either the normalized or non-normalized trace was visible
            # before. This way, toggling normalization does not override
            # the visibility settings of the individual traces.
            any_visible = nt.visible or nnt.visible
            nt.visible = any_visible and normalized
            nnt.visible = any_visible and (not normalized)

    def set_representation_linestyle(
        self,
        representation: str,
        linestyle: str
    ) -> None:
        for trace in self.fig.data:
            if trace.meta['representation'] == representation:
                trace.line.update({'dash' : linestyle})

    def set_imag_linestyle(
        self,
        linestyle: str
    ) -> None:
        self.set_representation_linestyle('imag', linestyle)

    def set_real_linestyle(
        self,
        linestyle: str
    ) -> None:
        self.set_representation_linestyle('real', linestyle)

    def set_abs_linestyle(
        self,
        linestyle: str
    ) -> None:
        self.set_representation_linestyle('abs', linestyle)

    def set_legend_visibility(
        self,
        visible: bool
    ) -> None:
        self.fig.update_layout(showlegend=visible)



def wire_callbacks(
    signal_display: SignalDisplay,
    dashboard: SignalDisplayDashboard,
    controller: SimulationController
) -> None:
    """
    Connect simulation controller and dashboard controls with
    visualizing signal display object.
    """
    dashboard.imag_control_box.visibility_checkbox.observe(
        lambda change: signal_display.set_imag_visibility(change['new']),
        names='value'
    )
    dashboard.real_control_box.visibility_checkbox.observe(
        lambda change: signal_display.set_real_visibility(change['new']),
        names='value'
    )
    dashboard.abs_control_box.visibility_checkbox.observe(
        lambda change: signal_display.set_abs_visibility(change['new']),
        names='value'
    )
    dashboard.normalization_buttons.observe(
        lambda change: signal_display.set_normalized_visibility(change['new'] == 'Normalized'),
        names='value'
    )
    dashboard.imag_control_box.linstyle_selector.observe(
        lambda change: signal_display.set_imag_linestyle(change['new']),
        names='value'
    )
    dashboard.real_control_box.linstyle_selector.observe(
        lambda change: signal_display.set_real_linestyle(change['new']),
        names='value'
    )
    dashboard.abs_control_box.linstyle_selector.observe(
        lambda change: signal_display.set_abs_linestyle(change['new']),
        names='value'
    )
    dashboard.imag_control_box.opacity_slider.observe(
        lambda change: signal_display.set_imag_opacity(change['new']),
        names='value'
    )
    dashboard.real_control_box.opacity_slider.observe(
        lambda change: signal_display.set_real_opacity(change['new']),
        names='value'
    )
    dashboard.abs_control_box.opacity_slider.observe(
        lambda change: signal_display.set_abs_opacity(change['new']),
        names='value'
    )
    dashboard.legend_checkbox.observe(
        lambda change: signal_display.set_legend_visibility(change['new']),
        names='value'
    )
    controller.add_on_run_complete_callback(
        lambda output: signal_display.reinitialize_traces(
            output, dashboard.query_state()
        )
    )