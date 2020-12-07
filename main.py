import copy
import os
import logging

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from config import config
from pof import Component
from pof.units import valid_units
from pof.loader.asset_model_loader import AssetModelLoader
from pof.interface.dashlogger import DashLogger
from pof.interface.layouts import layout, make_component_layout, cf  # TODO fix this
from pof.interface.figures import (
    update_condition_fig,
    update_cost_fig,
    update_pof_fig,
    make_sensitivity_fig,
)

t_end = 200

# Quick test to make sure everything is works
file_name = r"Asset Model - Pole - Timber.xlsx"
filename = os.getcwd() + r"\data\inputs" + os.sep + file_name

aml = AssetModelLoader(filename)
comp_data = aml.load()
comp = Component.from_dict(comp_data["pole"])

# Turn off logging level to speed up implementation
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger("werkzeug").setLevel(logging.WARNING)

# Build App
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

# Globals
pof_sim = copy.copy(comp)
sens_sim = copy.copy(comp)


# Layout
var_to_scale = cf.scaling
app.layout = layout(comp)

collapse_ids = comp.get_objects()


@app.callback(
    [Output(f"{prefix}-collapse", "is_open") for prefix in collapse_ids],
    [Input(f"{prefix}-collapse-button", "n_clicks") for prefix in collapse_ids],
    [State(f"{prefix}-collapse", "is_open") for prefix in collapse_ids],
)
def toggle_collapses(*args):
    ctx = dash.callback_context

    state_id = ""
    collapse_id = (
        ctx.triggered[0]["prop_id"].split(".")[0].replace("-collapse-button", "")
    )
    if collapse_id in collapse_ids:  # TODO change to is not None

        state_id = collapse_id + "-collapse.is_open"
        ctx.states[state_id] = not ctx.states[state_id]

    is_open = tuple(ctx.states.values())

    return is_open


ms_fig_update = comp.get_dash_ids()
param_inputs = [
    Input(dash_id, "checked") if "active" in dash_id else Input(dash_id, "value")
    for dash_id in ms_fig_update
]

# Update --> Simulate --> Figures


@app.callback(Output("update_state", "children"), param_inputs)
def update_parameter(graph_y_limit_active, graph_y_limit, *args):
    """Update a the pof object whenever an input is changes"""

    # Check the parameters that changed
    ctx = dash.callback_context
    dash_id = None
    value = None

    # If any parameters have changed update the objecte
    if ctx.triggered:
        dash_id = ctx.triggered[0]["prop_id"].split(".")[0]
        value = ctx.triggered[0]["value"]

        # Scale the value if req
        var = dash_id.split("-")[-1]
        if var in var_to_scale:
            value = value / var_to_scale.get(var, 1)

        # update the model
        comp.update(dash_id, value)

    return f"Update State: {dash_id} - {value}"


@app.callback(
    Output("sim_state", "children"),
    Output("sim_state_err", "children"),
    Input("sim_n_active", "checked"),
    Input("n_iterations-input", "value"),
    Input("update_state", "children"),
    Input("sens_time_unit-dropdown", "value"),
)
def update_simulation(active, n_iterations, state, time_unit):
    """ Triger a simulation whenever an update is completed or the number of iterations change"""
    global pof_sim
    global sim_err_count

    pof_sim.cancel_sim()

    # time.sleep(1)
    if active:
        comp.units = time_unit
        pof_sim = copy.copy(comp)

        pof_sim.mp_timeline(t_end=t_end, n_iterations=n_iterations)

        if not pof_sim.up_to_date:
            sim_err_count = sim_err_count + 1
            return dash.no_update, f"Errors: {sim_err_count}"
        return f"Sim State: {pof_sim.n_iterations} - {n_iterations}", ""
    else:
        return f"Sim State: not updating", ""


fig_start = 0
fig_end = 0

# After a simulation the following callbacks are triggered


@app.callback(
    [
        Output("cost-fig", "figure"),
        Output("pof-fig", "figure"),
        Output("cond-fig", "figure"),
        Output("fig_state", "children"),
    ],
    Input("sim_state", "children"),
    State("sim_n_active", "checked"),
)
def update_figures(state, active, *args):
    if active:

        global fig_start
        global fig_end

        fig_start = fig_start + 1
        cost_fig = update_cost_fig(pof_sim)  # legend = dropdown value
        pof_fig = update_pof_fig(pof_sim)
        cond_fig = update_condition_fig(pof_sim)

        fig_state = f"Fig State: {fig_start} - {fig_end}"
        fig_end = fig_end + 1
        return cost_fig, pof_fig, cond_fig, fig_state
    else:
        raise PreventUpdate


@app.callback(Output("ffcf", "children"), [Input("sim_state", "children")])
def update_ffcf(*args):
    n_cf = len(pof_sim.expected_cf())
    n_ff = len(pof_sim.expected_ff())

    try:
        ratio = round(n_ff / (n_cf + n_ff), 2)
    except:
        ratio = "--.--"

    return f"Conditional {n_cf} : {n_ff} Functional. {ratio}%"


@app.callback(
    Output("insp_interval-fig", "figure"),
    Input("sim_sens_active-check", "checked"),
    Input("n_sens_iterations-input", "value"),
    Input("sens_var_id-dropdown", "value"),
    Input("sens_var_y-dropdown", "value"),
    Input("sens_lower-input", "value"),
    Input("sens_upper-input", "value"),
    Input("sens_step_size-input", "value"),
    Input("sens_t_end-input", "value"),
    Input("cost-fig", "figure"),  # TODO change this trigger
)
def update_sensitivity(
    active,
    n_iterations,
    var_id,
    y_axis,
    lower,
    upper,
    step_size,
    t_end,
    *args,
):
    """ Trigger a sensitivity analysis of the target variable"""
    # Copy from the main model
    global sens_sim
    sens_sim.cancel_sim()

    if active:
        sens_sim = copy.deepcopy(comp)

        insp_interval_fig = make_sensitivity_fig(
            sens_sim,
            var_name=var_id,
            y_axis=y_axis,
            lower=lower,
            upper=upper,
            step_size=step_size,
            t_end=t_end,
            n_iterations=n_iterations,
        )

        return insp_interval_fig
    else:
        raise PreventUpdate


# The following progress bars are always running


@app.callback(
    [Output("n-progress", "value"), Output("n-progress", "children")],
    [Input("progress-interval", "n_intervals")],
)
def update_progress(n):
    if pof_sim.n is None:
        raise Exception("no process started")
    progress = int(pof_sim.progress() * 100)

    return progress, f"{progress} %" if progress >= 5 else ""


@app.callback(
    [Output("n_sens-progress", "value"), Output("n_sens-progress", "children")],
    [Input("progress-interval", "n_intervals")],
)
def update_progress_sens(n):
    if sens_sim.n is None:
        raise Exception("no process started")
    progress = int(sens_sim.sens_progress() * 100)

    return progress, f"{progress} %" if progress >= 5 else ""


if __name__ == "__main__":
    app.run_server(debug=True)