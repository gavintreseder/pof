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
from pof.interface.layouts import make_layout, cf  # TODO fix the need to import cf
from pof.interface.figures import (
    update_condition_fig,
    update_pof_fig,
    make_sensitivity_fig,
)

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
app.layout = make_layout(comp)

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
def update_parameter(*args):
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
    Input("t_end-input", "value"),
    Input("n_iterations-input", "value"),
    Input("update_state", "children"),
    Input("sens_time_unit-dropdown", "value"),
)
def update_simulation(active, t_end, n_iterations, state, time_unit):
    """ Triger a simulation whenever an update is completed or the number of iterations change"""
    global pof_sim

    pof_sim.cancel_sim()

    # time.sleep(1)
    if active:
        comp.units = time_unit  # TODO Mel move this to a param update
        pof_sim = copy.copy(comp)

        # Complete the simulations
        pof_sim.mp_timeline(t_end=t_end, n_iterations=n_iterations)

        # Generate the dataframe for reporting
        pof_sim.expected_risk_cost_df()

        if not pof_sim.up_to_date:
            return dash.no_update, f"Update cancelled"

    else:
        return dash.no_update, "Not active"

    return f"Sim State: {pof_sim.n_iterations} - {n_iterations}", ""


fig_start = 0
fig_end = 0

# ========================================================
# After a simulation the following callbacks are triggered
# ========================================================

# TODO Mel Repeat this for each of the y_var. It will need a different method for y_max = if the costs don't stack


@app.callback(
    Output("cost_var_y-input", "value"),  # Output('fig_limit_maint_strat', 'children')
    Input("sim_state", "children"),
    Input("sens_var_y-dropdown", "value"),  # TODO change name of this
    Input("axis_lock-checkbox", "checked"),  # TODO should this be a state?
)
def save_figure_limits(__, y_axis, axis_lock):
    """ Save the figure limits so they can be used for the axis lock"""
    try:
        if not axis_lock:
            y_max = pof_sim.df_erc.groupby("time")[y_axis].sum().max() * 1.05
        else:
            ctx = dash.callback_context
            dash_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if dash_id == "sens_var_y-dropdown":
                y_max = pof_sim.df_erc.groupby("time")[y_axis].sum().max() * 1.05
            else:
                return dash.no_update
    except:
        y_max = None
    return y_max


@app.callback(
    Output("cond-fig", "figure"),
    Output("ms-fig", "figure"),
    Output("pof-fig", "figure"),
    Input("sim_state", "children"),
    # Input("cond_var_y-input", "value"),
    Input("cost_var_y-input", "value"),
    Input("pof_var_y-input", "value"),
    Input("sens_var_y-dropdown", "value"),  # TODO change to an appropriate name
    State("t_end-input", "value"),
    State("sim_n_active", "checked"),
    State("axis_lock-checkbox", "checked"),
)
def update_figures(
    state,
    # cond_var_y,
    ms_var_y,
    pof_var_y,
    y_axis,
    t_end,
    active,
    axis_lock,
    *args,
):
    if active:
        cond_var_y = None  # Temp fix
        if not axis_lock:
            ms_var_y = None
            cond_var_y = None
            pof_var_y = None

        ms_fig = pof_sim.plot_erc(y_axis=y_axis, y_max=ms_var_y, t_end=t_end)
        pof_fig = update_pof_fig(pof_sim, t_end=t_end, y_max=pof_var_y)
        cond_fig = update_condition_fig(
            pof_sim,
            t_end=t_end,
            y_max=cond_var_y,
        )
    else:
        raise PreventUpdate

    return cond_fig, ms_fig, pof_fig


@app.callback(Output("ffcf", "children"), [Input("sim_state", "children")])
def update_ffcf(*args):
    """ Returns the ratio between conditional failures, functional failures and in service assets"""
    n_cf = len(pof_sim.expected_cf())
    n_ff = len(pof_sim.expected_ff())

    try:
        ratio = round(n_ff / (n_cf + n_ff), 2)
    except:
        ratio = "--.--"

    return f"Conditional {n_cf} : {n_ff} Functional. {ratio}%"


# Calcaulate sensitivity

# Plot Sensitivity

# @app.callback(
#     Output("insp_interval-fig", "figure"),
#     Input("sim_sens_active-check", "checked"),
#     Input("n_sens_iterations-input", "value"),
#     Input("sens_var_id-dropdown", "value"),
#     Input("sens_var_y-dropdown", "value"),
#     State("t_end-input", "value"),
#     State("sens_var_y-input", "value"),
# )

# def update_sensitivity_figure()
#     """ Updates the sensitivity figure whenever a new sensitivity simulation is completed"""


@app.callback(
    Output("sensitivity-fig", "figure"),
    Input("sim_sens_active-check", "checked"),
    Input("n_sens_iterations-input", "value"),
    Input("sens_var_id-dropdown", "value"),
    Input("sens_var_y-dropdown", "value"),
    Input("sens_lower-input", "value"),
    Input("sens_upper-input", "value"),
    Input("sens_step_size-input", "value"),
    Input("t_end-input", "value"),
    Input("sens_var_y-input", "value"),
    Input("ms-fig", "figure"),  # TODO change this trigger
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
    y_max,
    *args,
):
    """ Trigger a sensitivity analysis of the target variable"""
    # Copy from the main model
    global sens_sim
    sens_sim.cancel_sim()

    if active:
        sens_sim = copy.deepcopy(comp)

        sens_fig = make_sensitivity_fig(
            sens_sim,
            var_name=var_id,
            y_axis=y_axis,
            lower=lower,
            upper=upper,
            step_size=step_size,
            t_end=t_end,
            y_max=y_max,
            n_iterations=n_iterations,
        )

        return sens_fig
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