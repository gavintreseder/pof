import copy
import logging

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from config import config
from pof.loader.asset_model_loader import AssetModelLoader
from pof import Component, FailureMode, Task
from pof.interface.dashlogger import DashLogger
from pof.interface.layouts import *
from pof.interface.figures import (
    update_condition_fig,
    update_cost_fig,
    update_pof_fig,
    make_sensitivity_fig,
    make_inspection_interval_fig,
)

# Quick test to make sure everything is works
comp = Component.demo()

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


def layout():
    mcl = make_component_layout(comp)
    update_list = [
        {"label": option, "value": option} for option in comp.get_update_ids()
    ]

    layout = html.Div(
        [
            html.Div(id="log"),
            html.Div(children="Update State:", id="update_state"),
            html.Div(
                [
                    html.P(children="Sim State", id="sim_state"),
                    html.P(id="sim_state_err", style={"color": "red"}),
                ]
            ),
            html.Div(children="Fig State", id="fig_state"),
            html.P(id="ffcf"),
            html.Div(
                [
                    dcc.Interval(id="progress-interval", n_intervals=0, interval=50),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.InputGroupAddon(
                                        [
                                            dbc.Checkbox(
                                                id="sim_n_active", checked=True
                                            ),
                                            dcc.Input(
                                                id="n_iterations-input",
                                                value=10,
                                                type="number",
                                            ),
                                        ],
                                        addon_type="prepend",
                                    ),
                                    dbc.Progress(id="n-progress"),
                                ]
                            ),
                            dbc.Col(
                                [
                                    dbc.InputGroupAddon(
                                        [
                                            dbc.Checkbox(
                                                id="sim_n_sens_active", checked=False
                                            ),
                                            dcc.Input(
                                                id="n_sens_iterations-input",
                                                value=10,
                                                type="number",
                                            ),
                                        ],
                                        addon_type="prepend",
                                    ),
                                    dbc.Progress(id="n_sens-progress"),
                                ]
                            ),
                        ]
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(),
                    dbc.Col(
                        dcc.Dropdown(
                            id="demo-dropdown",
                            options=update_list,
                            value=comp.get_update_ids()[-1],
                        )
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="pof-fig")),
                    dbc.Col(dcc.Graph(id="insp_interval-fig")),
                ]
            ),
            dbc.Row(
                [dbc.Col(dcc.Graph(id="cond-fig")), dbc.Col(dcc.Graph(id="cost-fig"))]
            ),
            mcl,
        ]
    )

    return layout


# Layout
var_to_scale = cf.scaling
app.layout = layout

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
        value = value / var_to_scale.get(dash_id.split("-")[-1], 1)

        # update the model
        comp.update(dash_id, value)

    return f"Update State: {dash_id} - {value}"


@app.callback(
    [Output("sim_state", "children"), Output("sim_state_err", "children")],
    [
        Input("sim_n_active", "checked"),
        Input("n_iterations-input", "value"),
        Input("update_state", "children"),
    ],
    [State("sim_n_active", "checked"), State("n_iterations-input", "value")],
)
def update_simulation(
    active_input, n_iterations_input, state, active, n_iterations, *args
):
    """ Triger a simulation whenever an update is completed or the number of iterations change"""
    global pof_sim
    global sim_err_count

    pof_sim.cancel_sim()

    # time.sleep(1)
    if active:
        pof_sim = copy.copy(comp)

        pof_sim.mp_timeline(t_end=200, n_iterations=n_iterations)

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
    [Input("sim_state", "children")],
    [State("sim_n_active", "checked")],
)
def update_figures(state, active, *args):
    if active:

        global fig_start
        global fig_end

        fig_start = fig_start + 1
        cost_fig = update_cost_fig(pof_sim)  # legend = dropdown value
        pof_fig = update_pof_fig(pof_sim)
        cond_fig = update_condition_fig(pof_sim)
        fig_end = fig_end + 1
        return cost_fig, pof_fig, cond_fig, f"Fig State: {fig_start} - {fig_end}"
    else:
        raise PreventUpdate


@app.callback(Output("ffcf", "children"), [Input("sim_state", "children")])
def update_ffcf(*args):
    cf = len(pof_sim.expected_cf())
    ff = len(pof_sim.expected_ff())

    try:
        ratio = round(ff / (cf + ff), 2)
    except:
        ratio = "--.--"

    return f"Conditional {cf} : {ff} Functional. {ratio}%"


@app.callback(
    Output("insp_interval-fig", "figure"),
    [
        Input("sim_n_sens_active", "checked"),
        Input("n_sens_iterations-input", "value"),
        Input("demo-dropdown", "value"),
        Input("cost-fig", "figure"),
    ],
    [
        State("sim_n_sens_active", "checked"),
        State("n_sens_iterations-input", "value"),
        State("demo-dropdown", "value"),
    ],
)
def update_insp_interval(
    active_input,
    n_iterations_input,
    var_input,
    fig,
    active,
    n_iterations,
    var_name,
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
            var_name=var_name,
            lower=1,
            upper=10,
            step_size=1,
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