# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

import pprint
import copy

#from interface.layouts import generate_dist_form, generate_task_form, generate_impact_form, make_impact_condition_form
from failure_mode import FailureMode
from condition import Condition
from distribution import Distribution
from consequence import Consequence
from task import Inspection, OnConditionRepair, ImmediateMaintenance

from interface.layouts import make_failure_mode_layout

# Build App
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

mfml = make_failure_mode_layout(fm)

# Layout
app.layout = html.Div([
    html.Div(children='Test Output', id='test_output'),
    dcc.Graph(id="maintenance_strategy"),
    mfml,
])

sep= '-'
collapse_ids = []
for fm_name, fm in {'fm': fm}.items():
    collapse_ids = collapse_ids + [fm_name]
    for task in fm.tasks:
        collapse_ids = collapse_ids + [fm_name + sep + 'task' + sep + task]

@app.callback(
    #Output("test_output", "children"),
    [Output(f"{prefix}-collapse", "is_open") for prefix in collapse_ids],
    [Input(f"{prefix}-collapse-button", "n_clicks") for prefix in collapse_ids],
    [State(f"{prefix}-collapse", "is_open") for prefix in collapse_ids],
)
def toggle_collapses(*args):
    ctx = dash.callback_context

    state_id = ""
    collapse_id = ctx.triggered[0]['prop_id'].split('.')[0].replace('-collapse-button','')
    if collapse_id in collapse_ids: #TODO change to is not None

        state_id = collapse_id + '-collapse.is_open'
        ctx.states[state_id] = not ctx.states[state_id]

    is_open = tuple(ctx.states.values())

    return is_open

ms_fig_update = fm.get_dash_ids(prefix ='fm-')

@app.callback(
    Output("maintenance_strategy", "figure"), 
    [Input(dash_id,"checked") if 'active' in dash_id else Input(dash_id,"value") for dash_id in ms_fig_update]
)
def update_maintenance_strategy(*args):

    # Check the parameters that changed
    ctx = dash.callback_context
    dash_id = ctx.triggered[0]['prop_id'].split('.')[0]
    value = ctx.triggered[0]['value']

    #Temp utnil mutliple failure modes are set up
    dash_id = dash_id.replace('fm-', "")

    # Update the model
    fm_local = copy.deepcopy(fm)
    fm_local.dash_update(dash_id, value)

    fm_local.mc_timeline(t_end=200, n_iterations=100)
    df = fm_local.expected_cost_df()

    fig = px.area(
        df,
        x="time",
        y="cost_cumulative",
        color="task",
        title="Maintenance Strategy Costs",
    )

    #fig.update_yaxes(range=[0, graph_y_limit])

    return fig

if __name__ == "__main__":
    app.run_server(debug=True)
