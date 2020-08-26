# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd

import pprint
import copy

#from interface.layouts import generate_dist_form, generate_task_form, generate_impact_form, make_impact_condition_form
from component import Component
from failure_mode import FailureMode
from condition import Condition
from distribution import Distribution
from consequence import Consequence
from task import Inspection, OnConditionRepair, ImmediateMaintenance

from layout import make_component_layout

# Build App
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

comp = Component().set_demo()

mcl = make_component_layout(comp)

# Layout
graph_limit =  dbc.InputGroup(
    [
        dbc.InputGroupAddon(
            [
                dbc.Checkbox(id='graph_y_limit_active', checked=True),
                
            ],
            addon_type="prepend"
        ),
        dbc.Label("Graph Y Limit"),
        dbc.Input(
            type="number",
            id= 'graph_y_limit',
            #value = ,
            debounce=True,
        ),
    ],
)

graph_inputs = [Input('graph_y_limit_active', 'checked'), Input('graph_y_limit', 'value')]

# Layout
app.layout = html.Div([
    html.Div(children='Test Output', id='test_output'),
    graph_limit,
    dcc.Graph(id="maintenance_strategy"),
    mcl,
])

sep= '-'
collapse_ids = []
for comp_name, comp in {'comp' : comp}.items():
    comp_prefix = comp_name + sep
    collapse_ids = collapse_ids + [comp_name]
    # Failure modes
    fm_prefix = comp_prefix + 'failure_mode' + sep
    for fm_name, fm in comp.fm.items():
        collapse_ids = collapse_ids + [fm_prefix + fm_name]
        # Tasks
        task_prefix = fm_prefix + fm_name + sep + 'task' + sep
        for task in fm.tasks:
            collapse_ids = collapse_ids + [task_prefix + task]


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

ms_fig_update = comp.get_dash_ids(prefix ='comp-')

"""@app.callback(
    Output('graph_y_limit', 'value'), 
    graph_inputs + [Input('maintenance_strategy', 'figure')]
)
def update_graph_y_limit(graph_y_limit_active, graph_y_limit, fig):

    if not graph_y_limit_active:
        return comp.expected_risk_cost_df()['cost'].sum()"""


@app.callback(
    Output("maintenance_strategy", "figure"), 
    graph_inputs + [Input(dash_id,"checked") if 'active' in dash_id else Input(dash_id,"value") for dash_id in ms_fig_update]
)
def update_maintenance_strategy(graph_y_limit_active, graph_y_limit, *args):

    # Check the parameters that changed
    ctx = dash.callback_context
    dash_id = ctx.triggered[0]['prop_id'].split('.')[0]
    value = ctx.triggered[0]['value']

    #Temp utnil mutliple failure modes are set up
    dash_id = dash_id.replace('comp-', "")

    # Update the model
    comp.dash_update(dash_id, value)

    comp_local = copy.deepcopy(comp)

    comp_local.mc_timeline(t_end=200, n_iterations=100)
    df = comp_local.expected_risk_cost_df(t_end=200)

    fig = px.area(
        df,
        x="time",
        y="cost_cumulative",
        color="task",
        line_group='failure_mode',
        title="Maintenance Strategy Costs",
    )

    if graph_y_limit_active:
        fig.update_yaxes(range=[0, graph_y_limit])

    return fig

@app.callback(
    Output("test_output", "children"),
    [Input('comp-active', 'checked')]
)
def test(task_value):
    # Check the parameters that changed
    ctx = dash.callback_context
    dash_id = ctx.triggered[0]['prop_id'].split('.')[0]
    value = ctx.triggered[0]['value']

    return "value" + str(value)


if __name__ == "__main__":
    print ("app ok")
    app.run_server(debug=True)
