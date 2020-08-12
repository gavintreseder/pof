# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

import pprint
import copy

from failure_mode import FailureMode
from condition import Condition
from distribution import Distribution
from consequence import Consequence
from task import Inspection, OnConditionRepair, ImmediateMaintenance
#from layouts import generate_task_layout

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Simple failure mode demo
fm = FailureMode().set_default()

app.layout = html.Div(children=[
    html.H1(children='Probability of Failure'),

    html.Div(children='''
        Probability Inspection effective
    '''),
    dcc.Input(
        id='p_effective',
        type='number',
        value=50,
        debounce=True,
    ),
    html.Div(children='''
        Consequence
    '''),
    dcc.Input(
        id='consequence',
        type='number',
        value=50000,
        debounce=True,
    ),
    html.Div(children='''
        Inspection Interval
    '''),
    dcc.Input(
        id='inspection_interval',
        type='number',
        value=5,
        debounce=True,
    ),
    dcc.Graph(id='maintenance_strategy'),
    dcc.Checklist(
        id='task_checklist',
        options=[{'label':task, 'value' : task} for task in fm.tasks],
        value = [task_name for task_name, task in fm.tasks.items() if task.active],
    ),
    html.Div(children='''
        Graph y_limit
    '''),
    dcc.Input(
        id='graph_y_limit',
        type='number',
        value=20000,
        debounce=True,
    ),

])

@app.callback(
    Output(component_id='maintenance_strategy', component_property='figure'),
    [Input(component_id='graph_y_limit', component_property='value'),
    Input(component_id='p_effective', component_property='value'),
    Input(component_id='task_checklist', component_property='value'),
    Input(component_id='consequence', component_property='value'),
    Input(component_id='inspection_interval', component_property='value')]
)

def update_maintenance_strategy(graph_y_limit, p_effective, tasks, consequence, inspection_interval):

    fm_local = copy.deepcopy(fm)

    p_effective = p_effective if p_effective is not None else 0
    consequence = consequence if consequence is not None else 0
    inspection_interval = inspection_interval if inspection_interval is not None else 0
    graph_y_limit = graph_y_limit if graph_y_limit is not None else 10000


    if tasks is None:
        tasks = []
        
    for task_name, task in fm_local.tasks.items():
        if task_name in tasks:
            task.active = True
        else:
            task.active = False

    
    #fm.conditions['wall_thickness'] = Condition(100, 0, 'linear', [-5])
    fm_local.tasks['inspection'].p_effective = p_effective / 100
    fm_local.cof.risk_cost_total = consequence
    fm_local.tasks['inspection'].t_interval = inspection_interval
    fm_local.mc_timeline(t_end=200, n_iterations=100)
    df = fm_local.expected_cost_df()

    fig = px.area(df, x="time", y="cost_cumulative", color="task", title='Maintenance Strategy Costs')

    fig.update_yaxes(range=[0,graph_y_limit])

    return fig




"""@app.callback(
    Output(component_id='failure_mode_dict', component_property='children'),
    [Input(component_id='p_effective', component_property='value')]
)
def update_dict(p_effecitve):
    output_str = ''
    for task_name, task in fm.tasks.items():
        output_str = ('\n %s \n %s\n %s \n' %(
            output_str,
            task_name,
            pprint.pformat(task.__dict__)
        ))

    return output_str"""
"""


    dcc.Tabs(
        vertical=True,
        children = [
            dcc.Tab(
                label='Tab inner one',
                children=[
                    dcc.Tabs(
                        vertical = True,
                        children=[
                            dcc.Tab(label='Tab one'),
                            dcc.Tab(label='Tab two'),
                            dcc.Tab(label='Tab three'),
                            dcc.Tab(label='Tab four'),
                        ]
                    ),
                ],
            ),
            dcc.Tab(label='Tab tne'),
        ]
    ),
    html.H3(children='''Tasks'''),
    html.Div(id='failure_mode_dict'),
    html.Details([
        html.Summary('heading'),
        html.Div([
            html.Div('conetnte'),
            html.Details([
                html.Summary('heading 2'),
                html.Details('details2'),
            ])
        ])
    ])"""

if __name__ == '__main__':
    app.run_server(debug=True)