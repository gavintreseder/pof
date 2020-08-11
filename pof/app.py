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

from failure_mode import FailureMode
from condition import Condition
from distribution import Distribution
from consequence import Consequence
from task import Inspection, OnConditionRepair, ImmediateMaintenance

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#import layout as lay




# Simple failure mode demo
fm = FailureMode().set_default()

app.layout = html.Div(children=[
    html.H1(children='Probability of Failure'),

    html.Div(children='''
        Probability maintenance effective
    '''),
    dcc.Input(
        id='p_effective',
        type='number',
        value=50
    ),
    dcc.Graph(id='maintenance_strategy'),
    dcc.Checklist(
        id='task_checklist'
        options=[{'label':task, 'value' : task} for task in fm.tasks],
        labelStyle={'display': 'inline-block'}
    ),

    html.H3(children='''Tasks'''),
    html.Div(id='failure_mode_dict')
])



@app.callback(
    Output(component_id='maintenance_strategy', component_property='figure'),
    [Input(component_id='p_effective', component_property='value')]
)
def update_maintenance_strategy(p_effective):

    p_effective = p_effective if p_effective is not None else 0

    fm.reset()
    #fm.conditions['wall_thickness'] = Condition(100, 0, 'linear', [-5])
    fm.tasks['inspection'].p_effective = p_effective / 100
    fm.mc_timeline(t_end=200, n_iterations=100)
    df = fm.expected_cost_df()

    fig = px.area(df, x="time", y="cost_cumulative", color="task", title='Maintenance Strategy Costs')

    return fig

@app.callback(
    Output(component_id='maintenance_strategy', component_property='figure'),
    [Input(component_id='task_checklist', component_property='value')]
)
def update_maintenance_strategy(p_effective):

    

    fm.reset()
    #fm.conditions['wall_thickness'] = Condition(100, 0, 'linear', [-5])
    fm.tasks['inspection'].p_effective = p_effective / 100
    fm.mc_timeline(t_end=200, n_iterations=100)
    df = fm.expected_cost_df()

    fig = px.area(df, x="time", y="cost_cumulative", color="task", title='Maintenance Strategy Costs')

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

if __name__ == '__main__':
    app.run_server(debug=True)