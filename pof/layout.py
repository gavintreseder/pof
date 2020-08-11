
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd


layout = html.Div(children=[
    html.H1(children='Hello Dash'),
    dcc.Input(
        id='num-multi',
        type='number',
        value=5
    ),
    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

# Component
    # Failure Mode
        # Distribution
            # Alpha
            # Beta
            # Gamma

        # Tasks
            # Inspection
            # On Condition Repair
            # ImmediateMaintenance
    
        # Condition Loss

    # Condition