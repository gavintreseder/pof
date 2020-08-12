import dash_core_components as dcc
import dash_html_components as html

layout1 = html.Div([
    html.H3('App 1'),
    dcc.Dropdown(
        id='app-1-dropdown',
        options=[
            {'label': 'App 1 - {}'.format(i), 'value': i} for i in [
                'NYC', 'MTL', 'LA'
            ]
        ]
    ),
    html.Div(id='app-1-display-value'),
    dcc.Link('Go to App 2', href='/apps/app2')
])

layout2 = html.Div([
    html.H3('App 2'),
    dcc.Dropdown(
        id='app-2-dropdown',
        options=[
            {'label': 'App 2 - {}'.format(i), 'value': i} for i in [
                'NYC', 'MTL', 'LA'
            ]
        ]
    ),
    html.Div(id='app-2-display-value'),
    dcc.Link('Go to App 1', href='/apps/app1')
])

# Asset

    
    # Components

        # Condition

        # Task Groups ( move to assets)

        # Failure Modes

            # Condition Loss
            
            # Tasks

def generate_failure_mode_layout(fm):
    layout = html.Div([
        html.Details([              # Distribution
            html.Summary(fm._name),
            html.Div(
                children=[
                    html.Div('The current parameters for failure mode'),
                    html.Details([

                    ])
                ],
                style = {}

            ])
        ])
    ])

    return layout

def generate_task_layout(task, detail='simple'):

    layout = html.Div([
        html.Details([              # Distribution
            html.Summary(task._name),
            html.Div([
                html.Div('The current parameters for failure mode'),
                html.Details([
                    html.Div([

                    ])
                ])
            ])
        ])
    ])


def generate_task_layout(task, detail='simple'):

    form = dbc.Row(
        [
            dbc.Col(
                dbc.FormGroup(
                    [
                        dbc.Label("Probability Effective", html_for="p_effective"),
                        dbc.Input(
                            type="number",
                            id= task.name + "_p_effective",
                        ),
                    ]
                ),
                width=6,
            ),
            dbc.Col(
                dbc.FormGroup(
                    [
                        dbc.Label("Cost", html_for="cost"),
                        dbc.Input(
                            type="number",
                            id= task.name + "_cost",
                        ),
                    ]
                ),
                width=6,
            ),
            dbc.Col(
                dbc.FormGroup(
                    [
                        dbc.Label("Consequence", html_for="consequence"),
                        dbc.Input(
                            type="number",
                            id= task.name + "_consequence",
                        ),
                    ]
                ),
                width=6,
            ),

        ],
        form=True,
    )

    return layout
    # Task

    # Probability Effective

        # Cost
        # Consequence

        # Trigger

            # Time

            # State (n)
                # state_name
                # state (True/False)

            # Conditions (n)
                # condition_name
                # lower_threshold
                # upper_threshold

        # Impacts

            # Time

            # States (n)
                # state_name
                # state (True / False)

            # Conditions (n)
            
                # condition_name
                # method
                # axis
                # reduction_factor / target


            'wall_thickness': {'target': None,
   'reduction_factor': 0,
   'method': 'reduction_factor',
   'axis': 'condition'}},

   
   html.Div [


   ]