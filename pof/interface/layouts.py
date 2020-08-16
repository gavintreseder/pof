import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc


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

            )
        ])
    ])

    return layout


def generate_dist_form(distribution):
    """
    Takes a Distribution and generates the html form inputs
    """

    #TODO only works for a Weibull, doesn't take prefix a
    prefix = ""

    param_values = dict(
        alpha = 1,
        beta = 1,
        gamma = 1,
    )

    form = generate_horizontal_form(param_values=param_values, prefix=prefix)

    return form

def generate_task_layout(task, prefix=""):
    """

    """
    task_form = generate_task_form(task=task, prefix=prefix)

    task_layout = dbc.InputGroup(
        [
            dbc.InputGroupAddon(dbc.Checkbox(), addon_type="prepend"),

            dbc.Button(
                "Task Names",
                color="link",
                id="collapse-button",
            ),
            dbc.Col(
                [
                    #dbc.Label("Details...", className="mr-2"),
                    task_form,
                    dbc.Collapse(
                        dbc.Card(dbc.CardBody(
                            [
                                dbc.Col(dbc.Card(dbc.CardBody())),
                                #trigger_card, 
                                dbc.Col(dbc.Card(dbc.CardBody())),
                            ]
                        )),
                        id="collapse",
                        is_open=True
                    ),
                ]
            )

        ]
    )

    return task_layout

def generate_task_form(task, prefix="", detail='simple'):
    """
    Takes a Task and generates the html form inputs
    """

    form = dbc.Form(
        [
            dbc.Col(
                dbc.FormGroup(
                    [
                        dbc.Label("Probability Effective", html_for="p_effective"),
                        dbc.Input(
                            type="number",
                            id= prefix + "_p_effective",
                        ),
                    ],
                ),
            ),
            dbc.Col(
                dbc.FormGroup(
                    [
                        dbc.Label("Cost", html_for="cost"),
                        dbc.Input(
                            type="number",
                            id= prefix + "_cost",
                        ),
                    ],
                ),
            ),
            dbc.Col(
                dbc.FormGroup(
                    [
                        dbc.Label("Consequence", html_for="consequence"),
                        dbc.Input(
                            type="number",
                            id= prefix + "_consequence",
                        ),
                    ],
                ),
            ),

        ],
        inline=True,
    )

    return form


def generate_task_trigger_form(trigger):
    """
    Takes a Trigger and generates the html form inputs
    """

    form = NotImplemented

    return form


def generate_task_impact_form(impacts, prefix=""):
    """Takes a the impacts from a Trigger object and generates the html form inputs
    # TODO Implement times
    """

    forms = []
    # Conditions

    """for condition, impact in impacts['condition'].items():
        condition_prefix = "%s_%s_" %(prefix, condition)

        condition_form = make_impact_condition_form(impact, prefix=condition_prefix)
        forms = forms + [condition_form]

    # States"""
    state_form = generate_horizontal_form(impacts['state'], prefix=prefix)

    forms = forms + [state_form]

    form = dbc.Form(forms)

    return form

def make_impact_state_form(impact, prefix="", detail='simple'):


    return NotImplemented

def make_impact_condition_form(impact, prefix = "", detail='simple'):
    """Create a form for impact condition with method, axis and target
    """
    
    c_name = dbc.InputGroup(
        [
            dbc.InputGroupAddon(dbc.Checkbox(), addon_type="prepend"),
            dbc.Label("Cond Name", className="mr-2"),

        ]
    )

    target = dbc.FormGroup(
        [
            dbc.Label("Target", className="mr-2"),
            dbc.Input(
                type="number",
                id= prefix + "target",
                value=impact['target'],
            ),
        ],
        className="mr-3",
    )

    method = dbc.FormGroup(
        [
            dbc.Label("Method", className="mr-2"),
            dbc.Select(
                id=prefix + "method_dropdown",
                options=[{'label' : method, 'value' : method} for method in ['reduction_factor', 'tbc']],
                value=impact['method'],
            ),
        ],
        className="mr-3",
    )

    axis = dbc.FormGroup(
        [
            dbc.Label("Axis", className="mr-2"),
            dbc.Select(
                id=prefix + "axis_dropdown",
                options=[{'label' : axis, 'value' : axis} for axis in ['condition', 'time']],
                value=impact['axis'],
            ),
        ],
        className="mr-3",
    )

    form = dbc.Form([c_name, target, method, axis], inline=True, className="dash-bootstrap")

    return form


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

def generate_horizontal_form(param_values, prefix=""):
    """ Generate a html form for fields with a given prefix
    """

    param_inputs = []

    for param, value in param_values.items():
        param_input = dbc.Col(
            dbc.FormGroup(
                [
                    dbc.Label(param.capitalize(), className="mr-2"),
                    dbc.Input(
                        type="number",
                        id= prefix + param,
                        value=value,
                    ),
                ],
                className='mr-3',
            ),
        )
        param_inputs.append(param_input)

    form = dbc.Form(children=param_inputs, inline=True)

    return form



if __name__ == "__main__":
    print("layout methods - Ok")