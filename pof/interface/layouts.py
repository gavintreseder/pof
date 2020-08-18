import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

#from settings import Settings

IS_OPEN = False

# Asset

    
    # Components

        # Condition

        # Task Groups ( move to assets)

        # Failure Modes

            # Failure Distribution

            # Condition (Loss)
            
            # Tasks

            # Tasks
                # Params
                    # Probability Effective
                    # Cost
                    # Consequence

                # Trigger
                    # Time TODO
                    # State (n)
                        # state_name
                        # state (True/False)
                    # Conditions (n)
                        # condition_name
                        # lower_threshold
                        # upper_threshold
                # Impacts
                    # Time TODO
                    # States (n)
                        # state_name
                        # state (True / False)
                    # Conditions (n)
                        # condition_name
                        # method
                        # axis
                        # reduction_factor / target

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



def make_failure_mode_layout(failure_mode, prefix=""):
    """
    """
    
    # Get failure mode form
    dist_layout = make_dist_layout(failure_mode.failure_dist, prefix=prefix)

    # Get tasks layout
    tasks_layout = []
    for task_name, task in failure_mode.tasks.items():
        task_prefix = prefix + '-' + task_name
        tasks_layout = tasks_layout + [make_task_layout(task, task_name, task_prefix)]

    # Make the layout
    layout = dbc.InputGroup(
        [
            dbc.InputGroupAddon(dbc.Checkbox(), addon_type="prepend"),

            dbc.Button(
                failure_mode.name,
                color="link",
                id = prefix + "-collapse-button",
            ),
            dbc.Col(
                [
                    dist_layout,
                    dbc.Collapse(
                        dbc.Card(dbc.CardBody(
                                tasks_layout
                        )),
                        id = prefix + "-collapse",
                        is_open=IS_OPEN
                    ),
                ]
            )

        ]
    )

    return layout

def make_dist_layout(dist, prefix=""):
    """
    Takes a Distribution and generates the html form inputs
    """

    #TODO only works for a Weibull, doesn't take prefix a
    prefix = ""

    param_inputs = []

    for param, value in dist.params().items():
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


def make_tasks_layout(tasks, prefix=""):

    return NotImplemented

def make_task_layout(task, task_name="task", prefix=""):
    """

    """
    task_form = make_task_form(task=task, prefix=prefix)
    trigger_layout = make_task_trigger_layout(task.triggers(), prefix=prefix)
    impact_layout = make_task_impact_layout(task.impacts(), prefix=prefix)

    task_layout = dbc.InputGroup(
        [
            dbc.InputGroupAddon(dbc.Checkbox(), addon_type="prepend"),
            dbc.Button(
                task_name,
                color="link",
                id = prefix + "-collapse-button",
            ),
            dbc.Col(
                [
                    task_form,
                    dbc.Collapse(
                        dbc.Card(dbc.CardBody(
                            [
                                dbc.Col(trigger_layout),
                                dbc.Col(impact_layout),
                            ]
                        )),
                        id = prefix + "-collapse",
                        is_open=IS_OPEN
                    ),
                ]
            )
        ]
    )

    return task_layout

def make_task_form(task, prefix="", detail='simple'):
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
                            value = task.p_effective * 100,
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
                            value= task.cost,
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
                            value="Not Implemented",
                        ),
                    ],
                ),
            ),
        ],
        inline=True,
    )

    return form

# ******************* Triggers ******************


def make_task_trigger_layout(triggers, prefix=""):
    """
    Takes a Trigger and generates the html form inputs
    """
    prefix = prefix + '-trigger'
    state_layout = make_state_impact_layout(triggers['state'], prefix=prefix)
    condition_layout = make_condition_trigger_layout(triggers['condition'], prefix=prefix)

    layout = dbc.Card(
        [
            dbc.CardHeader("Triggers"),
            dbc.CardBody(
                [
                    state_layout,
                    condition_layout,
                ],
            ),
        ],
    )

    return layout


def make_condition_trigger_layout(triggers, prefix=""): # TODO make these sliders
    """ R
    """
    condition_inputs = []

    for condition, threshold in triggers.items():
        
        condition_prefix = "%s_%s_" %(prefix, condition)
        condition_input = dbc.InputGroup(
            [
                dbc.InputGroupAddon(
                    [
                        dbc.Checkbox(),
                        
                    ],
                    addon_type="prepend"
                ),
                dbc.Label(condition.capitalize(), className="mr-2"),
                dbc.Form(
                    [
                        dbc.Input(
                            type="number",
                            id= condition_prefix + "range-slider-lower",
                            value=threshold['lower'],
                        ),
                        dbc.Input(
                            type="number",
                            id= condition_prefix + "range-slider-upper",
                            value=threshold['upper'],
                        ),
                    ],
                    inline=True
                )
            ]
        )
        condition_inputs = condition_inputs + [condition_input]

    layout = dbc.Form(children=condition_inputs)
    
    return layout

def make_condition_trigger_form(prefix=""):
    form = NotImplemented
    return form



# ****************** Impacts *************************

def make_task_impact_layout(impacts, prefix=""):
    """Takes a the impacts from a Trigger object and generates the html form inputs
    # TODO Implement times
    """
    prefix = prefix + '-impact'
    state_layout = make_state_impact_layout(impacts['state'], prefix=prefix)
    condition_layout = make_condition_impact_layout(impacts['condition'], prefix=prefix)

    layout = dbc.Card(
        [
            dbc.CardHeader("Impacts"),
            dbc.CardBody(
                [
                    state_layout,
                    condition_layout,
                ],
            ),
        ],
    )

    return layout

def make_state_impact_layout(state_impacts, prefix="", detail='simple'):
    """ Creates an input form the state impacts
    """

    #TODO figure out how to take True or False as vlaues
    state_inputs = []
    prefix = prefix + '-state'
    for state, value in state_impacts.items():
        state_input = dbc.Col(
            dbc.InputGroup(
                [
                    dbc.InputGroupAddon(
                        [
                            dbc.Checkbox(),
                            dbc.Label(state.capitalize(), className="mr-2"),
                        ],
                        addon_type="prepend"),
                    dbc.Select(
                        options=[{'label' : option, 'value' : option} for option in ["True", "False"]],
                        id= prefix + state,
                        value=str(value),
                    ),
                ],
                className='mr-3',
            ),
        )
        state_inputs = state_inputs + [state_input]

    form = dbc.Form(children=state_inputs, inline=True)

    return form

def make_state_impact_form(state_impact, prefix=""):# Not use
    """ Creates a form for a state impact"""

    state = str("not implemented")
    value = state = str("not implemented")
    form = dbc.FormGroup(
        [
            dbc.Label(state.capitalize(), className="mr-2"),
            dbc.Select(
                options=[{'label' : option, 'value' : option} for option in ["True", "False"]],
                id= prefix + state,
                value=str(value),
            ),
        ],
        className='mr-3',
    )

    return NotImplemented #form

def make_condition_impact_layout(impacts, prefix=""):

    forms = []

    for condition, impact in impacts.items():
        condition_prefix = "%s_%s_" %(prefix, condition)

        condition_form = dbc.InputGroup(
            [
                dbc.InputGroupAddon(dbc.Checkbox(), addon_type="prepend"),
                dbc.InputGroupAddon(condition, addon_type="prepend"),
                make_condition_impact_form(impact, prefix=condition_prefix),
            ]
        )
        
        forms = forms + [condition_form]

    layout = dbc.Form(forms)

    return layout

def make_condition_impact_form(impact, prefix="", detail='simple'):
    """Create a form for impact condition with method, axis and target
    """

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

    form = dbc.Form([target, method, axis], inline=True, className="dash-bootstrap")

    return form


"""
                dbc.FormGroup(
                    [
                        dbc.Col(
                                dbc.Input(
                                type="number",
                                id= prefix + "range-slider-lower",
                                value=value,
                            ),
                            width=2
                        ),
                        dbc.Col(
                            dcc.RangeSlider(
                                id = prefix + 'range-slider',
                                min=0,
                                max=100,
                                step=1,
                                value =[treshold['lower'], treshold['upper']]
                            ),
                            width=8
                        ),
                        dbc.Col(
                            dbc.Input(
                                type="number",
                                id= prefix + "range-slider-upper",
                                value=value,
                            ),
                            width=2
                        ),
                    ],
"""

if __name__ == "__main__":
    print("layout methods - Ok")