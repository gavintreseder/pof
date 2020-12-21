"""
Generates a layout for the project

    Components
        name
        Failure Mode(s)
            ...

    Failure Mode
        name
        Distribution(s)
            Untreated only
        Task(s)

    Task
        name
        Probability Effective
        Cost
        Trigger
            State(s)
                name
                state (True/False)
            Condition(s)
                condition_name
                lower_threshold
                upper_threshold
        Impacts
            State(s)
                name
                state (True/False)
            Condition(s)
                name
                method
                axis
                reduction_factor / target

# Condition
# Task Groups ( move to assets)
"""

import logging

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from pof.interface.cfg import Config as cf
from pof import Component, FailureMode, Task
from pof.units import valid_units
from config import config

SCALING = cf.scaling
defaults = config["Layouts"]
TIME_VARS = [
    "t_end",
    "pf_interval",
    "pf_std",
    "t_delay",
    "t_interval",
    "alpha",
    "beta",
    "gamma",
]

SCALING = {"p_effective": 100}
IS_OPEN = defaults.get("collapse_open")


# *************** Main ******************************


def scale_input(pof_obj, attr: str, value: float, units: str = None) -> float:

    # TODO combine this with update and set methods
    if attr in SCALING:
        value = value / SCALING.get(attr, 1)

    if units is not None:
        if pof_obj.units is not None:

            if attr in TIME_VARS:
                ratio = valid_units.get(units) / valid_units.get(pof_obj.units)
                value = value * ratio

        else:
            logging.warning("Please set units for the pof_obj before scaling")

    return value


def make_layout(comp):
    # Dropdown list values
    update_list_sens_x = [
        {"label": option, "value": option}
        for option in comp.get_update_ids(numericalOnly=True)
    ]
    sens_x_default = comp.get_update_ids(numericalOnly=True)[0]

    y_values_sens = [
        "cost",
        "cost_cumulative",
        "cost_annual",
        "quantity",
        "quantity_cumulative",
    ]
    update_list_y_sens = [
        {"label": option, "value": option} for option in y_values_sens
    ]
    y_values_ms = ["cost", "cost_cumulative"]
    update_list_y_ms = [{"label": option, "value": option} for option in y_values_ms]
    y_value_default = defaults.get("y_value_default")

    update_list_unit = [{"label": option, "value": option} for option in valid_units]
    unit_default = "years"

    # Generate row data
    inputs = make_input_component(
        update_list_unit=update_list_unit, unit_default=unit_default
    )

    sim_progress = make_sim_progress()
    sim_sens_progress = make_sim_sens_progress()

    sim_inputs = make_sim_inputs(
        update_list_y=update_list_y_ms, y_value_default=y_value_default
    )
    sim_sens_inputs = make_sim_sens_inputs(
        update_list_y=update_list_y_sens,
        y_value_default=y_value_default,
        update_list_sens_x=update_list_sens_x,
        sens_x_default=sens_x_default,
    )

    mcl = make_component_layout(comp)
    sim = make_sim_layout()

    indicators_default = comp.get_ind_default_values()
    ind_inputs = make_indicator_inputs_form(indicators=indicators_default)
    consequence_default = comp.get_consequence_default()
    consequence_input = make_consequence_input(consequence_default=consequence_default)

    # Make layout
    layouts = html.Div(
        [
            html.Div(id="log"),
            html.Div(children=None, id="graph_limits"),
            dbc.Row(
                [
                    dbc.Col(
                        [inputs],
                    ),
                    dbc.Col(
                        [
                            dbc.Row([consequence_input]),
                            dbc.Row([ind_inputs]),
                        ]
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="pof-fig")),
                    dbc.Col(dcc.Graph(id="cond-fig")),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="ms-fig")),
                    dbc.Col(dcc.Graph(id="sensitivity-fig")),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="task_forecast-fig")),
                    dbc.Col(dcc.Graph(id="forecast_table-fig")),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Row(
                                [sim_progress],
                            ),
                            dbc.Row(
                                [sim_inputs],
                            ),
                        ]
                    ),
                    dbc.Col(
                        [
                            dbc.Row(
                                [sim_sens_progress],
                            ),
                            dbc.Row([sim_sens_inputs]),
                        ]
                    ),
                ]
            ),
            mcl,
            sim,
        ]
    )

    return layouts


# ******************* Validation ******************


def make_pof_obf_layout(pof_obj, prefix="", sep="-"):
    if isinstance(pof_obj, Component):
        layout = make_component_layout(pof_obj, prefix, sep)
    elif isinstance(pof_obj, FailureMode):
        layout = make_failure_mode_layout

    validate_layout(pof_obj, layout)

    return layout


# ******************* Validation ******************


def validate_layout(pof_obj, layout):

    objs = pof_obj.get_objects()
    collapse = [id + "-collapse" for id in objs if id + "-collapse" not in layout]
    collapse_button = [
        id + "-collapse-button" for id in objs if id + "-collapse-button" not in layout
    ]

    params = pof_obj.get_dash_ids(numericalOnly=False)

    layout_objects = collapse + collapse_button + params

    missing_objects = [obj for obj in layout_objects if obj not in layout]

    if missing_objects:
        logging.info("Missing objects from layout - %s", missing_objects)
        return False
    else:
        return True


# ******************* Component ******************


def make_component_layout(component, prefix="", sep="-"):
    """"""

    prefix = prefix + component.name + sep

    # Get tasks layout
    fms_layout = []
    for fm in component.fm.values():
        fms_layout = fms_layout + [
            make_failure_mode_layout(fm, prefix=prefix + "fm" + sep, sep=sep)
        ]

    # Make the layout
    layout = dbc.InputGroup(
        [
            dbc.InputGroupAddon(
                dbc.Checkbox(
                    id=prefix + "active", checked=component.active, disabled=True
                ),
                addon_type="prepend",
            ),
            dbc.Button(
                component.name,
                color="link",
                id=prefix + "collapse-button",
            ),
            dbc.Col(
                [
                    dbc.Collapse(
                        dbc.Card(dbc.CardBody(fms_layout)),
                        id=prefix + "collapse",
                        is_open=IS_OPEN,
                    ),
                ]
            ),
        ]
    )

    return layout


def make_failure_mode_layout(fm, prefix="", sep="-"):
    """"""

    prefix = prefix + fm.name + sep

    # Get failure mode form
    cond_inputs = make_cond_form_inputs(fm, prefix=prefix, sep=sep)

    dist_inputs = make_dist_form_inputs(fm.untreated, prefix=prefix, sep=sep)

    fm_form = dbc.Form(children=dist_inputs + cond_inputs, inline=True)

    # Get tasks layout
    tasks_layout = []
    for task in fm.tasks.values():
        tasks_layout.append(
            make_task_layout(task, prefix=prefix + "tasks" + sep, sep=sep)
        )

    # Make the layout
    layout = dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.InputGroupAddon(
                            dbc.Checkbox(id=prefix + "active", checked=fm.active),
                            addon_type="prepend",
                        ),
                        dbc.Button(
                            fm.name,
                            color="link",
                            id=prefix + "collapse-button",
                            style={"width": 200},
                        ),
                        dbc.Col(
                            [
                                fm_form,
                                dbc.Collapse(
                                    dbc.Card(dbc.CardBody(tasks_layout)),
                                    id=prefix + "collapse",
                                    is_open=IS_OPEN,
                                ),
                            ]
                        ),
                    ]
                )
            ]
        )
    )

    return layout


def make_dist_form_inputs(dist, prefix="", sep="-"):
    """
    Takes a Distribution and generates the html form inputs
    """

    prefix = prefix + "dists" + sep + dist.name + sep

    param_inputs = []

    for param, value in dist.params().items():
        param_input = dbc.Col(
            dbc.FormGroup(
                [
                    dbc.Label(param.capitalize(), className="mr-2"),
                    dbc.Input(
                        type="number",
                        id=prefix + param,
                        value=value,
                        debounce=True,
                        style={"width": 100},
                    ),
                ],
                className="mr-3",
            ),
        )
        param_inputs.append(param_input)

    return param_inputs


def make_cond_form_inputs(condition, prefix="", sep="-"):  # Not used
    """
    Takes a Condition and generates the form inputs
    """
    # TODO fix this up when condition is rewritten
    # prefix = prefix + 'Condition' + sep + condition.name + sep

    # Dropdown format for pf curve
    pf_curve = dbc.Col(
        dbc.FormGroup(
            [
                dbc.Label("PF Curve", className="mr-2"),
                dbc.Select(
                    options=[
                        {"label": option, "value": option}
                        for option in condition.PF_CURVES
                    ],
                    id=prefix + "pf_curve",
                    value=str(condition.pf_curve),
                ),
            ],
            className="mr-3",
        ),
    )

    # Input format for the others
    param_inputs = []
    for param in ["pf_interval", "pf_std"]:
        param_input = dbc.Col(
            dbc.FormGroup(
                [
                    dbc.Label(param.capitalize(), className="mr-2"),
                    dbc.Input(
                        type="number",
                        id=prefix + param,
                        value=getattr(condition, param),
                        min=0,
                        debounce=True,
                        style={"width": 100},
                    ),
                ],
                className="mr-3",
            ),
        )
        param_inputs.append(param_input)

    param_inputs.insert(0, pf_curve)

    return param_inputs


def make_task_layout(task, prefix="", sep="-"):
    """"""
    prefix = prefix + task.name + sep

    task_form = make_task_form(task=task, prefix=prefix)
    trigger_layout = make_task_trigger_layout(task.triggers, prefix=prefix)
    impact_layout = make_task_impact_layout(task.impacts, prefix=prefix)

    task_layout = dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.InputGroupAddon(
                            dbc.Checkbox(id=prefix + "active", checked=task.active),
                            addon_type="prepend",
                        ),
                        dbc.Button(
                            task.name,
                            color="link",
                            id=prefix + "collapse-button",
                            style={"width": 200},
                        ),
                        dbc.Col(
                            [
                                task_form,
                                dbc.Collapse(
                                    dbc.CardDeck(
                                        [
                                            trigger_layout,
                                            impact_layout,
                                        ]
                                    ),
                                    id=prefix + "collapse",
                                    is_open=IS_OPEN,
                                ),
                            ]
                        ),
                    ]
                )
            ]
        )
    )

    return task_layout


def make_task_form(task, prefix="", sep="-"):  # TODO make this better
    """
    Takes a Task and generates the html form inputs
    """
    var_dict = dict(t_delay="Time Delay", t_interval="Time Interval")
    time_details = []

    for key in var_dict:
        if task.trigger == "time":
            time_details.append(
                [
                    dbc.Label(var_dict[key]),
                    dbc.Input(
                        type="number",
                        id=prefix + key,
                        value=getattr(task, key),
                        min=0,
                        debounce=True,
                        style={"width": 100},
                    ),
                ],
            )
        else:
            time_details.append([])

    form = dbc.Form(
        [
            dbc.Col(
                dbc.InputGroup(
                    [
                        dbc.Label("Prob Effective", html_for="p_effective"),
                        dbc.Input(
                            type="number",
                            id=prefix + "p_effective",
                            value=task.p_effective * SCALING["p_effective"],
                            min=0,
                            max=100,
                            debounce=True,
                            style={"width": 100},
                        ),
                        dbc.InputGroupAddon("%", addon_type="append"),
                    ],
                ),
            ),
            dbc.Col(
                dbc.FormGroup(
                    [
                        dbc.Label("Cost", html_for="cost"),
                        dbc.Input(
                            type="number",
                            id=prefix + "cost",
                            value=task.cost,
                            min=0,
                            debounce=True,
                            style={"width": 100},
                        ),
                    ],
                ),
            ),
            dbc.Col(
                dbc.FormGroup(time_details[0]),
            ),
            dbc.Col(
                dbc.FormGroup(time_details[1]),
            ),
        ],
        inline=True,
    )

    return form


# ******************* Triggers ******************


def make_task_trigger_layout(triggers, prefix="", sep="-"):
    """
    Takes a Trigger and generates the html form inputs
    """
    prefix = prefix + "trigger" + sep
    state_layout = make_state_impact_layout(triggers["state"], prefix=prefix)
    condition_layout = make_condition_trigger_layout(
        triggers["condition"], prefix=prefix
    )

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


def make_condition_trigger_layout(
    triggers, prefix="", sep="-"
):  # TODO make these sliders
    """R"""
    condition_inputs = []

    for condition, threshold in triggers.items():

        cond_prefix = prefix + "condition" + sep + condition + sep
        condition_input = dbc.InputGroup(
            [
                dbc.InputGroupAddon(
                    [
                        dbc.Checkbox(id=cond_prefix + "active", checked=True),
                    ],
                    addon_type="prepend",
                ),
                dbc.Label(condition.capitalize(), className="mr-2"),
                dbc.Form(
                    [
                        dbc.Input(
                            type="number",
                            id=cond_prefix + "lower",
                            value=threshold["lower"],
                            min=0,
                            debounce=True,
                        ),
                        dbc.Input(
                            type="number",
                            id=cond_prefix + "upper",
                            value=threshold["upper"],
                            min=0,
                            debounce=True,
                        ),
                    ],
                    inline=True,
                ),
            ]
        )
        condition_inputs = condition_inputs + [condition_input]

    layout = dbc.Form(children=condition_inputs)

    return layout


# ****************** Impacts *************************


def make_task_impact_layout(impacts, prefix="", sep="-"):
    """Takes a the impacts from a Trigger object and generates the html form inputs
    # TODO Implement times
    """
    prefix = prefix + "impact" + sep
    state_layout = make_state_impact_layout(impacts["state"], prefix=prefix)
    condition_layout = make_condition_impact_layout(impacts["condition"], prefix=prefix)
    system_impact_layout = make_system_impact_layout(impacts["system"], prefix=prefix)

    layout = dbc.Card(
        [
            dbc.CardHeader("Impacts"),
            dbc.CardBody(
                [
                    state_layout,
                    condition_layout,
                    system_impact_layout,
                ],
            ),
        ],
    )

    return layout


def make_state_impact_layout(state_impacts, prefix="", sep="-"):
    """Creates an input form the state impacts"""

    # TODO figure out how to take True or False as vlaues
    state_inputs = []
    prefix = prefix + "state" + sep
    for state, value in state_impacts.items():
        if state == "failure":
            state_updated = "Functional Failure"
        else:
            state_updated = state

        state_input = dbc.Col(
            dbc.InputGroup(
                [
                    dbc.InputGroupAddon(
                        [
                            dbc.Checkbox(
                                id=prefix + state + sep + "active", checked=True
                            ),
                            dbc.Label(state_updated.capitalize(), className="mr-2"),
                        ],
                        addon_type="prepend",
                    ),
                    dbc.Select(
                        options=[
                            {"label": option, "value": option}
                            for option in ["True", "False"]
                        ],
                        id=prefix + state,
                        value=str(value),
                    ),
                ],
                className="mr-3",
            ),
        )
        state_inputs = state_inputs + [state_input]

    form = dbc.Form(children=state_inputs, inline=True)

    return form


def make_condition_impact_layout(impacts, prefix="", sep="-"):

    forms = []
    base_prefix = prefix
    for condition, impact in impacts.items():
        prefix = base_prefix + "condition" + sep + condition + sep

        condition_form = dbc.InputGroup(
            [
                dbc.InputGroupAddon(
                    dbc.Checkbox(id=prefix + "active", checked=True),
                    addon_type="prepend",
                ),
                dbc.InputGroupAddon(condition, addon_type="prepend"),
                make_condition_impact_form(impact, prefix=prefix),
            ]
        )

        forms = forms + [condition_form]

    layout = dbc.Form(forms)

    return layout


def make_system_impact_layout(impact, prefix="", sep="-"):

    layout = dbc.FormGroup(
        [
            dbc.Label("System", className="mr-2"),
            dbc.Select(
                id=prefix + "system",
                options=[
                    {"label": system, "value": system} for system in Task.SYSTEM_IMPACT
                ],
                value=impact,
            ),
        ],
        className="mr-3",
    )

    return layout


def make_condition_impact_form(impact, prefix="", sep="-"):
    """Create a form for impact condition with method, axis and target"""

    target = dbc.FormGroup(
        [
            dbc.Label("Target", className="mr-2"),
            dbc.Input(
                type="number",
                id=prefix + "target",
                value=impact["target"],
                min=0,
                debounce=True,
            ),
        ],
        className="mr-3",
    )

    method = dbc.FormGroup(
        [
            dbc.Label("Method", className="mr-2"),
            dbc.Select(
                id=prefix + "method",
                options=[
                    {"label": method, "value": method}
                    for method in ["reduction_factor", "tbc"]
                ],
                value=impact["method"],
            ),
        ],
        className="mr-3",
    )

    axis = dbc.FormGroup(
        [
            dbc.Label("Axis", className="mr-2"),
            dbc.Select(
                id=prefix + "axis",
                options=[
                    {"label": axis, "value": axis} for axis in ["condition", "time"]
                ],
                value=impact["axis"],
            ),
        ],
        className="mr-3",
    )

    form = dbc.Form([target, method, axis], inline=True, className="dash-bootstrap")

    return form


# *******************Sim meta data***********************
def make_input_component(update_list_unit, unit_default):
    form = dbc.FormGroup(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            "Axis Lock",
                            dbc.Checkbox(
                                id="axis_lock-checkbox",
                                checked=defaults.get("axis_lock"),
                            ),
                        ]
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            "Model Units",
                            dcc.Dropdown(
                                id="model_units-dropdown",
                                options=update_list_unit,
                                value=unit_default,
                            ),
                        ]
                    ),
                    dbc.Col(
                        [
                            "Input Units",
                            dcc.Dropdown(
                                id="input_units-dropdown",
                                options=update_list_unit,
                                value=unit_default,
                            ),
                        ]
                    ),
                    dbc.Col(),
                    dbc.Col(),
                    dbc.Col(),
                    dbc.Col(),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.InputGroupAddon(
                                [
                                    "t_end",
                                    dcc.Input(
                                        id="t_end-input",
                                        value=200,
                                        type="number",
                                        style={"width": 100},
                                    ),
                                ],
                                addon_type="prepend",
                            ),
                        ],
                    ),
                    dbc.Col(),
                    dbc.Col(),
                    dbc.Col(),
                    dbc.Col(),
                    dbc.Col(),
                ]
            ),
        ]
    )

    return form


def make_sim_progress():
    layout = dbc.InputGroup(
        [
            dcc.Interval(id="progress-interval", n_intervals=0, interval=100),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.InputGroupAddon(
                                [
                                    dbc.Checkbox(
                                        id="sim_n_active",
                                        checked=True,
                                    ),
                                    "Iterations",
                                    dcc.Input(
                                        id="n_iterations-input",
                                        value=10,
                                        type="number",
                                        style={"width": 100},
                                    ),
                                ],
                                addon_type="prepend",
                            )
                        ],
                        width="auto",
                    ),
                ]
            ),
            dbc.Col([dbc.Progress(id="n-progress")]),
        ]
    )

    return layout


def make_sim_sens_progress():
    layout = dbc.InputGroup(
        [
            dcc.Interval(id="sens_progress-interval", n_intervals=0, interval=100),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.InputGroupAddon(
                                [
                                    dbc.Checkbox(
                                        id="sim_sens_active-check",
                                        checked=False,
                                    ),
                                    "Iterations",
                                    dcc.Input(
                                        id="n_sens_iterations-input",
                                        value=10,
                                        type="number",
                                        style={"width": 100},
                                    ),
                                ],
                                addon_type="prepend",
                            ),
                        ],
                        width="auto",
                    ),
                ]
            ),
            dbc.Col([dbc.Progress(id="n_sens-progress")]),
        ]
    )

    return layout


def make_sim_inputs(update_list_y, y_value_default):
    form = dbc.FormGroup(
        [
            dbc.Row(
                [
                    "y-axis options",
                    dcc.Dropdown(
                        id="ms_var_y-dropdown",
                        options=update_list_y,
                        value=y_value_default,
                    ),
                ]
            )
        ]
    )

    return form


def make_sim_sens_inputs(
    update_list_y, y_value_default, update_list_sens_x, sens_x_default
):
    form = dbc.FormGroup(
        [
            dbc.Row(
                [
                    "y-axis options",
                    dcc.Dropdown(
                        id="sens_var_y-dropdown",
                        options=update_list_y,
                        value=y_value_default,
                    ),
                ]
            ),
            dbc.Row(
                [
                    "x-axis options",
                    dcc.Dropdown(
                        id="sens_var_id-dropdown",
                        options=update_list_sens_x,
                        value=sens_x_default,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.InputGroupAddon(
                                [
                                    "Lower",
                                    dcc.Input(
                                        id="sens_lower-input",
                                        value=0,
                                        type="number",
                                        style={"width": 100},
                                        debounce=True,
                                    ),
                                ],
                                addon_type="prepend",
                            ),
                        ],
                    ),
                    dbc.Col(
                        [
                            dbc.InputGroupAddon(
                                [
                                    "Upper",
                                    dcc.Input(
                                        id="sens_upper-input",
                                        value=10,
                                        type="number",
                                        style={"width": 100},
                                        debounce=True,
                                    ),
                                ],
                                addon_type="prepend",
                            ),
                        ],
                    ),
                    dbc.Col(
                        [
                            dbc.InputGroupAddon(
                                [
                                    "Step Size",
                                    dcc.Input(
                                        id="sens_step_size-input",
                                        value=1,
                                        type="number",
                                        style={"width": 100},
                                        debounce=True,
                                    ),
                                ],
                                addon_type="prepend",
                            ),
                        ],
                    ),
                ]
            ),
        ]
    )

    return form


def make_sim_layout():
    layout = dbc.InputGroup(
        [
            dbc.Button(
                "Sim metadata",
                color="link",
                id="sim_params-collapse-button",
            ),
            dbc.Collapse(
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                html.Div(
                                    children="Update State:",
                                    id="update_state",
                                ),
                                html.Div(
                                    [
                                        html.P(
                                            children="Sim State",
                                            id="sim_state",
                                        ),
                                        html.P(
                                            id="sim_state_err",
                                            style={"color": "red"},
                                        ),
                                    ]
                                ),
                                html.Div(
                                    children="Fig State",
                                    id="fig_state",
                                ),
                                html.P(id="ffcf"),
                            ]
                        )
                    ]
                ),
                id="sim_params-collapse",
            ),
        ]
    )

    return layout


# *************** Indicator & Consequence Inputs ************
def make_indicator_inputs_form(indicators):

    ind_inputs = []
    for key in indicators.keys():
        name = str(key).replace("_", " ").capitalize()
        ind_input = dbc.InputGroup(
            [
                dbc.Col([dbc.InputGroupAddon(name, addon_type="prepend")]),
                make_param_inputs(indicators, key),
            ],
        )
        ind_inputs = ind_inputs + [ind_input]

    layout = dbc.InputGroup(
        [
            dbc.Button(
                "Indicator inputs",
                color="link",
                id="indicator_inputs-collapse-button",
            ),
            dbc.Collapse(
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                dbc.Form(ind_inputs),
                            ]
                        )
                    ]
                ),
                id="indicator_inputs-collapse",
                is_open=IS_OPEN,
            ),
        ]
    )

    return layout


def make_param_inputs(indicators, key):
    param_inputs = []
    for param, value in indicators[key].items():
        param_input = dbc.InputGroup(
            [
                dbc.InputGroupAddon(param.capitalize(), className="I1"),
                dbc.Input(
                    type="number",
                    id=key + "_" + param + "_input",
                    value=value,
                    debounce=True,
                    style={"width": 100},
                ),
            ]
        )
        param_inputs = param_inputs + [param_input]

    form = dbc.Form(param_inputs, inline=True)

    return form


def make_consequence_input(consequence_default):
    layout = dbc.InputGroup(
        [
            dbc.Button(
                "Consequence input",
                color="link",
                id="consequence_input-collapse-button",
            ),
            dbc.Collapse(
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                dbc.Input(
                                    type="number",
                                    id="consequence_input",
                                    value=consequence_default,
                                    debounce=True,
                                    style={"width": 100},
                                ),
                            ]
                        )
                    ]
                ),
                id="consequence_input-collapse",
                is_open=IS_OPEN,
            ),
        ]
    )

    return layout


if __name__ == "__main__":
    print("layout methods - Ok")