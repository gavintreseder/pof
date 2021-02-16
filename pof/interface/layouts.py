"""
Generates a layout for the project

    Systems
        name
        Component(s)

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

from pof import Component, FailureMode, Task
from pof.system import System
from pof.units import valid_units
from config import config
from statistics import mean
from datetime import datetime

cf_layouts = config["Layouts"]
cf_comp = config["Component"]
cf_sys = config["System"]
cf_main = config["Main"]
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
SCALING = config["Scaling"]
IS_OPEN = cf_layouts.get("collapse_open")


# *************** Main ******************************


def scale_input(
    attr: str, value: float, units_before: str = None, units_after: str = None
) -> float:

    # TODO combine this with update and set methods
    if attr in SCALING:
        value = value / SCALING.get(attr, 1)

    if units_after is not None:
        if units_before is not None:

            if attr in TIME_VARS:
                ratio = valid_units.get(units_after) / valid_units.get(units_before)
                value = value * ratio

        else:
            logging.warning("Please set units for the pof_obj before scaling")

    return value


def make_layout(system):
    # Dropdown list values
    y_values_ms = ["cost", "cost_cumulative"]
    update_list_y_ms = [{"label": option, "value": option} for option in y_values_ms]
    y_value_default = cf_layouts.get("y_value_default")

    update_list_unit = [{"label": option, "value": option} for option in valid_units]
    unit_default = cf_main.get("input_units_default")
    unit_default_model = cf_main.get("model_units_default")

    if cf_main.get("system"):
        comp_list = [
            {"label": comp.name, "value": comp.name}
            for comp in system.comp.values()
            if comp.active
        ]
    else:
        comp_list = [{"label": cf_main.get("name"), "value": cf_main.get("name")}]

    comp_default = comp_list[0]["value"]

    # Generate row data
    inputs = make_input_section(
        update_list_unit=update_list_unit,
        unit_default=unit_default,
        unit_default_model=unit_default_model,
        comp_list=comp_list,
        comp_default=comp_default,
    )

    sim_progress = make_sim_progress()
    sim_sens_progress = make_sim_sens_progress()

    sim_inputs = make_sim_inputs(
        update_list_y=update_list_y_ms, y_value_default=y_value_default
    )
    sim_sens_inputs = make_sim_sens_inputs(system, comp_name=comp_default)

    if cf_main.get("system"):
        make_chart_layout = make_system_layout(system)
    else:
        make_chart_layout = make_component_layout(system)

    sim = make_sim_layout()

    file_input = make_file_name_input()
    save_button = make_save_button()
    load_button = make_load_button()

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
                            dbc.Row([file_input]),
                            dbc.Row([html.Div(id="save_button", children=save_button)]),
                            dbc.Row([html.Div(id="load_button", children=load_button)]),
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
                            dbc.Row(
                                [
                                    html.Div(
                                        id="sim_sens_input", children=sim_sens_inputs
                                    )
                                ]
                            ),
                        ]
                    ),
                ]
            ),
            html.Div(id="param_layout", children=make_chart_layout),
            sim,
        ]
    )

    return layouts


# ******************* Validation ******************


def make_pof_obj_layout(pof_obj, prefix="", sep="-"):
    if isinstance(pof_obj, System):
        layout = make_system_layout(pof_obj, prefix, sep)
    elif isinstance(pof_obj, Component):
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


# ****************** System **********************


def make_system_layout(system, prefix="", sep="-"):
    """"""

    prefix = prefix + system.name + sep

    # Get the component layout
    comp_layout = []
    for comp in system.comp.values():
        comp_layout = comp_layout + [
            make_component_layout(comp, prefix=prefix + "comp" + sep, sep=sep)
        ]

    layout = dbc.InputGroup(
        [
            dbc.InputGroupAddon(
                dbc.Checkbox(
                    id=prefix + "active", checked=system.active, disabled=True
                ),
                addon_type="prepend",
            ),
            dbc.Button(
                system.name,
                color="link",
                id=prefix + "collapse-button",
            ),
            dbc.Col(
                [
                    dbc.Row(
                        dbc.Collapse(
                            dbc.Card(dbc.CardBody(dbc.Row(comp_layout))),
                            id=prefix + "collapse",
                            is_open=IS_OPEN,
                        )
                    )
                ]
            ),
        ]
    )

    return layout


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

    # Make the consequence layout
    consequence_input = make_consequence_input(component, prefix=prefix)

    # Make the indicator layout
    ind_inputs = make_indicator_inputs_form(component, prefix=prefix)

    # Make the layout
    layout = dbc.InputGroup(
        [
            dbc.InputGroupAddon(
                dbc.Checkbox(id=prefix + "active", checked=component.active),
                addon_type="prepend",
            ),
            dbc.Button(
                component.name,
                color="link",
                id=prefix + "collapse-button",
            ),
            dbc.Col(
                [
                    dbc.Row(
                        [
                            dbc.Collapse(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            dbc.Row(fms_layout),
                                            dbc.Row(consequence_input),
                                            dbc.Row(ind_inputs),
                                        ]
                                    )
                                ),
                                id=prefix + "collapse",
                                is_open=IS_OPEN,
                            ),
                        ]
                    )
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
                        step="any",
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
                        step="any",
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
                            value=task.p_effective * SCALING.get("p_effective"),
                            min=0,
                            max=100,
                            debounce=True,
                            style={"width": 100},
                            step="any",
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
                        dbc.Checkbox(
                            id=cond_prefix + "active", checked=True, disabled=True
                        ),
                    ],
                    addon_type="prepend",
                ),
                dbc.FormGroup(
                    [
                        dbc.Label(condition.capitalize(), className="mr-2"),
                        dbc.Input(
                            type="number",
                            id=cond_prefix + "lower",
                            value=threshold["lower"],
                            min=0,
                            debounce=True,
                            step="any",
                        ),
                        dbc.Input(
                            type="number",
                            id=cond_prefix + "upper",
                            value=threshold["upper"],
                            min=0,
                            debounce=True,
                            step="any",
                        ),
                    ],
                ),
            ]
        )
        condition_inputs = condition_inputs + [condition_input]

    layout = dbc.Form(children=condition_inputs, inline=True, className="mr-3")

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
                                id=prefix + state + sep + "active",
                                checked=True,
                                disabled=True,
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
                    dbc.Checkbox(id=prefix + "active", checked=True, disabled=True),
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
def make_input_section(
    update_list_unit, unit_default, unit_default_model, comp_list, comp_default
):
    form = dbc.FormGroup(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Checkbox(
                                id="axis_lock-checkbox",
                                checked=cf_layouts.get("axis_lock"),
                            ),
                            "Axis Lock",
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
                                value=unit_default_model,
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
                                        value=100,  # TODO move this to config.toml
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
            dbc.Row(
                [
                    dbc.Col(
                        [
                            "Graph Component",
                            dcc.Dropdown(
                                id="comp_graph-dropdown",
                                options=comp_list,
                                value=comp_default,
                            ),
                        ]
                    ),
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
    form = dbc.InputGroup(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.InputGroupAddon(
                                [
                                    "y-axis options",
                                    dcc.Dropdown(
                                        id="ms_var_y-dropdown",
                                        options=update_list_y,
                                        value=y_value_default,
                                        style={"width": 500},
                                    ),
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
    )

    return form


def make_sim_sens_inputs(system, comp_name=None):

    if cf_main.get("system"):
        update_list_sens_x = [
            {"label": option, "value": option}
            for option in system.get_update_ids(numericalOnly=True, comp_name=comp_name)
        ]
    else:
        update_list_sens_x = [
            {"label": option, "value": option}
            for option in system.get_update_ids(numericalOnly=True)
        ]
    sens_x_default = update_list_sens_x[0]["value"]

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

    y_value_default = cf_layouts.get("y_value_default")

    return make_sim_sens_inputs_layout(
        update_list_y=update_list_y_sens,
        y_value_default=y_value_default,
        update_list_sens_x=update_list_sens_x,
        sens_x_default=sens_x_default,
    )


def make_sim_sens_inputs_layout(
    update_list_y, y_value_default, update_list_sens_x, sens_x_default
):
    form = dbc.InputGroup(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.InputGroupAddon(
                                [
                                    "y-axis options",
                                    dcc.Dropdown(
                                        id="sens_var_y-dropdown",
                                        options=update_list_y,
                                        value=y_value_default,
                                        style={"width": 500},
                                    ),
                                ],
                                addon_type="prepend",
                            ),
                        ]
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.InputGroupAddon(
                                [
                                    "x-axis options",
                                    dcc.Dropdown(
                                        id="sens_var_id-dropdown",
                                        options=update_list_sens_x,
                                        value=sens_x_default,
                                        style={"width": 500},
                                    ),
                                ],
                                addon_type="prepend",
                            )
                        ]
                    )
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


def make_file_name_input():
    layout = dbc.InputGroupAddon(
        [
            "File Name",
            dcc.Input(
                id="file_name-input",
                value=cf_main.get("file_name_default"),
                type="text",
                style={"width": 500},
                debounce=True,
            ),
        ],
        addon_type="prepend",
    )

    return layout


def make_save_button():
    layout = html.Div(
        [
            dbc.Button(
                "Save Model",
                color="secondary",
                outline=True,
                id="save-button",
                className="save",
            ),
            dbc.Label(
                "Error " + str(datetime.now()),
                id="save_error-input",
                hidden=True,
            ),
            dbc.Label(
                "Success " + str(datetime.now()),
                id="save_success-input",
                hidden=True,
            ),
        ]
    )

    return layout


def make_load_button():
    layout = html.Div(
        [
            dbc.Button(
                "Load Model",
                color="secondary",
                outline=True,
                id="load-button",
                className="save",
            ),
            dbc.Label(
                "Error " + str(datetime.now()),
                id="load_error-input",
                hidden=True,
            ),
            dbc.Label(
                "Success " + str(datetime.now()),
                id="load_success-input",
                hidden=True,
            ),
        ]
    )

    return layout


# *************** Indicator & Consequence Inputs ************
def make_indicator_inputs_form(component, prefix=""):

    ind_inputs = []
    for ind in component.indicator.values():
        ind = ind.name
        name = str(ind).replace("_", " ").capitalize()
        ind_input = dbc.InputGroup(
            [
                dbc.Col([dbc.InputGroupAddon(name, addon_type="prepend")]),
                make_param_inputs(component, ind, prefix=prefix),
            ],
        )
        ind_inputs = ind_inputs + [ind_input]

    layout = dbc.InputGroup(
        [
            dbc.Button(
                "Indicator inputs",
                color="link",
                id=prefix + "indicator-collapse-button",
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
                id=prefix + "indicator-collapse",
                is_open=IS_OPEN,
            ),
        ]
    )

    return layout


def make_param_inputs(component, ind, prefix="", sep="-"):
    param_inputs = []
    params = cf_comp.get("indicator_input_fields")

    for param in params:
        param_input = dbc.InputGroup(
            [
                dbc.InputGroupAddon(param.capitalize(), className="I1"),
                dbc.Input(
                    type="number",
                    id=prefix + "indicator" + sep + ind + sep + param,
                    value=getattr(component.indicator[ind], param),
                    debounce=True,
                    style={"width": 100},
                ),
            ]
        )
        param_inputs = param_inputs + [param_input]

    form = dbc.Form(param_inputs, inline=True)

    return form


def make_consequence_input(component, prefix="", sep="-"):
    vals = []
    for fm in component.fm.values():
        vals.append(getattr(fm.consequence, "cost"))

    layout = dbc.InputGroup(
        [
            dbc.Button(
                "Consequence input",
                color="link",
                id=prefix + "consequence-collapse-button",
            ),
            dbc.Collapse(
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                dbc.Input(
                                    type="number",
                                    id=prefix + "consequence" + sep + "cost",
                                    value=mean(vals),
                                    debounce=True,
                                    style={"width": 100},
                                ),
                            ]
                        )
                    ]
                ),
                id=prefix + "consequence-collapse",
                is_open=IS_OPEN,
            ),
        ]
    )

    return layout


if __name__ == "__main__":
    print("layout methods - Ok")