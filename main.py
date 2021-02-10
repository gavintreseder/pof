import copy
import os
import logging

import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from config import config
from pof import Component
from pof.units import valid_units
from pof.loader.asset_model_loader import AssetModelLoader
from pof.interface.dashlogger import DashLogger
from pof.interface.layouts import (
    make_layout,
    scale_input,
    make_component_layout,
    make_save_button,
    make_load_button,
)

from pof.data.asset_data import SimpleFleet
from pof.paths import Paths

cf = config["Main"]

# Turn off logging level to speed up implementation
logging.getLogger().setLevel(logging.INFO)
logging.getLogger("werkzeug").setLevel(logging.WARNING)

# Build App
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

# ========================================================
# Load the data sets.
# ========================================================

# Forecast years
START_YEAR = 2015
END_YEAR = 2024
CURRENT_YEAR = 2020

paths = Paths()

# Population Data
file_path = paths.input_path + os.sep
FILE_NAME = r"population_summary.csv"

sfd = SimpleFleet(file_path + FILE_NAME)
sfd.load()
sfd.calc_age_forecast(START_YEAR, END_YEAR, CURRENT_YEAR)

global comp
global pof_sim
global sens_sim
global collapse_ids
global sim_triggers
global param_inputs

aml = AssetModelLoader(paths.demo_path + os.sep + "Asset Model - Pole - Timber.xlsx")
comp_data = aml.load(paths.demo_path + os.sep + "Asset Model - Pole - Timber.xlsx")
comp = Component.from_dict(comp_data["pole"])
comp.units = cf.get("file_units_default")  # Set the units to be file units initially
comp.fleet_data = sfd  # TODO fix by creating asset class

# Instantiate global variables
pof_sim = copy.copy(comp)
sens_sim = copy.copy(comp)
collapse_ids = comp.get_objects()
collapse_ids.append("sim_params")
sim_triggers = comp.get_dash_ids(numericalOnly=False)
param_inputs = [
    Input(dash_id, "checked") if "active" in dash_id else Input(dash_id, "value")
    for dash_id in sim_triggers
]

# Layout
app.layout = make_layout(comp)


@app.callback(
    Output("param_layout", "children"),
    Output("load_button", "children"),
    Output("load_error-input", "hidden"),
    Output("load_success-input", "hidden"),
    Output("save_button", "children"),
    Output("save_error-input", "hidden"),
    Output("save_success-input", "hidden"),
    Output("file_name-input", "value"),
    Input("load-button", "n_clicks"),
    Input("save-button", "n_clicks"),
    State("file_name-input", "value"),
)
def save_load_file(click_load, click_save, file_name_input):
    """ Load the data of the user input file """

    # TODO change hide to just set the text to be something, i.e. error_msg = "" when things are ok, error_msg = 'there's a mistake'
    # Think about what happens if you want different error messages long term?

    global comp
    global pof_sim
    global sens_sim
    global collapse_ids
    global sim_triggers
    global param_inputs

    load_error_hide = True
    load_success_hide = True
    save_error_hide = True
    save_success_hide = True

    # Save function
    if click_save:
        try:
            # If file_name_new ends in xlsx, replace with json
            if file_name_input[-4:] == "xlsx":
                file_name_json = file_name_input[: len(file_name_input) - 4] + "json"
            else:
                file_name_json = file_name_input

            # Save to json
            comp.save(file_name_json, file_units=cf.get("file_units_default"))
            save_success_hide = False

        except Exception as error:
            logging.error("Error saving file", exc_info=error)
            save_error_hide = False

    # Define the path of the file to load
    if click_save:  # If the save function has been called, the file name is now a json
        file_name_output = file_name_json
        file_path_output = paths.model_path + os.sep
    else:  # Otherwise the file name is as input
        file_name_output = file_name_input
        if file_name_input == cf.get(
            "file_name_default"
        ):  # The default file is in the pof path
            file_path_output = paths.demo_path + os.sep
        else:  # All other files are saved in the model path
            file_path_output = paths.model_path + os.sep
    logging.info(file_path_output + file_name_output)

    # Load the file
    if os.path.exists(file_path_output + file_name_output):
        aml = AssetModelLoader(file_path_output + file_name_output)
        comp_data = aml.load(file_path_output + file_name_output)
        comp = Component.from_dict(comp_data["pole"])

        comp.units = cf.get(
            "file_units_default"
        )  # Set the units to be file units initially

        comp.fleet_data = sfd  # TODO fix by creating asset class

        load_success_hide = False
    else:
        load_error_hide = False

    # Redefine global variables
    pof_sim = copy.copy(comp)
    sens_sim = copy.copy(comp)

    collapse_ids = comp.get_objects()
    collapse_ids.append("sim_params")

    sim_triggers = comp.get_dash_ids(numericalOnly=False)
    param_inputs = [
        Input(dash_id, "checked") if "active" in dash_id else Input(dash_id, "value")
        for dash_id in sim_triggers
    ]

    return (
        make_component_layout(comp),
        make_load_button(),
        load_error_hide,
        load_success_hide,
        make_save_button(),
        save_error_hide,
        save_success_hide,
        file_name_output,
    )


# ========================================================
# Collapsable objects to hide information when not needed.
# ========================================================

# Get the dash ids for all the objects that have a collapse button


@app.callback(
    [Output(f"{prefix}-collapse", "is_open") for prefix in collapse_ids],
    [Input(f"{prefix}-collapse-button", "n_clicks") for prefix in collapse_ids],
    [State(f"{prefix}-collapse", "is_open") for prefix in collapse_ids],
)
def toggle_collapses(*args):
    """ Expands and collapses hidden dash components in the interface"""
    ctx = dash.callback_context

    state_id = ""
    collapse_id = (
        ctx.triggered[0]["prop_id"].split(".")[0].replace("-collapse-button", "")
    )
    if collapse_id in collapse_ids:  # TODO change to is not None

        state_id = collapse_id + "-collapse.is_open"
        ctx.states[state_id] = not ctx.states[state_id]

    is_open = tuple(ctx.states.values())

    return is_open


# ========================================================
# UPDATE --> Simulate --> Figures
#        --> Simulate --> Figures
# ========================================================

# TODO make a better unit update


@app.callback(
    Output("update_state", "children"),
    Input(
        "model_units-dropdown", "value"
    ),  # TODO Change the layout name so this updates normally
    Input("input_units-dropdown", "value"),
    param_inputs,
)
def update_parameter(model_units, input_units, *args):
    """Update a the pof object whenever an input is changes""",
    global comp

    # Check the parameters that changed
    ctx = dash.callback_context
    dash_id = None
    value = None

    # If any parameters have changed update the objecte
    if ctx.triggered:
        dash_id = ctx.triggered[0]["prop_id"].split(".")[0]
        value = ctx.triggered[0]["value"]

        if dash_id == "model_units-dropdown":
            comp.units = model_units
        elif dash_id == "input_units-dropdown":
            pass

        else:
            # Scale the value if req
            var = dash_id.split("-")[-1]
            value = scale_input(
                attr=var, value=value, units_before=comp.units, units_after=input_units
            )
            # update the model
            comp.update(dash_id, value)

    return f"Update State: {dash_id} - {value}"


@app.callback(
    Output("sim_state", "children"),
    Output("sim_state_err", "children"),
    Input("update_state", "children"),
    Input("sim_n_active", "checked"),
    Input("t_end-input", "value"),
    Input("n_iterations-input", "value"),
    Input("param_layout", "children"),
    State("input_units-dropdown", "value"),
)
def update_simulation(__, active, t_end, n_iterations, ___, input_units):
    """ Triger a simulation whenever an update is completed or the number of iterations change"""
    global comp
    global pof_sim
    global sfd

    pof_sim.cancel_sim()

    # time.sleep(1)
    if active:
        pof_sim = copy.copy(comp)

        # Scale t_end # TODO generalise funciton and move
        t_end = int(
            scale_input(
                "t_end", t_end, units_before=pof_sim.units, units_after=input_units
            )
        )

        # Complete the simulations
        pof_sim.mp_timeline(t_end=t_end, n_iterations=n_iterations)

        # Produce reports
        pof_sim.expected_risk_cost_df(t_end=t_end)
        pof_sim.calc_pof_df(t_end=t_end)
        pof_sim.calc_df_task_forecast(sfd.df_age_forecast)
        # pof_sim.calc_summary(sfd.df_age)
        pof_sim.calc_df_cond(t_end=t_end)

        if not pof_sim.up_to_date:
            return dash.no_update, "Update cancelled"

    else:
        return dash.no_update, "Not active"

    return f"Sim State: {pof_sim.n_iterations} - {n_iterations}", ""


@app.callback(
    Output("cond-fig", "figure"),
    Output("ms-fig", "figure"),
    Output("pof-fig", "figure"),
    Output("task_forecast-fig", "figure"),
    Output("forecast_table-fig", "figure"),
    Input("sim_state", "children"),
    Input("ms_var_y-dropdown", "value"),
    Input("axis_lock-checkbox", "checked"),
    State("input_units-dropdown", "value"),
    State("sim_n_active", "checked"),
    State("cond-fig", "figure"),
    State("ms-fig", "figure"),
    State("pof-fig", "figure"),
    State("task_forecast-fig", "figure"),
)
def update_figures(
    __,
    y_axis,
    axis_lock,
    input_units,
    active,
    prev_cond_fig,
    prev_ms_fig,
    prev_pof_fig,
    prev_task_fig,
    *args,
):
    global pof_sim
    global sfd

    if active:

        ctx = dash.callback_context
        dash_id = ctx.triggered[0]["prop_id"].split(".")[0]
        keep_axis = dash_id == "sim_state" and axis_lock

        pof_fig = pof_sim.plot_pof(
            keep_axis=keep_axis, units=input_units, prev=prev_pof_fig
        )

        ms_fig = pof_sim.plot_ms(
            y_axis=y_axis, keep_axis=keep_axis, units=input_units, prev=prev_ms_fig
        )

        cond_fig = pof_sim.plot_cond(
            keep_axis=keep_axis, units=input_units, prev=prev_cond_fig
        )

        task_forecast_fig = pof_sim.plot_task_forecast(
            keep_axis=keep_axis, prev=prev_task_fig
        )

        # pop_table_fig = pof_sim.plot_pop_table()

        forecast_table_fig = pof_sim.plot_summary(sfd.df_age)

    else:
        raise PreventUpdate

    return (
        cond_fig,
        ms_fig,
        pof_fig,
        task_forecast_fig,
        forecast_table_fig,
    )


@app.callback(
    Output("sensitivity-fig", "figure"),
    Input("sim_state", "children"),
    Input("sim_sens_active-check", "checked"),
    Input("n_sens_iterations-input", "value"),
    Input("sens_var_id-dropdown", "value"),
    Input("sens_var_y-dropdown", "value"),
    Input("sens_lower-input", "value"),
    Input("sens_upper-input", "value"),
    Input("sens_step_size-input", "value"),
    Input("t_end-input", "value"),
    Input("axis_lock-checkbox", "checked"),
    # Input("ms-fig", "figure"),  # TODO change this trigger
    State("sensitivity-fig", "figure"),
    State("input_units-dropdown", "value"),
)
def update_sensitivity(
    __,
    active,
    n_iterations,
    var_id,
    y_axis,
    lower,
    upper,
    step_size,
    t_end,
    axis_lock,
    prev_sens,
    input_units,
    *args,
):
    """ Trigger a sensitivity analysis of the target variable"""
    # Copy from the main model
    global comp
    global sens_sim
    sens_sim.cancel_sim()

    if active:

        ctx = dash.callback_context
        dash_id = ctx.triggered[0]["prop_id"].split(".")[0]
        keep_axis = dash_id == "sim_state" and axis_lock

        sens_sim = copy.deepcopy(comp)

        # Scale the inputs if needed
        var = var_id.split("-")[-1]
        lower = scale_input(
            var, lower, units_before=sens_sim.units, units_after=input_units
        )
        upper = scale_input(
            var, upper, units_before=sens_sim.units, units_after=input_units
        )
        step_size = scale_input(
            var, step_size, units_before=sens_sim.units, units_after=input_units
        )
        t_end = int(
            scale_input(
                "t_end", t_end, units_before=sens_sim.units, units_after=input_units
            )
        )

        sens_sim.expected_sensitivity(
            var_id=var_id,
            lower=lower,
            upper=upper,
            t_end=t_end,
            step_size=step_size,
            n_iterations=n_iterations,
        )

        sens_fig = sens_sim.plot_sens(
            var_id=var_id,
            y_axis=y_axis,
            keep_axis=keep_axis,
            prev=prev_sens,
            units=input_units,
        )

        return sens_fig
    else:
        raise PreventUpdate


# ==============================================
# The following progress bars are always running
# ==============================================


@app.callback(
    Output("n-progress", "value"),
    Output("n-progress", "children"),
    Input("progress-interval", "n_intervals"),
)
def update_progress(__):
    if pof_sim.n is None:
        raise Exception("no process started")
    progress = int(pof_sim.progress() * 100)

    return progress, f"{progress} %" if progress >= 5 else ""


@app.callback(
    [Output("n_sens-progress", "value"), Output("n_sens-progress", "children")],
    [Input("sens_progress-interval", "n_intervals")],
)
def update_progress_sens(__):
    if sens_sim.n is None:
        raise Exception("no process started")
    progress = int(sens_sim.sens_progress() * 100)

    return progress, f"{progress} %" if progress >= 5 else ""


if __name__ == "__main__":
    app.run_server(debug=True)
