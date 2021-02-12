# ************ Packages ********************
import copy
import logging
from typing import Dict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import json

# Change the system path if an individual file is being run
if __package__ is None or __package__ == "":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import config
from pof.component import Component, sort_df
from pof.pof_base import PofBase
from pof.pof_container import PofContainer
import pof.demo as demo
from pof.interface.figures import (
    make_ms_fig,
    make_sensitivity_fig,
    update_pof_fig,
    update_condition_fig,
    make_task_forecast_fig,
    make_pop_table_fig,
    make_table_fig,
)
from pof.units import scale_units, unit_ratio
from pof.paths import Paths

DEFAULT_ITERATIONS = 10

cf = config.get("System")


class System(PofBase):
    """
    Parameters:

    Methods:


    Usage:


    """

    TIME_VARIABLES = []
    POF_VARIABLES = ["comp"]

    def __init__(
        self,
        name: str = "sys",
        active: bool = True,
        comp: Dict = None,
        *args,
        **kwargs,
    ):

        super().__init__(name=name, *args, **kwargs)

        self.active = active
        self.comp = PofContainer()

        self.set_component(comp)

        # Simulation traking
        self._sim_counter = 0
        self.stop_simulation = False

        # Dash Tracking
        self.up_to_date = True
        self.n = 0
        self.n_iterations = 10
        self.n_sens = 0
        self.n_sens_iterations = 10

        self.n_comp = 0
        self.comp_total = 3

        # Reporting
        self.df_pof = None
        self.df_cond = None
        self.df_erc = None
        self.df_sens = None
        self.df_task = None

    # ****************** Load data ******************

    def set_component(self, comp_input):
        """Takes a dictionary of Component objects or Component data and sets the system components """
        self.set_obj("comp", Component, comp_input)

    # ***************** Timeline ********************

    def cancel_sim(self):
        """ Cancels the simulation """
        self.up_to_date = False
        self.n_comp = 0
        self.n = 0
        self.n_sens = 0

    def mc_timeline(self, t_end, t_start=0, n_iterations=DEFAULT_ITERATIONS):
        """ Simulate the timeline mutliple times"""
        self.reset()

        for i in tqdm(range(n_iterations)):
            self.init_timeline(t_end=t_end, t_start=t_start)

            self.save_timeline(i)
            self.increment_counter()
            self.reset_for_next_sim()

    def mp_timeline(self, t_end, t_start=0, n_iterations=DEFAULT_ITERATIONS):
        """ Simulate the timeline mutliple times and exit immediately if updated"""
        self.reset()
        self.up_to_date = True

        self.n_comp = 0
        self.comp_total = len(self.comp)

        self.n = 0
        self.n_iterations = n_iterations

        try:
            for __ in tqdm(range(self.n_iterations)):
                if not self.up_to_date:
                    break

                self.sim_timeline(t_end=t_end, t_start=t_start)
                self.save_timeline(self.n_comp)
                self.increment_counter()
                self.reset_for_next_sim()

                self.n += 1
        except Exception as error:
            if self.up_to_date:
                raise error
            else:
                logging.warning("Error caught during cancel_sim")

    def sim_timeline(self, t_end, t_start=0):
        """ Simulates the timelines for all components attached to this system """

        self.init_timeline(t_start=t_start, t_end=t_end)

        for comp in self.comp.values():
            comp.sim_timeline(t_start=t_start, t_end=t_end)

    def init_timeline(self, t_end, t_start=0):
        """ Initialise the timeline """
        for comp in self.comp.values():
            comp.init_timeline(t_start=t_start, t_end=t_end)
            self.n_comp += 1

    def increment_counter(self):
        """ Increment the sim counter by 1 """
        self._sim_counter += 1

        for comp in self.comp.values():
            comp.increment_counter()

    def save_timeline(self, idx):
        """ Saves the timeline for each component """
        for comp in self.comp.values():
            comp.save_timeline(idx)

    # ****************** Progress *******************

    def progress(self) -> float:
        """ Returns the progress of the primary simulation """
        return (self.n_comp * self.n) / (
            self.comp_total * self.n_iterations * self.n_iterations
        )

    def sens_progress(self) -> float:
        """ Returns the progress of the sensitivity simulation """
        return (self.n_comp * self.n_sens) / (
            self.comp_total * self.n_sens_iterations * self.n_sens_iterations
        )

    # ****************** Reports ****************

    def expected_risk_cost_df(self, t_start=0, t_end=None):
        """ Create df_erc for all components """

        # Create the dataframe
        df = pd.DataFrame()

        # Append the values for each component to the dataframe
        for comp in self.comp.values():
            df = df.append(
                comp.expected_risk_cost_df(
                    t_start=t_start, t_end=t_end, comp_name=comp.name
                )
            )

        self.df_erc = df

    def calc_pof_df(self, t_end=None):
        """ Create df_pof for all components """

        # Create the dataframe
        df = pd.DataFrame()

        # Append the values for each component to the dataframe
        for comp in self.comp.values():
            df = df.append(comp.calc_pof_df(t_end=t_end, comp_name=comp.name))

        self.df_pof = df

    def calc_df_task_forecast(self, df_age_forecast, age_units="years"):
        """ Create df_task for all components """

        # Create the dataframe
        df = pd.DataFrame()

        # Append the values for each component to the dataframe
        for comp in self.comp.values():
            df = df.append(
                comp.calc_df_task_forecast(
                    df_age_forecast=df_age_forecast,
                    age_units=age_units,
                    comp_name=comp.name,
                )
            )

        self.df_task = df

    def calc_df_cond(self, t_start=0, t_end=None):
        """ Create df_cond for all components """

        # Create the dataframe
        df = pd.DataFrame()

        # Append the values for each component to the dataframe
        for comp in self.comp.values():
            df = df.append(
                comp.calc_df_cond(t_start=t_start, t_end=t_end, comp_name=comp.name)
            )

        self.df_cond = df

    def expected_condition(self):
        """ Create expected condition for all components """

        # Create the dictionary
        expected = {}

        # Add the key and value for each component
        for comp in self.comp.values():
            expected[comp.name] = comp.expected_condition()

        return expected

    def expected_sensitivity(
        self,
        var_id,
        lower,
        upper,
        step_size=1,
        n_iterations=100,
        t_end=100,
    ):
        """ Create df_sens for all components """

        self.n_sens_iterations = n_iterations

        # Create the dataframe
        df = pd.DataFrame()

        # Append the values for each component to the dataframe
        for comp in self.comp.values():
            df = df.append(
                comp.expected_sensitivity(
                    var_id=var_id,
                    lower=lower,
                    upper=upper,
                    step_size=step_size,
                    n_iterations=n_iterations,
                    t_end=t_end,
                    comp_name=comp.name,
                )
            )
            self.n_sens = comp.n_sens

        self.df_sens = df

    # ***************** Figures *****************

    # TODO change default to first value from const

    def plot_ms(
        self,
        y_axis="cost_cumulative",
        keep_axis=False,
        units: str = None,
        prev=None,
        comp_name=None,
    ):
        """ Returns a cost figure if df has aleady been calculated"""
        # TODO Add conversion for units when plotting if units != self.units

        df, units = scale_units(
            df=self.df_erc, input_units=units, model_units=self.units
        )

        df = df[df["comp"] == comp_name]

        return make_ms_fig(
            df=df,
            y_axis=y_axis,
            keep_axis=keep_axis,
            units=units,
            prev=prev,
        )

    def plot_pof(self, keep_axis=False, units=None, prev=None, comp_name=None):
        """ Returns a pof figure if df has aleady been calculated"""

        df, units = scale_units(
            df=self.df_pof, input_units=units, model_units=self.units
        )

        df = df[df["comp"] == comp_name]

        return update_pof_fig(df=df, keep_axis=keep_axis, units=units, prev=prev)

    def plot_cond(self, keep_axis=False, units=None, prev=None, comp_name=None):
        """ Returns a condition figure if df has aleady been calculated"""

        df, units = scale_units(
            df=self.df_cond, input_units=units, model_units=self.units
        )

        df = df[df["comp"] == comp_name]

        ecl_all = self.expected_condition()
        ecl_comp = ecl_all[comp_name]

        return update_condition_fig(
            df=df,
            ecl=ecl_comp,
            keep_axis=keep_axis,
            units=units,
            prev=prev,
        )

    def plot_task_forecast(self, keep_axis=False, prev=None, comp_name=None):
        """ Return a task figure if df has aleady been calculated """

        df = self.df_task[self.df_task["comp"] == comp_name]

        return make_task_forecast_fig(
            df=df,
            keep_axis=keep_axis,
            prev=prev,
        )

    def plot_sens(
        self,
        y_axis="cost_cumulative",
        keep_axis=False,
        units=None,
        var_id="",
        prev=None,
        comp_name=None,
    ):
        """ Returns a sensitivity figure if df_sens has aleady been calculated"""
        var_name = var_id.split("-")[-1]

        df_plot = self.sens_summary(var_name=var_name)

        df = sort_df(
            df=df_plot, column="source", var=var_name
        )  # Sens ordered here as x var is needed

        df, units = scale_units(df, input_units=units, model_units=self.units)

        df = df[df["comp"] == comp_name]

        return make_sensitivity_fig(
            df_plot=df,
            var_name=var_name,
            y_axis=y_axis,
            keep_axis=keep_axis,
            units=units,
            prev=prev,
        )

    def sens_summary(self, var_name="", summarise=True):
        """ Add direct and total to df_sens for the var_id and return the df to plot """
        # if summarise: #TODO

        df = self.df_sens

        # Add direct and indirect
        df_total = df.groupby(by=[var_name]).sum()
        df_direct = (
            df_total - df.loc[df["source"] == "risk"].groupby(by=[var_name]).sum()
        )
        summary = {
            "total": df_total,
            "direct": df_direct,
            # "risk": df.loc[df["source"] == "risk"],
        }

        df_plot = pd.concat(summary, names=["source"]).reset_index()
        df_plot["active"] = df_plot["active"].astype(bool)
        df_plot = df_plot.append(df)
        # df_plot = df_plot.append(df.loc[df["source"] != "risk"])

        return df_plot

    def plot_summary(self, df_cohort=None):
        df = pd.DataFrame()

        for comp in self.comp.values():
            df = df.append(comp.calc_summary(df_cohort=df_cohort, comp_name=comp.name))

        fig = make_table_fig(df)

        return fig

    # ****************** Reset ******************

    def reset_condition(self):
        """ Reset condition parameters to their initial state """

        for comp in self.comp.values():
            comp.reset_condition()

    def reset_for_next_sim(self):
        """ Reset parameters back to the initial state"""

        for comp in self.comp.values():
            comp.reset_for_next_sim()

    def reset(self):
        """ Reset all parameters back to the initial state and reset sim parameters"""

        # Reset failure modes
        for comp in self.comp.values():
            comp.reset()

        # Reset counters
        self._sim_counter = 0
        # self._t_in_service = []
        self.stop_simulation = False

        # Reset stored reports
        self.df_erc = None
        self.df_sens = None
        self.df_pof = None
        self.df_cond = None
        self.df_task = None

    # ****************** Interface ******************

    def get_objects(self, prefix="", sep="-"):
        """ Returns a list of objects to populate the layout """
        objects = [prefix + self.name]

        # Create the system prefix ("system"-)
        prefix = prefix + self.name + sep

        # Add hte comp objects onto "system-"
        for comp in self.comp.values():
            objects = objects + comp.get_objects(prefix=prefix + "comp" + sep)

        return objects

    def get_dash_ids(self, numericalOnly: bool, prefix="", sep="-", active=None):
        """Return a list of dash ids for values that can be changed"""
        if active is None or (self.active == active):
            # System ids
            prefix = prefix + self.name + sep
            sys_ids = [prefix + param for param in ["active"]]

            # Component ids
            comp_ids = []
            for comp in self.comp.values():
                comp_ids = comp_ids + comp.get_dash_ids(
                    numericalOnly=numericalOnly,
                    prefix=prefix + "comp" + sep,
                    sep=sep,
                    active=active,
                )

            dash_ids = sys_ids + comp_ids
        else:
            dash_ids = []

        return dash_ids

    def get_update_ids(self, numericalOnly=bool, prefix="", sep="-"):
        """ Get the ids for all objects that should be updated"""

        # Get the system ids
        ids_sys = self.get_dash_ids(numericalOnly=numericalOnly, active=True)

        # Get the component ids
        for comp in self.comp.values():
            ids_comp = comp.get_update_ids(
                numericalOnly=numericalOnly, prefix=prefix, sep=sep
            )

        ids = ids_sys + ids_comp

        return ids

    def save(self, file_name, file_units=None):
        """ Save a json file with a system """

        # Scale the units back to file_units
        self.units = file_units

        # Create the data set
        data = self.to_dict()

        # Save to json
        with open(Paths().model_path + "\\" + file_name, "w") as json_file:
            json.dump(data, json_file)

    # ****************** Demonstration parameters ******************

    @classmethod
    def demo(cls):
        """ Loads a demonstration data set if no parameters have been set already"""

        return cls.load(demo.system_data["overhead_network"])


if __name__ == "__main__":
    system = System()
    print("System - Ok")