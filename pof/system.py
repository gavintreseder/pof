# ************ Packages ********************
import copy
import logging
from typing import Dict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

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
cf_main = config.get("Main")


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
        # self._sim_counter = 0
        self.stop_simulation = False

        # Dash Tracking
        self.up_to_date = True
        self.sim_progress = 0
        self.sim_sens_progress = 0

        self.n_comp = 0
        self.n_comp_sens = 0
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
        self.sim_progress = 0
        self.sim_sens_progress = 0

    def mc_timeline(self, t_end, t_start=0, n_iterations=DEFAULT_ITERATIONS):
        """ Simulate the timeline mutliple times"""
        self.reset()

        for comp in self.comp.values():
            if comp.active:
                comp.mc_timeline(t_end=t_end, t_start=t_start)

                self.n_comp += 1

    def mp_timeline(self, t_end, t_start=0, n_iterations=DEFAULT_ITERATIONS):
        """ Simulate the timeline mutliple times and exit immediately if updated"""
        self.reset()
        self.up_to_date = True

        self.n_comp = 0
        self.comp_total = len([comp.name for comp in self.comp.values() if comp.active])

        try:
            for comp in self.comp.values():
                if comp.active:
                    comp.mp_timeline(
                        t_end=t_end, t_start=t_start, n_iterations=n_iterations
                    )

                    self.sim_progress = comp.progress()
                    self.n_comp += 1

                    self.reset_for_next_sim()
        except Exception as error:
            if self.up_to_date:
                raise error
            else:
                logging.warning("Error caught during cancel_sim")

    # def sim_timeline(self, t_end, t_start=0):
    #     """ Simulates the timelines for all components attached to this system """

    #     self.init_timeline(t_start=t_start, t_end=t_end)

    #     for comp in self.comp.values():
    #         if comp.active:
    #             comp.sim_timeline(t_start=t_start, t_end=t_end)
    #             self.sim_progress = comp.progress()

    # def init_timeline(self, t_end, t_start=0):
    #     """ Initialise the timeline """
    #     for comp in self.comp.values():
    #         if comp.active:
    #             comp.init_timeline(t_start=t_start, t_end=t_end)
    #             self.n_comp += 1

    # def next_tasks(self):
    #     return NotImplemented

    # def complete_tasks(self):
    #     return NotImplemented

    # def renew(self):
    #     return NotImplemented

    # def increment_counter(self):
    #     """ Increment the sim counter by 1 """
    #     self._sim_counter += 1

    #     for comp in self.comp.values():
    #         if comp.active:
    #             comp.increment_counter()

    # def save_timeline(self, idx):
    #     """ Saves the timeline for each component """
    #     for comp in self.comp.values():
    #         if comp.active:
    #             comp.save_timeline(idx)

    # ****************** Progress *******************

    def progress(self) -> float:
        """ Returns the progress of the primary simulation """
        return self.sim_progress * self.n_comp / self.comp_total

    def sens_progress(self) -> float:
        """ Returns the progress of the sensitivity simulation """
        return self.sim_sens_progress * self.n_comp / self.comp_total

    def total_iterations(self) -> float:
        """ Returns the total simulations run """
        total_iterations = 0

        for comp in self.comp.values():
            if comp.active:
                total_iterations += comp.n

        return total_iterations

    def total_sens_iterations(self) -> float:
        """ Returns the total simulations run for sens """
        total_iterations = 0

        for comp in self.comp.values():
            if comp.active:
                total_iterations += comp.n_sens

        return total_iterations

    # ****************** Reports ****************

    def expected_risk_cost_df(self, t_start=0, t_end=None):
        """ Create df_erc for all components """

        # Create the dataframe
        df = pd.DataFrame()

        # Append the values for each component to the dataframe
        for comp in self.comp.values():
            if comp.active:
                df = df.append(comp.expected_risk_cost_df(t_start=t_start, t_end=t_end))

                df["comp"] = comp.name

        self.df_erc = df

    def calc_pof_df(self, t_end=None):
        """ Create df_pof for all components """

        # Create the dataframe
        df = pd.DataFrame()

        # Append the values for each component to the dataframe
        for comp in self.comp.values():
            if comp.active:
                df = df.append(comp.calc_pof_df(t_end=t_end))

                df["comp"] = comp.name

        self.df_pof = df

    def calc_df_task_forecast(self, df_age_forecast, age_units="years"):
        """ Create df_task for all components """

        # Create the dataframe
        df = pd.DataFrame()

        # Append the values for each component to the dataframe
        for comp in self.comp.values():
            if comp.active:
                df = df.append(
                    comp.calc_df_task_forecast(
                        df_age_forecast=df_age_forecast,
                        age_units=age_units,
                    )
                )

                df["comp"] = comp.name

        self.df_task = df

    def calc_df_cond(self, t_start=0, t_end=None):
        """ Create df_cond for all components """

        # Create the dataframe
        df = pd.DataFrame()

        # Append the values for each component to the dataframe
        for comp in self.comp.values():
            if comp.active:
                df = df.append(comp.calc_df_cond(t_start=t_start, t_end=t_end))

                df["comp"] = comp.name

        self.df_cond = df

    def expected_condition(self):
        """ Create expected condition for all components """

        # Create the dictionary
        expected = {}

        # Add the key and value for each component
        for comp in self.comp.values():
            if comp.active:
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

        # Create the dataframe
        df = pd.DataFrame()
        self.n_comp = 0
        self.comp_total = len([comp.name for comp in self.comp.values() if comp.active])

        # Append the values for each component to the dataframe
        for comp in self.comp.values():
            if comp.active:
                df = df.append(
                    comp.expected_sensitivity(
                        var_id=var_id,
                        lower=lower,
                        upper=upper,
                        step_size=step_size,
                        n_iterations=n_iterations,
                        t_end=t_end,
                    )
                )
                self.sim_sens_progress = comp.sens_progress()

                self.n_comp += 1

                df["comp"] = comp.name

        self.df_sens = df

    # ****************** Reset ******************

    def reset_condition(self):
        """ Reset condition parameters to their initial state """

        for comp in self.comp.values():
            if comp.active:
                comp.reset_condition()

    def reset_for_next_sim(self):
        """ Reset parameters back to the initial state"""

        for comp in self.comp.values():
            if comp.active:
                comp.reset_for_next_sim()

    def reset(self):
        """ Reset all parameters back to the initial state and reset sim parameters"""

        # Reset failure modes
        for comp in self.comp.values():
            if comp.active:
                comp.reset()

        # Reset counters
        # self._sim_counter = 0
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
            if comp.active:
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

    def get_update_ids(
        self, numericalOnly: bool = True, prefix="", sep="-", filter_ids: dict = None
    ):
        """ Get the ids for all objects that should be updated"""

        prefix = prefix + self.name + sep

        # Get the component ids
        for comp in self.comp.values():
            if comp.name == filter_ids["comp"] and comp.active:
                ids = comp.get_update_ids(
                    numericalOnly=numericalOnly, prefix=prefix + "comp" + sep, sep=sep
                )

        return ids

    # ****************** Demonstration parameters ******************

    @classmethod
    def demo(cls):
        """ Loads a demonstration data set if no parameters have been set already"""

        return cls.load(demo.system_data[cf_main.get("name")])


if __name__ == "__main__":
    system = System()
    print("System - Ok")