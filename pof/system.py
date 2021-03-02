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
        self._sim_counter = 0
        self.stop_simulation = False

        # Dash Tracking
        self.up_to_date = True
        self._in_service = True
        self.n = 0
        self.n_sens = 0
        self.n_iterations = 10
        self.n_sens_iterations = 10

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
        self.n = 0
        self.n_sens = 0

    def mc_timeline(self, t_end, t_start=0, n_iterations=DEFAULT_ITERATIONS):
        """ Simulate the timeline mutliple times"""
        self.reset()

        for i in tqdm(range(n_iterations)):
            self.sim_timeline(t_end=t_end, t_start=t_start)
            self.save_timeline(i)
            self.increment_counter()
            self.reset_for_next_sim()

    def mp_timeline(self, t_end, t_start=0, n_iterations=DEFAULT_ITERATIONS):
        """ Simulate the timeline mutliple times and exit immediately if updated"""
        self.reset()
        self.up_to_date = True
        self.n = 0
        self.n_iterations = n_iterations

        try:
            for __ in tqdm(range(self.n_iterations)):
                if not self.up_to_date:
                    break

                # Complete a simulation
                self.sim_timeline(t_end=t_end, t_start=t_start)
                self.save_timeline(self.n)
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

        t_now = t_start
        self._in_service = True

        while t_now < t_end and self._in_service:

            t_next, next_comp_tasks = self.next_tasks(t_now)

            self.complete_tasks(t_next, next_comp_tasks)

            t_now = t_next + 1

    def init_timeline(self, t_end, t_start=0):
        """ Initialise the timeline """
        for comp in self.comp.values():
            if comp.active:
                comp.init_timeline(t_start=t_start, t_end=t_end)

    def next_tasks(self, t_start=0):
        """
        Returns a dictionary with the component triggered
        """
        # logging.info(self.units)
        task_schedule = dict()
        for comp_name, comp in self.comp.items():
            if comp.active:
                t_next, task_names = comp.next_tasks(t_start=t_start)

                if t_next in task_schedule:
                    task_schedule[t_next][comp_name] = task_names
                else:
                    task_schedule[t_next] = dict()
                    task_schedule[t_next][comp_name] = task_names

                t_next = min(task_schedule.keys())
                # if t_next < t_start:
                #     break

        return t_next, task_schedule[t_next]

    def complete_tasks(self, t_next, comp_tasks):

        for comp_name, comp_tasks in comp_tasks.items():

            system_impacts = self.comp[comp_name].complete_tasks(t_next, comp_tasks)

            if "system" in system_impacts and cf.get("allow_system_impact"):
                logging.debug("System %s reset by Component %s", self._name, comp_name)
                self.renew(t_renew=t_next + 1)

                break

    def renew(self, t_renew):

        # Fail
        if config.get("Component").get("remain_failed"):
            for comp in self.comp.values():
                comp.fail(t_renew)

            self._in_service = False

        # Replace
        else:
            for comp in self.comp.values():
                comp.renew(t_renew)

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
        return self.n / self.n_iterations

    def sens_progress(self) -> float:
        """ Returns the progress of the sensitivity simulation """
        return (self.n_sens * self.n_iterations + self.n) / (
            self.n_iterations * self.n_sens_iterations + self.n
        )

    # ****************** Reports ****************

    def expected_risk_cost_df(self, t_start=0, t_end=None):
        """ Create df_erc for all components """

        # Create the dataframe
        df = pd.DataFrame()

        # Append the values for each component to the dataframe
        for comp in self.comp.values():
            if comp.active:
                df_comp = comp.expected_risk_cost_df(t_start=t_start, t_end=t_end)
                df_comp["comp"] = comp.name

                df = df.append(df_comp)

        self.df_erc = df

        return self.df_erc

    def calc_pof_df(self, t_end=None):
        """ Create df_pof for all components """

        # Create the dataframe
        df = pd.DataFrame()

        # Append the values for each component to the dataframe
        for comp in self.comp.values():
            if comp.active:
                df_comp = comp.calc_pof_df(t_end=t_end)
                df_comp["comp"] = comp.name

                df = df.append(df_comp)

        self.df_pof = df

    def calc_df_task_forecast(self, df_age_forecast, age_units="years"):
        """ Create df_task for all components """

        # Create the dataframe
        df = pd.DataFrame()

        # Append the values for each component to the dataframe
        for comp in self.comp.values():
            if comp.active:
                df_comp = comp.calc_df_task_forecast(
                    df_age_forecast=df_age_forecast,
                    age_units=age_units,
                )
                df_comp["comp"] = comp.name

                df = df.append(df_comp)

        self.df_task = df

    def calc_df_cond(self, t_start=0, t_end=None):
        """ Create df_cond for all components """

        # Create the dataframe
        df = pd.DataFrame()

        # Append the values for each component to the dataframe
        for comp in self.comp.values():
            if comp.active:
                df_comp = comp.calc_df_cond(t_start=t_start, t_end=t_end)
                df_comp["comp"] = comp.name

                df = df.append(df_comp)

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
        """
        Returns dataframe of sensitivity data for a given variable name using a given lower, upper and step_size.
        """
        rc = dict()
        self.reset()

        # Progress bars
        self.n_sens = 0
        self.n_sens_iterations = len(np.arange(lower, upper + step_size, step_size))

        var = var_id.split("-")[-1]

        prefix = ["quantity", "cost"]
        suffix = ["", "_annual", "_cumulative"]
        cols = [f"{pre}{suf}" for pre in prefix for suf in suffix]

        for i in np.arange(lower, upper + step_size, step_size):
            if not self.up_to_date:
                return "sim cancelled"
            try:
                # Reset component
                self.reset()

                # Update annd simulate a timeline
                self.update(var_id, i)
                self.mp_timeline(t_end=t_end, n_iterations=n_iterations)
                df_rc = self.expected_risk_cost_df()

                # Summarise outputs
                df_rc = df_rc.groupby(by=["comp", "task", "active"])[cols].max()
                df_rc[var] = i

                rc[i] = df_rc

                self.n_sens = self.n_sens + 1

            except Exception as error:
                logging.error("Error at %s", exc_info=error)

        df = (
            pd.concat(rc)
            .reset_index()
            .drop(["level_0"], axis=1)
            .rename(columns={"task": "source"})
        )

        self.df_sens = df

        # Set df_sens for each component
        for comp in self.comp.values():
            if comp.active:
                comp.df_sens = df[df["comp"] == comp.name]

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
        self._sim_counter = 0
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