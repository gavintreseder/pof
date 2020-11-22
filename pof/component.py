"""

Author: Gavin Treseder
"""

# ************ Packages ********************
from typing import Dict
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

# Change the system path if an individual file is being run
if __package__ is None or __package__ == "":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import config
from pof.failure_mode import FailureMode
from pof.helper import fill_blanks
from pof.indicator import Indicator
from pof.load import Load
import pof.demo as demo

DEFAULT_ITERATIONS = 100

cf = config.get("Component")


class Component(Load):
    """
    Parameters:

    Methods:


    Usage:


    """

    def __init__(
        self,
        name: str = "comp",
        active: bool = True,
        indicator: Dict = None,
        fm: Dict = None,
        *args,
        **kwargs,
    ):

        super().__init__(name=name, *args, **kwargs)

        self.active = active
        self.indicator = dict()
        self.fm = dict()

        self.set_indicator(indicator)
        self.set_failure_mode(fm)

        # Link failure mode indicators to the component indicators
        self.link_indicators()

        # Trial for indicator
        """self.indicator["safety_factor"] = PoleSafetyFactor(component=self)
        self.indicator["slow_degrading"] = Condition.load(
            demo.condition_data["slow_degrading"]
        )  # TODO fix this call
        self.indicator["fast_degrading"] = Condition.load(
            demo.condition_data["fast_degrading"]
        )  # TODO fix this call"""

        # TODO link failure_modes to indicators

        # Simulation traking
        self._in_service = True
        self._sim_counter = 0
        self._t_replacement = []
        self.stop_simulation = False

        # Dash Tracking
        self.up_to_date = True
        self.n = 0
        self.n_iterations = 10
        self.n_sens = 0
        self.n_sens_iterations = 10

    # ***************** Property methods ************

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, value):
        """ Takes a unit and updates any time values to reflect the new units"""

        # TODO add a valid unit to the test data set
        # TODO add an invalid unit to the test data set
        # TODO update the complete test data set

        # TODO move this property to load

        # Check that it is a valid value
        # TODO Create a new file with a dict that has the ratio between units all based in hours and import it at the top

        # Get the ratio between the current units and the new units
        # TODO Repeat this function across all of the pof objects
        # Potentially move it to Load and then each file only needs a class variable with a list of attrs that are times
        # and list of attrs that have a unit method

        # Adjust every time value by this ratio and make sure they ints?

        self._units = value

    # ****************** Load data ******************

    def load_asset_data(self, *args, **kwargs):

        # TODO Hook up data
        self.info = dict(
            pole_load=10,
            pole_strength=20,
        )

        # Set perfect indicator values?? TODO

        # Set indicators
        for indicator in self.indicator.values():

            # Set perfect

            # Set current
            NotImplemented

    def set_indicator(self, indicator_input):
        """
        Takes a dictionary of Indicator objects or indicator data and sets the component indicators
        """
        self.set_obj("indicator", Indicator, indicator_input)

    def set_failure_mode(self, fm_input):
        """
        Takes a dictionary of FailureMode objects or FailureMode data and sets the component failure modes
        """

        self.set_obj("fm", FailureMode, fm_input)

    def link_indicators(self):

        for fm in self.fm.values():
            fm.set_indicators(self.indicator)

        # TODO move this logic of an indicator manager
        for ind in self.indicator.values():
            if ind.__class__.__name__ == "PoleSafetyFactor":
                ind.link_component(self)

    # ****************** Set data ******************

    def mc(self, t_end, t_start, n_iterations):
        """ Complete MC simulation and calculate all the metrics for the component"""

        return NotImplemented

    # ****************** Timeline ******************

    def cancel_sim(self):
        self.up_to_date = False

    def next_sim(self, t_end, t_start=0, n_iterations=None, multiple=5):

        if n_iterations is not None:
            self.n_iterations = n_iterations

        while self.n < self.n_iterations:
            # Concept for continuin loop
            # if self.n == self.n_iterations:
            #    self.n_iterations = min(self.n * trigger_mutliple, n_iterations)

            if not self.up_to_date:
                self.reset()
                self.n = 1
                self.n_iterations = min(self.n * multiple, n_iterations)
                self.up_to_date = True

            # Simulate
            self.sim_timeline(t_end=t_end, t_start=t_start)
            self.save_timeline(self.n)
            self.increment_counter()
            self.reset_for_next_sim()

            self.n = self.n + 1

        self.up_to_date = True

    def mp_timeline(self, t_end, t_start=0, n_iterations=10):
        """ Simulate the timeline mutliple times and exit immediately if updated"""
        self.reset()
        self.up_to_date = True
        self.n = 1
        self.n_iterations = n_iterations

        # while self.n < n_iterations and self.up_to_date:

        for i in range(self.n_iterations):
            if not self.up_to_date:
                break
            # Do work
            self.sim_timeline(t_end=t_end, t_start=t_start)
            self.save_timeline(self.n)
            self.increment_counter()
            self.reset_for_next_sim()

            self.n = self.n + 1

    def mc_timeline(self, t_end, t_start=0, n_iterations=DEFAULT_ITERATIONS):
        """ Simulate the timeline mutliple times"""
        self.reset()

        for i in tqdm(range(n_iterations)):
            self.sim_timeline(t_end=t_end, t_start=t_start)

            self.save_timeline(i)
            self.increment_counter()
            self.reset_for_next_sim()

    def sim_timeline(self, t_end, t_start=0):
        """ Simulates the timelines for all failure modes attached to this component"""

        # Initialise the failure modes
        self.init_timeline(t_start=t_start, t_end=t_end)

        t_now = t_start
        self._in_service = True

        while t_now < t_end and self._in_service:

            t_next, next_fm_tasks = self.next_tasks(t_now)

            self.complete_tasks(t_next, next_fm_tasks)

            t_now = t_next + 1

    def init_timeline(self, t_end, t_start=0):
        """ Initialise the timeline"""
        for fm in self.fm.values():
            fm.init_timeline(t_start=t_start, t_end=t_end)

    def next_tasks(self, t_start):
        """
        Returns a dictionary with the failure mode triggered
        """
        # TODO make this more efficent
        # TODO make this work if no tasks returned. Expect an error now

        # Get the task schedule for next tasks
        task_schedule = dict()
        for fm_name, fm in self.fm.items():

            t_next, task_names = fm.next_tasks(t_start=t_start)

            if t_next in task_schedule:
                task_schedule[t_next][fm_name] = task_names
            else:
                task_schedule[t_next] = dict()
                task_schedule[t_next][fm_name] = task_names

        t_next = min(task_schedule.keys())

        return t_next, task_schedule[t_next]

    def complete_tasks(self, t_next, fm_tasks):
        """Complete any tasks in the dictionary fm_tasks at t_now"""

        # TODO add logic around all the different ways of executing
        # TODO check task groups
        # TODO check value?
        # TODO add task impacts

        for fm_name, task_names in fm_tasks.items():
            system_impact = self.fm[fm_name].complete_tasks(t_next, task_names)

            if bool(system_impact) and cf.get("allow_system_impact"):
                logging.debug(f"Component {self._name} reset by FailureMode {fm_name}")
                self.renew(t_renew=t_next + 1)

                break

    def renew(
        self,
        t_renew,
    ):
        """
        Renew the component because a task has triggered an as-new change or failure
        """

        # Reset the indicators
        for ind in self.indicator.values():
            ind.reset_to_perfect()

        # Fail
        if config.get("FailureMode").get("remain_failed"):
            for fm in self.fm.values():
                fm.fail(t_renew)

            self._in_service = False

        # Replace
        else:
            for fm in self.fm.values():
                fm.renew(t_renew)

        self._t_replacement.append(t_renew)

    def increment_counter(self):
        self._sim_counter = self._sim_counter + 1

        for fm in self.fm.values():
            fm.increment_counter()

    def save_timeline(self, idx):
        for fm in self.fm.values():
            fm.save_timeline(idx)

        for ind in self.indicator.values():
            ind.save_timeline(idx)

    # ****************** Expected ******************

    # TODO RUL

    def expected(self):

        # Run expected method on all failure modes

        # Run
        NotImplemented

    # PoF

    def expected_cf(self):
        """ Returns the conditional failures for the component """
        return self._t_replacement

    def expected_ff(self):
        """Returns the functional failures for the component"""
        ff = []
        for fm in self.fm.values():
            ff.extend(fm.expected_ff())

        return ff

    def expected_untreated(self, t_start=0, t_end=100):

        sf = dict(all=dict(pof=np.full((t_end - t_start + 1), 1)))
        for fm in self.fm.values():
            sf[fm.name] = dict()
            sf[fm.name]["pof"] = fm.untreated.sf(t_start=t_start, t_end=t_end)
            sf[fm.name]["fm_active"] = fm.active

            if fm.active:
                sf["all"]["pof"] = sf["all"]["pof"] * sf[fm.name]["pof"]
                sf["all"]["fm_active"] = True

        # Treat the failure modes as a series and combine together
        # cdf = {fm: 1 - sf for fm, sf in sf.items()}
        cdf = dict()

        for fm in sf:
            cdf[fm] = dict()
            cdf[fm]["pof"] = 1 - sf[fm]["pof"]
            cdf[fm]["fm_active"] = sf[fm]["fm_active"]
            cdf[fm]["time"] = np.linspace(
                t_start, t_end, t_end - t_start + 1, dtype=int
            )

        return cdf

    def expected_pof(self, t_start=0, t_end=100):

        sf = self.expected_sf(t_start, t_end)

        cdf = dict()

        for fm in sf:
            cdf[fm] = dict()
            cdf[fm]["pof"] = 1 - sf[fm]["pof"]
            cdf[fm]["fm_active"] = sf[fm]["fm_active"]
            cdf[fm]["time"] = np.linspace(
                t_start, t_end, t_end - t_start + 1, dtype=int
            )

        return cdf

    def expected_sf(self, t_start=0, t_end=100):

        # Calcuate the failure rates for each failure mode
        sf = dict(all=dict(pof=np.full((t_end - t_start + 1), 1)))
        sf["all"]["fm_active"] = False

        for fm_name, fm in self.fm.items():
            pof = fm.expected_pof()
            sf[fm_name] = dict()
            sf[fm_name]["pof"] = pof.sf(t_start, t_end)
            sf[fm_name]["fm_active"] = fm.active

            if fm.active:
                sf["all"]["pof"] = sf["all"]["pof"] * sf[fm_name]["pof"]
                sf["all"]["fm_active"] = True

        # Treat the failure modes as a series and combine together
        # sf['all'] = np.array([fm.sf(t_start, t_end) for fm in self.fm.values()]).prod(axis=0)

        # TODO Fit a new Weibull for the new failure rate....

        return sf

    def expected_risk_cost_df(self, t_start=0, t_end=None):
        """ A wrapper for expected risk cost that returns a dataframe"""
        erc = self.expected_risk_cost()

        if t_end == None:
            t_end = t_start
            for timeline in erc.values():
                for task in timeline.values():
                    if isinstance(task, bool):
                        continue
                    else:
                        t_end = max(max(task["time"], default=t_start), t_end)

        df = pd.DataFrame().from_dict(erc, orient="index")
        df.index.name = "failure_mode"
        df = df.reset_index().melt(id_vars="failure_mode", var_name="task")
        df = pd.concat(
            [df.drop(columns=["value"]), df["value"].apply(pd.Series)], axis=1
        )[["failure_mode", "task", "time", "cost", "fm_active", "task_active"]].dropna()

        df = df.apply(fill_blanks, axis=1, args=(t_start, t_end))
        df_cost = df.explode("cost")["cost"]
        df = df.explode("time")
        df["cost"] = df_cost

        # Add a cumulative cost
        df["cost_cumulative"] = df.groupby(by=["failure_mode", "task"])[
            "cost"
        ].transform(pd.Series.cumsum)

        return df

    def expected_risk_cost_df_legacy_method(self, t_start=0, t_end=None):
        """ A wrapper for expected risk cost that returns a dataframe"""

        # TODO encapsualte in failure_mode and task

        # Create the erc_df
        d_comp = {}
        for fm in self.fm.values():
            d_fm = {}
            for task in fm.tasks.values():
                d_fm[task.name] = pd.DataFrame(task.expected_costs())
            df_fm = (
                pd.concat(d_fm, names=["source", "drop"])
                .reset_index()
                .drop("drop", axis=1)
            )
            d_comp[fm.name] = df_fm

        df_comp = (
            pd.concat(d_fm, names=["failure_mode", "drop"])
            .reset_index()
            .drop("drop", axis=1)
        )

        # Get the desired time steps
        t_start = 0  # int(df_comp['time'].min()) if t_start is None
        t_end = 200  # int(df_comp['time'].max()) if t_end is None
        time = np.linspace(t_start, t_end, t_end - t_start + 1).astype(int)

        df = df_comp[["failure_mode", "source", "active"]].drop_duplicates()
        df_time = pd.DataFrame({"time": time})

        # Cross join
        df["key"] = 1
        df_time["key"] = 1
        df = pd.merge(df, df_time, on="key")
        df = pd.merge(
            df, df_fm, on=["failure_mode", "source", "time", "active"], how="left"
        ).drop("key", axis=1)

        # Fill blanks
        df["cost"].fillna(0, inplace=True)

        # Calculate other forms of cost
        df["cost_cumulative"] = df.groupby(by=["failure_mode", "source"])[
            "cost"
        ].transform(pd.Series.cumsum)
        return df

    def expected_risk_cost(self):
        return {fm.name: fm.expected_risk_cost() for fm in self.fm.values()}

    def expected_condition(self, conf=0.95):
        return {
            ind.name: ind.expected_condition(conf) for ind in self.indicator.values()
        }

    # ****************** Optimal? ******************

    def expected_inspection_interval(self, t_max, t_min=0, step=1, n_iterations=100):
        # TODO add an optimal onto this

        self.n_sens = 1
        self.n_sens_iterations = int((t_max - t_min) / step + 1)

        rc = dict()
        self.reset()

        for i in range(t_min, t_max, step):

            # Set t_interval
            for fm in self.fm.values():
                if "inspection" in list(fm.tasks):
                    fm.tasks["inspection"].t_interval = i

            # self.mc_timeline(t_end=100, n_iterations=n_iterations)
            self.mp_timeline(t_end=100, n_iterations=n_iterations)

            rc[i] = self.expected_risk_cost_df().groupby(by=["task"])["cost"].sum()
            rc[i]["inspection_interval"] = i

            # Reset component
            self.reset()

            self.n_sens = self.n_sens + 1

        df = (
            pd.DataFrame()
            .from_dict(rc, orient="index")
            .rename(columns={"risk": "risk_cost"})
        )
        df["direct_cost"] = df.drop(["inspection_interval", "risk_cost"], axis=1).sum(
            axis=1
        )
        df["total"] = df["direct_cost"] + df["risk_cost"]

        return df

    def progress(self) -> float:
        """ Returns the progress of the primary simulation"""
        return self.n / self.n_iterations

    def sens_progress(self) -> float:
        """ Returns the progress of the sensitivity simulation"""
        return (self.n_sens * self.n_iterations + self.n) / (
            self.n_iterations * self.n_sens_iterations
        )

    def expected_sensitivity(
        self, var_name, t_min, t_max, step_size=1, n_iterations=100
    ):
        """
        Returns dataframe of sensitivity data for a given variable name using a given confidence.
        """
        rc = dict()
        ta = pd.DataFrame()
        fma = pd.DataFrame()
        self.reset()

        var = var_name.split("-")[-1]

        for i in np.arange(t_min, t_max, step_size):
            try:
                self.update(var_name, i)
                self.mc_timeline(t_end=100, n_iterations=n_iterations)
                erc = self.expected_risk_cost_df()
                rc[i] = erc.groupby(by=["task"])["cost"].sum()

                rc[i][var] = i

                # Reset component
                self.reset()

            except Exception as error:
                logging.error("Error at %s", exc_info=error)

        ta = erc.groupby(by=["task"])["task_active"].all().reset_index()
        fma = erc.groupby(by=["task", "failure_mode"])["fm_active"].all().reset_index()

        df_active = ta.merge(fma, on="task").rename(columns={"task": "source"})

        df = (
            pd.DataFrame()
            .from_dict(rc, orient="index")
            .rename(columns={"risk": "risk_cost"})
        )

        return df, df_active

    # ****************** Reset ******************

    def reset_condition(self):

        for fm in self.fm.values():
            fm.reset_condition()

    def reset_for_next_sim(self):
        """ Reset parameters back to the initial state"""

        for fm in self.fm.values():
            fm.reset_for_next_sim()

    def reset(self):
        """ Reset all parameters back to the initial state and reset sim parameters"""

        # Reset failure modes
        for fm in self.fm.values():
            fm.reset()

        # Reset counters
        self._sim_counter = 0
        self._t_replacement = []
        self.stop_simulation = False

    # ****************** Interface ******************

    def update_from_dict(self, data):
        """ Adds an additional update method for task groups"""

        # Loop through all the varaibles to update
        for attr, detail in data.items():
            if attr == "task_group":
                self.update_task_group({attr: detail})

            else:
                super().update_from_dict({attr: detail})

    def update_task_group(self, data):
        """ Update all the tasks with that task_group across the objects"""
        # TODO replace with task group manager

        for fm in self.fm.values():
            self.update_task_group(data)

    def get_dash_ids(self, prefix="", sep="-"):
        """ Return a list of dash ids for values that can be changed"""

        # Component
        prefix = prefix + self.name + sep
        comp_ids = [prefix + param for param in ["active"]]

        # Tasks
        fm_ids = []
        for fm in self.fm.values():
            fm_ids = fm_ids + fm.get_dash_ids(prefix=prefix + "fm" + sep)

        dash_ids = comp_ids + fm_ids

        return dash_ids

    def get_update_ids(self, prefix="", sep="-"):
        """ Get the ids for all objects that should be updated"""
        # TODO remove this once task groups added to the interface
        # TODO fix encapsulation

        ids = self.get_dash_ids()

        update_ids = dict()
        for fm in self.fm.values():
            for task in fm.tasks.values():
                if task.task_group_name not in update_ids:
                    update_ids[
                        task.task_group_name + "t_interval"
                    ] = f"{self.name}{sep}task_group_name{sep}{task.task_group_name}{sep}t_interval"

                    update_ids[
                        task.task_group_name
                    ] = f"{self.name}{sep}task_group_name{sep}{task.task_group_name}{sep}t_delay"

        ids = ids + list(update_ids.values())
        return ids

    def get_objects(self, prefix="", sep="-"):

        prefix = prefix
        objects = [prefix + self.name]

        prefix = prefix + self.name + sep

        for fms in self.fm.values():
            objects = objects + fms.get_objects(prefix=prefix + "fm" + sep)

        return objects

    def get_timeline(self):

        print(NotImplemented)

    # ****************** Demonstration parameters ******************

    @classmethod
    def demo(cls):
        """ Loads a demonstration data set if no parameters have been set already"""

        return cls.load(demo.component_data["comp"])


if __name__ == "__main__":
    component = Component()
    print("Component - Ok")

    """import doctest
    doctest.testmod()"""