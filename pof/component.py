"""

Author: Gavin Treseder
"""

# ************ Packages ********************
from typing import Dict
import logging
import os

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Change the system path if an individual file is being run
if __package__ is None or __package__ == "":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import config
from pof.failure_mode import FailureMode
from pof.helper import fill_blanks
from pof.indicator import Indicator
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
from pof.data.asset_data import SimpleFleet
from pof.loader.asset_model_loader import AssetModelLoader

DEFAULT_ITERATIONS = 10


cf = config.get("Component")


class Component(PofBase):
    """
    Parameters:

    Methods:


    Usage:


    """

    TIME_VARIABLES = []
    POF_VARIABLES = ["indicator", "fm"]

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

        self.fleet_data = "temp fix until creating asset class"

        self.active = active
        self.indicator = PofContainer()
        self.fm = PofContainer()

        self.set_indicator(indicator)
        self.set_failure_mode(fm)

        # Link failure mode indicators to the component indicators
        self.link_indicators()

        # Simulation traking
        self._in_service = True
        self._sim_counter = 0
        self._t_in_service = []
        self.stop_simulation = False

        # Dash Tracking
        self.up_to_date = True
        self.n = 0
        self.n_iterations = 10
        self.n_sens = 0
        self.n_sens_iterations = 10

        # Reporting
        self.df_pof = None
        self.df_cond = None
        self.df_erc = None
        self.df_sens = None
        self.df_task = None

    # ****************** Load data ******************

    def load_asset_data(
        self,
    ):

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
            raise NotImplementedError

    def set_indicator(self, indicator_input):
        """Takes a dictionary of Indicator objects or indicator data and sets the component indicators"""
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

    def mc(self, t_end, t_start=0, n_iterations=DEFAULT_ITERATIONS):
        """ Complete MC simulation and calculate all the metrics for the component"""

        # Simulate a timeline
        self.mp_timeline(t_end=t_end, t_start=t_start, n_iterations=n_iterations)

        # Produce reports
        self.expected_risk_cost_df(t_end=t_end)
        self.calc_pof_df(t_end=t_end)
        self.calc_df_task_forecast()
        self.calc_df_cond()

        return NotImplemented

    # ****************** Timeline ******************

    def cancel_sim(self):
        self.up_to_date = False
        self.n = 0
        self.n_sens = 0

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

        if self._in_service:
            self._t_in_service.append(t_now)

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
                logging.debug(
                    "Component %s reset by FailureMode %s", self._name, fm_name
                )
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

    def increment_counter(self):
        self._sim_counter += 1

        for fm in self.fm.values():
            fm.increment_counter()

    def save_timeline(self, idx):
        for fm in self.fm.values():
            fm.save_timeline(idx)

        for ind in self.indicator.values():
            ind.save_timeline(idx)

    # ****************** Expected ******************

    def population_table(self):
        """ Reports a summary of the in service, FF and CF outcomes over the MC simulation """
        pop_table = {}
        _ff = 0
        _cf = 0
        for fm in self.fm.values():
            _ff = _ff + len(fm.expected_ff())
            _cf = _cf + len(fm.expected_cf())
        pop_table["summary"] = {
            "is": self._sim_counter - _ff - _cf,
            "cf": _cf,
            "ff": _ff,
        }

        df_pop_summary = pd.DataFrame.from_dict(pop_table).T
        col_order = ["is", "cf", "ff"]
        df = df_pop_summary.reindex(columns=col_order)

        return df

    def expected_forecast_table(self, cohort=None):
        """ Reports a summary for each of the failure modes and the expected outcomes over the MC simulation"""
        ff_cf = {}
        for fm in self.fm.values():
            if fm.active:
                insp_effective = fm.inspection_effectiveness()
                _ff = len(fm.expected_ff())
                _cf = len(fm.expected_cf())
                n = self._sim_counter

                ff_cf[fm.name] = {
                    "ie": insp_effective,
                    "is": n,  # - _cf - _ff,
                    "ff": _ff,
                    "cf": _ff,
                }

        df = pd.DataFrame.from_dict(ff_cf).T
        df.index.name = "fm"

        if cohort is not None:
            sample_size = self._sim_counter / cohort["assets"].sum()
            df[["ff_pop", "cf_pop"]] = df[["ff", "cf"]] / sample_size

        df["fm"] = df.index

        # Calculate the total row
        df.loc["total", "fm"] = "total"
        # df.loc['total', 'ie'] = (1 - df['ie'])
        sum_cols = ["ff", "cf", "ff_pop", "cf_pop"]
        df.loc["total", sum_cols] = df[sum_cols].sum()

        # Calculate the ratio of cf to ff
        ff_percent = round((df["cf"] / (df["cf"] + df["ff"]) * 100), 2)
        cf_percent = round((df["cf"] / (df["cf"] + df["ff"]) * 100), 2)
        df["cf:ff (%)"] = f"{cf_percent}:{ff_percent}"

        # Format for display
        col_order = [
            "fm",
            "ie",
            "cf:ff (%)",
            "is",
            "ff",
            "cf",
            "ff_pop",
            "cf_pop",
        ]
        df = df.reindex(columns=col_order)

        df["ie"] = df["ie"].mul(100).round(2)

        # df.rename({"ie": "ie (%)"}, inplace=True)
        df

        return df

    def expected_cf(self):
        """ Returns the conditional failures for the component """
        t_cf = []
        for fm in self.fm.values():
            t_cf.extend(fm.expected_cf())

        return t_cf

    def expected_ff(self):
        """Returns the functional failures for the component"""
        t_ff = []
        for fm in self.fm.values():
            t_ff.extend(fm.expected_ff())

        return t_ff

    def expected_life(self):
        e_l = (
            sum(self._t_in_service + self.expected_cf() + self.expected_ff())
            / self._sim_counter
        )

        return e_l

    def expected_untreated(self, t_start=0, t_end=100):

        sf = dict(all=dict(pof=np.full((t_end - t_start + 1), 1)))
        for fm in self.fm.values():
            sf[fm.name] = dict()
            sf[fm.name]["pof"] = fm.untreated.sf(t_start=t_start, t_end=t_end)
            sf[fm.name]["active"] = fm.active

            if fm.active:
                sf["all"]["pof"] = sf["all"]["pof"] * sf[fm.name]["pof"]
                sf["all"]["active"] = True

        # Treat the failure modes as a series and combine together
        # cdf = {fm: 1 - sf for fm, sf in sf.items()}
        cdf = dict()

        for fm in sf:
            cdf[fm] = dict()
            cdf[fm]["pof"] = 1 - sf[fm]["pof"]
            cdf[fm]["active"] = sf[fm]["active"]
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
            cdf[fm]["active"] = sf[fm]["active"]
            cdf[fm]["time"] = np.linspace(
                t_start, t_end, t_end - t_start + 1, dtype=int
            )

        return cdf

    def expected_sf(self, t_start=0, t_end=100):

        # Calcuate the failure rates for each failure mode
        sf = dict(all=dict(pof=np.full((t_end - t_start + 1), 1)))
        sf["all"]["active"] = False

        for fm_name, fm in self.fm.items():
            if fm.active:
                pof = fm.expected_pof()
                sf[fm_name] = dict()
                sf[fm_name]["pof"] = pof.sf(t_start, t_end)
                sf[fm_name]["active"] = fm.active

                sf["all"]["pof"] = sf["all"]["pof"] * sf[fm_name]["pof"]
                sf["all"]["active"] = True

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
        )[
            [
                "failure_mode",
                "task",
                "active",
                "time",
                "quantity",
                "cost",
            ]
        ].dropna()

        fill_cols = ["cost", "quantity"]  # time not needed
        df_filled = df.apply(fill_blanks, axis=1, args=(t_start, t_end, fill_cols))
        df = df_filled.explode("time")

        for col in fill_cols:
            df[col] = df_filled.explode(col)[col]
            df[col + "_cumulative"] = df.groupby(by=["failure_mode", "task"])[
                col
            ].transform(pd.Series.cumsum)
            df[col + "_annual"] = df[col] / self.expected_life()

            # df[col + "_lifecycle"] = df[col] / df['_annual'] self.expected_life()

        # Formatting
        self.df_erc = df_order(df=df, column="task")
        # TODO Mel change the function name to be order_df df_order makes it look like a variable not a function

        return self.df_erc

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
        # t_end = t_end  # int(df_comp['time'].max()) if t_end is None
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

    # **************** Interface ********************

    def progress(self) -> float:
        """ Returns the progress of the primary simulation"""
        return self.n / self.n_iterations

    def sens_progress(self) -> float:
        """ Returns the progress of the sensitivity simulation"""
        return (self.n_sens * self.n_iterations + self.n) / (
            self.n_iterations * self.n_sens_iterations
        )

    def expected_sensitivity(
        self, var_id, lower, upper, step_size=1, n_iterations=100, t_end=100
    ):
        """
        Returns dataframe of sensitivity data for a given variable name using a given lower, upper and step_size.
        """
        rc = dict()
        self.reset()

        # Progress bars
        self.n_sens = 0
        self.n_sens_iterations = int((upper - lower) / step_size + 1)

        var = var_id.split("-")[-1]

        prefix = ["quantity", "cost"]
        suffix = ["", "_annual", "_cumulative"]
        cols = [f"{pre}{suf}" for pre in prefix for suf in suffix]

        for i in np.arange(lower, upper + 1, step_size):
            if not self.up_to_date:
                break
            try:
                # Reset component
                self.reset()

                # Update annd simulate a timeline
                self.update(var_id, i)
                self.mp_timeline(t_end=t_end, n_iterations=n_iterations)
                df_rc = self.expected_risk_cost_df()

                # Summarise outputs
                df_rc = df_rc.groupby(by=["task", "active"])[cols].max()
                df_rc[var] = i

                rc[i] = df_rc

                self.n_sens = self.n_sens + 1

            except Exception as error:
                logging.error("Error at %s", exc_info=error)

        self.df_sens = (
            pd.concat(rc)
            .reset_index()
            .drop(["level_0"], axis=1)
            .rename(columns={"task": "source"})
        )

        return self.df_sens

    # ****************** Reports ****************

    def calc_df_erc(self):
        if self.up_to_date:
            if self.df_erc is not None:
                df_erc = self.df_erc
            else:
                self.df_erc = self.expected_risk_cost_df()

        raise NotImplementedError()

    def calc_pof_df(self, t_end=None):
        # TODO this could be way more efficient
        pof = dict(
            maint=pd.DataFrame(self.expected_pof(t_end=t_end)),
            no_maint=pd.DataFrame(self.expected_untreated(t_end=t_end)),
        )

        # OLD CODE
        # df = pd.concat(pof).rename_axis(["strategy", "time"]).reset_index()
        # df.index.names = ["strategy", "key"]
        # df = df.rename(columns={"variable": "source"})

        # df = df.melt(
        #     id_vars=["time", "strategy", "fm_active"],
        #     var_name="source",
        #     value_name="pof",
        # )

        df = pd.concat(pof).rename_axis(["strategy", "key"]).reset_index()

        df_pof = df[df["key"] == "pof"]
        df_pof = pd.melt(
            df_pof, id_vars=["strategy"], value_vars=df_pof.columns[2:]
        ).rename(columns={"variable": "source", "value": "pof"})
        df_pof = df_pof.explode("pof", ignore_index=True)

        df_active = df[df["key"] == "active"]
        df_active = pd.melt(
            df_active, id_vars=["strategy"], value_vars=df_active.columns[2:]
        ).rename(columns={"variable": "source", "value": "active"})

        df_time = df[df["key"] == "time"]
        df_time = pd.melt(
            df_time, id_vars=["strategy"], value_vars=df_time.columns[2:]
        ).rename(columns={"variable": "source", "value": "time"})
        df_time = df_time.explode("time", ignore_index=True)["time"]

        df = df_pof.merge(df_active, on=["strategy", "source"])
        df["time"] = df_time

        self.df_pof = df

        return self.df_pof

    def calc_df_task_forecast(self):
        """ Create the task plot dataframe """

        df = fleet_data.get_task_forecast(df_erc=self.df_erc)

        df_ordered = df_order(df=df, column="task")

        self.df_task = df_ordered

        return self.df_task

    def calc_df_cond(self):
        ecl = self.expected_condition()

        df = pd.DataFrame()
        i = 1
        for cond_name, cond in ecl.items():
            data = np.append(cond["upper"], cond["lower"][::-1])
            df["y" + str(i)] = data
            i = i + 1

        self.df_cond = df

        return self.df_cond

    # ***************** Figures *****************

    # TODO change default to first value from const

    def plot_ms(
        self,
        y_axis="cost_cumulative",
        y_max=None,
        prev=None,
    ):
        """ Returns a cost figure if df has aleady been calculated"""
        # TODO Add conversion for units when plotting if units != self.units
        return make_ms_fig(
            df=self.df_erc,
            y_axis=y_axis,
            y_max=y_max,
            units=self.units,
            prev=prev,
        )

    def plot_pof(self, y_max=None, prev=None):
        """ Returns a pof figure if df has aleady been calculated"""

        return update_pof_fig(df=self.df_pof, y_max=y_max, units=self.units, prev=prev)

    def plot_cond(self, y_max=None, prev=None):
        """ Returns a condition figure if df has aleady been calculated"""
        return update_condition_fig(
            df=self.df_cond,
            ecl=self.expected_condition(),
            y_max=y_max,
            units=self.units,
            prev=prev,
        )

    def plot_task_forecast(
        self,
        y_max=None,
        prev=None,
    ):
        """ Return a task figure if df has aleady been calculated """

        return make_task_forecast_fig(
            df=self.df_task,
            y_max=y_max,
            prev=prev,
        )

    def plot_sens(
        self,
        y_axis="cost_cumulative",
        y_max=None,
        units=NotImplemented,  # TODO add a plot here to make sure it
        var_id="",
        prev=None,
    ):
        """ Returns a sensitivity figure if df_sens has aleady been calculated"""
        var_name = var_id.split("-")[-1]
        df = df_order(
            df=self.df_sens, column="source", var=var_name
        )  # Sens ordered here as x var is needed

        return make_sensitivity_fig(
            df=df,
            var_name=var_name,
            y_axis=y_axis,
            y_max=y_max,
            units=self.units,
            prev=prev,
        )

    def plot_pop_table(self):

        df = self.population_table()
        fig = make_pop_table_fig(df)

        return fig

    def plot_summary(self, cohort=None):

        df = self.expected_forecast_table(cohort=cohort)
        fig = make_table_fig(df)

        return fig

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
        self._t_in_service = []
        self.stop_simulation = False

        # Reset stored reports
        self.df_erc = None
        self.df_sens = None

    # ****************** Interface ******************

    def update_from_dict(self, data):
        """ Adds an additional update method for task groups"""

        # Loop through all the varaibles to update
        for attr, detail in data.items():
            if attr == "task_group_name":
                self.update_task_group(detail)

            else:
                super().update_from_dict({attr: detail})

    def update_task_group(self, data):
        """ Update all the tasks with that task_group across the objects"""
        # TODO replace with task group manager

        for fm in self.fm.values():
            fm.update_task_group(data)

    def get_dash_ids(self, numericalOnly: bool, prefix="", sep="-", active=None):
        """ Return a list of dash ids for values that can be changed"""

        if active is None or (self.active == active):
            # Component
            prefix = prefix + self.name + sep
            comp_ids = [prefix + param for param in ["active"]]

            # Tasks
            fm_ids = []
            for fm in self.fm.values():
                fm_ids = fm_ids + fm.get_dash_ids(
                    numericalOnly=numericalOnly,
                    prefix=prefix + "fm" + sep,
                    sep=sep,
                    active=active,
                )

            dash_ids = comp_ids + fm_ids
        else:
            dash_ids = []

        return dash_ids

    def get_update_ids(self, numericalOnly=bool, prefix="", sep="-"):
        """ Get the ids for all objects that should be updated"""
        # TODO remove this once task groups added to the interface
        # TODO fix encapsulation

        ids = self.get_dash_ids(numericalOnly=numericalOnly, active=True)

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

        ids = list(update_ids.values()) + ids
        return ids

    def get_objects(self, prefix="", sep="-"):

        prefix = prefix
        objects = [prefix + self.name]

        prefix = prefix + self.name + sep

        for fms in self.fm.values():
            objects = objects + fms.get_objects(prefix=prefix + "fm" + sep)

        return objects

    # ****************** Demonstration parameters ******************

    @classmethod
    def demo(cls):
        """ Loads a demonstration data set if no parameters have been set already"""

        return cls.load(demo.component_data["comp"])


def df_order(df=None, column=None, var=None):
    """
    sorts the dataframes for the graphs with total, risk and direct first
    """
    if column is None:
        raise ValueError("Column must be defined")

    # List distinct values in input column
    if column == "task":
        values = df["task"].unique().tolist()
        values.sort()
    elif column == "source":
        values = df["source"].unique().tolist()
        values.sort()

    # Create a list of columns to sort by
    if var is not None:
        columns = [var, column]
    elif "year" in df.columns.tolist():
        columns = ["year", column]
    elif "time" in df.columns.tolist():
        columns = ["time", column]
    else:
        columns = [column]

    # Define custom order required
    start_order = ["total", "risk", "direct"]
    set_order = []

    # Add custom columns to the set order
    for var in start_order:
        if var in values:
            set_order.append(var)

    # Add all other columns to set order alphabetically
    for var in values:
        if var not in set_order:
            set_order.append(var)

    # Create a dictionary of return order
    return_order = {}
    i = 1
    for var in set_order:
        return_order[var] = i
        i = i + 1

    # Sort by sort column list
    df[column] = pd.Categorical(df[column], return_order)
    df_ordered = df.sort_values(by=columns)

    return df_ordered


if __name__ == "__main__":
    component = Component()
    print("Component - Ok")

    """import doctest
    doctest.testmod()"""