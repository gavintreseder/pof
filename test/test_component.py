"""
    Filename: test_component.py
    Description: Contains the code for testing the Component class
    Author: Gavin Treseder | gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
 
"""

import unittest
from unittest.mock import MagicMock
from unittest.mock import patch
import numpy as np
import pandas as pd
import scipy.stats as ss

import utils

from pof.component import Component
from config import config
import pof.demo as demo

from pof.interface.figures import update_condition_fig


cf = config["Component"]


class TestComponent(unittest.TestCase):
    """
    Unit tests for the Component class
    """

    def test_class_imports_correctly(self):
        self.assertIsNotNone(Component)

    def test_class_instantiate(self):
        comp = Component()
        self.assertIsNotNone(comp)

    @patch("cf.USE_DEFAULT", True)
    def test_class_instantiate_no_input_use_default_true(self):
        """ Tests the creation of a class instance with no inputs when the global default flag is set to true"""
        comp = Component()
        self.assertIsNotNone(comp)

    @patch("cf.USE_DEFAULT", False)
    def test_class_instantiate_no_input_use_default_false(self):
        """ Tests the creation of a class instance with no inputs when the global default flag is set to false"""
        with self.assertRaises(
            Exception,
            msg="Indicator should not be able to link if there isn't an indicator by that name",
        ):
            comp = Component()

    def test_from_dict(self):
        comp = Component.from_dict(demo.component_data["comp"])

    ## *************** Test set_demo ***********************

    def test_demo(self):
        comp = Component.demo()
        self.assertIsNotNone(comp)

    # *************** Test init_timeline ***********************

    def test_init_timeline(self):
        t_end = 200
        comp = Component.demo()
        comp.init_timeline(t_end)

        for fm in comp.fm.values():
            t_fm_timeline_end = fm.timeline["time"][-1]

            self.assertEqual(t_end, t_fm_timeline_end)

    # *************** Test complete_tasks ***********************

    def test_complete_tasks_one_fm_one_task(self):
        fm_next_tasks = dict(
            slow_aging=["inspection"],
        )
        t_now = 5
        comp = Component.demo()
        comp.init_timeline(200)
        comp.complete_tasks(t_now, fm_next_tasks)

        for fm_name, fm in comp.fm.items():
            for task_name, task in fm.tasks.items():

                if fm_name in list(fm_next_tasks):
                    if task_name in fm_next_tasks[fm_name]:
                        self.assertEqual([t_now], task.t_completion)
                    else:
                        self.assertEqual([], task.t_completion)
                else:
                    self.assertEqual([], task.t_completion)

    def test_complete_tasks_two_fm_two_task(self):
        fm_next_tasks = dict(
            slow_aging=["inspection", "cm"],
            fast_aging=["inspection", "cm"],
        )
        t_now = 5
        comp = Component.demo()
        comp.init_timeline(200)
        comp.complete_tasks(t_now, fm_next_tasks)

        for fm_name, fm in comp.fm.items():
            for task_name, task in fm.tasks.items():

                if fm_name in list(fm_next_tasks):
                    if task_name in fm_next_tasks[fm_name]:
                        self.assertEqual([t_now], task.t_completion)
                    else:
                        self.assertEqual([], task.t_completion)
                else:
                    self.assertEqual([], task.t_completion)

    # *************** Test next_tasks ***********************

    def test_next_tasks_one_fm_one_task(self):

        t_now = None
        test_next_task = dict(
            slow_aging=(5, ["inspection"]),
            fast_aging=(10, ["inspection", "cm"]),
            random=(15, ["inspection"]),
        )

        expected = {k: v[1] for k, v in test_next_task.items() if v[0] == 5}

        comp = Component.demo()

        for fm_name, fm in comp.fm.items():
            fm.next_tasks = MagicMock(return_value=test_next_task[fm_name])

        t_next, next_task = comp.next_tasks(t_now)

        self.assertEqual(next_task, expected)
        self.assertEqual(t_next, 5)

    def test_next_tasks_many_fm_many_task(self):

        times = dict(
            slow_aging=[5, 5, 5],
            fast_aging=[10, 5, 5],
            random=[10, 10, 5],
        )

        for i in range(3):
            t_now = None
            test_next_task = dict(
                slow_aging=(times["slow_aging"][i], ["inspection"]),
                fast_aging=(times["fast_aging"][i], ["inspection", "cm"]),
                random=(times["random"][i], ["inspection"]),
            )

            expected = {k: v[1] for k, v in test_next_task.items() if v[0] == 5}

            comp = Component.demo()

            for fm_name, fm in comp.fm.items():
                fm.next_tasks = MagicMock(return_value=test_next_task[fm_name])

            t_next, next_task = comp.next_tasks(t_now)

            self.assertEqual(next_task, expected)
            self.assertEqual(t_next, 5)

    # *************** Test sim_timeline ***********************

    def test_sim_timeline_active_all(self):
        comp = Component.demo()

        comp.sim_timeline(200)

    def test_sim_timline_active_one(self):
        comp = Component.demo()

        comp.fm[list(comp.fm)[0]].active = False
        comp.sim_timeline(200)

    def test_mc_timeline(self):
        comp = Component.demo()

        comp.mc_timeline(t_end=100)

    # ************ Test expected methods *****************

    def test_expected_condition_no_timeline(self):
        comp = Component.demo()
        comp.expected_condition()

        # TODO add some checks

    def test_expected_condition_with_timelines(self):
        comp = Component.demo()
        comp.mc_timeline(10)
        comp.expected_condition()

    # ************ Test update methods *****************

    def test_update(self):
        # TODO test all values
        comp = Component.demo()

        comp.update("comp-fm-slow_aging-active", False)
        self.assertEqual(comp.fm["slow_aging"].active, False)

    def test_update_from_str(self):

        expected_list = [True]

        comp = Component.demo()
        dash_ids = comp.get_dash_ids()

        for dash_id in dash_ids:

            for expected in expected_list:

                comp.update(dash_id, expected)

                val = NotImplemented

                self.assertEqual(val, expected, msg="Error: dash_id %s" % (dash_id))

    def test_expected_inspection_interval(self):

        NotImplemented

    # ************ Test reset methods *****************

    def test_reset(self):

        expected = 0

        comp = Component.demo()
        comp.mc_timeline(5)
        comp.reset()

        self.assertEqual(comp._sim_counter, expected)

    def test_delete(self):
        comp = Component().demo()

        d = {}
        d_split = {}
        for id_ in comp.get_dash_ids():

            id_str = id_.split("comp" + "-", 1)[1]
            id_str = id_str.split("-")

            value = comp

            for s in id_str:
                if s == "fm":
                    value = value.fm
                elif s == "active":
                    value = value.active
                    if isinstance(value, str):
                        continue
                    elif isinstance(value, bool):
                        continue
                    else:
                        d[id_] = value
                        d_split[id_] = id_.split("-")
                elif s == "pf_curve":
                    value = value.pf_curve
                    if isinstance(value, str):
                        continue
                    elif isinstance(value, bool):
                        continue
                    else:
                        d[id_] = value
                        d_split[id_] = id_.split("-")
                elif s == "pf_interval":
                    value = value.pf_interval
                    if isinstance(value, str):
                        continue
                    elif isinstance(value, bool):
                        continue
                    else:
                        d[id_] = value
                        d_split[id_] = id_.split("-")
                elif s == "pf_std":
                    value = value.pf_std
                    if isinstance(value, str):
                        continue
                    elif isinstance(value, bool):
                        continue
                    else:
                        d[id_] = value
                        d_split[id_] = id_.split("-")
                elif s == "dists":
                    value = value.dists
                elif s == "alpha":
                    value = value.alpha
                    if isinstance(value, str):
                        continue
                    elif isinstance(value, bool):
                        continue
                    else:
                        d[id_] = value
                        d_split[id_] = id_.split("-")
                elif s == "beta":
                    value = value.beta
                    if isinstance(value, str):
                        continue
                    elif isinstance(value, bool):
                        continue
                    else:
                        d[id_] = value
                        d_split[id_] = id_.split("-")
                elif s == "gamma":
                    value = value.gamma
                    if isinstance(value, str):
                        continue
                    elif isinstance(value, bool):
                        continue
                    else:
                        d[id_] = value
                        d_split[id_] = id_.split("-")
                elif s == "tasks":
                    value = value.tasks
                elif s == "p_effective":
                    value = value.p_effective
                    if isinstance(value, str):
                        continue
                    elif isinstance(value, bool):
                        continue
                    else:
                        d[id_] = value
                        d_split[id_] = id_.split("-")
                elif s == "cost":
                    value = value.cost
                    if isinstance(value, str):
                        continue
                    elif isinstance(value, bool):
                        continue
                    else:
                        d[id_] = value
                        d_split[id_] = id_.split("-")
                elif s == "t_interval":
                    value = value.t_interval
                    if isinstance(value, str):
                        continue
                    elif isinstance(value, bool):
                        continue
                    else:
                        d[id_] = value
                        d_split[id_] = id_.split("-")
                elif s == "t_delay":
                    value = value.t_delay
                    if isinstance(value, str):
                        continue
                    elif isinstance(value, bool):
                        continue
                    else:
                        d[id_] = value
                        d_split[id_] = id_.split("-")
                elif s == "states":
                    value = value.states
                elif s == "impact":
                    value = value.impacts
                elif s == "trigger":
                    value = value.triggers
                elif s == id_str[-1]:
                    if isinstance(value[s], str):
                        continue
                    elif isinstance(value[s], bool):
                        continue
                    else:
                        d[id_] = value[s]
                        d_split[id_] = id_.split("-")
                else:
                    value = value[s]
        df = pd.DataFrame(d.items(), columns=["name", "mean"])
        df["sd"] = 5
        df_inspection_slow = df[df["name"].str.contains("cost")]
        df_inspection_slow = df_inspection_slow[
            df_inspection_slow["name"].str.contains("slow")
        ]
        df_inspection_slow["sd"] = df_inspection_slow["mean"] * 0.2
        conf = 0.9
        n_increments = 3
        # if it's even add one

        # empty dict
        d_all = {}

        # Loop through the rows #TODO look at df.apply
        for index, row in df_inspection_slow.iterrows():

            # Convert the mean and sd into a range of values
            [lower, upper] = ss.norm.interval(
                alpha=conf, loc=row["mean"], scale=row["sd"]
            )

            # Get the snesitityv for a single row
            df_sens = comp.sensitivity(
                var_name=row["name"],
                lower=lower,
                upper=upper,
                n_increments=n_increments,
                n_iterations=5,
            )
            df_sens = df_sens.reset_index(drop=True)

            # Scale everything to the mean
            cols = ["direct_cost", "risk_cost", "total"]
            mean_idx = list(df_sens.index)[int((len(list(df_sens.index)) - 1) / 2)]
            df_sens.loc[:, cols] = df_sens[cols] / df_sens.loc[mean_idx, cols]

            # Add columns for sorting
            # df_sens = df_sens.rename(columns = {'target':'value'}) # target
            df_sens["percent_change"] = df_sens["value"] / df_sens["value"][mean_idx]
            df_sens["conf"] = ss.norm.cdf(
                df_sens["value"], loc=row["mean"], scale=row["sd"]
            )
            df_sens["conf"] = df_sens["conf"].round(2)

            # add var_name
            df_sens["var_name"] = row["name"]

            # Putthe df in the dict
            d_all[row["name"]] = df_sens


if __name__ == "__main__":
    unittest.main()
