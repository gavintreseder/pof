# import unittest
# from unittest.mock import Mock, MagicMock, patch
# import copy

# import utils

# from pof.task import Task, ScheduledTask, ConditionTask, Inspection
# import pof.demo as demo

# import fixtures


# class TestCommon(unittest.TestCase):
#     def setUp(self):
#         self._class = Mock(return_value=None)

#     def test_class_imports_correctly(self):
#         self.assertIsNotNone(self._class)

#     def test_class_instantiate(self):
#         task = self._class()
#         self.assertIsNotNone(task)


# class TestTask(TestCommon, unittest.TestCase):
#     def setUp(self):
#         super().setUp()
#         self._class = Task

#     # **************** test_load ***********************

#     def test_load_empty(self):
#         task = Task.load()
#         self.assertIsNotNone(task)

#     def test_load_valid_dict(self):
#         task = Task.load(demo.inspection_data["instant"])
#         self.assertIsNotNone(task)

#     # **************** test_update ***********************

#     def test_update(self):

#         test_data_1 = copy.deepcopy(fixtures.on_condition_replacement_data)
#         test_data_1["cost"] = 0
#         test_data_1["triggers"]["condition"]["fast_degrading"]["upper"] = 90
#         test_data_2 = copy.deepcopy(fixtures.on_condition_replacement_data)

#         # Test all the options
#         t1 = Task.from_dict(test_data_1)
#         t2 = Task.from_dict(test_data_2)

#         t1.update_from_dict(
#             {"cost": 5000, "trigger": {"condition": {"fast_degrading": {"upper": 20}}}}
#         )

#         # self.assertEqual(t1.__dict__, t2.__dict__)
#         self.assertEqual(t1.cost, t2.cost)
#         self.assertEqual(t1.triggers, t2.triggers)

#     def test_update_error(self):

#         test_data = copy.deepcopy(fixtures.on_condition_replacement_data)

#         t = Task.from_dict(test_data)

#         update = {"alpha": 10, "beta": 5}

#         self.assertRaises(KeyError, t.update_from_dict, update)


# class TestScheuduledTask(TestCommon):
#     def setUp(self):
#         super().setUp()
#         self._class = Task

#     # def test_sim_timeline(self):

#     #     # Task params
#     #     param_t_interval = [0, 1, 5]
#     #     param_t_delay = [0, 1, 5]

#     #     # Sim_timeline params
#     #     param_inputs = [(0, 0, 100)]

#     #     # for t_interval in param_t_interval:
#     #     #     for t_delay in param_t_delay:

#     #     #         for t_elapse, t_start, t_end in param_inputs:

#     #     #             with self.subTest():
#     #     #                 task = ScheduledTask(t_delay=t_delay, t_interval=t_interval)


# class TestConditionTask(TestCommon, unittest.TestCase):
#     def setUp(self):
#         super().setUp()
#         self._class = ConditionTask

#     def test_imports_correctly(self):
#         self.assertTrue(True)

#     def test_instantiate(self):
#         task = ConditionTask()
#         self.assertIsNotNone(task)

#     # **************** test_load ***********************

#     def test_load_empty(self):
#         task = ConditionTask.load()
#         self.assertIsNotNone(task)

#     def test_load_valid_dict(self):
#         task = ConditionTask.load(demo.on_condition_replacement_data)
#         self.assertIsNotNone(task)

#     def test_update(self):

#         test_data_1 = copy.deepcopy(fixtures.on_condition_replacement_data)
#         test_data_1["cost"] = 0
#         test_data_1["triggers"]["condition"]["fast_degrading"]["upper"] = 90
#         test_data_2 = copy.deepcopy(fixtures.on_condition_replacement_data)

#         # Test all the options
#         t1 = ConditionTask.from_dict(test_data_1)
#         t2 = ConditionTask.from_dict(test_data_2)

#         t1.update_from_dict(
#             {"cost": 5000, "trigger": {"condition": {"fast_degrading": {"upper": 20}}}}
#         )

#         # self.assertEqual(t1.__dict__, t2.__dict__)
#         self.assertEqual(t1.cost, t2.cost)
#         self.assertEqual(t1.triggers, t2.triggers)

#     def test_update_error(self):

#         test_data = copy.deepcopy(fixtures.on_condition_replacement_data)

#         t = ConditionTask.from_dict(test_data)

#         update = {"alpha": 10, "beta": 5}

#         self.assertRaises(KeyError, t.update_from_dict, update)


# class TestInspection(TestCommon, unittest.TestCase):
#     def setUp(self):
#         super().setUp()
#         self._task = Inspection

#     def test_imports_correctly(self):
#         self.assertTrue(True)

#     def test_instantiate(self):
#         task = Inspection()
#         self.assertIsNotNone(task)

#     # **************** test_load ***********************

#     def test_load_empty(self):
#         task = Inspection.load()
#         self.assertIsNotNone(task)

#     def test_load_valid_dict(self):
#         task = Inspection.load(demo.inspection_data["instant"])
#         self.assertIsNotNone(task)

#     # **************** test_update ***********************

#     def test_update(self):

#         test_data_1 = copy.deepcopy(fixtures.inspection_data["instant"])
#         test_data_1["cost"] = 0
#         test_data_1["triggers"]["condition"]["instant"]["upper"] = 90
#         test_data_2 = copy.deepcopy(fixtures.inspection_data["instant"])

#         t1 = Inspection.from_dict(test_data_1)
#         t2 = Inspection.from_dict(test_data_2)

#         t1.update_from_dict(
#             {
#                 "cost": 50,
#                 "trigger": {"condition": {"instant": {"upper": 0}}},
#             }
#         )

#         # self.assertEqual(t1, t2)
#         self.assertEqual(t1.cost, t2.cost)
#         self.assertEqual(t1.triggers, t2.triggers)

#     def test_update_error(self):

#         test_data = copy.deepcopy(fixtures.inspection_data["instant"])

#         t = Inspection.from_dict(test_data)

#         update = {"alpha": 10, "beta": 5}

#         self.assertRaises(KeyError, t.update_from_dict, update)


# del TestCommon

# if __name__ == "__main__":
#     unittest.main()