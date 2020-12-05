"""
    Filename: test_pofbasecommon.py
    Description: Contains the code for testing the Constructor class
    Author:
        Gavin Treseder | gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
        Illyse Schram  | ischram@kpmg.com.au | illyse.schram@essentialenergy.com.au
"""

import copy
import unittest
from unittest.mock import Mock

import fixtures
from test_pofbase_common import TestPofBaseCommon
import testconfig  # pylint: disable=unused-import
from pof.load import Load
from pof.units import valid_units


class TestLoad(TestPofBaseCommon, unittest.TestCase):
    def setUp(self):

        # PofBase
        self._class = Load
        self._data_valid = [{"name": "name"}]
        self._data_invalid_values = [{"name": 1234}]
        self._data_invalid_types = [{"invalid_type": "invalid_type"}]
        self._data_complete = copy.deepcopy(fixtures.complete["load"])

        # Mock the pof object
        self.pof_obj = Mock()
        self.pof_obj.name = "mock_name"
        self.pof_obj.load.return_value = Mock()

        # Create a load object with every type of data store
        self.load = Load(name="before_update")
        self.load.obj = Load(name="before_update")
        self.load.dict_obj = dict(test_key=Load(name="before_update"))

    def test_scale_units(self):
        """Case when Time_variables has length > 1, updates all variables
        && when Pof_variables has a dict and a string, updates all correctly"""

        for key in valid_units:
            # Arrange
            load = Load()
            current_unit = "hours"

            load.TIME_VARIABLES = ["test_time_1", "test_time_2"]
            load.test_time_1 = 1
            load.test_time_2 = 1

            load.POF_VARIABLES = ["obj", "dict_obj"]
            load.obj = Load(units=current_unit)
            load.dict_obj = dict(obj_key=Load(units=current_unit))

            # Act
            load._scale_units(key, current_unit)

            # Assert - Time_variables
            for var in load.TIME_VARIABLES:
                self.assertAlmostEquals(getattr(load, var), 1 / valid_units[key])

            # Assert - Pof_variables
            self.assertEqual(load.obj.units, key)
            self.assertEqual(load.dict_obj["obj_key"].units, key)

    def test_scale_units_zero(self):
        """ Case when getattr of a time_variable is 0, should return 0"""

        # Arrange
        load = Load()
        current_unit = "hours"

        load.TIME_VARIABLES = ["test_time_3"]
        load.test_time_3 = 0
        load.POF_VARIABLES = []

        for key in valid_units:
            # Act
            load._scale_units(key, current_unit)

            # Assert
            for var in load.TIME_VARIABLES:
                self.assertAlmostEquals(getattr(load, var), 0)

    def test_scale_units_error(self):
        """ Case when var is None raise value error """
        # Arrange
        load = Load()
        current_unit = "hours"

        load.POF_VARIABLES = ["test_pof_3"]
        load.test_pof_3 = None

        for key in valid_units:
            # Assert && Act
            with self.assertRaises(ValueError):
                load._scale_units(key, current_unit)

    # ------------ set_containter_attr with empty container -----------

    def test_set_obj_from_dict(self):

        # Arrange
        load = Load()
        load.dict_obj = None
        test_data = dict(name="test")
        expected = Load(name="test")

        # Act
        load.set_obj("dict_obj", Load, test_data)

        # Assert
        self.assertEqual(load.dict_obj[expected.name], expected)

    def test_set_obj_from_dict_of_dicts(self):

        load = Load()
        test_data = dict(pof_object=dict(name="test"))
        expected = Load(name="test")

        load.set_obj("obj", Load, test_data)

        self.assertEqual(load.obj[expected.name], expected)

    def test_set_obj_from_dict_of_objects(self):

        # Arrange
        load = Load()
        load.dict_obj = None
        test_data = {"test_key": Load(name="test")}
        expected = Load(name="test")

        # Act
        load.set_obj("dict_obj", Load, test_data)

        # Assert
        self.assertEqual(load.dict_obj[expected.name], expected)

    def test_set_obj_from_object(self):

        load = Load()
        test_data = Load(name="test")
        expected = Load(name="test")

        load.set_obj("obj", Load, test_data)

        self.assertEqual(load.obj[expected.name], expected)

    # ------------ set_containter_attr with existing data -----------

    def test_set_obj_existing_data_from_dict(self):

        load = Load()
        load.obj = dict(test=Load(name="this_should_change"))
        test_data = dict(name="test")
        expected = Load(name="test")

        load.set_obj("obj", Load, test_data)

        self.assertEqual(load.obj[expected.name], expected)

    def test_set_obj_existing_data_from_dict_of_dicts(self):

        load = Load()
        load.obj = dict(test=Load(name="this_should_change"))
        test_data = dict(pof_object=Load(name="test"))
        expected = Load(name="test")

        load.set_obj("obj", Load, test_data)

        self.assertEqual(load.obj[expected.name], expected)

    def test_set_obj_existing_data_from_dict_of_objects(self):

        load = Load()
        load.obj = dict(test=Load(name="this_should_change"))
        test_data = dict(pof_object=Load(name="test"))
        expected = Load(name="after_update")
        key_before_update = "test"

        load.set_obj("obj", Load, test_data)
        test_data["pof_object"].name = "after_update"

        self.assertEqual(load.obj[key_before_update], expected)

    def test_set_obj_existing_data_from_object(self):

        load = Load()
        load.obj = dict(test=Load(name="this_should_change"))
        test_data = Load(name="test")
        expected = Load(name="after_update")
        key_before_update = "test"

        load.set_obj("obj", Load, test_data)
        test_data.name = "after_update"

        self.assertEqual(load.obj[key_before_update], expected)

    # -------------------- Test update -----------------------------

    def test_update_errors_caught_and_logged(self):
        """ check that an attriubte that doens't exist returns a Key Error"""
        # Arrange
        load = Load(name="before_update")
        load.data = "before_update"
        load.obj = Load(name="before_update")
        load.dict_data = {"test_key": "before_update"}
        load.dict_obj = {"test_key": Load(name="before_update")}

        param_tests = [
            ({"invalid_attribute": "after_update"}),
            ({"obj": {"invalid_attribute": "after_update"}},),
            ({"dict_data": {"invalid_key": "after_update"}}),
            ({"dict_obj": {"invalid_key": {"name": "after_update"}}},),
        ]

        for test_data in param_tests:
            with self.tc.assertLogs(level="DEBUG") as log:
                # Act
                load.update(test_data)

                # Assert
                # TODO check errorhas been looged
                # self.tc.assertTrue("Update Failed" in log.output[-1])

    # -------------------- Test update_from_str -----------------------------

    def test_update_from_str(self):
        """ Check a string is converted to a dict for subsequent updates"""
        # Arrange
        load = Load(name="test_load")
        load.data = "before_update"

        id_str = "Load-test_load-data"
        value = "after_update"
        sep = "-"

        # Act
        load.update_from_str(id_str=id_str, value=value, sep=sep)

        # Assert
        self.assertEqual(load.data, "after_update")

    # -------------------- Test update_from_dict --------------------

    def test_update_from_dict_to_update_data(self):
        """ Check an attribute can be updated"""
        # Arrange
        load = Load(name="no_update")
        load.obj = Load(name="no_update")
        load.dict_obj = {"test_key": Load(name="no_update")}
        test_data = {"name": "after_update"}

        # Act
        load.update_from_dict(test_data)

        # Assert
        self.assertEqual(load.name, "after_update")

    def test_update_from_dict_to_update_object(self):
        """ Check an attribute that is another pof object can be updated"""
        # Arrange
        load = Load(name="no_update")
        load.obj = Load(name="no_update")
        test_data = {"obj": {"name": "after_update"}}

        # Act
        load.update_from_dict(test_data)

        # Assert
        self.assertEqual(load.obj.name, "after_update")

    def test_update_from_dict_to_update_dict_of_data(self):
        """ Check a dictionary of data can be updated"""

        # Arrange
        load = Load(name="no_update")
        load.dict_data = {"test_key": "no_update"}
        test_data = {"dict_data": {"test_key": "after_update"}}

        # Act
        load.update_from_dict(test_data)

        # Assert
        self.assertEqual(load.dict_data["test_key"], "after_update")

    def test_update_from_dict_to_update_dict_of_objects(self):
        """ Check a dictionary of pof objects can be updated"""

        # Arrange
        load = Load(name="no_update")
        load.dict_obj = {"test_key": Load(name="no_update")}
        test_data = {"dict_obj": {"test_key": {"name": "after_update"}}}

        # Act
        load.update_from_dict(test_data)

        # Assert
        self.assertEqual(load.dict_obj["test_key"].name, "after_update")

    def test_update_from_dict_with_multiple_values_at_once(self):
        """ Check multiple attributes that exist can be updated at once"""

        # Arrange
        load = Load(name="before_update")
        load.obj = Load(name="before_update")
        load.dict_data = {"test_key": "before_update"}
        load.dict_obj = {"test_key": Load(name="before_update")}
        test_data = dict(
            name="after_update",
            obj={"name": "after_update"},
            dict_data={"test_key": "after_update"},
            dict_obj={"test_key": {"name": "after_update"}},
        )

        # Act
        load.update_from_dict(test_data)

        # Assert
        self.assertEqual(load.name, "after_update")
        self.assertEqual(load.obj.name, "after_update")
        self.assertEqual(load.dict_data["test_key"], "after_update")
        self.assertEqual(load.dict_obj["test_key"].name, "after_update")

    def test_update_from_dict_with_errors_expected(self):
        """ check that an attriubte that doens't exist returns a Key Error"""
        # Arrange
        load = Load(name="before_update")
        load.data = "before_update"
        load.obj = Load(name="before_update")
        load.dict_data = {"test_key": "before_update"}
        load.dict_obj = {"test_key": Load(name="before_update")}

        param_tests = [
            (
                AttributeError,
                {"invalid_attribute": "after_update"},
            ),
            (
                AttributeError,
                {"obj": {"invalid_attribute": "after_update"}},
            ),
            (
                KeyError,
                {"dict_data": {"invalid_key": "after_update"}},
            ),
            (
                KeyError,
                {"dict_obj": {"invalid_key": {"name": "after_update"}}},
            ),
        ]

        for error, test_data in param_tests:
            # Act / Assert
            with self.assertRaises(error):
                load.update_from_dict(test_data)


if __name__ == "__main__":
    unittest.main()
