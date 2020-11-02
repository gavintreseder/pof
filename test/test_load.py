"""
    Filename: test_load.py
    Description: Contains the code for testing the Constructor class
    Author:
        Gavin Treseder | gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
        Illyse Schram  | ischram@kpmg.com.au | illyse.schram@essentialenergy.com.au
"""

import unittest
from unittest.mock import Mock, patch

import fixtures
import testconfig
from test_pofbase import TestPofBase
from pof.load import Load

class TestLoad(unittest.TestCase):
    def setUp(self):

        # Mock the pof object
        self.pof_obj = Mock()
        self.pof_obj.name = "mock_name"
        self.pof_obj.load.return_value = Mock()

        # Create a load object with every type of data store
        self.load = Load(name="before_update")
        self.load.obj = Load(name="before_update")
        self.load.dict_obj = dict(test_key=Load(name="before_update"))

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
            # Act
            load.update(test_data)

            # Assert
            # TODO add context manager to check logger

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
                KeyError,
                {"invalid_attribute": "after_update"},
            ),
            (
                KeyError,
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
