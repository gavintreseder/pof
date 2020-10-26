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
from pof.load import Load


# TODO think how to move update tests onto main file


class TestPofBase(object):
    """
    An abstract test class that contains a collection of tests to test a from_dict method for a pof object

    Needs to include a setUp() method that includes:
        self._class
                        Used for calling class methods

        self._class_name
                        Used for patching config

        self._data_valid
        self._data_invalid_values
        self._data_invalid_types

    """

    def setUp(self):
        """
        Set up the test so errors are returned if test data is not overloaded
        """

        # Config for checking default behaviour
        self.blank_config = Mock()
        self.blank_config.get.return_value = None
        self.blank_config.getboolean.return_value = None
        self.blank_config.getint.return_value = None

        # Class data
        self._class = Mock(return_value=None)
        self._class.from_dict.return_value = None

        # Valid and invalid Data that will cause errors if not overloaded
        self._data_valid = Mock(return_value=None)
        self._data_invalid_values = [{"name": "name"}]
        self._data_invalid_types = [{"name": "name"}]

    # ---------------- Class Instantiate ------------------------

    def test_class_instantiate_with_no_data(self):
        instance = self._class()
        self.assertIsNotNone(instance)

    def test_class_instantiate_with_valid_data(self):
        instance = self._class(**self._data_valid)
        self.assertIsNotNone(instance)

    # ---------------- Test from_dict ----------------

    def test_from_dict_no_data(self):
        with self.assertRaises(TypeError):
            self._class.from_dict()

    def test_from_dict_with_valid_data(self):
        instance = self._class.from_dict(self._data_valid)
        self.assertIsNotNone(instance)

    def test_from_dict_with_invalid_data_config_default(self):

        # TODO Mock cf.get_boolean('on_error_default')
        # Arrange
        class_config = self._class.__module__ + ".cf"

        with patch(class_config, Mock()):
            self.from_dict_invalid_data()

    def test_from_dict_with_invalid_data_config_none(self):

        # Arrange
        class_config = self._class.__module__ + ".cf"
        load_config = "pof.load.cf"  # TODO make this work for any namespace

        with patch(class_config, self.blank_config):
            with patch(load_config, self.blank_config):

                # Act / Assert
                self.from_dict_invalid_data()

    def from_dict_invalid_data(self):
        """Check invalid data"""
        for invalid_type in self._data_invalid_types:
            with self.assertRaises(TypeError):
                self._class.from_dict(invalid_type)

        for invalid_value in self._data_invalid_values:
            with self.assertRaises(ValueError):
                self._class.from_dict(invalid_value)

    # ************ Test load ***********************

    # def test_load(self):
    #     fm = FailureMode.load()
    #     self.assertIsNotNone(fm)

    # def test_load_no_data_no_config(self):
    #     with patch("pof.failure_mode.cf", self.blank_config):
    #         with self.assertRaises(
    #             ValueError,
    #             msg="Error expected with no input",
    #         ):
    #             FailureMode.load()

    # def test_load_data_demo_data(self):
    #     try:
    #         fm = FailureMode.load(demo.failure_mode_data["slow_aging"])
    #         self.assertIsNotNone(fm)
    #     except ValueError:
    #         self.fail("ValueError returned")
    #     except:
    #         self.fail("Unknown error")


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

    def test_set_container_attr_from_dict(self):

        load = Load()
        load.obj = None

        test_data = dict(name="test")
        expected = Load(name="test")

        load._set_container_attr("obj", Load, test_data)

        self.assertEqual(load.obj[expected.name], expected)

    def test_set_container_attr_from_dict_of_dicts(self):

        load = Load()
        test_data = dict(pof_object=dict(name="test"))
        expected = Load(name="test")

        load._set_container_attr("obj", Load, test_data)

        self.assertEqual(load.obj[expected.name], expected)

    def test_set_container_attr_from_dict_of_objects(self):

        load = Load()
        test_data = dict(pof_object=Load(name="test"))
        expected = Load(name="test")

        load._set_container_attr("obj", Load, test_data)

        self.assertEqual(load.obj[expected.name], expected)

    def test_set_container_attr_from_object(self):

        load = Load()
        test_data = Load(name="test")
        expected = Load(name="test")

        load._set_container_attr("obj", Load, test_data)

        self.assertEqual(load.obj[expected.name], expected)

    # ------------ set_containter_attr with existing data -----------

    def test_set_container_attr_existing_data_from_dict(self):

        load = Load()
        load.obj = dict(test=Load(name="this_should_change"))
        test_data = dict(name="test")
        expected = Load(name="test")

        load._set_container_attr("obj", Load, test_data)

        self.assertEqual(load.obj[expected.name], expected)

    def test_set_container_attr_existing_data_from_dict_of_dicts(self):

        load = Load()
        load.obj = dict(test=Load(name="this_should_change"))
        test_data = dict(pof_object=Load(name="test"))
        expected = Load(name="test")

        load._set_container_attr("obj", Load, test_data)

        self.assertEqual(load.obj[expected.name], expected)

    def test_set_container_attr_existing_data_from_dict_of_objects(self):

        load = Load()
        load.obj = dict(test=Load(name="this_should_change"))
        test_data = dict(pof_object=Load(name="test"))
        expected = Load(name="after_update")
        key_before_update = "test"

        load._set_container_attr("obj", Load, test_data)
        test_data["pof_object"].name = "after_update"

        self.assertEqual(load.obj[key_before_update], expected)

    def test_set_container_attr_existing_data_from_object(self):

        load = Load()
        load.obj = dict(test=Load(name="this_should_change"))
        test_data = Load(name="test")
        expected = Load(name="after_update")
        key_before_update = "test"

        load._set_container_attr("obj", Load, test_data)
        test_data.name = "after_update"

        self.assertEqual(load.obj[key_before_update], expected)

    # -------------------- Test update -----------------------------

    def test_update_with_string(self):
        # TODO
        NotImplemented

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
            (KeyError, {"dict_data": {"invalid_key": "after_update"}}),
            (
                KeyError,
                {"dict_obj": {"invalid_key": {"name": "after_update"}}},
            ),
        ]

        for error, test_data in param_tests:
            # Act / Assert
            with self.assertRaises(error):
                load.update_from_dict(test_data)
