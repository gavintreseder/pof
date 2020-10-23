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

    # ---------------- Load from_dict ----------------

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

    # ------------ set_containter_attr with empty container -----------

    def test_set_container_attr_from_dict(self):

        load = Load()
        load.data = None

        test_data = dict(name="test")
        expected = Load(name="test")

        load._set_container_attr("data", Load, test_data)

        self.assertEqual(load.data[expected.name], expected)

    def test_set_container_attr_from_dict_of_dicts(self):

        load = Load()
        test_data = dict(pof_object=dict(name="test"))
        expected = Load(name="test")

        load._set_container_attr("data", Load, test_data)

        self.assertEqual(load.data[expected.name], expected)

    def test_set_container_attr_from_dict_of_objects(self):

        load = Load()
        test_data = dict(pof_object=Load(name="test"))
        expected = Load(name="test")

        load._set_container_attr("data", Load, test_data)

        self.assertEqual(load.data[expected.name], expected)

    def test_set_container_attr_from_object(self):

        load = Load()
        test_data = Load(name="test")
        expected = Load(name="test")

        load._set_container_attr("data", Load, test_data)

        self.assertEqual(load.data[expected.name], expected)

    # ------------ set_containter_attr with existing data -----------

    def test_set_container_attr_existing_data_from_dict(self):

        load = Load()
        load.data = dict(test=Load(name="this_should_change"))
        test_data = dict(name="test")
        expected = Load(name="test")

        load._set_container_attr("data", Load, test_data)

        self.assertEqual(load.data[expected.name], expected)

    def test_set_container_attr_existing_data_from_dict_of_dicts(self):

        load = Load()
        load.data = dict(test=Load(name="this_should_change"))
        test_data = dict(pof_object=Load(name="test"))
        expected = Load(name="test")

        load._set_container_attr("data", Load, test_data)

        self.assertEqual(load.data[expected.name], expected)

    def test_set_container_attr_existing_data_from_dict_of_objects(self):

        load = Load()
        load.data = dict(test=Load(name="this_should_change"))
        test_data = dict(pof_object=Load(name="test"))
        expected = Load(name="after_update")
        key_before_update = "test"

        load._set_container_attr("data", Load, test_data)
        test_data["pof_object"].name = "after_update"

        self.assertEqual(load.data[key_before_update], expected)

    def test_set_container_attr_existing_data_from_object(self):

        load = Load()
        load.data = dict(test=Load(name="this_should_change"))
        test_data = Load(name="test")
        expected = Load(name="after_update")
        key_before_update = "test"

        load._set_container_attr("data", Load, test_data)
        test_data.name = "after_update"

        self.assertEqual(load.data[key_before_update], expected)
