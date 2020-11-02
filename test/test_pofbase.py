"""
    Filename: test_pofbase.py
    Description: Contains the code for testing elements common to all pof objects
    Author:
        Gavin Treseder | gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
        Illyse Schram  | ischram@kpmg.com.au | illyse.schram@essentialenergy.com.au
"""

import unittest
from unittest.mock import Mock, patch


class TestPofBase(object):
    """
    An abstract test class that contains a collection of tests to test a from_dict method for a pof object

    Needs to include a setUp() method that includes:
        self._class
                        Used for calling class methods

        self._data_valid
        self._data_invalid_values
        self._data_invalid_types

    """

    tc = unittest.TestCase()

    def setUp(self):
        """
        Set up the test so errors are returned if test data is not overloaded
        """

        # Config for checking default behaviour
        self.blank_config = Mock()
        self.blank_config.get.return_value = None

        # Class data
        self._class = Mock(spec=object, return_value=None)
        self._class.from_dict = Mock(return_value=None)

        # Valid and invalid Data that will cause errors if not overloaded
        self._data_valid = Mock(return_value=None)
        self._data_invalid_values = [{"name": "name"}]
        self._data_invalid_types = [{"name": "name"}]

    # ---------------- Class Instantiate ------------------------

    def test_class_instantiate_with_no_data(self):
        """ Check class instantiate works with all no data"""
        # Arrange
        class_config = "pof.load.cf"
        for param in [True, False, None]:
            with patch.dict(class_config, {"handle_invalid_data": param}):

                # Act
                instance = self._class()

                # Assert
                self.tc.assertIsNotNone(instance)

    def test_class_instantiate_with_valid_data(self):
        """ Check class instantiate works with valid data"""

        # Arrange
        class_config = "pof.load.cf"
        for param in [True, False, None]:
            with patch.dict(class_config, {"handle_invalid_data": param}):

                # Act
                instance = self._class(**self._data_valid)

                # Assert
                self.tc.assertIsNotNone(instance)

    def test_class_instantiate_with_invalid_data(self):

        # Arrange
        class_config = "pof.load.cf"
        with patch.dict(class_config, {"handle_invalid_data": True}):
            invalid_data = self._data_invalid_types
            for data in invalid_data:

                # Act
                instance = self._class(**data)

                # Assert
                self.tc.assertIsNotNone(instance)
                self.tc.assertTrue(isinstance(instance, self._class))

        with patch.dict(class_config, {"handle_invalid_data": False}):
            invalid_data = self._data_invalid_types
            for data in invalid_data:

                # Act / Assert
                with self.tc.assertRaises(TypeError):
                    instance = self._class(**data)

    # ---------------- Test from_dict ----------------

    def test_from_dict_no_data(self):
        with self.tc.assertRaises(TypeError):
            self._class.from_dict()

    def test_from_dict_with_valid_data(self):
        instance = self._class.from_dict(self._data_valid)
        self.tc.assertIsNotNone(instance)

    def test_from_dict_with_invalid_data(self):
        """ Check invalid data is handled correctly"""

        # Arrange
        class_config = "pof.load.cf"
        with patch.dict(class_config, {"handle_invalid_data": False}):

            # Act / Assert
            for invalid_type in self._data_invalid_types:
                with self.tc.assertRaises(TypeError):
                    self._class.from_dict(invalid_type)

            for invalid_value in self._data_invalid_values:
                with self.tc.assertRaises(ValueError):
                    self._class.from_dict(invalid_value)

    # ************ Test load ***********************

    def test_load_with_empty(self):
        instance = self._class.load()
        self.tc.assertIsNotNone(instance)
        self.tc.assertTrue(isinstance(instance, self._class))

    def test_load_valid_dict(self):
        # Arrange
        instance_from_dict = self._class.from_dict(self._data_valid)

        # Act
        instance = self._class.load(self._data_valid)

        # Assert
        self.tc.assertTrue(isinstance(instance, self._class))
        self.tc.assertEqual(instance, instance_from_dict)

    def test_load_with_invalid_data_errors_managed(self):

        # Arrange
        # class_config = self._class.__module__ + ".cf"
        class_config = "pof.load.cf"

        with patch.dict(class_config, {"handle_invalid_data": False}):
            with patch.dict(class_config, {"on_error_use_default": False}):

                # Act / Assert
                for data in self._data_invalid_types:
                    with self.tc.assertRaises(TypeError):
                        self._class.from_dict(data)

                for data in self._data_invalid_values:
                    with self.tc.assertRaises(ValueError):
                        self._class.from_dict(data)

            with patch.dict(class_config, {"on_error_use_default": True}):
                invalid_data = self._data_invalid_types + self._data_invalid_values
                for data in invalid_data:

                    # Act  / Assert
                    instance = self._class.load(data)

                    # Assert
                    self.tc.assertIsNotNone(instance)
                    self.tc.assertTrue(isinstance(instance, self._class))

    # ************ Test load ***********************

    def test_demo(self):

        # Arrange / Act / Assert
        self.tc.assertIsNotNone(self._class.demo())

    # def test_load_error(self):

    #     with patch.dict(class_config, {"on_error_use_default": True}):

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
