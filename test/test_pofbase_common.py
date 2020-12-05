"""
    Filename: test_pofbasecommon.py
    Description: Contains the code for testing all pof objects that inherit from PofBase
    Author:
        Gavin Treseder | gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
        Illyse Schram  | ischram@kpmg.com.au | illyse.schram@essentialenergy.com.au
"""

import unittest
from unittest.mock import Mock, patch

import testconfig  # pylint: disable=unused-import

# from pof.units import valid_units


class TestPofBaseCommon(object):
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
        self._class = Mock(spec=Mock, return_value=Mock)
        self._class.from_dict = Mock(return_value=None)
        self._class._scale_units = Mock(return_value=None)  # TODO

        # Valid and invalid Data that will cause errors if not overloaded
        self._data_valid = [Mock(return_value=None)]
        self._data_invalid_values = [{"name": "name"}]
        self._data_invalid_types = [{"name": "name"}]
        self._data_complete = {"mock": Mock(return_value=None)}

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

                for data in self._data_valid:
                    # Act
                    instance = self._class(**data)

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
        for data in self._data_valid:
            instance = self._class.from_dict(data)
            self.tc.assertIsNotNone(instance)

    def test_from_dict_with_invalid_data(self):
        """ Check invalid data is handled correctly"""

        # Arrange
        class_config = "pof.load.cf"
        with patch.dict(class_config, {"handle_invalid_data": False}):

            # Act / Assert
            # for invalid_type in self._data_invalid_types:
            #     with self.tc.assertRaises(TypeError):
            #         self._class.from_dict(invalid_type)

            for invalid_value in self._data_invalid_values:
                with self.tc.assertRaises(ValueError):
                    self._class.from_dict(invalid_value)

    def test_from_dict_with_complete_data(self):
        # Arrange
        class_config = "pof.load.cf"

        with patch.dict(class_config, {"handle_invalid_data": False}):
            with patch.dict(class_config, {"on_error_use_default": False}):
                for data in self._data_complete.values():
                    # Act
                    instance = self._class.from_dict(data)

                    # Assert
                    self.tc.assertIsNotNone(instance)
                    self.tc.assertTrue(isinstance(instance, self._class))

    # ************ Test load ***********************

    def test_load_with_empty(self):
        instance = self._class.load()
        self.tc.assertIsNotNone(instance)
        self.tc.assertTrue(isinstance(instance, self._class))

    def test_load_valid_dict(self):
        for data in self._data_valid:
            # Arrange
            instance_from_dict = self._class.from_dict(data)

            # Act
            instance = self._class.load(data)

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

    # ************ Test __ methods __

    def test_equivalence(self):

        # Arrange / Act
        inst_1 = self._class.demo()
        inst_2 = self._class.demo()
        inst_3 = self._class(name="a different name")

        # Assert
        self.tc.assertTrue(inst_1 == inst_2)
        self.tc.assertTrue(inst_2 == inst_1)
        self.tc.assertTrue(inst_1 != inst_3)
        self.tc.assertTrue(inst_3 != inst_1)
        self.tc.assertTrue(not (inst_1 is inst_2))
        self.tc.assertTrue(not (inst_2 is inst_3))

    # *************** Test Update *************

    def test_update(self):

        # Arrange
        data = self._data_complete[0]
        instance_0 = self._class.from_dict(data)
        instance_1 = self._class.from_dict(self._data_complete[1])

        with patch.dict("pof.load.cf", {"handle_update_error": False}):

            # Act
            instance_1.update(data)

        msg = []
        for key, val in instance_0.__dict__.items():
            if instance_1.__dict__[key] != val:
                msg.append((key, val))

        # Assert
        self.tc.assertEqual(instance_0, instance_1, msg=msg)

    def test_update_errors_logged(self):
        """ Checks that an error is raised when invalid input data is provided"""
        # Arrange
        class_config = "pof.load.cf"
        instance = self._class.demo()

        invalid_data = self._data_invalid_types + self._data_invalid_values

        for data in invalid_data:
            with patch.dict(class_config, {"handle_update_error": True}):
                with self.tc.assertLogs(level="DEBUG") as log:
                    # Act
                    instance.update(data)

                    # Assert
                    self.tc.assertTrue("Update Failed" in log.output[-1])

    def test_update_errors_raised(self):
        """ Checks that an error is raised when invalid input data is provided"""
        # Arrange
        class_config = "pof.load.cf"
        instance = self._class.demo()

        invalid_data = self._data_invalid_types + self._data_invalid_values

        for data in invalid_data:
            with patch.dict(class_config, {"handle_update_error": False}):
                with self.tc.assertRaises((ValueError, AttributeError, KeyError)):
                    # Act
                    instance.update(data)

    # ************* Test units ******************

    def test_units_undefined(self):
        """ Case when current_units is not defined, sets to None & self._units = value """

        # Arrange
        input_data = ["seconds", "minutes", "hours", "days", "weeks", "months", "years"]

        for val in input_data:
            # Act
            instance = self._class.from_dict(self._data_complete[0])
            instance.units = val

            # Assert
            self.tc.assertEqual(instance.units, val)

    def test_units_invalid(self):
        """ Case when value not in valid units raises error """

        # Arrange
        input_data = ["test"]

        for val in input_data:
            # Act
            instance = self._class.from_dict(self._data_complete[0])

            # Assert
            with self.tc.assertRaises(ValueError):
                instance.units = val

    def test_units_defined(self):
        """Case when current_units is defined, but equal to value, set self._units = value
        && Case when current_units is not None & not equal to value, calls self._scale_units then sets self._units = value"""

        # Arrange
        input_data = ["months", "years"]

        # Act
        instance = self._class.from_dict(self._data_complete[0])
        mock = Mock()

        for val in input_data:
            instance.units = val

            mock._scale_units(val, "years")

            # if getattr(self.tc, instance.units) == val:
            # mock that the scale units call was triggered
            # mock._scale_units.assert_called_with(val, getattr(self.tc, instance.units))

            # Assert
            self.tc.assertEqual(instance.units, val)

    def test_scale_units_integration(self):
        """ Integration test for scale_units -- Scale down and then back up """

        # Arrange
        instance = self._class.from_dict(self._data_complete[0])
        instance.pf_interval = 10
        instance.pf_std = 0.0001
        current_value = []
        months_value = []
        return_value = []

        # Act
        current_units = instance.units
        for var in instance.TIME_VARIABLES:
            current_value.append(getattr(instance, var))

        instance.units = "months"
        for var in instance.TIME_VARIABLES:
            months_value.append(getattr(instance, var))

        instance.units = current_units
        for var in instance.TIME_VARIABLES:
            return_value.append(getattr(instance, var))

        # Assert
        i = 0
        for var in instance.TIME_VARIABLES:
            self.tc.assertEqual(return_value[i], current_value[i])
            i = i + 1
