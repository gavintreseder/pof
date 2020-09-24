"""
    Filename: test_load.py
    Description: Contains the code for testing the Constructor class
    Author: Gavin Treseder | gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
 
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import copy

import utils

from pof.load import Load
import pof.demo as demo
import fixtures


"""
class TestLoad(unittest.TestCase):
    @patch(cf.on_error_use_default, False)
    def test_first_case(self):
        self.assertRaises(Load.load, "Invalid Data")

    @patch(cf.on_error_use_default, False)
    def test_n(self):
        ldr = Load()
        ldr = Load("Invalid_data")
        self.assert"""


class TestLoad(unittest.TestCase):
    def test_set_container_attr_object(self):
        load = Load()
        test_data = Mock()
        test_data.name = "mock_name"
        d_type = Mock

        load._set_container_attr("dummy_att", d_type, test_data)

        self.assertEqual(load.dummy_att[test_data.name], test_data)

    def test_set_container_attr_dict_of_object(self):
        load = Load()
        test_data = dict(mock_data=Mock())
        test_data["mock_data"].name = "mock_name"
        d_type = Mock

        load._set_container_attr("dummy_att", d_type, test_data)

        self.assertEqual(
            load.dummy_att[test_data["mock_data"].name], test_data["mock_data"]
        )

    def test_set_container_attr_dict_of_dict(self):

        load = Load()
        test_data = dict(mock_data=dict(name="mock_name"))
        d_type = Mock()
        d_type.load.return_value = Mock(name="mock_name")

        load._set_container_attr("dummy_att", d_type, test_data)

        self.assertEqual(load.dummy_att[test_data.name], test_data)

    def test_set_container_attr_dict(self):

        load = Load()
        test_data = dict(name="mock_name")
        d_type = Mock()
        d_type.load.return_value = Mock(name="mock_name")

        load._set_container_attr("dummy_att", d_type, test_data)

        self.assertEqual(load.dummy_att[test_data.name], test_data)
