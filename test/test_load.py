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


class TestLoad(unittest.TestCase):
    """def setUp(self):

    # Mock the pof object
    self.pof_obj = Mock()
    self.pof_obj.name = "mock_name"
    self.pof_obj.load.return_value = Mock()"""

    def test_imports_correctly(self):
        self.assertIsNotNone(Load)

    def test_class_instantiate(self):
        self.assertIsNotNone(Load())

    def test_set_container_attr_from_dict(self):

        load = Load()
        test_data = dict(name="test")
        expected = Load(name="test")

        load._set_container_attr("name", Load, test_data)

        self.assertEqual(load.name[expected.name], expected)

    def test_set_container_attr_from_dict_of_dicts(self):

        load = Load()
        test_data = dict(pof_object=dict(name="test"))
        expected = Load(name="test")

        load._set_container_attr("name", Load, test_data)

        self.assertEqual(load.name[expected.name], expected)

    def test_set_container_attr_from_dict_of_objects(self):

        load = Load()
        test_data = dict(pof_object=Load(name="test"))
        expected = Load(name="test")

        load._set_container_attr("name", Load, test_data)

        self.assertEqual(load.name[expected.name], expected)

    def test_set_container_attr_from_object(self):

        load = Load()
        test_data = Load(name="test")
        expected = Load(name="test")

        load._set_container_attr("name", Load, test_data)

        self.assertEqual(load.name[expected.name], expected)

    def test_set_container_attr_existing_data_from_dict(self):

        load = Load(name=dict(test=Load(name="this_should_change")))
        test_data = dict(name="test")
        expected = Load(name="test")

        load._set_container_attr("name", Load, test_data)

        self.assertEqual(load.name[expected.name], expected)

    def test_set_container_attr_existing_data_from_dict_of_dicts(self):

        load = Load(name=dict(test=Load(name="this_should_change")))
        test_data = dict(pof_object=Load(name="test"))
        expected = Load(name="test")

        load._set_container_attr("name", Load, test_data)

        self.assertEqual(load.name[expected.name], expected)

    def test_set_container_attr_existing_data_from_dict_of_objects(self):

        load = Load(name=dict(test=Load(name="this_should_change")))
        test_data = dict(pof_object=Load(name="test"))
        expected = Load(name="after_update")
        key_before_update = "test"

        load._set_container_attr("name", Load, test_data)
        test_data["pof_object"].name = "after_update"

        self.assertEqual(load.name[key_before_update], expected)

    def test_set_container_attr_existing_data_from_object(self):

        load = Load(name=dict(test=Load(name="this_should_change")))
        test_data = Load(name="test")
        expected = Load(name="after_update")
        key_before_update = "test"

        load._set_container_attr("name", Load, test_data)
        test_data.name = "after_update"

        self.assertEqual(load.name[key_before_update], expected)