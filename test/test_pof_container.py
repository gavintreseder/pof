"""
    Filename: test_pof_container.py
    Description: Contains the code for testing the PofContainer class
    Authors:
        Gavin Treseder | gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
"""

import unittest
from unittest.mock import Mock

import testconfig  # pylint: disable=unused-import
from pof.pof_container import PofContainer
from pof.load import Load


class TestPofContainer(unittest.TestCase):
    """
    Unit tests for the PofContainer class

    """

    def setUp(self):

        # Mocked approach
        # pof_obj_1 = Mock(data="pof_data_1")
        # pof_obj_1.name = "pof_obj_1"
        # pof_obj_1.update_from_dict

        # pof_obj_2 = Mock(data="pof_data_2")
        # pof_obj_2.name = "pof_obj_2"

        # Using a pof object
        pof_obj_1 = Load(name="pof_obj_1")
        pof_obj_1.pof_data = "no_change"

        pof_obj_2 = Load(name="pof_obj_2")
        pof_obj_2.pof_data = "no_change"

        pof_obj_3 = Load(name="pof_obj_3")
        pof_obj_3.pof_data = "no_change"

        self.pof_con = PofContainer(
            pof_obj_1=pof_obj_1, pof_obj_2=pof_obj_2, pof_obj_3=pof_obj_3
        )

    def test_update_from_dict_name_changed(self):
        """ Check the key and name are changed if they are already in use"""
        # Arrange
        pof_con = self.pof_con
        update = {
            "pof_obj_1": {"name": "updated_name"},
            "pof_obj_2": {"name": "updated_name"},
            "pof_obj_3": {"name": "updated_name"},
        }

        # Act
        pof_con.update_from_dict(update)

        # Assert
        for obj in pof_con.values():
            actual = obj.pof_data
            expected = "no_change"
            self.assertEqual(actual, expected)

        for suffix in ["", "|1", "|2"]:
            expected = "updated_name" + suffix
            actual = pof_con[expected].name
            self.assertEqual(actual, expected)

    def test_update_from_dict_name_not_changed(self):
        """ Check keys and names aren't changed when other data is updated"""

        param_list = [
            ("pof_obj_1", "updated_data"),
            ("pof_obj_2", "updated_data"),
            ("pof_obj_3", "no_change"),
        ]

        # Arrange
        pof_con = self.pof_con
        expected = "updated_data"
        update = {
            "pof_obj_1": {"pof_data": expected},
            "pof_obj_2": {"pof_data": expected},
        }

        # Act
        pof_con.update_from_dict(update)

        # Assert
        for name, expected in param_list:
            actual = pof_con[name].pof_data
            self.assertEqual(actual, expected)

        self.assertEqual(len(pof_con), 3, msg="Length should not change")


if __name__ == "__main__":
    unittest.main()
