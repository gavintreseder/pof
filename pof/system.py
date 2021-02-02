# ************ Packages ********************
import copy
import logging
from typing import Dict

import numpy as np
import pandas as pd
import json

# Change the system path if an individual file is being run
if __package__ is None or __package__ == "":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pof.pof_base import PofBase
from config import config
from pof.pof_container import PofContainer
from pof.component import Component
from pof.paths import Paths
import pof.demo as demo

cf = config.get("System")


class System(PofBase):
    """
    Parameters:

    Methods:


    Usage:


    """

    TIME_VARIABLES = {}
    POF_VARIABLES = ["comp"]

    def __init__(
        self,
        name: str = "sys",
        active: bool = True,
        comp: Dict = None,
        *args,
        **kwargs,
    ):

        super().__init__(name=name, *args, **kwargs)

        self.active = active
        self.comp = PofContainer()

        self.set_component(comp)

    # ****************** Load data ******************

    def set_component(self, comp_input):
        """Takes a dictionary of Component objects or Component data and sets the system components """
        self.set_obj("comp", Component, comp_input)

    # ****************** Interface ******************

    def get_objects(self, prefix="", sep="-"):
        """ Returns a list of objects to populate the layout """
        objects = [prefix + self.name]

        # Create the system prefix ("system"-)
        prefix = prefix + self.name + sep

        # Add hte comp objects onto "system-"
        for comp in self.comp.values():
            objects = objects + comp.get_objects(prefix=prefix + "comp" + sep)

        return objects

    def get_dash_ids(self, numericalOnly: bool, prefix="", sep="-", active=None):
        """Return a list of dash ids for values that can be changed"""
        if active is None or (self.active == active):
            # System ids
            prefix = prefix + self.name + sep
            sys_ids = [prefix + param for param in ["active"]]

            # Component ids
            comp_ids = []
            for comp in self.comp.values():
                comp_ids = comp_ids + comp.get_dash_ids(
                    numericalOnly=numericalOnly,
                    prefix=prefix + "comp" + sep,
                    sep=sep,
                    active=active,
                )

            dash_ids = sys_ids + comp_ids
        else:
            dash_ids = []

        return dash_ids

    def get_update_ids(self, numericalOnly=bool, prefix="", sep="-"):
        """ Get the ids for all objects that should be updated"""

        # Get the system ids
        ids_sys = self.get_dash_ids(numericalOnly=numericalOnly, active=True)

        # Get the component ids
        for comp in self.comp.values():
            ids_comp = comp.get_update_ids(
                numericalOnly=numericalOnly, prefix=prefix, sep=sep
            )

        ids = ids_sys + ids_comp

        return ids

    def save(self, file_name):
        """ Save a json file with a system """

        # Create the data set
        data = self.to_dict()

        # Save to json
        with open(Paths().model_path + "\\" + file_name, "w") as json_file:
            json.dump(data, json_file)

    # ****************** Demonstration parameters ******************

    @classmethod
    def demo(cls):
        """ Loads a demonstration data set if no parameters have been set already"""

        return cls.load(demo.system_data["overhead_network"])


if __name__ == "__main__":
    system = System()
    print("System - Ok")