from pathlib import Path

import os
from os.path import dirname, realpath

str(Path(dirname(realpath(__file__))).parents[1])

# Change the system path if an individual file is being run
if __package__ is None or __package__ == "":
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class Paths:
    """
    A class which defines the file paths required
    """

    def __init__(self):

        self.root_path = str(Path(dirname(realpath(__file__))).parents[1]) + os.sep
        self.pof_path = self.root_path + "pof" + os.sep
        self.demo_path = self.pof_path + "data" + os.sep + "inputs" + os.sep
        self.input_path = self.root_path + "inputs" + os.sep
        self.test_path = self.input_path + "test_inputs" + os.sep
        self.csv_path = self.input_path + "csvs" + os.sep
        self.model_path = self.input_path + "model" + os.sep
        self.output_path = self.root_path + "outputs" + os.sep


if __name__ == "__main__":
    paths = Paths()
    print("Paths - Ok")