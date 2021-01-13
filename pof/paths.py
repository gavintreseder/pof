import os
from os.path import dirname, realpath
from pathlib import Path

# Change the system path if an individual file is being run
if __package__ is None or __package__ == "":
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class Paths:
    """
    A class which defines the file paths required
    """

    def __init__(self):

        self.root_path = str(Path(dirname(realpath(__file__))).parents[1])
        self.pof_path = self.root_path + r"\pof"
        self.input_path = self.root_path + r"\inputs"
        self.csv_path = self.input_path + r"\csvs"
        self.model_path = self.input_path + r"\model"
        self.output_path = self.root_path + r"\outputs"

if __name__ == "__main__":
    paths = Paths()
    print("Paths - Ok")