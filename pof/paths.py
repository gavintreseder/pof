import os
from os.path import dirname

# Change the system path if an individual file is being run
if __package__ is None or __package__ == "":
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class Paths:
    def __init__(self):

        self._root_path = dirname(os.getcwd())
        self._pof_path = self._root_path + r"\pof"
        self._input_path = self._root_path + r"\inputs"
        self._csv_path = self._input_path + r"\csvs"
        self._demo_path = self._pof_path + r"\data\inputs"
        self._output_path = self._root_path + r"\outputs"


if __name__ == "__main__":
    paths = Paths()
    print("Paths - Ok")