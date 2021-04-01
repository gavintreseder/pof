import os

# Change the system path if an individual file is being run
if __package__ is None or __package__ == "":
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class Paths:
    """
    A class which defines the file paths required
    """

    def __init__(self):

        self.root_path = os.path.dirname(os.getcwd()) + os.sep
        self.pof_path = self.root_path + r"\pof" + os.sep
        self.demo_path = self.pof_path + r"\data\inputs" + os.sep
        self.input_path = self.root_path + r"\inputs" + os.sep
        self.test_path = self.input_path + r"\test_inputs" + os.sep
        self.csv_path = self.input_path + r"\csvs" + os.sep
        self.model_path = self.input_path + r"\model" + os.sep
        self.output_path = self.root_path + r"\outputs" + os.sep


if __name__ == "__main__":
    paths = Paths()
    print("Paths - Ok")