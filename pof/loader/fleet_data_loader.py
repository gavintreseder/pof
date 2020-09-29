import pandas as pd


class FleetDataLoader:
    """
    Handles the loading of data from input files and should create a FleetData object when it all works
    """

    # file_paths
    # file_types

    def from_txt(self, path):
        df = pd.read_csv(path, delimiter="\t")
        # df = pd.read_csv(path + ".txt", delimiter="\t", encoding='utf-16') # for consequence model
        return df

        NotImplemented

    def from_csv(self, path):

        df = pd.read_csv(path)
        # Open the file
        return df