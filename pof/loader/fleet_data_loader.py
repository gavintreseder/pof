import pandas as pd


class FleetDataLoader:
    """
    Handles the loading of data from input files and should create a FleetData object when it all works
    """

    # file_paths
    # file_types
    def from_file(self, path: str):
        """
        Selects the appropriate load function based on the string type
        """
        if path.endswith(".txt"):
            df = self.from_txt(path)
        elif path.endswith(".csv"):
            df = self.from_csv(path)
        else:
            raise ValueError("Invalid path/filetype")
        return df

    def from_txt(self, path):
        """"""
        try:
            df = pd.read_csv(path, delimiter="\t")
        except UnicodeDecodeError:
            df = pd.read_csv(
                path, delimiter="\t", encoding="utf-16"
            )  # for consequence model

        return df

        NotImplemented

    def from_csv(self, path):

        df = pd.read_csv(path)
        # Open the file
        return df