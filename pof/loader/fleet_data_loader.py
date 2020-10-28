import pandas as pd
import dask.dataframe as dd


class FleetDataLoader:
    """
    Handles the loading of data from input files and should create a FleetData object when it all works
    """

    # file_paths
    # file_types
    def from_file(self, path: str, columns=None, dtype=None):
        """
        Selects the appropriate load function based on the string type
        """
        if path.endswith(".txt"):
            df = self.from_txt(path, columns, dtype)
        elif path.endswith(".csv"):
            df = self.from_csv(path, columns, dtype)
        else:
            raise ValueError("Invalid path/filetype")
        return df

    def from_txt(self, path, columns=None, dtype=None):
        """"""
        try:
            df = dd.read_csv(
                path,
                delimiter="\t",
                blocksize=16 * 1024 * 1024,
                usecols=columns,
                dtype=dtype,
            )
            # txt_chunks = pd.read_csv(
            # path, delimiter="\t", usecols=columns, chunksize=100
            # )
            # df = pd.concat(chunk for chunk in txt_chunks)

        except UnicodeDecodeError:
            # df = dd.read_csv(path, delimiter="\t", encoding="ISO-8859-1")
            txt_chunks = pd.read_csv(
                path,
                delimiter="\t",
                usecols=columns,
                encoding="utf-16",
                chunksize=10000,
            )
            df = pd.concat(chunk for chunk in txt_chunks)

        return df

    def from_csv(self, path, columns=None, dtype=None):

        df = dd.read_csv(
            path,
            blocksize=16 * 1024 * 1024,
            usecols=columns,
            dtype=dtype,
        )

        return df