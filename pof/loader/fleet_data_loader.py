import pandas as pd
import dask.dataframe as dd
import numpy as np


class FleetDataLoader:
    """
    Handles the loading of data from input files and should create a FleetData
    object when it all works
    """

    def from_file(self, path: str, columns=None, dtype=None, nrows=None):
        """
        Selects the appropriate load function based on the string type
        """
        if path.endswith(".txt"):
            dataframe = self.from_txt(path, columns, dtype, nrows)
        elif path.endswith(".csv"):
            dataframe = self.from_csv(path, columns, dtype, nrows)
        else:
            raise ValueError("Invalid path/filetype")
        return dataframe

    def from_txt(self, path, columns=None, dtype=None, nrows=None):
        """
        Load function for .txt files
        """
        try:
            dataframe = dd.read_csv(
                path,
                delimiter="\t",
                blocksize=16 * 1024 * 1024,
                usecols=columns,
                dtype=dtype,
            )

        except UnicodeDecodeError:

            txt_chunks = pd.read_csv(
                path,
                delimiter="\t",
                usecols=columns,
                encoding="utf-16",
                chunksize=10000,
            )
            dataframe_pandas = pd.concat(chunk for chunk in txt_chunks)
            dataframe = dd.from_pandas(dataframe_pandas, npartitions=1).reset_index()

        if nrows:
            dataframe = dataframe.head(n=nrows)
            dataframe = dd.from_pandas(dataframe, npartitions=1).reset_index()

        return dataframe

    def from_csv(self, path, columns=None, dtype=None, nrows=None):
        """
        Load function for .csv files
        """

        dataframe = dd.read_csv(
            path,
            blocksize=16 * 1024 * 1024,
            usecols=columns,
            dtype=dtype,
        )

        if nrows:
            dataframe = dataframe.head(n=nrows)
            dataframe = dd.from_pandas(dataframe, npartitions=1).reset_index()

        return dataframe

    def cleanse_data(
        self,
        df,
        replace=None,
        replace_with=None,
        numerical_columns=None,
        columns_rename=None,
    ):
        """
        Cleanses data by replacing string characters, cleansing numerical data columns and
        renaming specified column headings
        """

        df = self.replace_substring(df=df, replace=replace, replace_with=replace_with)
        df = self.replace_columns(df=df, columns_rename=columns_rename)
        df = self.cleanse_numerical(df=df, numerical_columns=numerical_columns)
        df = self.asset_id_to_string(df=df)
        df = self.date_formatter(df=df)
        df = self.dataframe_repartition(df=df)

        return df

    def replace_substring(self, df, replace, replace_with):
        """
        Replaces list of prohibited characters present in data with given replacement character
        """

        if replace:
            for r in replace:
                df_replaced = df.applymap(
                    lambda s: s.replace(r, replace_with[0]) if isinstance(s, str) else s
                )
            df_replaced = df.applymap(
                lambda s: s.replace(" ", "") if isinstance(s, str) else s
            )
        else:
            df_replaced = df

        return df_replaced

    def cleanse_numerical(self, df, numerical_columns):
        """
        Converts string numerical data columns to floats and replaces any strings, e.g. "UNKNOWN", to a given value (default np.nan)
        """
        if numerical_columns:
            for c in numerical_columns:
                if df[c].dtype == "object":
                    df[c] = df[c].where(df[c].str.isnumeric(), np.nan).astype("float")

        return df

    def asset_id_to_string(self, df):
        """
        Converts asset id columns to strings
        """

        df["asset_id"] = df["asset_id"].astype("string")

        return df

    def dataframe_repartition(self, df):
        """
        Repartitions the dataframe and resets the index
        """

        df = df.repartition(npartitions=1).reset_index(drop=True)

        return df

    def replace_columns(self, df, columns_rename):
        """
        Replaces known column headings to be changed for easier identification
        """

        if columns_rename:
            replaced = []
            for col in list(df.columns):
                new_col = col.lower()
                new_col = new_col.replace(" ", "_")
                new_col = new_col.replace("$", "")
                replaced.append(new_col)
                for name, rename in columns_rename.items():
                    if new_col == name:
                        new_col = rename
                        replaced[-1] = new_col

            df.columns = replaced

        return df

    def date_formatter(self, df):
        """
        Converts date data to the correct format
        """

        if "installation_date" in df.columns:
            df.installation_date = dd.to_datetime(
                df.installation_date, format="%Y%m%d", errors="coerce"
            ).dt.date

        if "date" in df.columns:
            df.date = df.date.apply(lambda x: x[:10])
            df.date = dd.to_datetime(df.date, format="%Y-%m-%d").dt.date

        if "replace_date" in df.columns:
            df.replace_date = dd.to_datetime(df.replace_date, format="%Y-%m-%d").dt.date

        return df
