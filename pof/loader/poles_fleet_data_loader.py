import os
import datetime

import pandas as pd
import numpy as np
import dask.array as da
import dask.dataframe as dd
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory

if __package__ is None or __package__ == "":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pof.loader.fleet_data_loader import FleetDataLoader
from pof.loader.fleet_data import FleetData


# Constant
prohibbited_characters = [".", ";", "-"]
replacement_character = ["_"]


class PolesFleetDataLoader(FleetDataLoader):
    """
    Anything extra that awe
    """

    def __init__(self, df_csq=None, df_asset_info=None, condition_data=None):
        self.df_csq = df_csq
        self.df_asset_info = df_asset_info
        self.condition_data = condition_data
        # TODO: make sure all cls variable initialised
        self.asset_path = None

        self.load_asset_()

    def get_path(self):
        try:
            file_path
        except:
            file_path = askdirectory(initialdir=os.getcwd())

        self.asset_path = file_path + "\\ACS - Poles - Asset Details.csv"
        self.condition_path = file_path + "\\ACS - Poles - Condition History.csv"
        self.csq_path = file_path + "\\ACS - Poles - Consequence Model Output.csv"
        self.intervention_path = file_path + "\\ACS - Poles - Intervention History.csv"

    def load_asset_(self, path=None):
        if path is None:
            self.get_path()

        self.load_csq_model()
        self.load_asset_info()
        self.load_condition_data()

    def load_csq_model(self):
        """
        Legacy not needed
        """
        print("loading consequence model")
        csq_columns = [
            "ASSET_ID",
            "BushfirePriority",
            "TransformerOnPole",
            "PremiseCount",
            "Travel Time to Pole (hrs)",
        ]

        self.df_csq = self.from_file(self.csq_path, csq_columns)
        self.df_csq = self.df_csq.rename(columns={"ASSET_ID": "Asset ID"})
        self.df_csq["Asset ID"] = self.df_csq["Asset ID"].astype(str)

        self.df_csq = self.replace_substring(self.df_csq)
        self.df_csq = self.replace_columns(self.df_csq)
        self.df_csq = self.df_csq.repartition(npartitions=1).reset_index(drop=True)
        print("consequence model loaded")

    def load_asset_info(self):
        print("loading asset information")
        csq_columns = [
            "ASSET_ID",
            "C_Safety_Dollars",
            "C_Network_Dollars",
            "C_Bushfire_Dollars",
            "C_Environment_Dollars",
            "C_Financial_Dollars",
            "Total Consequence $",
        ]
        dtype = {"Pole Length": "object"}

        self.df_asset_info = self.from_file(self.asset_path, dtype=dtype)
        self.df_asset_info["Date Installed"] = dd.to_datetime(
            self.df_asset_info["Date Installed"], format="%Y%m%d", errors="coerce"
        )

        intervention_columns = ["Asset ID", "Pseudo Asset ID"]

        csq = self.from_file(self.csq_path, csq_columns)
        csq = csq.rename(columns={"ASSET_ID": "Asset ID"})

        csq["Asset ID"] = csq["Asset ID"].astype(str)
        self.df_asset_info["Asset ID"] = self.df_asset_info["Asset ID"].astype(str)

        self.df_asset_info = self.df_asset_info.merge(csq, on="Asset ID")

        intervention = self.from_file(
            self.intervention_path, intervention_columns
        ).drop_duplicates()

        intervention["Asset ID"] = intervention["Asset ID"].astype(str)

        self.df_asset_info = self.df_asset_info.merge(intervention, on="Asset ID")

        self.df_asset_info = self.replace_substring(self.df_asset_info)
        self.df_asset_info = self.replace_columns(self.df_asset_info)
        self.df_asset_info = self.df_asset_info.repartition(npartitions=1).reset_index(
            drop=True
        )
        print("asset information loaded")

    def load_condition_data(self):
        "loading condition data"
        intervention_columns = ["Asset ID", "Pseudo Asset ID"]

        df_condition = self.from_file(
            self.condition_path,
            dtype={"After Value": "object", "Before Value": "object"},
        )
        df_condition["Date Changed"] = dd.to_datetime(
            df_condition["Date Changed"], format="%Y-%m-%d %H:%M:%S"
        ).dt.normalize()

        # used to drop additional columns later
        cond_columns = df_condition.columns

        df_intervention = self.from_file(
            self.intervention_path, intervention_columns
        ).drop_duplicates()

        # create column of replacement dates
        df_intervention["Replace Date"] = df_intervention["Pseudo Asset ID"].map(
            lambda x: x.split("-", 1)[1] if len(x.split("-", 1)) == 2 else np.nan
        )
        df_intervention["Replace Date"] = dd.to_datetime(
            df_intervention["Replace Date"], format="%Y-%m-%d"
        )

        df_intervention["Asset ID"] = df_intervention["Asset ID"].astype(str)
        df_condition["Asset ID"] = df_condition["Asset ID"].astype(str)

        df_condition = df_condition.merge(df_intervention, on="Asset ID")

        # compare dates if replacement date < condition date the use Pseudo Asset ID
        df_condition["Asset ID"] = da.where(
            df_condition["Replace Date"].to_dask_array()
            > df_condition["Date Changed"].to_dask_array(),
            df_condition["Asset ID"].to_dask_array(),
            df_condition["Pseudo Asset ID"].to_dask_array(),
        )

        df_condition = df_condition[cond_columns]

        df_condition = self.replace_substring(df_condition)
        df_condition = self.replace_columns(df_condition)
        df_condition = df_condition.repartition(npartitions=1).reset_index(drop=True)

        self.condition_data = df_condition
        print("condition data loaded")

    def replace_substring(
        self, df, replace=prohibbited_characters, replace_with=replacement_character
    ):

        for r in replace:
            df_replaced = df.applymap(
                lambda s: s.replace(r, replace_with[0]) if isinstance(s, str) else s
            )
        df_replaced = df.applymap(
            lambda s: s.replace(" ", "") if isinstance(s, str) else s
        )
        # df_replaced = df.where(~isinstance(df, str), df.replace(r, replace_with[0]))

        return df_replaced

    def cleanse_numerical(self, df, columns, replace_with=np.nan):

        for c in columns:
            df[c] = df[c].where(df[c].str.isnumeric(), replace_with).astype("float")

        return df

    def replace_columns(self, df):

        specific_columns_rename = dict(
            detail_code="condition_name",
            date_changed="date",
            before_value="condition_before",
            after_value="condition_after",
            total_consequence_="total_csq",
            date_installed="installation_date",
        )

        replaced = []
        for col in list(df.columns):
            new_col = col.lower()
            new_col = new_col.replace(" ", "_")
            new_col = new_col.replace("$", "")
            replaced.append(new_col)
            for name, rename in specific_columns_rename.items():
                if new_col == name:
                    new_col = rename
                    replaced[-1] = new_col

        df.columns = replaced

        return df

    def get_fleet_data(self):
        """
        Creates a FleetData object
        """
        print("fleet data object created")
        return FleetData(
            asset_info=self.df_asset_info,
            condition=self.condition_data,
            csq_info=self.df_csq,
        )


if __name__ == "__main__":
    print("ok")