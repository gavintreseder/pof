import pandas as pd
import datetime
import numpy as np
import os

from pof.loader.fleet_data_loader import FleetDataLoader
from pof.loader.fleet_data import FleetData

import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory

try:
    file_path
except:
    file_path = askdirectory(initialdir=os.getcwd())

# I don't think this is right..
asset_path = file_path + "\\ACS - Poles - Asset Details.csv"
condition_path = file_path + "\\ACS - Poles - Condition History.csv"
csq_path = file_path + "\\ACS - Poles - Consequence Model Output.csv"
intervention_path = file_path + "\\ACS - Poles - Intervention History.csv"


class PolesFleetDataLoader(FleetDataLoader):
    """
    Anything extra that awe
    """

    def __init__(self, df_csq=None, df_asset_info=None, condition_data=None):
        self.df_csq = df_csq
        self.df_asset_info = df_asset_info
        self.condition_data = condition_data

    def load_asset_(self):
        self.load_csq_model()
        self.load_asset_info()
        self.load_condition_data()

    def load_csq_model(self):
        csq_columns = [
            "Asset ID",
            "BushfirePriority",
            "TransformerOnPole",
            "PremiseCount",
            "Travel Time to Pole (hrs)",
        ]

        self.df_csq = self.from_csv(csq_path)
        self.df_csq = self.df_csq.rename(columns={"ASSET_ID": "Asset ID"})
        self.df_csq = self.df_csq[csq_columns]

    def load_asset_info(self):
        csq_columns = [
            "C_Safety_Dollars",
            "C_Network_Dollars",
            "C_Bushfire_Dollars",
            "C_Environment_Dollars",
            "C_Financial_Dollars",
            "Total Consequence $",
        ]

        self.df_asset_info = self.from_csv(asset_path)
        self.df_asset_info["Date Installed"] = pd.to_datetime(
            self.df_asset_info["Date Installed"], format="%Y%m%d", errors="coerce"
        )

        csq = self.from_csv(csq_path)
        csq = csq.rename(columns={"ASSET_ID": "Asset ID"})
        csq = csq[csq_columns]

        self.df_asset_info = self.df_asset_info.merge(csq, on="Asset ID")

        intervention = self.from_csv(intervention_path)
        intervention = intervention[["Asset ID", "Pseudo Asset ID"]].drop_duplicates()

        self.df_asset_info = self.df_asset_info.merge(intervention, on="Asset ID")

    def load_condition_data(self):
        df_condition = self.from_csv(condition_path)
        df_condition["Date Changed"] = pd.to_datetime(
            df_condition["Date Changed"], format="%Y-%m-%d %H:%M:%S"
        ).dt.normalize()

        # used to drop additional columns later
        cond_columns = df_condition.columns

        df_intervention = self.from_csv(intervention_path)
        df_intervention = df_intervention[
            ["Asset ID", "Pseudo Asset ID"]
        ].drop_duplicates()

        # create column of replacement dates
        df_intervention["Replace Date"] = df_intervention["Pseudo Asset ID"].map(
            lambda x: x.split("-", 1)[1] if len(x.split("-", 1)) == 2 else np.nan
        )
        df_intervention["Replace Date"] = pd.to_datetime(
            df_intervention["Replace Date"], format="%Y-%m-%d"
        )

        df_condition = df_condition.merge(df_intervention, on="Asset ID")

        # compare dates if replacement date < condition date the use Pseudo Asset ID
        df_condition["Asset ID"] = np.where(
            df_condition["Replace Date"] > df_condition["Date Changed"],
            df_condition["Asset ID"],
            df_condition["Pseudo Asset ID"],
        )

        df_condition = df_condition[cond_columns]

        # create dictionary
        c_dict = (
            df_condition.groupby("Asset ID")[df_condition.columns[1:]]
            .apply(lambda g: g.values.tolist())
            .to_dict()
        )

        self.condition_data = c_dict

    def get_fleet_data(self):
        """
        Creates a FleetData object
        """
        return FleetData(
            asset_info=self.df_asset_info,
            condition=self.condition_data,
            csq_info=self.df_csq,
        )
