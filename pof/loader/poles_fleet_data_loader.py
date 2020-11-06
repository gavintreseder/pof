import logging
import os

from tkinter.filedialog import askdirectory
import numpy as np
import dask.array as da
import dask.dataframe as dd

from pof.loader.fleet_data_loader import FleetDataLoader
from pof.loader.fleet_data import FleetData

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class PolesFleetDataLoader(FleetDataLoader):
    """
    Anything extra that awe
    """

    pandas_or_dask = "pandas"

    file_path = "C:\\Users\\ischram\\OneDrive - KPMG\\Desktop\\Data\\csv"

    nrows = None

    prohibbited_characters = [".", ";", "-"]
    replacement_character = ["_"]

    specific_columns_rename = dict(
        detail_code="condition_name",
        date_changed="date",
        before_value="condition_before",
        after_value="condition_after",
        total_consequence_="total_consequence",
        date_installed="installation_date",
    )

    csq_columns = dict(
        asset_info=[
            "asset_id",
            "c_safety_dollars",
            "c_network_dollars",
            "c_bushfire_dollars",
            "c_environment_dollars",
            "c_financial_dollars",
            "total_consequence",
        ],
        csq=[
            "asset_id",
            "bushfirepriority",
            "transformeronpole",
            "premisecount",
            "travel_time_to_pole_(hrs)",
            "travel_distance_to_pole_(km)",
        ],
    )

    intervention_columns = ["asset_id", "pseudo_asset_id"]

    numerical_columns = dict(
        asset_info=[
            "latitude",
            "longitude",
            "premise_count",
            "pole_length",
            "pole_strength",
        ],
        csq=[
            "premisecount",
            "travel_time_to_pole_(hrs)",
            "travel_distance_to_pole_(km)",
        ],
        condition=[
            "condition_before",
            "condition_after",
        ],
    )

    dtype = dict(
        asset_info={"Pole Length": "object"},
        condition={"After Value": "object", "Before Value": "object"},
        csq={"Radial": "object", "Radial(Y/N)": "object"},
    )

    def __init__(
        self,
        df_csq=None,
        df_asset_info=None,
        condition_data=None,
    ):
        self.df_csq = df_csq
        self.df_asset_info = df_asset_info
        self.condition_data = condition_data

        self.asset_path = None
        self.condition_path = None
        self.csq_path = None
        self.intervention_path = None

        self.load()

    def get_path(self):
        """
        Requests user to input file path if not specified
        """

        logging.debug("Retrieving file path")

        file_path = (
            askdirectory(initialdir=os.getcwd())
            if self.file_path is None
            else self.file_path
        )

        self.asset_path = file_path + "\\" + "ACS - Poles - Asset Details.csv"
        self.condition_path = file_path + "\\" + "ACS - Poles - Condition History.csv"
        self.csq_path = file_path + "\\" + "ACS - Poles - Consequence Model Output.csv"
        self.intervention_path = (
            file_path + "\\" + "ACS - Poles - Intervention History.csv"
        )

        logging.debug("File path retrieved")

    def load(self):
        """
        Loads poles data
        """

        df_dict = self.read_data()

        self.merge_data(
            df_csq=df_dict["csq"],
            df_asset_info=df_dict["asset_info"],
            df_condition=df_dict["condition"],
            df_intervention=df_dict["intervention"],
        )

        if self.pandas_or_dask == "pandas":
            self.to_pandas()

        logging.debug("Load complete")

    def to_pandas(self):

        logging.debug("Converting dask dataframes to pandas")

        self.df_csq = self.df_csq.compute()
        self.df_asset_info = self.df_asset_info.compute()
        self.condition_data = self.condition_data.compute()

    def read_data(self):

        self.get_path()

        df_csq = self.from_file(
            path=self.csq_path, dtype=self.dtype["csq"], nrows=self.nrows
        )
        df_csq = self.cleanse_data(
            df=df_csq,
            replace=self.prohibbited_characters,
            replace_with=self.replacement_character,
            numerical_columns=self.numerical_columns["csq"],
            columns_rename=self.specific_columns_rename,
        )

        df_asset_info = self.from_file(
            path=self.asset_path, dtype=self.dtype["asset_info"], nrows=self.nrows
        )
        df_asset_info = self.cleanse_data(
            df=df_asset_info,
            replace=self.prohibbited_characters,
            replace_with=self.replacement_character,
            numerical_columns=self.numerical_columns["asset_info"],
            columns_rename=self.specific_columns_rename,
        )

        df_condition = self.from_file(
            path=self.condition_path, dtype=self.dtype["condition"], nrows=self.nrows
        )
        df_condition = self.cleanse_data(
            df=df_condition,
            replace=self.prohibbited_characters,
            replace_with=self.replacement_character,
            numerical_columns=self.numerical_columns["condition"],
            columns_rename=self.specific_columns_rename,
        )

        df_intervention = self.from_file(path=self.intervention_path, nrows=self.nrows)
        df_intervention = self.cleanse_data(
            df=df_intervention,
            replace=self.prohibbited_characters,
            replace_with=self.replacement_character,
            numerical_columns=None,
            columns_rename=self.specific_columns_rename,
        )

        df_dict = dict(
            csq=df_csq,
            asset_info=df_asset_info,
            condition=df_condition,
            intervention=df_intervention,
        )

        return df_dict

    def merge_data(self, df_csq, df_asset_info, df_condition, df_intervention):

        logging.debug("got into merge_data")

        self.load_csq_model(df_csq=df_csq)
        self.load_asset_info(
            df_csq=df_csq, df_asset_info=df_asset_info, df_intervention=df_intervention
        )
        self.load_condition_data(
            df_condition=df_condition, df_intervention=df_intervention
        )

    def load_csq_model(self, df_csq):
        """
        Legacy not needed. Loads poles consequence model data
        """

        logging.debug("Loading consequence model")

        self.df_csq = df_csq[self.csq_columns["csq"]]

        logging.debug("Consequence model loaded")

    def load_asset_info(self, df_csq, df_asset_info, df_intervention):
        """
        Loads poles asset information
        """

        logging.debug("Loading asset information")

        df_intervention = df_intervention[self.intervention_columns].drop_duplicates()

        df_csq = df_csq[self.csq_columns["asset_info"]].drop_duplicates()

        df_asset_info = df_asset_info.merge(
            df_csq, how="left", on="asset_id"
        ).reset_index(drop=True)

        df_asset_info = df_asset_info.merge(
            df_intervention, how="left", on="asset_id"
        ).reset_index(drop=True)

        self.df_asset_info = df_asset_info

        logging.debug("Asset information loaded")

    def load_condition_data(self, df_condition, df_intervention):
        """
        Loads poles condition data
        """

        logging.debug("Loading condition data")

        cond_columns = df_condition.columns

        df_intervention = df_intervention[self.intervention_columns].drop_duplicates()

        df_intervention["replace_date"] = df_intervention["pseudo_asset_id"].map(
            lambda x: x.split("-", 1)[1] if len(x.split("-", 1)) == 2 else np.nan
        )
        df_intervention = self.date_formatter(df_intervention)

        df_condition = df_condition.merge(
            df_intervention, how="left", on="asset_id"
        ).reset_index(drop=True)

        df_condition["pseudo_asset_id"] = df_condition["pseudo_asset_id"].where(
            df_condition["replace_date"] > df_condition["date"],
            df_condition["asset_id"],
        )

        self.condition_data = df_condition[cond_columns]

        logging.debug("Condition data loaded")

    def get_fleet_data(self):
        """
        Creates a FleetData object
        """

        logging.debug("Creating FleetData object")

        return FleetData(
            asset_info=self.df_asset_info,
            condition=self.condition_data,
            csq_info=self.df_csq,
            pandas_or_dask=self.pandas_or_dask,
        )


if __name__ == "__main__":
    print("ok")
