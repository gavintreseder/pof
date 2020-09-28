import pandas as pd
import datetime


def PolesFleetDataLoader(FleetDataLoader):
    """
    Anything extra that awe
    """
    path = "C:\\Users\\ischram\\OneDrive - KPMG\\Desktop\\Data"
    filename = ""  # TODO: Illyse: will this be separate? go through folder for name or will user input name?

    # Load consequence model
    df_csq = self.from_csv(path)
    df_csq = df_csq.rename(
        columns={"ASSET_ID": "Asset ID"}
    )  # TODO:Illyse:where will this be handled??

    # Load the asset data
    df_asset_info = self.from_csv(path)
    df_asset_info["Date Installed"] = pd.to_datetime(
        df_asset_info["Date Installed"], format="%Y%m%d", errors="coerce"
    )  # TODO:Illyse: where will this be handled??

    # Load the condition data
    condition_data = self.from_csv(path)
    condition_data["Date Changed"] = pd.to_datetime(
        condition_data["Date Changed"], format="%Y-%m-%d %H:%M:%S"
    )  # TODO:Illyse: where will this be handled??

    condition_dict = {}
    for asset_id in condition_data["Asset ID"].drop_duplicates():
        condition_dict[asset_id] = []
        for detail in condition_data[condition_data["Asset ID"] == asset_id][
            [
                "Detail Code",
                "Detail Description",
                "Date Changed",
                "Before Value",
                "After Value",
            ]
        ].values:
            v = []
            for value in detail:
                v.append(value)
            condition_dict[asset_id].append(v)

    condition_data = condition_dict

    def load_asset_(self):
        NotImplemented

    # Read in the data
    # df of asset informatoin

    # dict of condition data (tuples, list, dict)

    # Create fleet data oject

    def get_fleet_data(self):
        """
        Creates a FleetData object
        """
        return FleetData(asset_info=df_asset_info, condition=condition_data)


def FleetDataLoader(self):
    """
    Handles the loading of data from input files and should create a FleetData object when it all works
    """

    # file_paths
    # file_types

    def from_txt(self):
        df = pandas.read_csv(path + ".txt", delimiter="\t")
        # df = pandas.read_csv(path + ".txt", delimiter="\t", encoding='utf-16') # for consequence model

        NotImplemented

    def from_csv(self, path):

        df = pd.read_csv(path + ".csv")
        # Open the file


def FleetData(self):

    """
    A class that contains all the fleet data in the structured format that we want to use
    """

    # Primary Key -> Asset ID
    # Asset IDs

    def get_asset(self, key=None):
        """
        Takes a key and returns as AssetData class for that asset
        """

        asset = "asset that matches that key"

        NotImplemented

        return asset

    def get_representative_asset(self):

        """
        Return the 'average asset'
        """

    def get_youngest_asset(self):

        NotImplemented


def AssetData(self):

    # Primary Key - Asset ID

    # Differentiators
    # Pole Material
    # Pole Treatment

    # Asset Info

    # Condition
    # AGD = 100
    # AGD 20 Jun 2008 -> 120

    # Age

    # Task History

    def __init__(self):
        NotImplemented

    def get(self, attribute=None):

        NotImplemented

        return attribute


# Comments

fdl = PolesFleetDataLoader()
fdl.load()


asset_ids = fdl.get_keys()


for asset_id in asset_ids:
    asset_data = fdl.get(asset_id)
    asset_model.load_asset_data(asset_data)

    asset_model.do_some_cool_things

    asset_data.get("wall_thickness", "current")
    asset_data.get("wall_thickness", "perfect")
    asset_data.get("wall_thickness", "failed")
