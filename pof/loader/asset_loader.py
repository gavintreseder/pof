import pandas as pd
import datetime


class AssetData:

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
        # for particular asset??
        # assuming dataframe of condition information
        # if 'condition':
        # get condition data

        # if 'wall_thickness':
        # filter for wall_thickness data

        # if 'current':
        # get most recent condition value
        # if 'perfect':
        # get maximum condiiton value

        return attribute


# Comments
"""
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


fdl = PoleFleetDataLoader().load(folder_path
asset_data = fdl.get(key=12343455)
asset_data.get('wall_thickness', 'perfect')
asset_data.get('wall_thickness', 'current')
 
asset_data.get('condition', 'wall_thickness', 'current')
    """
