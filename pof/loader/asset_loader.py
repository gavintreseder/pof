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

    def get(self, data, attribute, key):

        if data == "condition":
            if attribute == "external_diameter":
                value = self.get_external_diameter(key)

            elif attribute == "wall_thickness":
                value = self.get_wall_thickness(key)

            else:
                raise ValueError(
                    "Condition attribute must be 'external_diameter' or 'wall_thickness'"
                )

        return value

    def get_external_diameter(self, key):
        condition = self.condition_data.copy()

        if key == "perfect":
            c = condition[condition["Detail Code"] == "DAGD"]
            # gets max value in either before or after
            external_diameter = max(
                [
                    c[c["Before Value"] == c["Before Value"].max()][
                        "Before Value"
                    ].values[0],
                    c[c["After Value"] == c["After Value"].max()]["After Value"].values[
                        0
                    ],
                ]
            )
        elif key == "current":
            c = condition[condition["Detail Code"] == "DCZD"]
            # gets after value from most recent
            external_diameter = c[c["Date Changed"] == c["Date Changed"].max()][
                "After Value"
            ].values[0]
        else:
            raise ValueError("External Diameter key must be 'perfect' or 'current'")

        return external_diameter

    def get_wall_thickness(self, key):
        condition = self.condition_data.copy()
        c = condition[condition["Detail Code"] == "DPWT"]

        if key == "perfect":
            # gets max value in either before or after
            wall_thickness = max(
                [
                    c[c["Before Value"] == c["Before Value"].max()][
                        "Before Value"
                    ].values[0],
                    c[c["After Value"] == c["After Value"].max()]["After Value"].values[
                        0
                    ],
                ]
            )
        elif key == "current":
            # gets after value from most recent
            wall_thickness = c[c["Date Changed"] == c["Date Changed"].max()][
                "After Value"
            ].values[0]
        else:
            raise ValueError("Wall Thickness key must be 'perfect' or 'current'")

        return wall_thickness


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
