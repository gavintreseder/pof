

def PolesFleetDataLoader(FleetDataLoader):
    """
    Anything extra that awe 
    """


    # Load consequence model
    csq = self.from_csv(path)

    # Load the asset data


    # Load the condition data

    condition

    def load_asset_


    # Read in the data
        # df of asset informatoin

        # dict of condition data (tuples, list, dict)

    # Create fleet data oject

    def get_fleet_data(self):
        """
        Creates a FleetData object
        """
        return FleetData(asset_info = df_asset_info, condition = condition_data)


def FleetDataLoader(self):
    """
    Handles the loading of data from input files and should create a FleetData object when it all works
    """

    # file_paths
    # file_types

    def from_txt(self):

        NotImplemented


    def from_csv(self, path): 

        df = pd.read_csv(path)
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

        asset = 'asset that matches that key'

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

    def __init__(self)



    def get(self, attribute=None):

        NotImplemented

        return attribute



# Comments

fdl = PoleFleetDataLoader()
fdl.load()


asset_ids = fdl.get_keys()


for asset_id in asset_ids:
    asset_data = fdl.get(asset_id)
    asset_model.load_asset_data(asset_data)

    asset_model.do_some_cool_things


    asset_data.get('wall_thickness', 'current')
    asset_data.get('wall_thickness', 'perfect')
    asset_data.get('wall_thickness', 'failed')