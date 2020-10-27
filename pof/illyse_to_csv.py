if __package__ is None or __package__ == "":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pof.loader.fleet_data import FleetData
from pof.loader.poles_fleet_data_loader import PolesFleetDataLoader

print("starting")
pfd = PolesFleetDataLoader()
print("between")
g = pfd.get_fleet_data()
print("okay")

# add progress bar
# tqdm - simple but may not work with dask
# Illyse script
# converted intp csv
# input and output tkinter - for script
# docstring
# check memory error
#   clean up
# stretch goals
# del(pfd) after created fleet data object
# create function pfd.close()
