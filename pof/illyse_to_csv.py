from pympler import classtracker


if __package__ is None or __package__ == "":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pof.loader.fleet_data import FleetData
from pof.loader.poles_fleet_data_loader import PolesFleetDataLoader

# imports data
pfd = PolesFleetDataLoader()

# creates fleet data object from data
fd = pfd.get_fleet_data()

# attributes to keep in summary
attributes = {
    "pole_material": [],
    "pole_strength": [],
    "age": [],
    "DAGD_perfect_condition": [],
    "DCZD_condition_loss": [True],
    "DPWT_perfect_condition": [],
    "total_csq": [],
}
# attributes to remove from summary
remove = None

tr = classtracker.ClassTracker()  # pympler
tr.track_object(fd)  # pympler

# population summary (dask object)
summary = fd.get_population_summary(by=attributes, remove=remove, n_bins=10)

tr.create_snapshot()  # pympler
tr.stats.print_summary()  # pympler

# population summary (pandas object)
# x = summary.compute()


############################################

# add progress bar
# tqdm - simple but may not work with dask
# Illyse script
# DONE: converted intp csv
# input and output tkinter - for script
# docstring
# check memory error
#   clean up
# stretch goals
# del(pfd) after created fleet data object
# create function pfd.close()
