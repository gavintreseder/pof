import logging
from dask.diagnostics import ProgressBar

logging.getLogger().setLevel(logging.DEBUG)


if __package__ is None or __package__ == "":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pof.loader.fleet_data import FleetData
from pof.loader.poles_fleet_data_loader import PolesFleetDataLoader

with ProgressBar():
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

    # population data
    population_data = fd.get_population_data(by=attributes, remove=remove, n_bins=10)

    # population summary
    population_summary = fd.get_population_summary(
        by=attributes, remove=remove, n_bins=10
    )


############################################

# stretch goals
# del(pfd) after created fleet data object
# create function pfd.close()
