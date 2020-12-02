#  Add root folder to python path
import sys
import os

sys.path.append(os.path.dirname(os.getcwd()))

import numpy as np
import pandas as pd
import scipy.stats as ss

import plotly.express as px

from component import Component

comp = Component.demo()

comp.mc_timeline(t_end=10, n_iterations=3)

df, df_active = comp.expected_sensitivity(
    var_name="comp-fm-random-dists-untreated-beta", lower=1, upper=10, n_iterations=5
)
