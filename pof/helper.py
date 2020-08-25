import collections
import numpy as np


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def fill_blanks(row, t_start, t_end):

    n = t_end - t_start + 1
    time = np.linspace(t_start, t_end, n, dtype=int)
    cost = np.full(n, 0)
    
    if row['time'].size:
        cost[row['time']] = row['cost']

    row['time'] = time
    row['cost'] = cost
    return row