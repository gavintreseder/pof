import collections
import inspect

import numpy as np


def flatten(d, parent_key="", sep="_"):
    """
    Takes
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def fill_blanks(row, t_start, t_end, cols):

    n = t_end - t_start + 1

    row_time = [item for item in row["time"] if item < n]

    for col in cols:
        temp_row = np.full(n, 0, dtype=row[col].dtype)
        temp_row[row_time] = row[col][: len(row_time)]
        row[col] = temp_row

    row["time"] = np.linspace(t_start, t_end, n, dtype=int)

    return row


def str_to_dict(id_str, value, sep="-"):

    id_str = id_str.split(sep)

    dict_data = {}
    for key in reversed(id_str):
        if dict_data == {}:
            dict_data = {key: value}
        else:
            dict_data = {key: dict_data}
    return dict_data


def get_signature(obj):
    """ Get the constructor signature"""
    signature = inspect.signature(obj).parameters

    if bool(obj.__bases__):
        for parent in obj.__bases__:
            parent_signature = get_signature(parent)
            signature = {**signature, **parent_signature}
            # TODO consider making the order consistent
            # for key, value in parent_signature.items():
            #    signature.setdefault({key: value})

    return signature


def valid_signature(obj, inputs):
    """ Returns whether an object can be created with the inputs provided based on the signature"""

    factory = getattr(obj, "factory", None)
    if callable(factory):
        obj = obj.factory(**inputs)

    signature = get_signature(obj)
    valid = [attr in signature for attr in inputs]

    return all(valid)