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


def id_update(instance, id_str, value, sep='-', children=None):
    """Updates an object using an id"""

    # Remove the class type and class name from the dash_id
    id_str = id_str.split(instance.name + sep, 1)[1]
    var = id_str.split(sep)[0]

    # Check if the variable is an attribute of the class
    if var in instance.__dict__:

        # Check if the variable is a dictionary
        if isinstance(instance.__dict__[var], dict):

            var_2 = id_str.split(sep)[1]

            # Check if the variable is a class with its own update methods
            # isinstance(instance.__dict__[var][var_2], children):
            if var_2 in [child.__name__ for child in children]:
                var_3 = id_str.split(sep)[2]
                instance.__dict__[var][var_3].update(id_str, value, sep)
            else:
                instance.__dict__[var][var_2] = value
        else:
            instance.__dict__[var] = value

    # Check if the variable is a class instance
    else:

        var = id_str.split(sep)[1]

        if var in instance.__dict__ and isinstance(instance.__dict__[var], children):
            instance.__dict__[var].update(id_str, value, sep)
        else:
            print("Invalid id \"%s\" %s not in class" % (id_str, var))


def str_to_dict(id_str, value, sep='-'):

    id_str = id_str.split(sep)

    dict_data = {}
    for key in reversed(id_str):
        if dict_data == {}:
            dict_data = {key: value}
        else:
            dict_data = {key: dict_data}
    return dict_data


# is it in the class

    # Attribute
    # Child class
    # Dict
    # attribute
    # class

# is it in a dict of that objects

# FailureMode to fm
# tasks


# Is it in the class
    # attribute
    # Child class
