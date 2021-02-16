"""
    Filename: Base.py
    Description: Contains the code for implementing the base pof class from which other pof classes inherit
    Authors: Gavin Treseder
        gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
"""

from collections.abc import Iterable
import logging
from typing import Dict


from flatten_dict import flatten, unflatten
import numpy as np
import pandas as pd
import scipy.stats as ss
import plotly.express as px
import plotly.graph_objects as go

if __package__ is None or __package__ == "":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pof.pof_container import PofContainer
from pof.helper import str_to_dict, valid_signature, get_signature
from config import config
from pof.units import valid_units
from pof.pof_container import PofContainer

cf = config["PofBase"]
cf_main = config["Main"]


class PofBase:
    """
    A class with methods for that are common to all pof classes.
    """

    # Class Variables
    TIME_VARIABLES = []
    POF_VARIABLES = []

    def __init__(self, name="pofbase", units="years", *args, **kwargs):

        self.name = name
        self.units = units
        # self.graph_units = units  # TODO temp fix

        # Dash feature
        self.up_to_date = True

        if args or kwargs:
            msg = f"Invalid Data {args} - {kwargs}"
            if cf.get("handle_invalid_data", False):
                logging.warning(msg)
            else:
                raise TypeError(msg)

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    # def __repr__(self):
    #     # TODO alternative version that keens things simple
    #     # sig = get_signature(self)
    #     # _repr = f"'{self.__class__.__name__}("
    #     # for param in sig:
    #     #     _repr.join()
    #     keys = sorted(self.__dict__)
    #     items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
    #     return "{}({})".format(type(self).__name__, ", ".join(items))

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):

        if isinstance(value, str):
            # probhitted = ["-"] #TODO expand list
            # if any(x in a_string for x in matches):
            self._name = value
        elif value is None:
            self._name = None
        else:
            raise ValueError("name must be a string")

    @classmethod
    def load(cls, details=None):
        """
        Loads the data with extra error checking and default logic
        """
        try:
            instance = cls.from_dict(details)

        except (ValueError, TypeError) as error:
            logging.warning(error)
            logging.warning("Error loading %s data from dictionary", cls.__name__)
            if cf.get("on_error_use_default", False):
                logging.info("Defaults used")
                instance = cls()
            else:
                raise error
        return instance

    @classmethod
    def from_dict(cls, details=None):
        """
        Unpacks the dictionary data and creates and object using the constructor
        """
        if isinstance(details, dict):
            instance = cls(**details)
        else:
            raise TypeError("Dictionary expected")

        return instance

    @classmethod
    def demo(cls):
        return cls("Not Implemented")

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, value):  # TODO - Make this dependant on csv
        """ Takes a unit and updates any time values to reflect the new units"""
        # TODO how to handle if they load None - file_units may not be years

        # Check if the uploaded unit is valid
        if value.lower() in valid_units:

            # Check if units is defined, if not set it to None
            current_units = getattr(self, "_units", None)

            # If current_units is not None and does not equal loaded_units call scale_units to scale
            if current_units is not None and current_units != value:
                self._scale_units(value, current_units)

            self._units = value

        else:
            raise ValueError(f"Unit must be in {list(valid_units)}")

    def _scale_units(self, new_units, current_units):
        """ Take current and loaded units and return the ratio between """
        # Determine the ratio between the current and uploaded unit
        ratio = (
            valid_units[current_units] / valid_units[new_units]
        )  # Current value over loaded value

        # Update the variables on this instance
        for var in self.TIME_VARIABLES:
            setattr(self, var, getattr(self, var) * ratio)

        # Update the variables on the child instance
        for var_name in self.POF_VARIABLES:
            var = getattr(self, var_name)
            if isinstance(var, dict) or isinstance(var, PofContainer):
                for val in var.values():
                    val.units = new_units
            elif var is not None:
                var.units = new_units
            else:
                raise ValueError(f"Something is wrong")

    def set_obj(self, attr, d_type, value):
        """

        value = {'tasks':{'inspection':{'t_interval':10}}}
        """

        # Create an empty dictionary if it doesn't exist #Dodgy fix because @property error
        if getattr(self, attr, None) is None:
            setattr(self, attr, dict())

        try:
            if value is None:
                setattr(self, attr, dict())

            # Add the value to the dictionary if it is an object of that type
            elif isinstance(value, d_type):
                getattr(self, attr)[value.name] = value

            # Check if the input is an iterable
            elif isinstance(value, Iterable):

                # Create an object from the dict
                if valid_signature(obj=d_type, inputs=value):
                    new_object = d_type.from_dict(value)
                    getattr(self, attr)[new_object.name] = new_object

                # Create an object from the dict of dict/objects
                else:
                    for val in value.values():

                        if isinstance(val, d_type):
                            getattr(self, attr)[val.name] = val

                        else:
                            new_object = d_type.from_dict(val)
                            getattr(self, attr)[new_object.name] = new_object

            else:
                raise ValueError

        except ValueError as error:  # TODO maybe cahnge this back?
            msg = "%s (%s) - %s cannot be set from %s" % (
                self.__class__.__name__,
                self.name,
                attr,
                value,
            )
            if value is None and cf.get("on_error_use_default") is True:
                logging.info(msg.join(" - Default used"))
            else:
                raise ValueError(msg) from error

    def get(self, string):
        return self.__dict__[string]

    def update(self, id_object, value=None):
        """ An update method with some error handling"""
        try:

            if isinstance(id_object, str):
                self.update_from_str(id_object, value, sep="-")

            elif isinstance(id_object, dict):
                self.update_from_dict(id_object)

            else:
                logging.warning(
                    'ERROR: Can only update "%s" from string or dict',
                    self.__class__.__name__,
                )
        except (KeyError, AttributeError, ValueError) as error:
            if config["PofBase"]["handle_update_error"]:
                logging.warning("Update Failed. {error}")
            else:
                raise error

    def update_from_str(self, id_str, value, sep="-"):
        """
        updates a single parameter using a string format
        """
        if self.name + sep in id_str:
            id_str = id_str.split(self.name + sep, 1)[1]

        dict_data = str_to_dict(id_str, value, sep)

        self.update_from_dict(dict_data)

    def update_from_dict(self, data: Dict):
        """Updates an attribute on a pof object using a

        Inputs:
            data: A nested dictionary of data to update

        Usage:
        >>> pof_base = PofBase()
        >>> pof_base.update({'name':'updated_name'})
        >>> pof_base.name

        'updated_name'
        """
        # Loop through all the varaibles to update
        for attr, value in data.items():

            attr_to_update = getattr(self, attr)

            # Check if has an update method
            if isinstance(attr_to_update, (PofBase, PofContainer)):
                attr_to_update.update_from_dict(value)

            # Check if it is a dictionary
            elif isinstance(attr_to_update, dict):

                still_to_update = {}
                for key, val in value.items():

                    var_to_update = getattr(self, attr).get(key, None)

                    # Check if is has an update method
                    if isinstance(var_to_update, (PofBase, PofContainer)):
                        var_to_update.update_from_dict(val)
                    else:
                        still_to_update[key] = val

                update_dict(data=attr_to_update, update=still_to_update)

            else:
                # Scale the units
                setattr(self, attr, value)

    def sensitivity(
        self,
        var_name,
        conf=0.9,
        lower=None,
        upper=None,
        n_increments=1,
        n_iterations=100,
    ):
        """
        Returns dataframe of sensitivity data for a given variable name or dataframe of variables using a given confidence.
        """

        if isinstance(var_name, str):
            df = self._sensitivity_single(
                var_name=var_name,
                lower=lower,
                upper=upper,
                n_increments=n_increments,
                n_iterations=n_iterations,
            )

        elif isinstance(var_name, pd.DataFrame):
            df = self._sensitivity_many(
                df=var_name,
                conf=conf,
                n_increments=n_increments,
                n_iterations=n_iterations,
            )

        else:
            raise ValueError(
                'ERROR: Cannot get "%s" sensitivity from string or dataframe'
                % (self.__class__.__name__)
            )

        return df

    def _sensitivity_many(self, df, conf, n_increments=50, n_iterations=100):
        """
        Returns dataframe of sensitivity data for a dataframe of variables using a given confidence.
        """

        d_all = {}

        for index, row in df.iterrows():

            [lower, upper] = ss.norm.interval(
                alpha=conf, loc=row["mean"], scale=row["sd"]
            )

            df_sens = self._sensitivity_single(
                var_name=row["name"],
                lower=lower,
                upper=upper,
                n_increments=n_increments,
                n_iterations=n_iterations,
            )
            df_sens = df_sens.reset_index(drop=True)

            # Scale everything to the mean
            cols = [("agg", "direct_cost"), ("agg", "risk_cost"), ("agg", "total")]
            mean_idx = list(df_sens.index)[int((len(list(df_sens.index)) - 1) / 2)]
            df_sens.loc[:, cols] = df_sens[cols] / df_sens.loc[mean_idx, cols]

            df_sens["percent_change"] = df_sens["value"] / df_sens["value"][mean_idx]
            df_sens["conf"] = ss.norm.cdf(
                df_sens["value"], loc=row["mean"], scale=row["sd"]
            )
            df_sens["conf"] = df_sens["conf"].round(2)
            df_sens["var_name"] = row["name"]

            d_all[row["name"]] = df_sens

        df_all = pd.concat(d_all.values(), ignore_index=True)

        return df_all

    def _sensitivity_single(
        self, var_name, lower, upper, n_increments=1, n_iterations=100
    ):
        """
        Returns dataframe of sensitivity data for a given variable name using a given confidence.
        """
        rc = dict()
        self.reset()

        if n_increments % 2 == 0:
            n_increments = n_increments + 1

        var = var_name.split("-")[-1]

        for i in np.linspace(lower, upper, n_increments):
            try:
                self.update(var_name, i)
                self.mc_timeline(
                    t_end=100, n_iterations=n_iterations
                )  # TODO Remove t_end hardcode
                agg = self.expected_risk_cost_df()
                agg["failure_mode"] = "agg"
                agg = agg.groupby(by=["failure_mode", "task"])["cost"].sum()
                rc[i] = (
                    self.expected_risk_cost_df()
                    .groupby(by=["failure_mode", "task"])["cost"]
                    .sum()
                )
                rc[i] = rc[i].append(agg)
                rc[i][var] = i

                # Reset component
                self.reset()

            except Exception as error:
                logging.error("Error at %s", exc_info=error)

        df = (
            pd.DataFrame()
            .from_dict(rc, orient="index")
            .rename(columns={"risk": "risk_cost"})
        )
        df[("agg", "direct_cost")] = df[("agg")].sum(axis=1)
        df[("agg", "direct_cost")] = df[("agg", "direct_cost")] - df[var]
        df[("agg", "direct_cost")] = (
            df[("agg", "direct_cost")] - df[("agg", "risk_cost")]
        )
        df[("agg", "total")] = df[("agg", "direct_cost")] + df[("agg", "direct_cost")]

        df = df.rename(columns={var: "value"})

        return df

    plot_type = ["line", "heatmap"]

    def make_sensitivity_plot(
        self, data, x_axis, y_axis, plot_type, failure_mode="agg", z_axis=None
    ):
        """
        Creates a sensitivity line plot or heatmap using the given sensitivity data.
        """
        if plot_type == "line":

            fig = px.line(
                data[(failure_mode)], x=data[x_axis], y=y_axis, color=data["var_name"]
            )
            fig.update_layout(
                autosize=False,
                width=1000,
                height=500,
                title="Cost affect on " + y_axis,
                xaxis_title=x_axis,
                legend_title="changing variable",
            )

        elif plot_type == "heatmap":

            fig = go.Figure(
                data=go.Heatmap(
                    x=data[x_axis],
                    y=data[y_axis],
                    z=data[(failure_mode, z_axis)],
                    hoverongaps=False,
                    colorscale="fall",
                )
            )
            fig.update_layout(
                autosize=False,
                width=1000,
                height=500,
                title="Cost affect on " + z_axis,
                xaxis_title=x_axis,
                legend_title=z_axis,
            )
        else:
            raise Exception("ERROR: Cannot plot")

        return fig.show()

    def expected_risk_cost_df(self, *args, **kwargs):
        raise NotImplementedError()

    def reset(self, *args, **kwargs):
        raise NotImplementedError()

    def mc_timeline(self, t_start=None, t_end=None, n_iterations=None):
        raise NotImplementedError()

    def to_dict(self):
        """ Create a dict of the comp object to save to a json file """

        # Get the first layer
        data_req = self.get_attr(data_req={})

        # Add the comp key
        data_req_sys = {}
        data_req_sys[cf_main.get("name")] = data_req

        # Unpack
        data_req_unpacked = self.unpack_container(data_req_sys)

        return data_req_unpacked

    def get_attr(self, data_req):
        """ Create a dictionary of signature and attribute values of the class """

        # Get the information needed to create object
        sig_list = list(get_signature(self.__class__))

        for attr in sig_list:
            if hasattr(self, attr):
                data_req[attr] = getattr(self, attr)

        return data_req

    def unpack_container(self, data_req):
        """ Unpack the PofContainer and PofBase objects to create a dict """
        # TODO This modifies the data set, it should make a copy and modify that

        # Loop through all the items to keep
        for attr, val in data_req.items():
            # Check if it is a container and unpack it
            if isinstance(val, PofContainer):
                data_req[attr] = val.data
            # Check if it is a pof base and unpack it
            elif isinstance(val, PofBase):
                data_req[attr] = {}
                sig_list = list(get_signature(val.__class__))
                sig_list = [n for n in sig_list if n != "component"]
                for var in sig_list:
                    if hasattr(val, var):
                        if getattr(val, var) != NotImplemented:
                            data_req[attr][var] = getattr(val, var)

        # Trigger the call again to unpack the next layer
        for attr, val in data_req.items():
            if isinstance(val, dict):
                for name, value in data_req[attr].items():
                    if isinstance(value, (PofContainer, PofBase)):
                        self.unpack_container(data_req[attr])

        return data_req


def update_dict(data: Dict, update: Dict):
    """ Update the inner value of the dictionary"""

    flat_data = flatten(data)
    flat_update = flatten(update)

    missing = set(flat_update).difference(set(flat_data))
    if not missing:

        # Update the data
        flat_data.update(flat_update)
        keys_changed = key_id_changed(flat_data, "name")
        updated = unflatten(flat_data)
        for key, val in updated.items():
            data[key] = val

        for path, new_key in keys_changed.items():
            replace(data=data, path=path, new_key=new_key)
    else:
        raise KeyError("Data %s cannot be updated with %s", data, update)

    return data


def key_id_changed(data: Dict, id: str = "name"):
    """ Get the keys of variables where the id has been changed"""
    id_keys = [key for key in list(data) if "name" in key]

    changed = {}
    for id_key in id_keys:
        idx = id_key.index("name")

        if id_key[idx - 1] != data[id_key]:
            changed[id_key[:idx]] = data[id_key]

    return changed


def replace(data: Dict, path, new_key):
    """ Change the key located at the end of path to the new_key"""
    cur = data
    for k in path[:-1]:
        cur = cur[k]

    cur[new_key] = cur.pop(path[-1])


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS, extraglobs={"pof_base": PofBase()})
    print("PofBase - Ok")
