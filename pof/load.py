"""
    Filename: indicator.py
    Description: Contains the code for implementing a load class
    Author: Gavin Treseder | gct999@gmail.com | gtreseder@kpmg.com.au | gavin.treseder@essentialenergy.com.au
"""


# from dataclasses import dataclass
from collections.abc import Iterable
import logging

from dataclass_property import dataclass, field_property
import numpy as np
import pandas as pd
import scipy.stats as ss
import plotly.express as px
import plotly.graph_objects as go

if __package__ is None or __package__ == "":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pof.helper import str_to_dict
from config import config

# Use config for load
cf = config["Load"]

"""
The load module is used to overload other pof classes so that they can use a common load methods
"""

# TODO add more robust error checking for types other than value error


@dataclass
class Load:
    """
    A class with methods for loading data that
    """

    name: str = field_property(default="Load")

    @name.getter
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
            raise TypeError("name must be a string")

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
            if cf.get("on_error_use_default"):
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
                if all([hasattr(d_type, attr) for attr in value]):
                    new_object = d_type.from_dict(value)
                    getattr(self, attr)[new_object.name] = new_object

                # Create an object from the dict of dict/objects
                else:
                    for key, val in value.items():

                        if isinstance(val, d_type):
                            getattr(self, attr)[val.name] = val

                        else:
                            new_object = d_type.from_dict(val)
                            getattr(self, attr)[new_object.name] = new_object

            else:
                raise ValueError

        except ValueError:  # TODO maybe cahnge this back?
            msg = "%s (%s) - %s cannot be set from %s" % (
                self.__class__.__name__,
                self.name,
                attr,
                value,
            )
            if value is None and cf.get("on_error_use_default") is True:
                logging.info(msg + "- Default used")
            else:
                raise ValueError(msg)

    def update(self, id_object, value=None):
        """ An update method with some error handling"""
        try:
            if isinstance(id_object, str):
                self.update_from_str(id_object, value, sep="-")

            elif isinstance(id_object, dict):
                self.update_from_dict(id_object)

            else:
                logging.warning(
                    'ERROR: Cannot update "%s" from string or dict',
                    self.__class__.__name__,
                )
        except KeyError as error:
            logging.warning(error)

    def update_from_str(self, id_str, value, sep="-"):
        """
        updates a single parameter using a string format
        """
        id_str = id_str.split(self.name + sep, 1)[1]

        dict_data = str_to_dict(id_str, value, sep)

        self.update_from_dict(dict_data)

    # def update_from_dict(self, *args, **kwargs):
    #     """
    #     Update_from_dict is overloaded in each of the child classes
    #     """
    #     raise NotImplementedError()

    def update_from_dict(self, data):
        """Updates an attribute on a pof object using a

        load = Load()
        load.update({'name':'updated_name'})
        load.name
        """
        # Loop through all the varaibles to update
        for attr, detail in data.items():

            # Check it is an attribute
            if hasattr(self, attr):
                var_to_update = getattr(self, attr)

                # Check if the object has a load method
                if isinstance(var_to_update, Load):
                    var_to_update.update_from_dict(detail)

                # Check if it is a dictionary
                elif isinstance(var_to_update, dict):

                    for key, val in detail.items():

                        var_to_update = getattr(self, attr).get(key, None)

                        # Check if it is a pof object with an update method
                        if isinstance(var_to_update, Load):
                            var_to_update.update_from_dict(val)

                        elif var_to_update is not None:
                            getattr(self, attr)[key] = val

                        else:
                            raise KeyError(
                                "%s - %s - %s - %s - does not have the value - %s"
                                % (self.__class__.__name__, self.name, attr, key, val),
                            )
                else:
                    setattr(self, attr, detail)

            else:
                raise KeyError(
                    "%s - %s - does not have the attribute - %s"
                    % (self.__class__.__name__, self.name, attr),
                )

    def validate_inputs(self, *args, **kwargs):
        # logging.debug(
        #     "%s - %s - Unused variables - %s - %s",
        #     self.__class__.__name__,
        #     self.name,
        #     args,
        #     kwargs,
        # )
        # TODO make it validate the inputs against that class
        return NotImplementedError

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
            print(
                'ERROR: Cannot get "%s" sensitivity from string or dataframe'
                % (self.__class__.__name__)
            )

        return df

    def _sensitivity_many(self, df, conf, n_increments=1, n_iterations=100):
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
                self.mc_timeline(t_end=100, n_iterations=n_iterations)
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

            except Exception as e:
                logging.error("Error at %s", exc_info=e)

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
            print("ERROR: Cannot plot")

        return fig.show()

    def expected_risk_cost_df(self, *args, **kwargs):
        raise NotImplementedError()

    def reset(self, *args, **kwargs):
        raise NotImplementedError()

    def mc_timeline(self, *args, **kwargs):
        raise NotImplementedError()


if __name__ == "__main__":
    load = Load()
    print("Load - Ok")
