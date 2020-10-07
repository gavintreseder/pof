from dataclasses import dataclass
from collections.abc import Iterable
import logging
import pandas as pd

from pof.helper import str_to_dict
from config import config

# Use config for load
cf = config["Load"]

"""
The load module is used to overload other pof classes so that they can use a common load methods
"""

# TODO add more robust error checking for types other than value error


@dataclass(repr=False)
class Load:
    """
    A class with methods for loading data that
    """

    # Overriden in children classes
    name: str = None

    @classmethod
    def load(cls, details=None):
        """
        Loads the data
        """
        return cls.from_dict(details)

    @classmethod
    def from_dict(cls, details=None):
        """
        Unpacks the dictionary data and creates and object using the constructor
        """
        try:
            instance = cls(**details)
        except ValueError as error:
            logging.warning("Error loading %s data from dictionary", cls.__name__)
            if cf.get("on_error_use_default"):
                logging.info("Defaults used")
                instance = cls()
            else:
                raise error

        return instance

    def _set_container_attr(self, attr, d_type, value):

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

                # TODO fix this
                # If this doesn't exist yet create it
                if "name" in value:
                    getattr(self, attr)[value["name"]] = d_type.load(value)

                # Iterate through and create objects using this method
                else:

                    for key, val in value.items():
                        if key in getattr(self, attr) and not isinstance(val, d_type):
                            getattr(self, attr)[key].update_from_dict(val)
                            # if "name" in val:
                            #     getattr(self, attr)[val["name"]] = d_type.load(val)
                            # else:
                            #     getattr(self, attr)[key].update_from_dict(val)
                        else:
                            self._set_container_attr(attr, d_type, val)

            else:
                raise ValueError

        except:
            if value is None and cf.get("on_error_use_default") is True:
                logging.info(
                    "%s (%s) - %s cannot be set from %s - Default Use",
                    self.__class__.__name__,
                    self.name,
                    attr,
                    value,
                )
            else:
                raise ValueError(
                    "%s (%s) - %s cannot be set from %s"
                    % (
                        self.__class__.__name__,
                        self.name,
                        attr,
                        value,
                    )
                )

    def update(self, id_object, value=None):
        """"""
        if isinstance(id_object, str):
            self.update_from_str(id_object, value, sep="-")

        elif isinstance(id_object, dict):
            self.update_from_dict(id_object)

        else:
            print(
                'ERROR: Cannot update "%s" from string or dict'
                % (self.__class__.__name__)
            )

    def update_from_str(self, id_str, value, sep="-"):
        """
        updates a single parameter using a string format
        """
        id_str = id_str.split(self.name + sep, 1)[1]

        dict_data = str_to_dict(id_str, value, sep)

        self.update_from_dict(dict_data)

    def update_from_dict(self, *args, **kwargs):
        """
        Update_from_dict ist overlaoded in each of the child classes
        """
        return "This must be overloaded"

    def sensitivity(self, var_name, lower, upper, step=1, n_iterations=10):
        """"""
        # TODO add an optimal onto this
        rc = dict()
        self.reset()

        var = var_name.split("-")[-1]

        for i in range(max(1, lower), upper, step):

            self.update(var_name, i)

            self.mc_timeline(t_end=100, n_iterations=n_iterations)

            rc[i] = self.expected_risk_cost_df().groupby(by=["task"])["cost"].sum()
            rc[i][var] = i

            # Reset component
            self.reset()

        df = (
            pd.DataFrame()
            .from_dict(rc, orient="index")
            .rename(columns={"risk": "risk_cost"})
        )
        df["direct_cost"] = df.drop([var, "risk_cost"], axis=1).sum(axis=1)
        df["total"] = df["direct_cost"] + df["risk_cost"]
        df = df[[var, "direct_cost", "risk_cost", "total"]]

        return df