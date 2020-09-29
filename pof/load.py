from dataclasses import dataclass
from collections.abc import Iterable
import logging

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

                if "name" in value:
                    getattr(self, attr)[value["name"]] = d_type.load(value)

                # Iterate through and create objects using this method
                else:

                    for key, val in value.items():
                        if key in getattr(self, attr) and not isinstance(val, d_type):
                            getattr(self, attr)[key].update_from_dict(val)
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