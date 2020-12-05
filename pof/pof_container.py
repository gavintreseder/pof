""" A custom container for pof objects

The PofContainer behaves exactly like a dictionary with an additional update_from_dict method to ensure the key always matches the object name

"""

from collections import UserDict
import logging


class PofContainer(UserDict):
    """A dictionary that changes the key if the name of the pof object it is storing changes"""

    def __repr__(self):
        return f"{type(self).__name__}({self.data})"

    def update_from_dict(self, data):
        """Updates the pof objects in the container based on a dictionary input"""

        for key, details in data.items():

            # Update with the dictionary
            self.data[key].update_from_dict(details)

            # Check if the name has been updated
            if key != self.data[key].name:

                logging.debug("Updating key to match name change")
                new_key = self.data[key].name

                # Change the key if it is already in the dict
                i = 1
                base_key = new_key
                while new_key in self.data:
                    logging.warning("Key %s is already in use", new_key)
                    new_key = f"{base_key}|{i}"
                    i = i + 1

                # Update the key
                self.data[key].name = new_key
                self.data[new_key] = self.data.pop(key)
                logging.debug("Key updated to %s", new_key)

    def update_from_dict_no_key_change(self, data):

        for key, details in data.items():

            # Check for a name change
            name = details.get("name", None)
            if name is not None and name in self.data:

                # Check if the name change will cause an error

                if name in self.data:
                    logging.warning(
                        "Name not updated. %s already exists in the container", name
                    )
                    del details["name"]

                self.data[name] = self.data.pop(key)

        return NotImplementedError()


# An example of a mutable mappin implementation

# class PofContainer(MutableMapping):
#     """A dictionary that changes the key if the name of the pof object it is storing changes"""

#     def __init__(self, *args, **kwargs):
#         self.store = dict()
#         self.update(dict(*args, **kwargs))  # use the free update to set keys

#     def __getitem__(self, key):
#         return self.store[key]

#     def __setitem__(self, key, value):
#         self.store[key] = value

#     def __delitem__(self, key):
#         del self.store[key]

#     def __iter__(self):
#         return iter(self.store)

#     def __len__(self):
#         return len(self.store)

#     def __repr__(self):
#         return f"{type(self).__name__}({self.store})"