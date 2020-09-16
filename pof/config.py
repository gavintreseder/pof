
# Change the system path is
if __package__ is None or __package__ == "":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass


@dataclass
class Config:

    on_error_use_default = True
    FILL_NONE_WITH_DEFAULT = True
    FILL_FROM_PARENT = True
    REPLACE_ON_FAILURE = True


@dataclass
class ComponentConfig(Config):

    NotImplemented


@dataclass
class FailureModeConfig(Config):
    """
    Contains the config parameters for the FailureMode class
    """

    FILL_NONE_WITH_DEFAULT = True


@dataclass
class TaskConfig(Config):

    NotImplemented