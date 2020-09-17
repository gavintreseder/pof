# Change the system path is
if __package__ is None or __package__ == "":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass


@dataclass
class Config:

    on_error_use_default: bool = True
    FILL_NONE_WITH_DEFAULT: bool = True
    FILL_FROM_PARENT: bool = True
    REPLACE_ON_FAILURE: bool = True

    # Flags for model logic
    USE_DEFAULT: bool = True


@dataclass
class ComponentConfig(Config):

    NotImplemented


@dataclass
class FailureModeConfig(Config):
    """
    Contains the config parameters for the FailureMode class
    """

    # Flags to drive model logic

    FILL_NONE_WITH_DEFAULT: bool = True

    PF_INTERVAL = 100
    PF_CURVES = ['step', 'linear', 'ssf_calc', 'dsf_calc']
    PF_CURVE = "step"
    STATES = dict(initiation=False, detection=False, failure=False)


@dataclass
class TaskConfig(Config):

    NotImplemented


@dataclass
class IndicatorConfig(Config):
    """
    Contains the config parameters for the Indicator class
    """


config = Config()
indicator = IndicatorConfig()
