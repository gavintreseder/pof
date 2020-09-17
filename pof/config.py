# Change the system path is
if __package__ is None or __package__ == "":
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass
from typing import List
from typing import Dict


@dataclass
class Config:
    """
    Params:

    """

    # Check if these are still used anywhere?
    on_error_use_default: bool = True
    FILL_NONE_WITH_DEFAULT: bool = True
    FILL_FROM_PARENT: bool = True
    REPLACE_ON_FAILURE: bool = True

    # Flags for model logic
    USE_DEFAULT: bool = True

    # Constants
    PF_CURVES: List = None

    def __post_init__(self):
        # Mutable data types (E.g. dict, list, array etc.) go here
        self.PF_CURVES = ["step", "linear", "ssf_calc", "dsf_calc"]


@dataclass
class ComponentConfig(Config):

    None


@dataclass
class FailureModeConfig(Config):
    """
    Contains the config parameters for the FailureMode class
    """

    # Flags to drive model logic
    FILL_NONE_WITH_DEFAULT: bool = True

    # Constants
    PF_INTERVAL: int = 100
    PF_CURVE: str = "step"
    STATES: Dict = None

    def __post_init__(self):
        super().__post_init__()

        self.PF_CURVES = dict(initiation=False, detection=False, failure=False)


@dataclass
class TaskConfig(Config):

    None


@dataclass
class IndicatorConfig(Config):
    """
    Contains the config parameters for the Indicator class
    """

    PF_CURVE: str = "step"
    PF_INTERVAL: int = 10
    PERFECT: bool = False
    FAILED: bool = True


@dataclass
class DistributionConfig(Config):

    alpha: float = 50
    beta: float = 1.5
    gamma: float = 10


@dataclass
class AssetModelLoaderConfig(Config):

    None


config = Config()

component_config = ComponentConfig()
failure_mode_config = FailureModeConfig()
indicator_config = IndicatorConfig()
task_config = TaskConfig()
distribution_config = DistributionConfig()

asset_model_loader_config = AssetModelLoaderConfig()


if __name__ == "__main__":
    print("Config - Ok")