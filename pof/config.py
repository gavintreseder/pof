

from dataclasses import dataclass

@dataclass
class Config:

    FILL_NONE_WITH_DEFAULT = True
    FILL_FROM_PARENT = True


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


"""
Create isntances of the objects so they can be accessed using:

import config
config.fm.FILL_NONE_WITH_DEFAULT

"""
fm = FailureModeConfig()
comp = ComponentConfig()
tsk = TaskConfig()
