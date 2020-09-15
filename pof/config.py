


class ComponentConfig:
    NotImplemented

class FailureModeConfig:
    FILL_NONE_WITH_DEFAULT = True

class TaskConfig:
    NotImplemented

"""
Create isntances of the objects so they can be accessed using:

import config
config.fm.FILL_NONE_WITH_DEFAULT

"""
fm = FailureModeConfig()
comp = ComponentConfig()
tsk = TaskConfig()
