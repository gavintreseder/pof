import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pof.component import Component
from pof.condition import Condition
from pof.failure_mode import FailureMode
from pof.task import Task
from pof.consequence import Consequence
from pof.distribution import Distribution

VALID_CHILDREN = (Component, Condition, FailureMode, Task, Distribution, Consequence)

def get_dash_id_value(instance, dash_id, prefix='', sep = "-"):

    # Remove the class type and class name from the dash_id
    dash_id = dash_id.split(instance.name + sep, 1)[1]
    var = dash_id.split(sep)[0]

    # Check if the variable is an attribute of the class
    if var in instance.__dict__:

        # Check if the variable is a dictionary
        if isinstance(instance.__dict__[var], dict): 
            var_2 = dash_id.split(sep)[1]

            # Check if the variable is a class with its own update methods
            if var_2 in [child.__name__ for child in VALID_CHILDREN]: #isinstance(instance.__dict__[var][var_2], children):
                var_3 = dash_id.split(sep)[2]
                val = get_dash_id_value(instance.__dict__[var][var_3], dash_id)
            else:
                val = get_dash_id_value(instance.__dict__[var][var_2], dash_id)
        else:
            val = instance.__dict__[var]

    else:
        
        var = dash_id.split(sep)[1]

        if var in instance.__dict__ and isinstance(instance.__dict__[var], VALID_CHILDREN):
            val = get_dash_id_value(instance.__dict__[var], dash_id)
        else:
            val = ("Invalid id \"%s\" %s not in class" %(dash_id, var))

    return val