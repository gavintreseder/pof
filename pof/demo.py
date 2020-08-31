from component import Component
from failure_mode import FailureMode
from task import Inspection
from distribution import Distribution


# TODO make this a class


# dict of variables
# dict of other classes
# variables
# other classes


# Check if it is a class

# Failure Modes



slow_aging = dict(
    Distribution = dict(
        name = "slow_aging",
        alpha = 100,
        beta = 2,
        gamma = 10,
    ),
    conditions = dict(
        Condition = wall_thickness
    )
)

slow_aging = dict(
    Distribution = dict(
        untreated= dict(
            slow_aging
        )
    )
)


# Conditions

wall_thickness = dict(

)

# Dists

slow_aging = dict(
    alpha= 100,
    beta = 2,
    gamma = 10,
)

# Impacts



# Tasks
scheduled_inspection = Inspection(t_interval=10)

scheduled_inspection.set_params(
        t_interval = 5,
        t_delay = 20,
        p_effective = 0.9,
        state_triggers = dict(),

        condition_triggers = dict(
            wall_thickness = dict(
                lower = 0,
                upper = 90,
            ),
        ),

        state_impacts = dict( 
            detection = True,
        ),

        condition_impacts = dict(
            wall_thickness = dict(
                target = None,
                reduction_factor = 0,
                method = 'reduction_factor',
                axis = 'condition',
             ),
        ),
)

# Failure Modes
early_life = FailureMode(alpha=10000, beta=0.5, gamma=0)
random = FailureMode(alpha=100, beta=1, gamma=0)
slow_aging = FailureMode(alpha=100, beta=1.5, gamma=20)
fast_aging = FailureMode(alpha=50, beta=3, gamma=20)

fm_demo = slow_aging

fm_demo.set_conditions(dict(
    wall_thickness = Condition(100, 0, 'linear', [-2]),
    external_diameter = Condition(100, 0, 'linear', [-5]),
))

fm_demo.set_tasks(dict(
    inspection = scheduled_inspection,
    #ocr = OnConditionRepair(activity='on_condition_repair').set_default(),
    cm = ImmediateMaintenance(activity='cm').set_default(),
))
