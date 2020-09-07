from pof.component import Component
from pof.condition import Condition
from pof.failure_mode import FailureMode
from pof.task import *
from pof.distribution import Distribution


# TODO make this a class


# dict of variables
# dict of other classes
# variables
# other classes

"""        if not self.conditions:
            self.set_conditions(
                dict(
                    wall_thickness=Condition(100, 0, "pf_linear", [-5], pf_interval=self.pf_interval, pf_std = self.pf_std),
                    external_diameter=Condition(100, 0, "pf_linear", [-2], pf_interval=self.pf_interval * 2, pf_std = self.pf_std),
                )
            )

        if not self.tasks:
            self.set_tasks(
                dict(
                    inspection=Inspection(t_interval=5, t_delay=10, name='inspection').set_default(),
                    on_condition_repair=OnConditionRepair(activity="on_condition_repair", name = 'on_condition_repair').set_default(),
                    cm=ImmediateMaintenance(activity="cm", name = 'cm').set_default(),
                )
            )

        if not self.states:
            self.set_states(dict(initiation=False, detection=False, failure=False,))

"""


"""
Usage

from distribution import Demo as demo

demo.slow_aging
demo.distribution.slow_aging
"""


distribution_data = dict(

    early_life = dict(
        name = "infant_mortality",
        alpha = 10000,
        beta = 0.5,
        gamma = 0,
    ),

    random = dict(
        name = "random",
        alpha = 100,
        beta = 1,
        gamma = 0,
    ),

    slow_aging = dict(
        name = "slow_aging",
        alpha = 100,
        beta = 2,
        gamma = 10,
    ),

    fast_aging = dict(
        name = "fast_aging",
        alpha = 100,
        beta = 3.5,
        gamma = 10,
    ),
)

inspection_data = dict(
    name = 'inspection',
    p_effective = 0.9,
    cost = 50,
    t_interval = 5,
    t_delay = 10,

    triggers = dict(
        condition = dict(
            wall_thickness=dict(
                lower=0,
                upper=90,
            ),
        ),
        state = dict(
            initiation=True
        ),
    ),

    impacts = dict(
        condition = dict(),
        state = dict(detection=True,)
    ),
)

inspection_data = dict(
    name = 'inspection',
    p_effective = 0.9,
    cost = 50,
    t_interval = 5,
    t_delay = 10,

    triggers = dict(
        condition = dict(
            wall_thickness=dict(
                lower=0,
                upper=90,
            ),
        ),
        state = dict(
            initiation=True
        ),
    ),

    impacts = dict(
        condition = dict(),
        state = dict(detection=True,)
    ),
)

# Condition examples

condition_data = dict(
    wall_thickness = dict(
        perfect = 100,
        failed = 0,
        curve = 'linear',
        pf_interval = 10,
        pf_std = 0.5,
    ),
    external_diameter = dict(
        perfect = 100,
        failed = 0,
        curve = 'linear',
        pf_interval = 10,
        pf_std = 0.5
    )
)

# States
state_data = dict(
    new = dict(
        initiation=False,
        detection=False,
        failure=False,
    ),
    inititated =dict(
        initiation=True,
        detection=False,
        failure=False,
    ),
    detected =dict(
        initiation=True,
        detection=True,
        failure=False,
    ),
    failed =dict(
        initiation=True,
        detection=True,
        failure=True,
    ),
)


failure_mode_data = dict(
    name = 'fm',
    untreated = distribution_data['slow_aging'],

    tasks = dict(
        inspection = inspection_data,

    ),
    states = state_data['new']

)



"""
class Demo:

    early_life = Distribution(data['early_life'])
    random = Distribution(data['random'])
    slow_aging = Distribution(data['slow_aging'])
    fast_aging = Distribution(data['fast_aging'])


class FMData:

    random_with_tasks = dict(
        # Failure properties

        # Failure Data
        name = 'random_with_tasks',
        untreated = dist_data.slow_aging,
        treated = dist_data.fast_aging,

        # Condition

        # Tasks
    )


class FMData:

    random_with_tasks = dict(
        untreated = Dataset.slow_aging,
        treated = Dataset.fast_aging,
    ),



# Conditions

wall_thickness = dict(

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


if __name__ == "__main__":
    data = Dataset
    demo = Demo
    data2 = FMData

    print("Demo - Ok")"""