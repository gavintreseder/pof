"""
Usage

from distribution import Demo as demo

demo.slow_aging
demo.distribution.slow_aging
"""

# fmt: off
# *********************** distribution data **********************************


dist_update_test_1 = dict(
    dist = dict(
        name='dist',
        alpha = 1,
        beta = 1,
        gamma = 1
    )
)

dist_update_test_2 = dict(
    dist = dict(
        name='dist_2',
        alpha = 2,
        beta = 2,
        gamma = 2
    )
)

distribution_data = dict(

    early_life=dict(
        name="early_life",
        alpha=1000000,
        beta=0.5,
        gamma=0,
    ),

    random=dict(
        name="random",
        alpha=1000,
        beta=1,
        gamma=0,
    ),

    slow_aging=dict(
        name="slow_aging",
        alpha=100,
        beta=2,
        gamma=10,
    ),

    fast_aging=dict(
        name="fast_aging",
        alpha=100,
        beta=3.5,
        gamma=10,
    ),
)

# *********************** condition data **********************************

condition_data = dict(
    wall__thickness=dict(
        name='wall_thickness',
        perfect=125,
        failed=0,
        pf_curve='linear',
        pf_interval=10,
        pf_std=0.5,
    ),
    extneral_diameter=dict(
        name='external_diameter',
        perfect=250,
        failed=0,
        pf_curve='linear',
        pf_interval=10,
        pf_std=0.5
    ),
    slow_degrading=dict(
        name='slow_degrading',
        perfect=100,
        failed=0,
        pf_curve='linear',
        pf_interval=20,
        pf_std=0.5
    ),
    fast_degrading=dict(
        name='fast_degrading',
        perfect=100,
        failed=0,
        pf_curve='linear',
        pf_interval=10,
        pf_std=0.5
    ),
    uncertain_degrading=dict(
        name='uncertain_degrading',
        perfect=100,
        failed=0,
        pf_curve='linear',
        pf_interval=30,
        pf_std=5
    ),
    instant=dict(
        name='instant',
        perfect=1,
        failed=0,
        pf_curve='linear',  # TODO change to step
        pf_interval=1,
        pf_std=0,
    )
)

# *********************** task data **********************************
inspection_data = dict(
    instant=dict(
        activity='Inspection',
        name='inspection',
        p_effective=0.9,
        cost=50,
        t_interval=5,
        t_delay=0,

        triggers=dict(
            condition=dict(
                instant=dict(
                    lower=0,
                    upper=0,
                )
            ),
            state=dict(
                initiation=True
            ),
        ),

        impacts=dict(
            condition=dict(),
            state=dict(detection=True,)
        ),
    ),
    degrading=dict(
        activity='Inspection',
        name='inspection',
        p_effective=0.9,
        cost=55,
        t_interval=5,
        t_delay=10,

        triggers=dict(
            condition=dict(
                fast_degrading=dict(
                    lower=0,
                    upper=90,
                ),
                slow_degrading=dict(
                    lower=0,
                    upper=90,
                ),
            ),
            state=dict(
                initiation=True
            ),
        ),

        impacts=dict(
            condition=dict(),
            state=dict(detection=True,)
        ),
    ),
)

on_condition_repair_data = dict(
    activity='ConditionTask',
    name='on_condition_repair',
    p_effective=0.7,
    cost=100,

    triggers=dict(
        condition=dict(
            fast_degrading=dict(lower=20, upper=90,),
            slow_degrading=dict(lower=20, upper=90,)
        ),
        state=dict(detection=True),
    ),

    impacts=dict(
        condition=dict(
            fast_degrading=dict(
                target=0.9,
                method="reduction_factor",
                axis="condition",
            ),
            slow_degrading=dict(
                target=0,
                method="reduction_factor",
                axis="condition",
            ),
        ),
        state=dict(initiation=False, detection=False, failure=False,),
    ),

)

on_condition_replacement_data = dict(
    activity='ConditionTask',
    name='on_condition_replacement',
    p_effective=1,
    cost=5000,

    triggers=dict(
        condition=dict(
            fast_degrading=dict(lower=0, upper=20,),
            slow_degrading=dict(lower=0, upper=20,)
        ),
        state=dict(detection=True, failure=True),
    ),

    impacts=dict(
        condition=dict(
            all=dict(
                target=1,
                method="reduction_factor",
                axis="condition",
            ),
        ),
        state=dict(initiation=False, detection=False, failure=False,),
    ),
)


# *********************** state data **********************************

state_data = dict(
    new=dict(
        initiation=False,
        detection=False,
        failure=False,
    ),
    inititated=dict(
        initiation=True,
        detection=False,
        failure=False,
    ),
    detected=dict(
        initiation=True,
        detection=True,
        failure=False,
    ),
    failed=dict(
        initiation=True,
        detection=True,
        failure=True,
    ),
)

# *********************** failure mode data ********************************** #TODO move condition

failure_mode_data = dict(
    early_life=dict(
        name='early_life',
        pf_curve = 'linear',
        pf_interval=10,
        untreated=distribution_data['early_life'],
        conditions=dict(
            instant=condition_data['instant'],
        ),

        tasks=dict(
            inspection=inspection_data['instant'],
            on_condition_replacement=on_condition_replacement_data
        ),
        states=state_data['new']
    ),
    random=dict(
        name='random',
        untreated=distribution_data['random'],
        conditions=dict(
            instant=condition_data['instant'],
        ),

        tasks=dict(
            inspection=inspection_data['instant'],
            on_condition_replacement=on_condition_replacement_data
        ),
        states=state_data['new']
    ),
    slow_aging=dict(
        name='slow_aging',
        untreated=distribution_data['slow_aging'],
        conditions=dict(
            slow_degrading=condition_data['slow_degrading'],
            fast_degrading=condition_data['fast_degrading']
        ),

        tasks=dict(
            inspection=inspection_data['degrading'],
            on_condition_repair=on_condition_repair_data,
            on_condition_replacement=on_condition_replacement_data
        ),
        states=state_data['new']
    ),
    fast_aging=dict(
        name='fast_aging',
        untreated=distribution_data['fast_aging'],

        conditions=dict(
            slow_degrading=condition_data['slow_degrading'],
            fast_degrading=condition_data['fast_degrading']
        ),

        tasks=dict(
            inspection=inspection_data['degrading'],
            on_condition_repair=on_condition_repair_data,
            on_condition_replacement=on_condition_replacement_data
        ),
        states=state_data['new']
    )
)

# *********************** component data **********************************

component_data = dict(

    comp=dict(

    )

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

fast_degrading = dict(

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
            fast_degrading = dict(
                lower = 0,
                upper = 90,
            ),
        ),

        state_impacts = dict( 
            detection = True,
        ),

        condition_impacts = dict(
            fast_degrading = dict(
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
    fast_degrading = Condition(100, 0, 'linear', [-2]),
    slow_degrading = Condition(100, 0, 'linear', [-5]),
))

fm_demo.set_tasks(dict(
    inspection = scheduled_inspection,
    #ocr = OnConditionRepair(activity='on_condition_repair').set_default(),
    cm = ImmediateMaintenance(activity='cm').set_default(),
))
"""

if __name__ == "__main__":

    print("Demo - Ok")
