"""
Usage

from distribution import Demo as demo

demo.slow_aging
demo.distribution.slow_aging
"""

# fmt: off
# *********************** distribution data **********************************

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
        alpha=50,
        beta=10,
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
    slow_degrading=dict(
        name='slow_degrading',
        perfect=100,
        failed=0,
        pf_curve='linear',
        pf_interval=10,
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
        pf_curve='step',
        pf_interval=1,
        pf_std=0,
    ),
    instant_10=dict(
        name='instant',
        perfect=1,
        failed=0,
        pf_curve='step',
        pf_interval=10,
        pf_std=0,
    ),
    instant_1=dict(
        name='instant',
        perfect=1,
        failed=0,
        pf_curve='step',
        pf_interval=1,
        pf_std=0,
    )
)

# *********************** task data **********************************
inspection_data = dict(
    instant=dict(
        task_type='Inspection',
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
        task_type='Inspection',
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

repair_data = dict(
    on_condition=dict(
        task_type='ConditionTask',
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
)

replacement_data =dict(
    instant = dict(
        task_type='ConditionTask',
        name='on_condition_replacement',
        p_effective=1,
        cost=5000,

        triggers=dict(
            condition=dict(
                instant=dict(lower=False, upper=True)
            ),
            state=dict(detection=True, failure=False),
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
            system=['component'],
        ),
    ),
    on_condition = dict(
        task_type='ConditionTask',
        name='on_condition_replacement',
        p_effective=1,
        cost=5000,

        triggers=dict(
            condition=dict(
                fast_degrading=dict(lower=0, upper=20,),
                slow_degrading=dict(lower=0, upper=20,)
            ),
            state=dict(detection=True),
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
            system=['component'],
        ),
    ),
    on_failure = dict(
        task_type='ConditionTask',
        name='on_failure_replacement',
        p_effective=1,
        cost=10000,

        triggers=dict(
            state=dict(failure=True),
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
            system=['component'],
        ),
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

# *********************** failure mode data ********************************** 

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
            on_condition_replacement=replacement_data['instant'],
            on_failure_replacement=replacement_data['on_failure'],
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
            on_condition_replacement=replacement_data['instant'],
            on_failure_replacement=replacement_data['on_failure'],
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
            on_condition_repair=repair_data['on_condition'],
            on_condition_replacement=replacement_data['on_condition'],
            on_failure_replacement=replacement_data['on_failure'],
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
            on_condition_repair=repair_data['on_condition'],
            on_condition_replacement=replacement_data['on_condition'],
            on_failure_replacement=replacement_data['on_failure'],
        ),
        states=state_data['new']
    )
)

component_data = dict(

    comp=dict(
        name = 'comp',
        fm = dict(
            early_life = failure_mode_data['early_life'],
            random = failure_mode_data['random'],
            slow_aging = failure_mode_data['slow_aging'],
            fast_aging = failure_mode_data['fast_aging'],
        ),
        indicator = dict(
            slow_degrading =condition_data['slow_degrading'],
            fast_degrading=condition_data['fast_degrading']
        )
    )
)

if __name__ == "__main__":

    print("Demo - Ok")
