"""
Usage

from distribution import Demo as demo

demo.slow_aging
demo.distribution.slow_aging
"""

import testconfig
import pof.demo as demo

# fmt: off
complete = {}
# *********************** distribution data **********************************

distribution_data = dict(demo.distribution_data)

complete['distribution_0'] = dict(
    dist = dict(
        name='dist_0',
        alpha = 0,
        beta = 0,
        gamma = 0
    )
)

complete['distribution_1'] = dict(
    dist = dict(
        name='dist_1',
        alpha = 1,
        beta = 1,
        gamma = 1
    )
)

# *********************** indicator data **********************************

condition_data = dict(demo.condition_data)

complete['indicator_0'] = dict(
    name='indicator_0',
    perfect=0,
    failed=1,
    pf_curve='linear',
    pf_interval=1,
    pf_std=0
)

complete['indicator_1'] = dict(
    name='indicator_1',
    perfect=1,
    failed=0,
    pf_curve='step',
    pf_interval=10,
    pf_std=1
)
# *********************** task data **********************************
inspection_data = dict(demo.inspection_data)

repair_data = dict(demo.repair_data)

replacement_data = dict(demo.replacement_data)

complete['inspection_0'] = dict(
    task_type='Inspection',
    name='inspection_0',
    p_effective=0,
    cost=0,
    t_interval=1,
    t_delay=0,

    triggers=dict(
        condition=dict(
            fast_degrading=dict(
                lower=0,
                upper=1,
            ),
        ),
        state=dict(
            initiation=False,
            detection= False,
            failure = False
        ),
    ),
    impacts=dict(
        condition=dict(
            fast_degrading=dict(
                target=0,
                method="reduction_factor",
                axis="condition",
            ),
        ),
        state=dict(
            initiation=False,
            detection= False,
            failure = False,
        ),
        system='failure_mode'
    ),
)

complete['inspection_1'] = dict(
    task_type='Inspection',
    name='inspection_1',
    p_effective=1,
    cost=1,
    t_interval=10,
    t_delay=1,

    triggers=dict(
        condition=dict(
            fast_degrading=dict(
                lower=1,
                upper=10,
            )
        ),
        state=dict(
            initiation=True,
            detection= True,
            failure = True
        ),
    ),

    impacts=dict(
        condition=dict(
            fast_degrading=dict(
                target=1,
                method="tbc",
                axis="time",
            ),
        ),
        state=dict(
            initiation=False,
            detection= False,
            failure = False
        ),
        system = 'component',
    ),
)


# *********************** state data **********************************

state_data = dict(demo.state_data)

# *********************** failure mode data **********************************

failure_mode_data = dict(demo.failure_mode_data)

complete['failure_mode_0']=dict(
    name='fm_0',
    pf_curve = 'linear',
    pf_interval=0,
    untreated=dict(complete['distribution_0']),
    conditions=dict(complete['indicator_0']),
    tasks=dict(complete['inspection_0']),
)

complete['failure_mode_1']=dict(
    name='fm_1',
    pf_curve = 'step',
    pf_interval=1,
    untreated=dict(complete['distribution_1']),
    conditions=dict(complete['indicator_1']),
    tasks=dict(complete['inspection_1']),
)

# *********************** component data **********************************

component_data = dict(demo.component_data)

complete['component_0'] = dict(
    name='component_0',
    fm=dict(complete['failure_mode_0']),
    indicator=dict(complete['indicator_0']),
)

complete['component_1'] = dict(
    name='component_1',
    fm=dict(complete['failure_mode_1']),
    indicator=dict(complete['indicator_1'])
)

if __name__ == "__main__":

    print("Fixtures - Ok")
