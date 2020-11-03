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

complete['distribution_0'] =  dict(
    name='dist_0',
    alpha = 0,
    beta = 0,
    gamma = 0
)

complete['distribution_1'] = dict(
    name='dist_1',
    alpha = 1,
    beta = 1,
    gamma = 1
)

# *********************** indicator data **********************************

condition_data = dict(demo.condition_data)

complete['condition_indicator_0'] = dict(
    name='condition_indicator_0',
    perfect=0,
    failed=1,
    pf_curve='linear',
    pf_interval=1,
    pf_std=0,
    initial=0,
    threshold_failure=0,
    threshold_detection=0,
)

complete['condition_indicator_1'] = dict(
    name='condition_indicator_1',
    perfect=1,
    failed=0,
    pf_curve='step',
    pf_interval=10,
    pf_std=1,
    initial=1,
    threshold_failure=1,
    threshold_detection=1,
)

# *********************** trigger_data *******************************



# *********************** task data **********************************
inspection_data = dict(demo.inspection_data)

repair_data = dict(demo.repair_data)

replacement_data = dict(demo.replacement_data)

# --------------- Task --------------------

complete['task_0'] = dict(
    task_type='Task',
    name='task_0',
    p_effective=0,
    cost=0,

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

complete['task_1'] = dict(
    task_type='Task',
    name='task_1',
    p_effective=1,
    cost=1,

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

# -------------- Scheduled Task ------------

complete['scheduled_task_0'] = dict(complete['task_0'])
complete['scheduled_task_0'].update(dict(
    task_type='ScheduledTask',
    name='scheduled_task_0',
    t_interval=1,
    t_delay=0,
))


complete['scheduled_task_1'] = dict(complete['task_1'])
complete['scheduled_task_1'].update(dict(
    task_type='ScheduledTask',
    name='scheduled_task_1',
    t_interval=10,
    t_delay=1,
))

# -------------- Condition Task ------------

complete['condition_task_0'] = dict(complete['task_0'])
complete['condition_task_0'].update(dict(
    task_type='ConditionTask',
    name='condition_task_0',
    task_completion = 'immediate',
))

complete['condition_task_1'] = dict(complete['task_1'])
complete['condition_task_1'].update(dict(
    task_type='ConditionTask',
    name='condition_task_1',
    task_completion = 'immediate',
))

# ------------- Inspection -----------------

complete['inspection_0'] = dict(complete['scheduled_task_0'])
complete['inspection_0'].update(dict(
    task_type='Inspection',
    name='inspection_0',
))

complete['inspection_1'] = dict(complete['scheduled_task_1'])
complete['inspection_1'].update(dict(
    task_type='Inspection',
    name='inspection_1',
))


# *********************** state data **********************************

state_data = dict(demo.state_data)

# *********************** failure mode data **********************************

failure_mode_data = dict(demo.failure_mode_data)

complete['failure_mode_0']=dict(
    name='fm_0',
    pf_curve = 'linear',
    pf_interval=0,
    untreated=dict(complete['distribution_0']),
    conditions=dict(complete['condition_indicator_0']),
    tasks=dict(complete['inspection_0']),
)

complete['failure_mode_1']=dict(
    name='fm_1',
    pf_curve = 'step',
    pf_interval=1,
    untreated=dict(complete['distribution_1']),
    conditions=dict(complete['condition_indicator_1']),
    tasks=dict(complete['inspection_1']),
)

# *********************** component data **********************************

component_data = dict(demo.component_data)

complete['component_0'] = dict(
    name='component_0',
    fm=dict(complete['failure_mode_0']),
    indicator=dict(complete['condition_indicator_0']),
)

complete['component_1'] = dict(
    name='component_1',
    fm=dict(complete['failure_mode_1']),
    indicator=dict(complete['condition_indicator_1'])
)

if __name__ == "__main__":
    print("Fixtures - Ok")
