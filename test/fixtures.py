"""
Usage

>>> import fixtures

fixtures.complete contains a complete data set for each object

>>> fixtures.complete['distribution][0]
{name=...}

fixtures.update contains a data set for testing the update methods 
>>> fixtures.complete['failure_mode']['update']


Notes:
Use copy.deepcopy to ensure tests do not change the same underlying data
"""

import copy

import testconfig  # pylint: disable=unused-import
import pof.demo as demo


# class Fixtures():

# fmt: off
complete = {
    'pof_base':{},
    'distribution':{},
    'condition_indicator':{},
    'task':{},
    'condition_task':{},
    'scheduled_task':{},
    'inspection':{},
    'consequence': {},
    'failure_mode':{},
    'component':{},
    'system':{}
}

update = copy.deepcopy(complete)

# *********************** load data *****************************************

complete['pof_base'][0] = dict(
    name='pof_base_0',
    units='years',
)

complete['pof_base'][1] = dict(
    name='pof_base_1',
    units='months',
)

complete['pof_base']['update'] = copy.deepcopy(complete['pof_base'][1])

# *********************** distribution data **********************************

distribution_data = copy.deepcopy(demo.distribution_data)

complete['distribution'][0] =  dict(
    name='dist_0',
    alpha = 0,
    beta = 0,
    gamma = 0
)

complete['distribution'][1] = dict(
    name='dist_1',
    alpha = 1,
    beta = 1,
    gamma = 1
)

complete['distribution']['update'] = copy.deepcopy(complete['distribution'][1])

# *********************** indicator data **********************************

condition_data = copy.deepcopy(demo.condition_data)

complete['condition_indicator'][0] = dict(
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

complete['condition_indicator'][1] = dict(
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

complete['condition_indicator']['update'] = copy.deepcopy(complete['condition_indicator'][1])

# *********************** trigger_data *******************************



# *********************** task data **********************************
inspection_data = copy.deepcopy(demo.inspection_data)

repair_data = copy.deepcopy(demo.repair_data)

replacement_data = copy.deepcopy(demo.replacement_data)

# --------------- Task --------------------

complete['task'][0] = dict(
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

complete['task'][1] = dict(
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

complete['task']['update'] = copy.deepcopy(complete['task'][1])

# -------------- Scheduled Task ------------

complete['scheduled_task'][0] = dict(
    complete['task'][0],
    task_type='ScheduledTask',
    name='scheduled_task_0',
    t_interval=1,
    t_delay=0,
)


complete['scheduled_task'][1] = dict(
    complete['task'][1],
    task_type='ScheduledTask',
    name='scheduled_task_1',
    t_interval=10,
    t_delay=1,
)

complete['scheduled_task']['update'] = copy.deepcopy(complete['scheduled_task'][1])

# -------------- Condition Task ------------

complete['condition_task'][0] = dict(
    complete['task'][0],
    task_type='ConditionTask',
    name='condition_task_0',
    task_completion = 'immediate',
)

complete['condition_task'][1] = dict(
    complete['task'][1],
    task_type='ConditionTask',
    name='condition_task_1',
    task_completion = 'immediate',
)

complete['condition_task']['update'] = copy.deepcopy(complete['condition_task'][1])

# ------------- Inspection -----------------

complete['inspection'][0] = dict(
    complete['scheduled_task'][0],
    task_type='Inspection',
    name='inspection_0',
)

complete['inspection'][1] = dict(
    complete['scheduled_task'][1],
    task_type='Inspection',
    name='inspection_1',
)

complete['inspection']['update'] = copy.deepcopy(complete['inspection'][1])

# *********************** consequence data **********************************

complete['consequence'][0] = dict(
    name='consequence_0',
    cost=10,
    group=None,
    units="years"
)


complete['consequence'][1] = dict(
    name='consequence_1',
    cost=20,
    group=None,
    units="years"
)

complete['consequence']['update'] = copy.deepcopy(complete['consequence'][1])

# *********************** state data **********************************

state_data = copy.deepcopy(demo.state_data)

# *********************** failure mode data **********************************

failure_mode_data = copy.deepcopy(demo.failure_mode_data)

failure_mode_data['predictable'] = dict(
    name='predictable',
    pf_curve='linear',
    pf_interval=0,
    untreated = distribution_data['predictable'],
)

complete['failure_mode'][0]=dict(
    name='fm_0',
    pf_curve = 'linear',
    pf_interval=0,
    untreated=complete['distribution'][0],
    conditions=complete['condition_indicator'][0],
    indicators=complete['condition_indicator'][0],
    consequence=complete['consequence'][0],
    tasks=complete['inspection'][0],
)

complete['failure_mode'][1]=dict(
    name='fm_1',
    pf_curve = 'step',
    pf_interval=1,
    untreated=complete['distribution'][1],
    conditions=complete['condition_indicator'][1],
    indicators=complete['condition_indicator'][1],
    consequence=complete['consequence'][1],
    tasks=complete['inspection'][1],
)

complete['failure_mode']['update'] = dict(
    complete['failure_mode'][1],
    #dists={'untreated':complete['distribution'][1]}, #TODO this was previously 'distribution_0'
    conditions={'condition_indicator_0':complete['condition_indicator'][1]},
    indicators={'condition_indicator_0':complete['condition_indicator'][1]},
    consequence=complete['consequence'][1],
    tasks={'inspection_0':complete['inspection'][1]},
)
#complete['failure_mode']['update'].update('name')

# *********************** component data **********************************

component_data = copy.deepcopy(demo.component_data)

complete['component'][0] = dict(
    name='component_0',
    fm=complete['failure_mode'][0],
    indicator=complete['condition_indicator'][0],
)

complete['component'][1] = dict(
    name='component_1',
    fm=complete['failure_mode'][1],
    indicator=complete['condition_indicator'][1],
)

complete['component']['update'] = dict(
    complete['component'][1],
    fm= {'fm_0':complete['failure_mode']['update']},
    indicator={'condition_indicator_0':complete['condition_indicator']['update']},
)

# *********************** system data **********************************

system_data = copy.deepcopy(demo.system_data)

complete['system'][0] = dict(
    name='system_0',
    comp=complete['component'][0],
)

complete['system'][1] = dict(
    name='system_1',
    comp=complete['component'][1]
)

complete['system']['update'] = dict(
    complete['system'][1],
    comp= {'component_0':complete['component']['update']},
)

complete = copy.deepcopy(complete)

# @classmethod
# def deepcopy(cls, string):

#     return copy.deepcopy cls.

if __name__ == "__main__":
    print("Fixtures - Ok")
