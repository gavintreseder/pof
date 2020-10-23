"""
Usage

from distribution import Demo as demo

demo.slow_aging
demo.distribution.slow_aging
"""

import pof.demo as demo

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

distribution_data = demo.distribution_data

# *********************** condition data **********************************

condition_data = demo.condition_data

# *********************** task data **********************************
inspection_data = demo.inspection_data

repair_data = demo.repair_data

replacement_data = demo.replacement_data

# *********************** state data **********************************

state_data = demo.state_data

# *********************** failure mode data ********************************** 

failure_mode_data = demo.failure_mode_data

# *********************** component data **********************************

component_data = demo.component_data

component_data_slow = dict(

    comp=dict(
        name = 'comp',
        fm = dict(
            slow_aging = failure_mode_data['slow_aging'],
        ),
        indicator = dict(
            slow_degrading =condition_data['slow_degrading'],
            fast_degrading=condition_data['fast_degrading']
        )
    )
)


if __name__ == "__main__":

    print("Fixtures - Ok")
