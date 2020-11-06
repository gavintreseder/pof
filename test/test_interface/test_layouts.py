import unittest
from unittest.mock import MagicMock, Mock, patch

import testconfig  # pylint: disable=unused-import
from pof import Component, FailureMode
from pof.interface.layouts import (
    make_component_layout,
    make_failure_mode_layout,
    validate_layout,
)


class TestLayout(unittest.TestCase):
    def test_make_component_layout(self):
        # Arrange
        comp = Component.demo()

        # Act
        layout = make_component_layout(comp)
        valid = validate_layout(comp, layout)

        # Assert
        self.assertTrue(valid)

    def test_make_failure_mode_layout(self):
        # Arrange
        fm = FailureMode.demo()

        # Act
        layout = make_failure_mode_layout(fm)
        valid = validate_layout(fm, layout)

        # Assert
        self.assertTrue(valid)


"""for ms in ms_fig_update:
    if not ms in str(mfml):
        print(ms)

        # Failure Mode Tests
mfml = make_failure_mode_layout(fm, prefix="fm-")"""


"""
# Task Tests
task = fm.tasks['ocr']
mtl = make_task_layout(task)

mtf = make_task_form(task, prefix="")

# Trigger Tests
triggers= fm.tasks['ocr'].triggers
mttl = make_task_trigger_layout(triggers)

condition_triggers = fm.tasks['ocr'].triggers['condition']
#mctf = make_condition_trigger_form(condition_triggers)


# Impact Tests

impacts= fm.tasks['ocr'].impacts
mtil = make_task_impact_layout(impacts)

state_impacts = fm.tasks['ocr'].impacts['state']
msil = make_state_impact_layout(state_impacts)

condition_impacts = fm.tasks['ocr'].impacts['condition']
mcil = make_condition_impact_layout(condition_impacts)

condition_impact = fm.tasks['ocr'].impacts['condition']['wall_thickness']
mtif = make_condition_impact_form(condition_impact)

# Distribution Tests
failure_dist = fm.failure_dist
mdl = make_dist_layout(failure_dist)"""