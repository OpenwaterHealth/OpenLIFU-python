from __future__ import annotations

import pytest

from openlifu.plan.param_constraint import ParameterConstraint

# ---- Tests for ParameterConstraint ----

@pytest.fixture()
def threshold_constraint():
    return ParameterConstraint(operator="<=", warning_value=5.0, error_value=7.0)

@pytest.fixture()
def range_constraint():
    return ParameterConstraint(operator="within", warning_value=(2.0, 4.0), error_value=(1.0, 5.0))

def test_invalid_no_thresholds():
    with pytest.raises(ValueError, match="At least one of warning_value or error_value must be set"):
        ParameterConstraint(operator="<=")

def test_invalid_warning_tuple_order():
    with pytest.raises(ValueError, match="Warning value must be a sorted tuple"):
        ParameterConstraint(operator="within", warning_value=(4.0, 2.0))

def test_invalid_error_type():
    with pytest.raises(ValueError, match="Error value must be a single value"):
        ParameterConstraint(operator=">", error_value=(1.0, 2.0))

@pytest.mark.parametrize(("value", "op", "threshold", "expected"), [
    (3, "<", 5, True),
    (5, "<", 5, False),
    (5, "<=", 5, True),
    (6, ">", 5, True),
    (5, ">=", 5, True),
    (3, "within", (2, 4), True),
    (2, "within", (2, 4), False),
    (1, "inside", (2, 4), False),
    (2, "inside", (2, 4), True),
    (3, "inside", (2, 4), True),
    (1, "outside", (2, 4), True),
    (2, "outside", (2, 4), False),
    (2, "outside_inclusive", (2, 4), True),
    (3, "outside_inclusive", (2, 4), False),
])
def test_compare(value, op, threshold, expected):
    assert ParameterConstraint.compare(value, op, threshold) == expected

def test_is_warning(threshold_constraint):
    assert not threshold_constraint.is_warning(4.0)
    assert threshold_constraint.is_warning(6.0)

def test_is_error(threshold_constraint):
    assert not threshold_constraint.is_error(6.0)
    assert threshold_constraint.is_error(8.0)

def test_is_warning_range(range_constraint):
    assert not range_constraint.is_warning(3.0)
    assert range_constraint.is_warning(4.5)

def test_is_error_range(range_constraint):
    assert not range_constraint.is_error(3.0)
    assert range_constraint.is_error(5.5)

@pytest.mark.parametrize(("value", "expected_status"), [
    (3.0, "ok"),
    (6.5, "warning"),
    (7.5, "error"),
])
def test_get_status_threshold(value, expected_status):
    constraint = ParameterConstraint(operator="<=", warning_value=5.5, error_value=7.0)
    assert constraint.get_status(value) == expected_status

@pytest.mark.parametrize(("value", "expected"), [
    (2.5, "ok"),
    (0.5, "warning"),
    (5.5, "error"),
])
def test_get_status_range(value, expected):
    constraint = ParameterConstraint(operator="within", warning_value=(1.0, 4.0), error_value=(0.0, 5.0))
    assert constraint.get_status(value) == expected
