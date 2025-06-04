from __future__ import annotations

import pytest

from openlifu.bf.apod_methods import MaxAngle, PiecewiseLinear, Uniform


# Test apodization methods with default parameters
@pytest.mark.parametrize("method_class", [Uniform, MaxAngle, PiecewiseLinear])
def test_apodization_methods_default_params(method_class):
    method = method_class()
    assert isinstance(method, method_class)

# Test apodization methods with custom parameters
@pytest.mark.parametrize(("method_class", "params"), [
    (Uniform, {}),
    (MaxAngle, {"max_angle": 45.0}),
    (PiecewiseLinear, {"zero_angle": 90.0, "rolloff_angle": 30.0}),
])
def test_apodization_methods_custom_params(method_class, params):
    method = method_class(**params)
    assert isinstance(method, method_class)
    for key, value in params.items():
        assert getattr(method, key) == value

# Test apodization methods with invalid parameters
@pytest.mark.parametrize(("method_class","invalid_params"), [
    (MaxAngle, {"max_angle": -10.0}),
    (PiecewiseLinear, {"zero_angle": 30.0, "rolloff_angle": 45.0}),
])
def test_apodization_methods_invalid_params(method_class, invalid_params):
    with pytest.raises((TypeError, ValueError)):
        method_class(**invalid_params)

# Test apodization methods with non-numeric parameters
@pytest.mark.parametrize(("method_class","invalid_params"), [
    (MaxAngle, {"max_angle": "invalid"}),
    (PiecewiseLinear, {"zero_angle": "invalid", "rolloff_angle": "invalid"}),
])
def test_apodization_methods_non_numeric_params(method_class, invalid_params):
    with pytest.raises(TypeError):
        method_class(**invalid_params)
