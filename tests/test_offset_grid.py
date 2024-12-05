import numpy as np
import pytest
from xarray import DataArray, Dataset

from openlifu.bf import offset_grid
from openlifu.geo import Point


@pytest.fixture()
def example_focus() -> DataArray:
    return Point(
        id="test_focus_point",
        radius=1.,
        position=np.array([0.5, 0.5, 0.5]),
        dims=("x", "y", "z"),
        units="mm"
    )

@pytest.fixture()
def example_xarr() -> DataArray:
    rng = np.random.default_rng(147)
    return Dataset(
            {
                'p': DataArray(
                    data=rng.random((3, 2, 3)),
                    dims=["x", "y", "z"],
                    attrs={'units': "Pa"}
                )
            },
            coords={
                'x': DataArray(dims=["x"], data=np.linspace(0, 1, 3), attrs={'units': "mm"}),
                'y': DataArray(dims=["y"], data=np.linspace(0, 1, 2), attrs={'units': "mm"}),
                'z': DataArray(dims=["z"], data=np.linspace(0, 1, 3), attrs={'units': "mm"})
            }
        )

def test_offset_grid(example_xarr: Dataset, example_focus: Point):
    """Test that the distance grid from the focus point is correct."""
    expected = np.array([
        [[[ 0.00000000e+00,  0.00000000e+00, -8.66025404e-04],
         [-3.53553391e-01, -2.04124145e-01,  2.87809109e-01],
         [-7.07106781e-01, -4.08248290e-01,  5.76484244e-01]],

        [[ 0.00000000e+00,  8.16496581e-01,  5.76484244e-01],
         [-3.53553391e-01,  6.12372436e-01,  8.65159378e-01],
         [-7.07106781e-01,  4.08248290e-01,  1.15383451e+00]]],


       [[[ 3.53553391e-01, -2.04124145e-01,  2.87809109e-01],
         [ 0.00000000e+00, -4.08248290e-01,  5.76484244e-01],
         [-3.53553391e-01, -6.12372436e-01,  8.65159378e-01]],

        [[ 3.53553391e-01,  6.12372436e-01,  8.65159378e-01],
         [ 0.00000000e+00,  4.08248290e-01,  1.15383451e+00],
         [-3.53553391e-01,  2.04124145e-01,  1.44250965e+00]]],


       [[[ 7.07106781e-01, -4.08248290e-01,  5.76484244e-01],
         [ 3.53553391e-01, -6.12372436e-01,  8.65159378e-01],
         [ 0.00000000e+00, -8.16496581e-01,  1.15383451e+00]],

        [[ 7.07106781e-01,  4.08248290e-01,  1.15383451e+00],
         [ 3.53553391e-01,  2.04124145e-01,  1.44250965e+00],
         [ 0.00000000e+00,  0.00000000e+00,  1.73118478e+00]]]])
    offset = offset_grid(example_xarr, example_focus)

    np.testing.assert_almost_equal(offset, expected)
