import numpy as np
import pytest
from xarray import DataArray, Dataset

from openlifu.bf import mask_focus
from openlifu.geo import Point


@pytest.fixture()
def example_focus() -> DataArray:
    return Point(
        id="test_focus_point",
        radius=1.,
        position=np.array([1.0, 0.0, 1.5]),
        dims=("x", "y", "z"),
        units="mm"
    )

@pytest.fixture()
def example_xarr() -> DataArray:
    rng = np.random.default_rng(147)
    return Dataset(
            {
                'p': DataArray(
                    data=rng.random((3, 3, 5)),
                    dims=["x", "y", "z"],
                    attrs={'units': "Pa"}
                )
            },
            coords={
                'x': DataArray(dims=["x"], data=np.linspace(-1, 1, 3), attrs={'units': "mm"}),
                'y': DataArray(dims=["y"], data=np.linspace(-1, 1, 3), attrs={'units': "mm"}),
                'z': DataArray(dims=["z"], data=np.linspace(0, 2, 5), attrs={'units': "mm"})
            }
        )

def test_mask_focus(example_xarr: Dataset, example_focus: Point):
    """Test that the distance grid from the focus point is correct."""
    expected = np.array(
        [[[False, False, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False]],

       [[False, False, False, False, False],
        [ True,  True,  True,  True, False],
        [False, False, False, False, False]],

       [[False, False,  True,  True,  True],
        [ True,  True,  True,  True,  True],
        [False, False,  True,  True,  True]]])
    mask = mask_focus(example_xarr, example_focus, distance=0.0011, aspect_ratio=(1.0, 1.0, 2.0))

    np.testing.assert_equal(mask, expected)
