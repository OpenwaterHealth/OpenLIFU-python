from __future__ import annotations

import numpy as np
import pytest
from xarray import DataArray, Dataset

from openlifu.util.units import (
    get_ndgrid_from_arr,
    getsiscale,
    rescale_coords,
    rescale_data_arr,
)


@pytest.fixture()
def example_xarr() -> Dataset:
    rng = np.random.default_rng(147)
    return Dataset(
            {
                'p': DataArray(
                    data=rng.random((3, 2)),
                    dims=["x", "y"],
                    attrs={'units': "Pa"}
                ),
                'it': DataArray(
                    data=rng.random((3, 2)),
                    dims=["x", "y"],
                    attrs={'units': "W/cm^2"}
                )
            },
            coords={
                'x': DataArray(dims=["x"], data=np.linspace(0, 1, 3), attrs={'units': "m"}),
                'y': DataArray(dims=["y"], data=np.linspace(0, 1, 2), attrs={'units': "m"})
            }
        )


def test_getsiscale():
    with pytest.raises(ValueError, match="Unknown prefix"):
        getsiscale('xx','distance')

    assert getsiscale('mm', 'distance') == 1e-3
    assert getsiscale('km', 'distance') == 1e3
    assert getsiscale('mm^2', 'area') == 1e-6
    assert getsiscale('mm^3', 'volume') == 1e-9
    assert getsiscale('ns', 'time') == 1e-9
    assert getsiscale('nanosecond', 'time') == 1e-9
    assert getsiscale('hour', 'time') == 3600.
    assert getsiscale('rad', 'angle') == 1.
    assert np.allclose(getsiscale('deg', 'angle'), np.pi/180.)
    assert getsiscale('MHz', 'frequency') == 1e6
    assert getsiscale('GHz', 'frequency') == 1e9
    assert getsiscale('THz', 'frequency') == 1e12


def test_rescale_data_arr(example_xarr: Dataset):
    """Test that an xarray data can be correctly rescaled."""
    expected_p = 1e-6 * example_xarr['p'].data
    expected_it = 1e4 * example_xarr['it'].data
    rescaled_p = rescale_data_arr(example_xarr['p'], units="MPa")
    rescaled_it = rescale_data_arr(example_xarr['it'], units="W/m^2")

    np.testing.assert_almost_equal(rescaled_p, expected_p)
    np.testing.assert_almost_equal(rescaled_it, expected_it)


def test_rescale_coords(example_xarr: Dataset):
    """Test that an xarray coords can be correctly rescaled."""
    expected_x = 1e3 * example_xarr['p'].coords['x'].data
    expected_y = 1e3 * example_xarr['p'].coords['y'].data
    rescaled = rescale_coords(example_xarr['p'], units="mm")

    np.testing.assert_almost_equal(rescaled['x'].data, expected_x)
    np.testing.assert_almost_equal(rescaled['y'].data, expected_y)


def test_get_ndgrid_from_arr(example_xarr: Dataset):
    """Test that an ndgrid can be constructed from the coordinates of an xarray."""
    expected_X = np.array([[0., 0.], [0.5, 0.5], [1., 1.]])
    expected_Y = np.array([[0., 1.], [0., 1.], [0., 1.]])
    expected = np.stack([expected_X, expected_Y], axis=-1)
    ndgrid = get_ndgrid_from_arr(example_xarr)

    np.testing.assert_equal(ndgrid, expected)
