import pytest
import numpy as np
from openlifu.util.units import (
    getsiscale
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
