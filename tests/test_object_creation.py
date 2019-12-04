import numpy as np
import pytest
import xarray as xr

from motion import Analogs, Markers, Angles
from ._constants import ANALOGS_DATA, MARKERS_DATA, EXPECTED_VALUES
from .utils import is_expected_array


def test_analogs_creation():
    a = Analogs()
    np.testing.assert_array_equal(x=a, y=xr.DataArray())
    assert a.dims == ("channel", "time_frame")

    b = Analogs(ANALOGS_DATA.values)
    is_expected_array(b, **EXPECTED_VALUES[56])

    with pytest.raises(ValueError):
        assert Analogs(MARKERS_DATA)


def test_markers_creation():
    a = Markers()
    np.testing.assert_array_equal(x=a, y=xr.DataArray())
    assert a.dims == ("axis", "channel", "time_frame")

    b = Markers(MARKERS_DATA.values)
    is_expected_array(b, **EXPECTED_VALUES[57])

    with pytest.raises(ValueError):
        assert Markers(ANALOGS_DATA)


def test_angles_creation():
    a = Angles()
    np.testing.assert_array_equal(x=a, y=xr.DataArray())

    b = Angles(MARKERS_DATA.values, time_frames=MARKERS_DATA.time_frame)
    is_expected_array(b, **EXPECTED_VALUES[57])

    with pytest.raises(ValueError):
        assert Angles(ANALOGS_DATA)
