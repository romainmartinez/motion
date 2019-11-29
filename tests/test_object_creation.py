import numpy as np
import pytest
import xarray as xr

from motion import Analogs, Markers
from ._constants import ANALOGS_DATA, MARKERS_DATA, EXPECTED_VALUES
from .utils import is_expected_array, print_expected_values


def test_analogs_creation():
    a = Analogs()
    np.testing.assert_array_equal(x=a, y=xr.DataArray())
    assert a.dims == ("channel", "time_frame")

    b = Analogs(ANALOGS_DATA.values)
    is_expected_array(b, **EXPECTED_VALUES.loc[56].to_dict())

    with pytest.raises(ValueError):
        assert Analogs(MARKERS_DATA)


def test_markers_creation():
    a = Markers()
    np.testing.assert_array_equal(x=a, y=xr.DataArray())
    assert a.dims == ("axis", "channel", "time_frame")

    b = Markers(MARKERS_DATA.values)
    is_expected_array(b, **EXPECTED_VALUES.loc[57].to_dict())
    print_expected_values(b)

    with pytest.raises(ValueError):
        assert Analogs(ANALOGS_DATA)
