import ezc3d

from ._constants import MARKERS_ANALOGS_C3D, EXPECTED_VALUES
from .utils import is_expected_array
import xarray as xr

def test_ezc3d():
    c3d = ezc3d.c3d(f"{MARKERS_ANALOGS_C3D}")

    is_expected_array(xr.DataArray(c3d['data']['points']), **EXPECTED_VALUES[65])
    is_expected_array(xr.DataArray(c3d["data"]["analogs"]), **EXPECTED_VALUES[66])
