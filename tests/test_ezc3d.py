import ezc3d

from ._constants import MARKERS_ANALOGS_C3D, EXPECTED_VALUES
from .utils import is_expected_array
import xarray as xr

import numpy as np


def test_ezc3d():
    c3d = ezc3d.c3d(f"{MARKERS_ANALOGS_C3D}")

    # c3d = ezc3d.c3d("markers_analogs.c3d")
    array = c3d["data"]["points"]
    decimal = 6

    np.testing.assert_array_equal(
        x=array.shape, y=(4, 51, 580), err_msg="Shape does not match"
    )
    raveled = array.ravel()
    np.testing.assert_array_almost_equal(
        x=raveled[0],
        y=44.16278839111328,
        err_msg="First value does not match",
        decimal=decimal,
    )
    np.testing.assert_array_almost_equal(
        x=raveled[-1], y=1.0, err_msg="Last value does not match", decimal=decimal,
    )
    np.testing.assert_array_almost_equal(
        x=np.nanmean(array),
        y=362.2979849093196,
        decimal=decimal,
        err_msg="Mean does not match",
    )
    np.testing.assert_array_almost_equal(
        x=np.nanmedian(array),
        y=337.7519226074219,
        decimal=decimal,
        err_msg="Median does not match",
    )
    np.testing.assert_allclose(
        actual=np.nansum(array),
        desired=42535594.91827867,
        rtol=0.05,
        err_msg="Sum does not match",
    )
    np.testing.assert_array_equal(
        x=np.isnan(array).sum(), y=915, err_msg="Nans value value does not match"
    )

    # is_expected_array(xr.DataArray(c3d["data"]["points"]), **EXPECTED_VALUES[65])
    # is_expected_array(xr.DataArray(c3d["data"]["analogs"]), **EXPECTED_VALUES[66])
