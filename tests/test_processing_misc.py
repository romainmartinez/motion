import numpy as np
import pytest
import xarray as xr

from tests._constants import MARKERS_DATA, ANALOGS_DATA, EXPECTED_VALUES
from tests.utils import is_expected_array


def test_proc_fft():
    is_expected_array(
        ANALOGS_DATA.proc.fft(freq=ANALOGS_DATA.rate),
        **EXPECTED_VALUES.loc[40].to_dict()
    )
    is_expected_array(
        ANALOGS_DATA.proc.fft(freq=ANALOGS_DATA.rate, only_positive=False),
        **EXPECTED_VALUES.loc[41].to_dict()
    )

    is_expected_array(
        MARKERS_DATA.proc.fft(freq=ANALOGS_DATA.rate),
        **EXPECTED_VALUES.loc[42].to_dict()
    )
    is_expected_array(
        MARKERS_DATA.proc.fft(freq=ANALOGS_DATA.rate, only_positive=False),
        **EXPECTED_VALUES.loc[43].to_dict()
    )


def test_proc_detect_onset():
    m = MARKERS_DATA[0, 0, :]
    r = xr.DataArray(m.proc.detect_onset(threshold=m.mean() + m.std()))
    is_expected_array(r, **EXPECTED_VALUES.loc[49].to_dict())

    r = xr.DataArray(
        m.proc.detect_onset(
            threshold=m.mean(), n_below=10, threshold2=m.mean() + m.std()
        )
    )
    is_expected_array(r, **EXPECTED_VALUES.loc[50].to_dict())

    np.testing.assert_array_equal(x=m.proc.detect_onset(threshold=m.mean() * 10), y=0)

    with pytest.raises(ValueError):
        assert MARKERS_DATA[0, :, :].proc.detect_onset(threshold=0)
        assert MARKERS_DATA[:, :, :].proc.detect_onset(threshold=0)


def test_proc_detect_outliers():
    is_expected_array(
        MARKERS_DATA.proc.detect_outliers(threshold=3),
        **EXPECTED_VALUES.loc[51].to_dict()
    )
    is_expected_array(
        MARKERS_DATA.proc.detect_outliers(threshold=1),
        **EXPECTED_VALUES.loc[52].to_dict()
    )

    is_expected_array(
        ANALOGS_DATA.proc.detect_outliers(threshold=3),
        **EXPECTED_VALUES.loc[53].to_dict()
    )
    is_expected_array(
        ANALOGS_DATA.proc.detect_outliers(threshold=1),
        **EXPECTED_VALUES.loc[54].to_dict()
    )
