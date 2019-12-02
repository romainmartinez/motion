import numpy as np

from tests._constants import MARKERS_DATA, ANALOGS_DATA, EXPECTED_VALUES
from tests.utils import is_expected_array


def test_proc_abs():
    is_expected_array(ANALOGS_DATA.meca.abs(), **EXPECTED_VALUES[1])
    is_expected_array(MARKERS_DATA.meca.abs(), **EXPECTED_VALUES[2])


def test_proc_matmul():
    m = MARKERS_DATA.sel(axis="x", channel="PSISl").meca.matmul(
        MARKERS_DATA.sel(axis="y", channel="PSISl")
    )
    m_ref = MARKERS_DATA.sel(axis="x", channel="PSISl") @ MARKERS_DATA.sel(
        axis="y", channel="PSISl"
    )
    np.testing.assert_array_almost_equal(m, 87604420.17968313, decimal=6)
    np.testing.assert_array_almost_equal(m, m_ref, decimal=6)

    a = ANALOGS_DATA.sel(channel="EMG1").meca.matmul(ANALOGS_DATA.sel(channel="EMG10"))
    a_ref = ANALOGS_DATA.sel(channel="EMG1") @ ANALOGS_DATA.sel(channel="EMG10")
    np.testing.assert_array_almost_equal(a, -1.41424996e-06, decimal=6)
    np.testing.assert_array_almost_equal(a, a_ref, decimal=6)


def test_proc_square_sqrt():
    is_expected_array(MARKERS_DATA.meca.square().meca.sqrt(), **EXPECTED_VALUES[3])

    is_expected_array(ANALOGS_DATA.meca.square().meca.sqrt(), **EXPECTED_VALUES[4])


def test_proc_norm():
    is_expected_array(MARKERS_DATA.meca.norm(dim="axis"), **EXPECTED_VALUES[44])
    is_expected_array(MARKERS_DATA.meca.norm(dim="channel"), **EXPECTED_VALUES[45])
    is_expected_array(MARKERS_DATA.meca.norm(dim="time_frame"), **EXPECTED_VALUES[46])

    is_expected_array(ANALOGS_DATA.meca.norm(dim="channel"), **EXPECTED_VALUES[47])
    is_expected_array(ANALOGS_DATA.meca.norm(dim="time_frame"), **EXPECTED_VALUES[48])


def test_proc_rms():
    m = MARKERS_DATA.meca.rms()
    a = ANALOGS_DATA.meca.rms()

    np.testing.assert_array_almost_equal(m, 496.31764559, decimal=6)
    np.testing.assert_array_almost_equal(a, 0.00011321, decimal=6)


def test_proc_center():
    is_expected_array(MARKERS_DATA.meca.center(), **EXPECTED_VALUES[5])
    is_expected_array(
        MARKERS_DATA.meca.center(MARKERS_DATA.isel(time_frame=0)), **EXPECTED_VALUES[6]
    )

    is_expected_array(ANALOGS_DATA.meca.center(), **EXPECTED_VALUES[7])
    is_expected_array(ANALOGS_DATA.meca.center(mu=2), **EXPECTED_VALUES[8])
    is_expected_array(
        ANALOGS_DATA.meca.center(ANALOGS_DATA.isel(time_frame=0)), **EXPECTED_VALUES[9]
    )


def test_proc_normalize():
    is_expected_array(MARKERS_DATA.meca.normalize(), **EXPECTED_VALUES[20])
    is_expected_array(MARKERS_DATA.meca.normalize(scale=1), **EXPECTED_VALUES[21])
    is_expected_array(
        MARKERS_DATA.meca.normalize(ref=MARKERS_DATA.sel(time_frame=5.76)),
        **EXPECTED_VALUES[22]
    )

    is_expected_array(ANALOGS_DATA.meca.normalize(), **EXPECTED_VALUES[23])
    is_expected_array(ANALOGS_DATA.meca.normalize(scale=1), **EXPECTED_VALUES[24])
    is_expected_array(
        ANALOGS_DATA.meca.normalize(ref=ANALOGS_DATA.sel(time_frame=5.76)),
        **EXPECTED_VALUES[25]
    )
