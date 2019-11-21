import numpy as np

from tests._constants import MARKERS_DATA, ANALOGS_DATA, EXPECTED_VALUES
from tests.utils import is_expected_array


def test_proc_abs():
    is_expected_array(ANALOGS_DATA.meca.abs(), **EXPECTED_VALUES.loc[1].to_dict())
    is_expected_array(MARKERS_DATA.meca.abs(), **EXPECTED_VALUES.loc[2].to_dict())


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
    is_expected_array(
        MARKERS_DATA.meca.square().meca.sqrt(), **EXPECTED_VALUES.loc[3].to_dict()
    )

    is_expected_array(
        ANALOGS_DATA.meca.square().meca.sqrt(), **EXPECTED_VALUES.loc[4].to_dict()
    )


def test_proc_norm():
    is_expected_array(
        MARKERS_DATA.meca.norm(dim="axis"), **EXPECTED_VALUES.loc[44].to_dict()
    )
    is_expected_array(
        MARKERS_DATA.meca.norm(dim="channel"), **EXPECTED_VALUES.loc[45].to_dict()
    )
    is_expected_array(
        MARKERS_DATA.meca.norm(dim="time_frame"), **EXPECTED_VALUES.loc[46].to_dict()
    )

    is_expected_array(
        ANALOGS_DATA.meca.norm(dim="channel"), **EXPECTED_VALUES.loc[47].to_dict()
    )
    is_expected_array(
        ANALOGS_DATA.meca.norm(dim="time_frame"), **EXPECTED_VALUES.loc[48].to_dict()
    )


def test_proc_rms():
    m = MARKERS_DATA.meca.rms()
    a = ANALOGS_DATA.meca.rms()

    np.testing.assert_array_almost_equal(m, 496.31764559, decimal=6)
    np.testing.assert_array_almost_equal(a, 0.00011321, decimal=6)


def test_proc_center():
    is_expected_array(MARKERS_DATA.meca.center(), **EXPECTED_VALUES.loc[5].to_dict())
    is_expected_array(
        MARKERS_DATA.meca.center(MARKERS_DATA.isel(time_frame=0)),
        **EXPECTED_VALUES.loc[6].to_dict()
    )

    is_expected_array(ANALOGS_DATA.meca.center(), **EXPECTED_VALUES.loc[7].to_dict())
    is_expected_array(
        ANALOGS_DATA.meca.center(mu=2), **EXPECTED_VALUES.loc[8].to_dict()
    )
    is_expected_array(
        ANALOGS_DATA.meca.center(ANALOGS_DATA.isel(time_frame=0)),
        **EXPECTED_VALUES.loc[9].to_dict()
    )


def test_proc_normalize():
    is_expected_array(
        MARKERS_DATA.meca.normalize(), **EXPECTED_VALUES.loc[20].to_dict()
    )
    is_expected_array(
        MARKERS_DATA.meca.normalize(scale=1), **EXPECTED_VALUES.loc[21].to_dict()
    )
    is_expected_array(
        MARKERS_DATA.meca.normalize(ref=MARKERS_DATA.sel(time_frame=5.76)),
        **EXPECTED_VALUES.loc[22].to_dict()
    )

    is_expected_array(
        ANALOGS_DATA.meca.normalize(), **EXPECTED_VALUES.loc[23].to_dict()
    )
    is_expected_array(
        ANALOGS_DATA.meca.normalize(scale=1), **EXPECTED_VALUES.loc[24].to_dict()
    )
    is_expected_array(
        ANALOGS_DATA.meca.normalize(ref=ANALOGS_DATA.sel(time_frame=5.76)),
        **EXPECTED_VALUES.loc[25].to_dict()
    )
