import numpy as np

from tests._constants import MARKERS_DATA, ANALOGS_DATA
from tests.utils import is_expected_array


def test_proc_abs():
    a = ANALOGS_DATA.proc.abs()
    is_expected_array(
        a,
        shape_val=(4, 11600),
        first_last_val=(2.6089122911798768e-05, 0.0),
        mean_val=4.88684185e-05,
        median_val=2.52773225e-05,
        sum_val=2.26749462,
        nans_val=0,
    )

    m = MARKERS_DATA.proc.abs()
    is_expected_array(
        m,
        shape_val=(4, 4, 580),
        first_last_val=(744.5535888671875, 1.0),
        mean_val=385.66622389,
        median_val=278.73994446,
        sum_val=3537330.60548401,
        nans_val=108,
    )


def test_proc_matmul():
    m = MARKERS_DATA.sel(axis="x", channel="PSISl").proc.matmul(
        MARKERS_DATA.sel(axis="y", channel="PSISl")
    )
    m_ref = MARKERS_DATA.sel(axis="x", channel="PSISl") @ MARKERS_DATA.sel(
        axis="y", channel="PSISl"
    )
    np.testing.assert_array_almost_equal(m, 87604420.17968313, decimal=6)
    np.testing.assert_array_almost_equal(m, m_ref, decimal=6)

    a = ANALOGS_DATA.sel(channel="EMG1").proc.matmul(ANALOGS_DATA.sel(channel="EMG10"))
    a_ref = ANALOGS_DATA.sel(channel="EMG1") @ ANALOGS_DATA.sel(channel="EMG10")
    np.testing.assert_array_almost_equal(a, -1.41424996e-06, decimal=6)
    np.testing.assert_array_almost_equal(a, a_ref, decimal=6)


def test_proc_square_sqrt():
    m = MARKERS_DATA.proc.square().proc.sqrt()
    is_expected_array(
        m,
        shape_val=(4, 4, 580),
        first_last_val=(744.5535888671875, 1.0),
        mean_val=385.66622389,
        median_val=278.73994446,
        sum_val=3537330.60548401,
        nans_val=108,
    )

    a = ANALOGS_DATA.proc.square().proc.sqrt()
    is_expected_array(
        a,
        shape_val=(4, 11600),
        first_last_val=(2.6089122911798768e-05, 0.0),
        mean_val=4.88684185e-05,
        median_val=2.52773225e-05,
        sum_val=2.26749462,
        nans_val=0,
    )


def test_proc_rms():
    m = MARKERS_DATA.proc.rms()
    a = ANALOGS_DATA.proc.rms()

    np.testing.assert_array_almost_equal(m, 496.31764559, decimal=6)
    np.testing.assert_array_almost_equal(a, 0.00011321, decimal=6)
