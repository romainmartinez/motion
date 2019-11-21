from tests._constants import ANALOGS_DATA, MARKERS_DATA, EXPECTED_VALUES
from tests.utils import is_expected_array


def test_proc_filters():
    freq = ANALOGS_DATA.rate
    order = 2

    is_expected_array(
        ANALOGS_DATA.meca.low_pass(freq=freq, order=order, cutoff=5),
        **EXPECTED_VALUES.loc[32].to_dict(),
    )
    is_expected_array(
        ANALOGS_DATA.meca.high_pass(freq=freq, order=order, cutoff=100),
        **EXPECTED_VALUES.loc[33].to_dict(),
    )
    is_expected_array(
        ANALOGS_DATA.meca.band_pass(freq=freq, order=order, cutoff=[10, 200]),
        **EXPECTED_VALUES.loc[34].to_dict(),
    )
    is_expected_array(
        ANALOGS_DATA.meca.band_stop(freq=freq, order=order, cutoff=[40, 60]),
        **EXPECTED_VALUES.loc[35].to_dict(),
    )

    freq = MARKERS_DATA.rate
    is_expected_array(
        MARKERS_DATA.meca.low_pass(freq=freq, order=order, cutoff=5),
        **EXPECTED_VALUES.loc[36].to_dict(),
    )
    is_expected_array(
        MARKERS_DATA.meca.high_pass(freq=freq, order=order, cutoff=10),
        **EXPECTED_VALUES.loc[37].to_dict(),
    )
    is_expected_array(
        MARKERS_DATA.meca.band_pass(freq=freq, order=order, cutoff=[1, 10]),
        **EXPECTED_VALUES.loc[38].to_dict(),
    )
    is_expected_array(
        MARKERS_DATA.meca.band_stop(freq=freq, order=order, cutoff=[5, 6]),
        **EXPECTED_VALUES.loc[39].to_dict(),
    )
