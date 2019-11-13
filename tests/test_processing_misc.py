from tests._constants import MARKERS_DATA, ANALOGS_DATA, EXPECTED_VALUES
from tests.utils import is_expected_array, print_expected_values


def test_proc_fft():
    print_expected_values(
        ANALOGS_DATA.proc.fft(freq=ANALOGS_DATA.rate, only_positive=False)
    )
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


def test_detect_outliers():
    ANALOGS_DATA.proc.detect_outliers(threshold=3)
