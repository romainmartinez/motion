import numpy as np

from tests._constants import MARKERS_DATA, ANALOGS_DATA, EXPECTED_VALUES
from tests.utils import is_expected_array


def test_proc_time_normalization():
    is_expected_array(
        MARKERS_DATA.proc.time_normalization(), **EXPECTED_VALUES.loc[26].to_dict()
    )
    is_expected_array(
        MARKERS_DATA.proc.time_normalization(n_frames=1000),
        **EXPECTED_VALUES.loc[27].to_dict()
    )
    time_vector = np.linspace(
        MARKERS_DATA.time_frame[0], MARKERS_DATA.time_frame[100], 100
    )
    is_expected_array(
        MARKERS_DATA.proc.time_normalization(time_vector=time_vector),
        **EXPECTED_VALUES.loc[28].to_dict()
    )

    is_expected_array(
        ANALOGS_DATA.proc.time_normalization(), **EXPECTED_VALUES.loc[29].to_dict()
    )
    is_expected_array(
        ANALOGS_DATA.proc.time_normalization(n_frames=1000),
        **EXPECTED_VALUES.loc[30].to_dict()
    )
    time_vector = np.linspace(
        ANALOGS_DATA.time_frame[0], ANALOGS_DATA.time_frame[100], 100
    )
    is_expected_array(
        ANALOGS_DATA.proc.time_normalization(time_vector=time_vector),
        **EXPECTED_VALUES.loc[31].to_dict()
    )
