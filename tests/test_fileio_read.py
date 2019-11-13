import pytest

import motion
from tests._constants import MARKERS_ANALOGS_C3D, EXPECTED_VALUES
from tests.utils import is_expected_array

_extensions = ["c3d"]
# analogs -------------------
_analogs_cases = [
    {"usecols": None, **EXPECTED_VALUES.loc[10].to_dict()},
    {
        "usecols": ["EMG1", "EMG10", "EMG11", "EMG12"],
        **EXPECTED_VALUES.loc[11].to_dict(),
    },
    {"usecols": [1, 3, 5, 7], **EXPECTED_VALUES.loc[12].to_dict()},
    {"usecols": "EMG1", **EXPECTED_VALUES.loc[13].to_dict()},
    {"usecols": 2, **EXPECTED_VALUES.loc[14].to_dict()},
]


@pytest.mark.parametrize(
    "usecols, shape_val, first_last_val, mean_val, median_val, sum_val, nans_val",
    [(d.values()) for d in _analogs_cases],
)
@pytest.mark.parametrize("extension", _extensions)
def test_read_analogs(
    usecols,
    shape_val,
    first_last_val,
    mean_val,
    median_val,
    sum_val,
    nans_val,
    extension,
):
    reader = getattr(motion, f"read_analogs_{extension}")
    data = reader(MARKERS_ANALOGS_C3D, prefix=".", usecols=usecols)

    is_expected_array(
        data, shape_val, first_last_val, mean_val, median_val, sum_val, nans_val
    )


@pytest.mark.parametrize("usecols", [2.0, [20.0]])
@pytest.mark.parametrize("extension", _extensions)
def test_read_catch_error(
    usecols, extension,
):
    reader = getattr(motion, f"read_analogs_{extension}")
    with pytest.raises(ValueError):
        assert reader(MARKERS_ANALOGS_C3D, usecols=usecols)


# markers -------------------
_markers_cases = [
    {"usecols": None, **EXPECTED_VALUES.loc[15].to_dict()},
    {
        "usecols": ["CLAV_post", "PSISl", "STERr", "CLAV_post"],
        **EXPECTED_VALUES.loc[16].to_dict(),
    },
    {"usecols": [1, 3, 5, 7], **EXPECTED_VALUES.loc[17].to_dict()},
    {"usecols": "CLAV_post", **EXPECTED_VALUES.loc[18].to_dict()},
    {"usecols": 2, **EXPECTED_VALUES.loc[19].to_dict()},
]


@pytest.mark.parametrize(
    "usecols, shape_val, first_last_val, mean_val, median_val, sum_val, nans_val",
    [(d.values()) for d in _markers_cases],
)
@pytest.mark.parametrize("extension", _extensions)
def test_read_markers(
    usecols,
    shape_val,
    first_last_val,
    mean_val,
    median_val,
    sum_val,
    nans_val,
    extension,
):
    reader = getattr(motion, f"read_markers_{extension}")
    data = reader(MARKERS_ANALOGS_C3D, prefix=":", usecols=usecols)

    is_expected_array(
        data, shape_val, first_last_val, mean_val, median_val, sum_val, nans_val
    )
