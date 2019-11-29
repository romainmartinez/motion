import pytest

import motion
from motion import Analogs, Markers
from tests._constants import (
    MARKERS_ANALOGS_C3D,
    ANALOGS_CSV,
    MARKERS_CSV,
    EXPECTED_VALUES,
)
from tests.utils import is_expected_array

_extensions = ["csv"]
# analogs -------------------
_analogs_cases = [
    {"usecols": None, **EXPECTED_VALUES.loc[10].to_dict()},
    {
        "usecols": ["EMG1", "EMG10", "EMG11", "EMG12"],
        **EXPECTED_VALUES.loc[11].to_dict(),
    },
    {"usecols": [1, 3, 5, 7], **EXPECTED_VALUES.loc[12].to_dict()},
    {"usecols": ["EMG1"], **EXPECTED_VALUES.loc[13].to_dict()},
    {"usecols": [2], **EXPECTED_VALUES.loc[14].to_dict()},
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
    decimal = 4
    if extension == "csv":
        Analogs.from_csv(
            ANALOGS_CSV, usecols=usecols, header=3, first_row=5, first_column=2
        )
        # if usecols is None:
        #     usecols = lambda a: a not in ["Frame", "Sub Frame"]
        # elif isinstance(usecols[0], int):
        #     usecols = [i + 2 for i in usecols]  # skip two first columns
        #     decimal = 0  # csv files are rounded
        # data = motion.read_analogs_csv(
        #     ANALOGS_CSV, usecols=usecols, skiprows=[0, 1, 2, 4], rate=2000
        # )
    elif extension == "c3d":
        data = motion.read_analogs_c3d(MARKERS_ANALOGS_C3D, prefix=".", usecols=usecols)
    else:
        raise ValueError("wrong extension provided")

    is_expected_array(
        data,
        shape_val,
        first_last_val,
        mean_val,
        median_val,
        sum_val,
        nans_val,
        decimal=decimal,
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
    {"usecols": ["CLAV_post"], **EXPECTED_VALUES.loc[18].to_dict()},
    {"usecols": [2], **EXPECTED_VALUES.loc[19].to_dict()},
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
    decimal = 4
    if extension == "csv":
        data = Markers.from_csv(
            MARKERS_CSV,
            usecols=usecols,
            header=2,
            first_row=5,
            first_column=2,
            prefix_delimiter=":",
        )
    elif extension == "c3d":
        data = motion.read_markers_c3d(MARKERS_ANALOGS_C3D, prefix=":", usecols=usecols)
    else:
        raise ValueError("wrong extension provided")

    is_expected_array(
        data,
        shape_val,
        first_last_val,
        mean_val,
        median_val,
        sum_val,
        nans_val,
        decimal=decimal,
    )
