import pytest

import motion
from tests._constants import MARKERS_ANALOGS_C3D
from tests.utils import is_expected_array

# analogs -------------------
_extensions = ["c3d"]
_analogs_cases = [
    {
        "usecols": None,
        "shape_val": (38, 11600),
        "first_last_val": [-0.022051572799682617, 0.0],
        "mean_val": -0.00167903,
        "median_val": 0.0,
        "sum_val": -740.11800894,
        "nans_val": 0,
    },
    {
        "usecols": ["EMG1", "EMG10", "EMG11", "EMG12"],
        "shape_val": (4, 11600),
        "first_last_val": [-2.6089122911798768e-05, 0.0],
        "mean_val": 7.37744731e-06,
        "median_val": 1.58788589e-06,
        "sum_val": 0.34231356,
        "nans_val": 0,
    },
    {
        "usecols": [1, 3, 5, 7],
        "shape_val": (4, 11600),
        "first_last_val": [-0.01039797067642212, 1.0517047485336661e-05],
        "mean_val": -0.00743816,
        "median_val": -2.68319245e-05,
        "sum_val": -345.13052138,
        "nans_val": 0,
    },
    {
        "usecols": "EMG1",
        "shape_val": (1, 11600),
        "first_last_val": [-2.6089122911798768e-05, -1.912586776597891e-05],
        "mean_val": -1.93823432e-05,
        "median_val": -2.14651582e-05,
        "sum_val": -0.22483518,
        "nans_val": 0,
    },
    {
        "usecols": 2,
        "shape_val": (1, 11600),
        "first_last_val": [-0.01544412225484848, 0.0],
        "mean_val": -0.08497782,
        "median_val": -0.01491006,
        "sum_val": -985.74269734,
        "nans_val": 0,
    },
]
_analogs_params = [
    (
        d["usecols"],
        d["shape_val"],
        d["first_last_val"],
        d["mean_val"],
        d["median_val"],
        d["sum_val"],
        d["nans_val"],
    )
    for d in _analogs_cases
]


@pytest.mark.parametrize(
    "usecols, shape_val, first_last_val, mean_val, median_val, sum_val, nans_val",
    _analogs_params,
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
    {
        "usecols": None,
        "shape_val": (4, 51, 580),
        "first_last_val": [44.16278839111328, 1.0],
        "mean_val": 362.29798491,
        "median_val": 337.75192261,
        "sum_val": 42535594.91827867,
        "nans_val": 915,
    },
    {
        "usecols": ["CLAV_post", "PSISl", "STERr", "CLAV_post"],
        "shape_val": (4, 4, 580),
        "first_last_val": [744.5535888671875, 1.0],
        "mean_val": 385.66622389,
        "median_val": 278.73994446,
        "sum_val": 3537330.60548401,
        "nans_val": 108,
    },
    {
        "usecols": [1, 3, 5, 7],
        "shape_val": (4, 4, 580),
        "first_last_val": [32.572296142578125, 1.0],
        "mean_val": 249.52316566,
        "median_val": 87.43008041,
        "sum_val": 2309586.42132697,
        "nans_val": 24,
    },
    {
        "usecols": "CLAV_post",
        "shape_val": (4, 1, 580),
        "first_last_val": [744.5535888671875, 1.0],
        "mean_val": 422.10368959,
        "median_val": 312.92277527,
        "sum_val": 976747.93771362,
        "nans_val": 6,
    },
    {
        "usecols": 2,
        "shape_val": (4, 1, 580),
        "first_last_val": [-93.72181701660156, 1.0],
        "mean_val": 242.87715054,
        "median_val": 124.81892014,
        "sum_val": 556188.67473757,
        "nans_val": 30,
    },
]
_markers_params = [
    (
        d["usecols"],
        d["shape_val"],
        d["first_last_val"],
        d["mean_val"],
        d["median_val"],
        d["sum_val"],
        d["nans_val"],
    )
    for d in _markers_cases
]


@pytest.mark.parametrize(
    "usecols, shape_val, first_last_val, mean_val, median_val, sum_val, nans_val",
    _markers_params,
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
