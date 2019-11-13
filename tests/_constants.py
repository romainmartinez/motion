from pathlib import Path

import pandas as pd

from motion import read_markers_c3d, read_analogs_c3d

# Path to data
if "tests" in f"{Path('.').absolute()}":
    DATA_FOLDER = Path("data")
else:
    DATA_FOLDER = Path("tests") / "data"


MARKERS_CSV = DATA_FOLDER / "markers.csv"
MARKERS_ANALOGS_C3D = DATA_FOLDER / "markers_analogs.c3d"
ANALOGS_CSV = DATA_FOLDER / "analogs.csv"
EXPECTED_VALUES_CSV = DATA_FOLDER / "is_expected_array_val.csv"

MARKERS_DATA = read_markers_c3d(
    MARKERS_ANALOGS_C3D,
    usecols=["CLAV_post", "PSISl", "STERr", "CLAV_post"],
    prefix=":",
)
ANALOGS_DATA = read_analogs_c3d(
    MARKERS_ANALOGS_C3D, usecols=["EMG1", "EMG10", "EMG11", "EMG12"], prefix="."
)

EXPECTED_VALUES = pd.read_csv(
    EXPECTED_VALUES_CSV,
    index_col=[0],
    converters={"shape_val": eval, "first_last_val": eval},
)
