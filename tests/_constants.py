from pathlib import Path

from motion import read_markers_c3d, read_analogs_c3d

# Path to data
if "tests" in f"{Path('.').absolute()}":
    DATA_FOLDER = Path('data')
else:
    DATA_FOLDER = Path("tests") / "data"


MARKERS_CSV = DATA_FOLDER / "markers.csv"
MARKERS_ANALOGS_C3D = DATA_FOLDER / "markers_analogs.c3d"
ANALOGS_CSV = DATA_FOLDER / "analogs.csv"

MARKERS_DATA = read_markers_c3d(MARKERS_ANALOGS_C3D, usecols=["CLAV_post", "PSISl", "STERr", "CLAV_post"], prefix=':')
ANALOGS_DATA = read_analogs_c3d(MARKERS_ANALOGS_C3D, usecols=["EMG1", "EMG10", "EMG11", "EMG12"], prefix='.')
