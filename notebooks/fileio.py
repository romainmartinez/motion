from pathlib import Path

from motion import read_analogs_c3d, read_markers_c3d

# Path to data
DATA_FOLDER = Path("..") / "tests" / "data"
MARKERS_CSV = DATA_FOLDER / "markers.csv"
MARKERS_ANALOGS_C3D = DATA_FOLDER / "markers_analogs.c3d"
ANALOGS_CSV = DATA_FOLDER / "analogs.csv"

analogs = read_analogs_c3d(MARKERS_ANALOGS_C3D, prefix=".")
markers = read_markers_c3d(MARKERS_ANALOGS_C3D, prefix=".")

# analogs = motion.read_analogs_csv(
#     ANALOGS_CSV, usecols=["EMG1", "EMG2", "EMG3"], skiprows=[0, 1, 2, 4], rate=100
# )

print("debug")

# markers = motion.read_markers_csv(
#     MARKERS_CSV,
#     usecols=["CLAV_post", "PSISl", "STERr", "CLAV_post"],
#     skiprows=[0, 1,2, 4],
#     rate=100,
#     prefix=":",
# )

# markers = motion.read_csv(MARKERS_CSV, header=[0, 1], skiprows=[0, 1, 2], rate=10)
