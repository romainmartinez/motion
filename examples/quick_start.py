import numpy as np
import pandas as pd
import xarray as xr

# Creating DataArray

fake_markers = np.random.randn(4, 23, 1000)
# fake_analogs = np.random.randn(1, 23, 1000)
time_frames = np.arange(1000)
channels = [f"channel_{i}" for i in range(fake_markers.shape[1])]

dims = ("data", "channels", "time_frames")

coords = {
    "data": ["x", "y", "z", "e"],
    "channels": channels,
    "time_frames": time_frames,
}
x = xr.DataArray(data=fake_markers, dims=dims, coords=coords)

# Indexing
## 1. Dimension: positional | index lookup: integer | example: d[:, 0]
x[:3, :, :]  # x, y, z all frames
x[[0, 1]]  # x, y all frames

## 2. Dimension: positional | index lookup: label | example: d.loc[:, 'IA']
x.loc["x"]  # x all frames
x.loc["x":"z"]  # x all frames
x.loc["x", "channel_1":"channel_5", 100:200]  # x channel 1 to 5 frames 100 to 200

## 3. Dimension: name | index lookup: integer | example: d.isel(space=0) or d[dict(space=0)]
x.isel(data=0)  # x all frames
x.isel(
    data=[0], channels=[0, 1, 2], time_frames=slice(100, 200)
)  # x channel 1 to 2 frames 100 to 200

## 4. Dimension: name | index lookup: label | example: d.sel(space="IA") or d.loc[dict(space="IA")]
x.sel(data="x")  # x all frames
x.sel(data=["x", "y"], channels=["channel_1", "channel_10"])  # x y channels 1 and 2

# Attributes
x.attrs["rate"] = 1000
x.attrs["units"] = "mm"
x.attrs["description"] = "A random variable created as an example."

# computation
## numpy like
x + 10
np.sin(x)
x.T
x.sum()

## aggregation
x.mean()  # mean all data (shape: 1)
x.mean(dim="data")  # mean over data (shape: (23, 1000)). Equivalent to x.mean(axis=0)
x.mean(
    dim=["data", "time_frames"]
)  # mean over data and time_frames (shape: (23,)) . Equivalent to x.mean(axis=[0, 2])

# Arithmetic operations broadcast based on dimension name. So you don’t need to insert dummy dimensions for alignment:
a = x.loc["x":"z"]  # shape (3, 23, 1000)
b = x.loc["e"]  # shape (23, 1000)
a + b  # shape (3, 23, 1000)

# It also means that in most cases you do not need to worry about the order of dimensions:
a - a.T  # shape (3, 23, 1000)

# Operations also align based on index labels:
x[:-1] - x[:1]

## groupby
x.groupby("channels").mean()  # mean of each channels
x.groupby("data").sum("channels").shape  # sum of channels for each data coord
x.groupby("data").apply(
    lambda x: x - x.min()
)  # substract min value for each data coord

## plotting
x.plot()  # plot histogram all data
x.loc["x", "channel_1"].plot()  # plot time serie of channel 1 at x
x.sel(channels="channel_1").plot.line(
    x="time_frames"
)  # plot time serie of channel 1 at x, y, z
x.sel(channels="channel_1").plot(
    x="time_frames", col="data"
)  # subplot x, y, z, e for channel_1

# pandas
s = x.to_series()
s.to_xarray()

# dataset is a dict-like container of aligned DataArray objects.
trials = xr.Dataset({"trial_1": x, "trial_2": x + 100})
# You can do almost everything you can do with DataArray objects with Dataset objects including indexing and arithmetic
# if you prefer to work with multiple variables at once.
trials.mean(dim="data")

# io
# NetCDF is the recommended file format for xarray objects.
# csv
csv_filename = "/home/romain/Documents/codes/motion/tests/data/markers.csv"
pd.read_csv(csv_filename, skiprows=[4], header=[2, 3]).head()
