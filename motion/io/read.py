from pathlib import Path
from typing import Union

import ezc3d
import numpy as np
import xarray as xr

import motion


def read_c3d(
    group: str,
    filename: Union[str, Path],
    usecols: Union[int, str, list],
    prefix: str,
    attrs: dict,
) -> xr.DataArray:
    reader = ezc3d.c3d(f"{filename}").c3d_swig

    channels = [
        label.split(prefix)[-1]
        for label in reader.parameters()
        .group(group)
        .parameter("LABELS")
        .valuesAsString()
    ]

    get_data_function = getattr(reader, f"get_{group.lower()}s")

    if usecols is None:
        data = get_data_function()
        channels_used = channels
    else:
        if isinstance(usecols, int):
            idx = [usecols]
        elif isinstance(usecols, str):
            idx = [channels.index(usecols)]
        elif isinstance(usecols, list):
            if isinstance(usecols[0], str):
                idx = [channels.index(channel) for channel in usecols]
            elif isinstance(usecols[0], int):
                idx = usecols
            else:
                raise ValueError("values inside usecols must be int or str")
        else:
            raise ValueError("usecols must be int, str or a list")

        data = get_data_function()[:, idx, :]
        channels_used = [channels[i] for i in idx]

    data_by_frame = 1 if group == "POINT" else reader.header().nbAnalogByFrame()

    attrs = attrs if attrs is not None else {}
    attrs["first_frame"] = reader.header().firstFrame() * data_by_frame
    attrs["last_frame"] = reader.header().lastFrame() * data_by_frame
    attrs["rate"] = reader.header().frameRate() * data_by_frame
    attrs["unit"] = (
        reader.parameters().group(group).parameter("UNITS").valuesAsString()[0]
    )

    time_frame = np.arange(
        start=0, stop=data.shape[-1] / attrs["rate"], step=1 / attrs["rate"]
    )

    if group == "POINT":
        coords = {
            "axis": ["x", "y", "z", "translation"],
            "channel": channels_used,
            "time_frame": time_frame,
        }
        array = motion.Markers(data=data, dims=coords.keys(), coords=coords)
    else:
        coords = {"channel": channels_used, "time_frame": time_frame}
        array = motion.Analogs(data=data[0, ...], dims=coords.keys(), coords=coords)

    array.attrs = attrs
    return array


def read_analogs_c3d(
    filename: Union[str, Path],
    usecols: Union[int, str, list] = None,
    prefix: str = None,
    attrs: dict = None,
) -> xr.DataArray:
    return read_c3d("ANALOG", filename, usecols, prefix, attrs)


def read_markers_c3d(
    filename: Union[str, Path],
    usecols: Union[int, str, list] = None,
    prefix: str = None,
    attrs: dict = None,
) -> xr.DataArray:
    return read_c3d("POINT", filename, usecols, prefix, attrs)


# csv -----------------------
# def _read_csv(
#     group: str,
#     filename: Union[str, Path],
#     prefix: str,
#     time_frame: Union[list, np.array],
#     rate: int,
#     attrs: dict,
#     **kwargs,
# ) -> Array:
#     x = pd.read_csv(filename, **kwargs)
#
#     ###
#     # header = 2
#     # if header:
#     #     with open(f"{filename}") as fd:
#     #         reader = csv.reader(fd)
#     #         for idx, row in enumerate(reader):
#     #             if idx == header:
#     #                 header_row = [col.split(prefix)[-1] for col in row]
#     #                 break
#     #
#     # idx = [header_row.index(target) for target in kwargs['usecols']]
#     # if group == 'POINT':
#     #     a = [[i, i+1, i+2] for i in idx]
#     #     [i for h in a for i in h]
#
#     # if 'usecols' in kwargs:
#     #     cols = x.columns.str.split(prefix).str[-1].to_list()
#     #     idx = [cols.index(target) for target in kwargs['usecols'] if target in c]
#     ####
#
#     if time_frame:
#         time_frame = x[time_frame]
#         x = x.drop(time_frame, axis=1)
#     else:
#         time_frame = np.arange(start=0, stop=x.shape[0] / rate, step=1 / rate)
#
#     coords = {
#         "channels": x.columns,
#         "time_frame": time_frame,
#     }
#
#     array = Array(data=x.T, dims=coords.keys(), coords=coords)
#
#     attrs = attrs if attrs is not None else {}
#     attrs["rate"] = rate
#
#     return array


# def read_analogs_csv(
#     filename: str,
#     prefix: str = None,
#     time_frame: Union[list, np.array] = None,
#     rate: int = 1,
#     attrs: dict = None,
#     **kwargs,
# ) -> Array:
#     return _read_csv("ANALOG", filename, prefix, time_frame, rate, attrs, **kwargs)
#
#
# def read_markers_csv(
#     filename: str,
#     prefix: str = None,
#     time_frame: Union[list, np.array] = None,
#     rate: int = 1,
#     attrs: dict = None,
#     **kwargs,
# ) -> Array:
#     return _read_csv("POINT", filename, prefix, time_frame, rate, attrs, **kwargs)
