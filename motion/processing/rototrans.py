from typing import Optional

import numpy as np
import xarray as xr

from motion import Angles


def rototrans_from_euler_angles(
    caller,
    angles: Optional[xr.DataArray] = None,
    angle_sequence: Optional[str] = None,
    translations: Optional[xr.DataArray] = None,
):
    if angles is None:
        angles = Angles()

    if translations is None:
        translations = Angles()

    if angle_sequence is None:
        angle_sequence = ""

    # Convert special zyzz angle sequence to zyz
    if angle_sequence == "zyzz":
        angles[2, :, :] -= angles[0, :, :]
        angle_sequence = "zyz"

    # If the user asked for a pure rotation
    if angles.time_frame.size != 0 and translations.time_frame.size == 0:
        translations = Angles(np.zeros((3, 1, angles.time_frame.size)))

    # If the user asked for a pure translation
    if angles.time_frame.size == 0 and translations.time_frame.size != 0:
        angles = Angles(np.zeros((0, 1, translations.time_frame.size)))

    # Sanity checks
    if angles.time_frame.size != translations.time_frame.size:
        raise IndexError(
            "Angles and translations must have the same number of frames. "
            f"You have translation = {translations.shape} and angles = {angles.shape}"
        )
    if angles.row.size != len(angle_sequence):
        raise IndexError(
            "Angles and angles_sequence must be the same size. "
            f"You have angles rows = {angles.row.size} and angle_sequence length = {len(angle_sequence)}"
        )
    if angles.time_frame.size == 0:
        return caller()

    empty_rt = np.repeat(
        np.eye(4)[..., np.newaxis], repeats=angles.time_frame.size, axis=2
    )
    rt = empty_rt.copy()
    for i in range(angles.row.size):
        a = angles.isel(row=i)
        matrix_to_prod = empty_rt.copy()
        if angle_sequence[i] == "x":
            # [[1, 0     ,  0     ],
            #  [0, cos(a), -sin(a)],
            #  [0, sin(a),  cos(a)]]
            matrix_to_prod[1, 1, :] = np.cos(a)
            matrix_to_prod[1, 2, :] = -np.sin(a)
            matrix_to_prod[2, 1, :] = np.sin(a)
            matrix_to_prod[2, 2, :] = np.cos(a)
        elif angle_sequence[i] == "y":
            # [[ cos(a), 0, sin(a)],
            #  [ 0     , 1, 0     ],
            #  [-sin(a), 0, cos(a)]]
            matrix_to_prod[0, 0, :] = np.cos(a)
            matrix_to_prod[0, 2, :] = np.sin(a)
            matrix_to_prod[2, 0, :] = -np.sin(a)
            matrix_to_prod[2, 2, :] = np.cos(a)
        elif angle_sequence[i] == "z":
            # [[cos(a), -sin(a), 0],
            #  [sin(a),  cos(a), 0],
            #  [0     ,  0     , 1]]
            matrix_to_prod[0, 0, :] = np.cos(a)
            matrix_to_prod[0, 1, :] = -np.sin(a)
            matrix_to_prod[1, 0, :] = np.sin(a)
            matrix_to_prod[1, 1, :] = np.cos(a)
        else:
            raise ValueError(
                "angle_sequence must be a permutation of axes (e.g. 'xyz', 'yzx', ...)"
            )
        rt = np.einsum("ijk,jlk->ilk", rt, matrix_to_prod)
    # Put the translations
    rt[:-1, -1:, :] = translations[:3, ...]
    return caller(rt)
