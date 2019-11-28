from typing import Optional

import numpy as np
import xarray as xr


def create_rototrans(
    rt: Optional[np.array] = None,
    angles: Optional[xr.DataArray] = None,
    angle_sequence: Optional[str] = None,
    translations: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    if angle_sequence:
        rt = rt_from_euler_angles(
            angles=angles, angle_sequence=angle_sequence, translations=translations
        )
    else:
        if rt is None:
            rt = np.eye(4)
        if rt.shape[0] != 4 or rt.shape[1] != 4:
            raise IndexError(
                f"rt must have first and second dimensions of length 4, you have: {rt.shape}"
            )
        if rt.ndim == 2:
            rt = rt[..., np.newaxis]
    return xr.DataArray(data=rt, dims=["row", "col", "time_frame"])


def rt_from_euler_angles(
    angles: Optional[xr.DataArray] = None,
    angle_sequence: Optional[str] = None,
    translations: Optional[xr.DataArray] = None,
):
    if angles is None:
        angles = xr.DataArray(
            data=np.ndarray(shape=(0, 0, 0)), dims=["row", "col", "time_frame"]
        )

    if translations is None:
        translations = xr.DataArray(
            data=np.ndarray(shape=(0, 0, 0)), dims=["row", "col", "time_frame"]
        )

    if angle_sequence is None:
        angle_sequence = ""

    # Convert special zyzz angle sequence to zyz
    if angle_sequence == "zyzz":
        angles[2, :, :] -= angles[0, :, :]
        angle_sequence = "zyz"

    # If the user asked for a pure rotation
    if angles.time_frame.size != 0 and translations.time_frame.size == 0:
        translations = xr.DataArray(
            data=np.zeros((3, 1, angles.time_frame.size)),
            dims=["row", "col", "time_frame"],
        )

    # If the user asked for a pure translation
    if angles.time_frame.size == 0 and translations.time_frame.size != 0:
        angles = xr.DataArray(
            data=np.zeros((0, 1, translations.time_frame.size)),
            dims=["row", "col", "time_frame"],
        )

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
        return create_rototrans()

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
    return create_rototrans(rt)


def get_euler_angles(rototrans: xr.DataArray, angle_sequence: str):
    """
    Get euler angles with specified angle sequence
    Parameters
    ----------
    rototrans
        Rototrans created with motion.create_rototrans()
    angle_sequence
        Euler sequence of angles. Valid values are all permutations of "xyz"
    Returns
    -------
    DataArray with the euler angles associated
    """
    if angle_sequence == "zyzz":
        angles = xr.DataArray(
            data=np.ndarray(shape=(3, 1, rototrans.time_frame.size)),
            dims=["row", "col", "time_frame"],
        )
    else:
        angles = xr.DataArray(
            data=np.ndarray(shape=(len(angle_sequence), 1, rototrans.time_frame.size)),
            dims=["row", "col", "time_frame"],
        )

    if angle_sequence == "x":
        angles[0, :, :] = np.arcsin(rototrans[2, 1, :])
    elif angle_sequence == "y":
        angles[0, :, :] = np.arcsin(rototrans[0, 2, :])
    elif angle_sequence == "z":
        angles[0, :, :] = np.arcsin(rototrans[1, 0, :])
    elif angle_sequence == "xy":
        angles[0, :, :] = np.arcsin(rototrans[2, 1, :])
        angles[1, :, :] = np.arcsin(rototrans[0, 2, :])
    elif angle_sequence == "xz":
        angles[0, :, :] = -np.arcsin(rototrans[1, 2, :])
        angles[1, :, :] = -np.arcsin(rototrans[0, 1, :])
    elif angle_sequence == "yx":
        angles[0, :, :] = -np.arcsin(rototrans[2, 0, :])
        angles[1, :, :] = -np.arcsin(rototrans[1, 2, :])
    elif angle_sequence == "yz":
        angles[0, :, :] = np.arcsin(rototrans[0, 2, :])
        angles[1, :, :] = np.arcsin(rototrans[1, 0, :])
    elif angle_sequence == "zx":
        angles[0, :, :] = np.arcsin(rototrans[1, 0, :])
        angles[1, :, :] = np.arcsin(rototrans[2, 1, :])
    elif angle_sequence == "zy":
        angles[0, :, :] = -np.arcsin(rototrans[0, 1, :])
        angles[1, :, :] = -np.arcsin(rototrans[2, 0, :])
    elif angle_sequence == "xyz":
        angles[0, :, :] = np.arctan2(rototrans[1, 2, :], rototrans[2, 2, :])
        angles[1, :, :] = np.arcsin(rototrans[0, 1, :])
        angles[2, :, :] = np.arctan2(-rototrans[0, 1, :], rototrans[0, 0, :])
    elif angle_sequence == "xzy":
        angles[0, :, :] = np.arctan2(rototrans[2, 1, :], rototrans[1, 1, :])
        angles[2, :, :] = np.arctan2(rototrans[0, 2, :], rototrans[0, 0, :])
        angles[1, :, :] = -np.arcsin(rototrans[0, 1, :])
    elif angle_sequence == "yzx":
        angles[2, :, :] = np.arctan2(-rototrans[1, 2, :], rototrans[1, 1, :])
        angles[0, :, :] = np.arctan2(-rototrans[2, 0, :], rototrans[0, 0, :])
        angles[1, :, :] = np.arcsin(rototrans[1, 2, :])
    elif angle_sequence == "zxy":
        angles[1, :, :] = np.arcsin(rototrans[2, 1, :])
        angles[2, :, :] = np.arctan2(-rototrans[2, 0, :], rototrans[2, 2, :])
        angles[0, :, :] = np.arctan2(-rototrans[0, 1, :], rototrans[1, 1, :])
    elif angle_sequence in ["zyz", "zyzz"]:
        angles[0, :, :] = np.arctan2(rototrans[1, 2, :], rototrans[0, 2, :])
        angles[1, :, :] = np.arccos(rototrans[2, 2, :])
        angles[2, :, :] = np.arctan2(rototrans[2, 1, :], -rototrans[2, 0, :])
    elif angle_sequence == "zxz":
        angles[0, :, :] = np.arctan2(rototrans[0, 2, :], -rototrans[1, 2, :])
        angles[1, :, :] = np.arccos(rototrans[2, 2, :])
        angles[2, :, :] = np.arctan2(rototrans[2, 0, :], rototrans[2, 1, :])

    return angles
