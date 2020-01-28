import numpy as np
import xarray as xr


def angles_from_rototrans(
    caller, rototrans: xr.DataArray, angle_sequence: str
) -> xr.DataArray:
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
        angles = caller(np.ndarray(shape=(3, 1, rototrans.time_frame.size)))
    else:
        angles = caller(
            np.ndarray(shape=(len(angle_sequence), 1, rototrans.time_frame.size))
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
        angles[0, :, :] = np.arctan2(-rototrans[1, 2, :], rototrans[2, 2, :])
        angles[1, :, :] = np.arcsin(rototrans[0, 2, :])
        angles[2, :, :] = np.arctan2(-rototrans[0, 1, :], rototrans[0, 0, :])
    elif angle_sequence == "xzy":
        angles[0, :, :] = np.arctan2(rototrans[2, 1, :], rototrans[1, 1, :])
        angles[2, :, :] = np.arctan2(rototrans[0, 2, :], rototrans[0, 0, :])
        angles[1, :, :] = np.arcsin(-rototrans[0, 1, :])
    elif angle_sequence == "yzx":
        angles[2, :, :] = np.arctan2(-rototrans[1, 2, :], rototrans[1, 1, :])
        angles[0, :, :] = np.arctan2(-rototrans[2, 0, :], rototrans[0, 0, :])
        angles[1, :, :] = np.arcsin(rototrans[1, 0, :])
    elif angle_sequence == "zxy":
        angles[1, :, :] = np.arcsin(rototrans[2, 1, :])
        angles[2, :, :] = np.arctan2(-rototrans[2, 0, :], rototrans[2, 2, :])
        angles[0, :, :] = np.arctan2(-rototrans[0, 1, :], rototrans[1, 1, :])
    elif angle_sequence in ["zyz", "zyzz"]:
        angles[0, :, :] = np.arctan2(rototrans[1, 2, :], rototrans[0, 2, :])
        angles[1, :, :] = np.arccos(rototrans[2, 2, :])
        angles[2, :, :] = np.arctan2(rototrans[2, 1, :], -rototrans[2, 0, :])
    elif angle_sequence == "zyx":
        angles[2, :, :] = np.arctan2(rototrans[2, 1, :], rototrans[2, 2, :])
        angles[1, :, :] = np.arcsin(-rototrans[2, 0, :])
        angles[0, :, :] = np.arctan2(rototrans[1, 0, :], rototrans[0, 0, :])
    elif angle_sequence == "zxz":
        angles[0, :, :] = np.arctan2(rototrans[0, 2, :], -rototrans[1, 2, :])
        angles[1, :, :] = np.arccos(rototrans[2, 2, :])
        angles[2, :, :] = np.arctan2(rototrans[2, 0, :], rototrans[2, 1, :])

    return angles
