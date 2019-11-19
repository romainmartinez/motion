import numpy as np
import xarray as xr


class RotoTrans(xr.DataArray):
    __slots__ = []

    def __init__(
        self,
        rt: np.array = np.eye(4),
        angles: xr.DataArray = None,
        angle_sequence: str = None,
        translations: xr.DataArray = None,
        *args,
        **kwargs,
    ):
        if angle_sequence:
            rt = self.rt_from_euler_angles(
                angles=angles, angle_sequence=angle_sequence, translations=translations
            )
        else:
            if rt.shape[0] != 4 or rt.shape[1] != 4:
                raise IndexError(
                    f"rt must have first and second dimensions of length 4, you have: {rt.shape}"
                )
            rt[3, :-1, ...] = 0
            rt[3, -1, ...] = 1

        if rt.ndim == 2:
            rt = rt[..., np.newaxis]

        super(RotoTrans, self).__init__(data=rt)

    @staticmethod
    def rt_from_euler_angles(
        angles: xr.DataArray = None,
        angle_sequence: str = None,
        translations: xr.DataArray = None,
    ):
        # Convert special zyzz angle sequence to zyz
        if angle_sequence == "zyzz":
            angles[2, :, :] -= angles[0, :, :]
            angle_sequence = "zyz"

        # If the user asked for a pure rotation
        if angles.get_num_frames() != 0 and translations.get_num_frames() == 0:
            translations = FrameDependentNpArray(
                np.zeros((3, 1, angles.get_num_frames()))
            )

        # If the user asked for a pure translation
        if angles.get_num_frames() == 0 and translations.get_num_frames() != 0:
            angles = FrameDependentNpArray(
                np.zeros((0, 1, translations.get_num_frames()))
            )

        # Sanity checks
        if angles.get_num_frames() != translations.get_num_frames():
            raise IndexError(
                "angles and translations must have the same number of frames"
            )
        if angles.shape[0] is not len(angle_sequence):
            raise IndexError("angles and angles_sequence must be the same size")
        if angles.get_num_frames() == 0:
            return RotoTrans()

        rt_out = np.repeat(np.eye(4)[:, :, np.newaxis], angles.get_num_frames(), axis=2)
        try:
            for i in range(len(angles)):
                a = angles[i, :, :]
                matrix_to_prod = np.repeat(
                    np.eye(4)[:, :, np.newaxis], angles.get_num_frames(), axis=2
                )
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
                        "angle_sequence must be a permutation of axes (e.g. "
                        "xyz"
                        ", "
                        "yzx"
                        ", ...)"
                    )
                rt_out = np.einsum("ijk,jlk->ilk", rt_out, matrix_to_prod)
        except IndexError:
            raise ValueError(
                "angle_sequence must be a permutation of axes (e.g. "
                "xyz"
                ", "
                "yzx"
                ", ...)"
            )

        # Put the translations
        rt_out[0:3, 3:4, :] = translations[0:3, :, :]

        return RotoTrans(rt_out)
