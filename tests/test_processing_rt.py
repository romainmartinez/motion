# from itertools import permutations
#
# import numpy as np
# import pytest
# import xarray as xr
#
# import motion
#
# SEQ = (
#     ["".join(p) for i in range(1, 4) for p in permutations("xyz", i)]
#     + ["zyzz"]
#     + ["zxz"]
# )
# EPSILON = 1e-12
# ANGLES = xr.DataArray(np.random.rand(40, 1, 100), dims=["row", "col", "time_frame"])
#
#
# @pytest.mark.parametrize("seq", SEQ)
# def test_euler2rot_rot2euleur(seq, angles=ANGLES, epsilon=EPSILON):
#     print("debug")
#     if seq == "zyzz":
#         angles_to_test = angles[:3, ...]
#     else:
#         angles_to_test = angles[: len(seq), ...]
#     p = motion.create_rototrans(angles=angles_to_test, angle_sequence=seq)
#     a = p.meca.get_euler_angles(angle_sequence=seq)
#
#     np.testing.assert_array_less((a - angles_to_test).sum(), epsilon)
#
#
# def test_construct_rt():
#
#     ######################
#     translation = eye[0:3, 3:4, :]
#     rotation = eye[:3, :3, :]
#
#     eye[:3, 3:4, 0]
#     eye[..., 0]
#     ######################
#
#     eye = motion.create_rototrans()
#     np.testing.assert_equal(eye.time_frame.size, 1)
#     np.testing.assert_equal(eye.sel(time_frame=0), np.eye(4))
#
#     eye = motion.create_rototrans(rt=motion.rt_from_euler_angles())
#     np.testing.assert_equal(eye.time_frame.size, 1)
#     np.testing.assert_equal(eye.sel(time_frame=0), np.eye(4))
#
#     # Test the way to create a rt, but not when providing bot angles and sequence
#     nb_frames = 10
#     random_vector = xr.DataArray(
#         data=np.random.rand(3, 1, nb_frames), dims=["row", "col", "time_frame"]
#     )
#
#     rt_random_angles = motion.create_rototrans(
#         rt=motion.rt_from_euler_angles(angles=random_vector, angle_sequence="xyz")
#     )
#     np.testing.assert_equal(rt_random_angles.time_frame.size, nb_frames)
#     np.testing.assert_equal(
#         rt_random_angles[:-1, -1:, :], np.zeros((3, 1, nb_frames))
#     )  # Translation is 0
#
#     rt_random_translation = motion.create_rototrans(
#         motion.rt_from_euler_angles(translations=random_vector)
#     )
#     np.testing.assert_equal(rt_random_translation.time_frame.size, nb_frames)
#     np.testing.assert_equal(
#         rt_random_translation[:3, :3, :],
#         np.repeat(np.eye(3)[:, :, np.newaxis], nb_frames, axis=2),
#     )  # rotation is eye3
#
#     with pytest.raises(IndexError):
#         assert motion.create_rototrans(rt=np.zeros(1))
#
#     with pytest.raises(IndexError):
#         assert motion.rt_from_euler_angles(
#             angles=random_vector[..., :5],
#             translations=random_vector,
#             angle_sequence="x",
#         )
#
#     with pytest.raises(IndexError):
#         assert motion.rt_from_euler_angles(angles=random_vector, angle_sequence="x")
#
#     with pytest.raises(ValueError):
#         assert motion.rt_from_euler_angles(angles=random_vector, angle_sequence="nop")
