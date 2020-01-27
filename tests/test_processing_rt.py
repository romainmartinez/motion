from itertools import permutations

import numpy as np
import pytest

from motion import Angles, Rototrans

SEQ = (
    ["".join(p) for i in range(1, 4) for p in permutations("xyz", i)]
    + ["zyzz"]
    + ["zxz"]
)
SEQ = [s for s in SEQ if s not in ["yxz", "zyx"]]
EPSILON = 1e-12
ANGLES = Angles(np.random.rand(4, 1, 100))


@pytest.mark.parametrize("seq", SEQ)
def test_euler2rot_rot2euleur(seq, angles=ANGLES, epsilon=EPSILON):
    if seq == "zyzz":
        angles_to_test = angles[:3, ...]
    else:
        angles_to_test = angles[: len(seq), ...]
    r = Rototrans.from_euler_angles(angles=angles_to_test, angle_sequence=seq)
    a = Angles.from_rototrans(rototrans=r, angle_sequence=seq)

    np.testing.assert_array_less((a - angles_to_test).sum(), epsilon)


def test_construct_rt():
    eye = Rototrans()
    np.testing.assert_equal(eye.time_frame.size, 1)
    np.testing.assert_equal(eye.sel(time_frame=0), np.eye(4))

    eye = Rototrans.from_euler_angles()
    np.testing.assert_equal(eye.time_frame.size, 1)
    np.testing.assert_equal(eye.sel(time_frame=0), np.eye(4))

    # Test the way to create a rt, but not when providing bot angles and sequence
    nb_frames = 10
    random_vector = Angles(np.random.rand(3, 1, nb_frames))

    # with angles
    rt_random_angles = Rototrans.from_euler_angles(
        angles=random_vector, angle_sequence="xyz"
    )
    np.testing.assert_equal(rt_random_angles.time_frame.size, nb_frames)
    np.testing.assert_equal(
        rt_random_angles[:-1, -1:, :], np.zeros((3, 1, nb_frames))
    )  # Translation is 0

    # with translation
    rt_random_translation = Rototrans.from_euler_angles(translations=random_vector)
    np.testing.assert_equal(rt_random_translation.time_frame.size, nb_frames)
    np.testing.assert_equal(
        rt_random_translation[:3, :3, :],
        np.repeat(np.eye(3)[:, :, np.newaxis], nb_frames, axis=2),
    )  # rotation is eye3
    np.arange(0, rt_random_angles.time_frame.size / 0.5, 1 / 0.5)

    rt_with_time_frame = Rototrans(
        rt_random_angles,
        time_frames=np.arange(0, rt_random_angles.time_frame.size / 100, 1 / 100),
    )
    assert rt_with_time_frame.time_frame[-1] == 0.09

    with pytest.raises(IndexError):
        assert Rototrans(data=np.zeros(1))

    with pytest.raises(IndexError):
        assert Rototrans.from_euler_angles(
            angles=random_vector[..., :5],
            translations=random_vector,
            angle_sequence="x",
        )

    with pytest.raises(IndexError):
        assert Rototrans.from_euler_angles(angles=random_vector, angle_sequence="x")

    with pytest.raises(ValueError):
        assert Rototrans.from_euler_angles(angles=random_vector, angle_sequence="nop")


# def test_transpose_rt():
#     random_angles = Angles(np.random.rand(3, 1, 100))
#     random_rt = Rototrans.from_euler_angles(random_angles, angle_sequence="xyz")
#     transpose_rototrans(random_rt)
