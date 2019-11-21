# def test_proc_get_euler_angles():
#     get_euler_angles(MARKERS_DATA, angle_sequence="x")
#     is_expected_array(MARKERS_DATA)
# Define all the possible angle_sequence to tests

from motion.rototrans import RotoTrans

SEQ = [
    "x",
    "y",
    "z",
    "xy",
    "xz",
    "yx",
    "yz",
    "zx",
    "zy",
    "xyz",
    "xzy",
    "yxz",
    "yzx",
    "zxy",
    "zyx",
    "zyzz",
]
# If the difference between the initial and the final angles are less than epsilon, tests is success
EPSILON = 1e-12


# Define some random data to tests


def test_construct_rt():
    eye = RotoTrans()
    # np.testing.assert_equal(eye.shape[-1], 1)
    # np.testing.assert_equal(eye[:, :, 0], np.eye(4))
    #
    # eye = RotoTrans(RotoTrans.rt_from_euler_angles())
    # np.testing.assert_equal(eye.get_num_frames(), 1)
    # np.testing.assert_equal(eye[:, :, 0], np.eye(4))
