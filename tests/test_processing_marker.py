import numpy as np
import pytest

from motion import Angles, Rototrans, Markers


def test_rotate():
    n_frames = 100
    n_markers = 10

    angles = Angles.from_random_data(size=(3, 1, n_frames))
    rt = Rototrans.from_euler_angles(angles, "xyz")
    markers = Markers.from_random_data(size=(3, n_markers, n_frames))

    rotated_markers = markers.meca.rotate(rt)

    expected_rotated_marker = np.ndarray((4, n_markers, n_frames))
    for marker in range(n_markers):
        for frame in range(n_frames):
            expected_rotated_marker[:, marker, frame] = np.dot(
                rt.isel(time_frame=frame),
                markers.isel(channel=marker, time_frame=frame),
            )

    np.testing.assert_array_almost_equal(
        rotated_markers, expected_rotated_marker, decimal=10
    )

    rotated_markers = markers.isel(time_frame=0).meca.rotate(rt.isel(time_frame=0))
    expected_rotated_marker = np.ndarray(rotated_markers.shape)
    for marker in range(n_markers):
        expected_rotated_marker[:, marker] = np.dot(
            rt.isel(time_frame=0), markers.isel(channel=marker, time_frame=0)
        )

    np.testing.assert_array_almost_equal(
        rotated_markers, expected_rotated_marker, decimal=10
    )

    rotated_markers = markers.meca.rotate(rt.isel(time_frame=0))
    expected_rotated_marker = np.ndarray(rotated_markers.shape)
    for marker in range(n_markers):
        expected_rotated_marker[:, marker] = np.dot(
            rt.isel(time_frame=0), markers.isel(channel=marker)
        )

    np.testing.assert_array_almost_equal(
        rotated_markers, expected_rotated_marker, decimal=10
    )

    with pytest.raises(ValueError):
        markers.isel(time_frame=0).meca.rotate(rt)
