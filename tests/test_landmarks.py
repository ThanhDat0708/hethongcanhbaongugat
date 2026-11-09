import pytest

from drowsiness.landmarks import FaceLandmarkDetector


def test_measure_calculates_ear_and_mar_from_face_landmarks():
    points = [(0.0, 0.0)] * 478
    for indices in (
        (33, 160, 158, 133, 153, 144),
        (362, 385, 387, 263, 373, 380),
    ):
        first, second, third, fourth, fifth, sixth = indices
        points[first] = (0.0, 0.0)
        points[second] = (0.2, 0.1)
        points[third] = (0.4, 0.1)
        points[fourth] = (1.0, 0.0)
        points[fifth] = (0.4, -0.1)
        points[sixth] = (0.2, -0.1)
    points[78] = (0.0, 0.0)
    points[308] = (1.0, 0.0)
    for upper, lower, x in ((13, 14, 0.5), (82, 87, 0.25), (312, 317, 0.75)):
        points[upper] = (x, 0.1)
        points[lower] = (x, -0.1)

    measurements = FaceLandmarkDetector.measure(
        [type("Landmark", (), {"x": x, "y": y})() for x, y in points],
        image_width=1000,
        image_height=1000,
    )

    assert measurements.ear == pytest.approx(0.2)
    assert measurements.mar == pytest.approx(0.3)
    assert measurements.face_box == (0, 0, 1000, 116)

    scaled = FaceLandmarkDetector.measure(
        [type("Landmark", (), {"x": x, "y": y})() for x, y in points],
        image_width=640,
        image_height=480,
    )
    assert scaled.ear == pytest.approx(0.15)
    assert scaled.mar == pytest.approx(0.225)
