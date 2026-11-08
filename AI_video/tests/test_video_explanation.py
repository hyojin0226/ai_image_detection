"""Numerical checks for spatial explanations, including no-signal behavior."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from video_explanation import positive_gradcam, overlay_heatmap, strongest_zone


class ExplanationTests(unittest.TestCase):
    def test_known_channel_weights_and_positive_localization(self):
        activation = np.zeros((2, 2, 2), np.float32)
        activation[0, 1] = [3, 1]
        activation[1, 0] = [0, 4]
        gradient = np.ones_like(activation)
        gradient[:, :, 1] = -1
        heat = positive_gradcam(activation, gradient)
        np.testing.assert_array_equal(heat, [[0, 1], [0, 0]])

    def test_no_positive_signal_does_not_invent_heat(self):
        self.assertIsNone(positive_gradcam(np.ones((2, 2, 3)), np.zeros((2, 2, 3))))
        self.assertIsNone(positive_gradcam(np.ones((2, 2, 3)), -np.ones((2, 2, 3))))

    def test_invalid_gradient_is_reported(self):
        with self.assertRaises(ValueError):
            positive_gradcam(np.ones((2, 2, 3)), np.full((2, 2, 3), np.nan))

    def test_overlay_keeps_unhighlighted_pixels_and_excludes_padding(self):
        frame = np.full((9, 9, 3), 100, np.uint8)
        frame[:, :3] = 0
        heat = np.zeros((9, 9), np.float32)
        heat[:3, :3] = 1  # Padding must not be described as a picture region.
        heat[6:, 6:] = .5
        overlay, _ = overlay_heatmap(frame, heat)
        np.testing.assert_array_equal(np.array(overlay)[:, :3], frame[:, :3])
        np.testing.assert_array_equal(np.array(overlay)[3:6, 3:6], frame[3:6, 3:6])
        self.assertEqual(strongest_zone(heat, frame), '오른쪽 아래')


if __name__ == '__main__':
    unittest.main()
