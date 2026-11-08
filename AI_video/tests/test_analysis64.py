"""Functional checks for segmented inference, ranking and video-score gradients."""
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import analyze_64frames as module


class SamplingTests(unittest.TestCase):
    def test_40_seconds_is_eight_groups_of_eight_unique_positions(self):
        frames = np.full((64, 4, 4, 3), 90, np.uint8)
        times = (np.arange(64) + .5) * 40 / 64
        groups, skipped, black = module.select_segments(frames, times, 40, lambda t: None)
        self.assertEqual((len(groups), len(skipped), len(black)), (8, 0, 0))
        self.assertEqual(len(set(np.concatenate([g['times'] for g in groups]))), 64)
        for group in groups:
            self.assertEqual(len(group['frames']), 8)
            self.assertTrue(np.all(group['times'] > group['start_sec']))
            self.assertTrue(np.all(group['times'] < group['end_sec']))

    def test_all_black_is_not_classified_as_real_and_no_duplicates_are_padded(self):
        frames = np.zeros((64, 4, 4, 3), np.uint8)
        groups, skipped, black = module.select_segments(frames, (np.arange(64) + .5) / 64 * 3, 3,
                                                       lambda t: np.zeros((4, 4, 3), np.uint8))
        self.assertEqual(groups, [])
        self.assertEqual(len(skipped), 8)
        self.assertEqual(len(black), 192)

    def test_black_replacement_keeps_real_dark_scenes_and_64_unique_frames(self):
        frames = np.full((64, 4, 4, 3), 8, np.uint8)
        frames[0] = 0
        dark = np.full((4, 4, 3), 4, np.uint8)
        self.assertFalse(module.is_black(dark))
        times = (np.arange(64) + .5) * 40 / 64
        groups, skipped, _ = module.select_segments(frames, times, 40, lambda t: dark)
        self.assertEqual((len(groups), len(skipped)), (8, 0))
        self.assertEqual(len(set(np.concatenate([g['times'] for g in groups]))), 64)
        self.assertTrue(all(not module.is_black(frame) for g in groups for frame in g['frames']))

    def test_main_eight_are_largest_global_score_effect_not_brightness(self):
        records = [{'index': i + 1, 'timestamp_sec': i, 'video_score_delta': (-1) ** i * i / 1000}
                   for i in range(64)]
        main, details = module.rank_frames(records)
        self.assertEqual(main, list(range(64, 56, -1)))
        self.assertEqual(details, list(range(1, 57)))


class FrozenHeadGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        module.verify()
        cls.members = module.load_members()
        cls.head = staticmethod(module.make_differentiable_head(cls.members))

    def test_differentiable_head_matches_numpy_and_finite_difference(self):
        import tensorflow as tf
        rng = np.random.default_rng(12)
        sequence = rng.normal(.5, .1, (8, 257))
        expected = module.score_sequences(sequence[None], self.members)[0]
        tensor = tf.constant(sequence, tf.float64)
        with tf.GradientTape() as tape:
            tape.watch(tensor)
            value = self.head(tensor)
        gradient = tape.gradient(value, tensor).numpy()
        self.assertAlmostEqual(float(value), float(expected), places=10)
        self.assertTrue(np.isfinite(gradient).all())
        index = np.unravel_index(np.abs(gradient).argmax(), gradient.shape)
        self.assertGreater(abs(gradient[index]), 1e-10)
        epsilon = 1e-5
        plus, minus = sequence.copy(), sequence.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical = (module.score_sequences(plus[None], self.members)[0]
                     - module.score_sequences(minus[None], self.members)[0]) / (2 * epsilon)
        np.testing.assert_allclose(gradient[index], numerical, rtol=2e-3, atol=1e-7)

    def test_constant_frames_have_finite_gradients_and_identical_forward_values(self):
        import tensorflow as tf
        sequence = np.full((8, 257), .2)
        tensor = tf.constant(sequence, tf.float64)
        with tf.GradientTape() as tape:
            tape.watch(tensor)
            value = self.head(tensor)
        gradient = tape.gradient(value, tensor).numpy()
        self.assertTrue(np.isfinite(gradient).all())
        self.assertAlmostEqual(float(value), float(module.score_sequences(sequence[None], self.members)[0]), places=10)

    def test_frame_ablation_measures_final_average_not_just_one_segment(self):
        rng = np.random.default_rng(14)
        sequence = rng.normal(.5, .2, (8, 8, 257)).astype(np.float32)
        scores = module.score_sequences(sequence, self.members)
        records = module.replacement_records(sequence, scores, self.members)
        changed = sequence.copy()
        changed[0, 0] = sequence[0].mean(axis=0)
        alternative = float(module.score_sequences(changed, self.members).mean())
        self.assertAlmostEqual(records[0]['replacement_video_score'], alternative, places=12)
        self.assertAlmostEqual(records[0]['video_score_delta'], float(scores.mean()) - alternative, places=12)


if __name__ == '__main__':
    unittest.main()
