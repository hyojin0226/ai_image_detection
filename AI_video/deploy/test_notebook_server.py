"""API boundary checks with a fake runner; no neural inference or downloads."""
import http.client
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from notebook_server import Jobs, make_server, frame_count_for_duration, select_duration_segments


class DurationSamplingTests(unittest.TestCase):
    def test_boundaries_and_invalid_durations(self):
        for duration, expected in [(3, 16), (10, 16), (10.001, 32), (20, 32), (20.001, 48), (30, 48), (30.001, 64), (40, 64)]:
            with self.subTest(duration=duration):
                self.assertEqual(frame_count_for_duration(duration), expected)
        for invalid in [0, 2.99, 40.001, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                frame_count_for_duration(invalid)

    def test_entire_timeline_and_eight_unique_frames_per_group(self):
        for duration in [8, 18, 28, 38]:
            count = frame_count_for_duration(duration)
            times = [(i + .5) * duration / count for i in range(count)]
            groups, skipped, _ = select_duration_segments([1] * count, times, duration, lambda t: 1, lambda frame: frame == 0)
            self.assertEqual(len(groups) * 8, count)
            self.assertFalse(skipped)
            self.assertEqual(groups[0]['start_sec'], 0)
            self.assertEqual(groups[-1]['end_sec'], duration)
            chosen = [t for group in groups for t in group['times']]
            self.assertEqual(len(set(chosen)), count)
            self.assertTrue(all(0 < t < duration for t in chosen))

    def test_black_only_exclusion_refills_or_explicitly_skips(self):
        times = [(i + .5) / 2 for i in range(16)]
        frames = [0] + [1] * 15
        groups, skipped, black = select_duration_segments(frames, times, 8, lambda t: 1, lambda frame: frame == 0)
        self.assertEqual(len(groups), 2)
        self.assertFalse(skipped)
        self.assertEqual(len(black), 1)
        self.assertEqual(len({t for g in groups for t in g['times']}), 16)
        groups, skipped, _ = select_duration_segments([0] * 16, times, 8, lambda t: 0, lambda frame: frame == 0)
        self.assertFalse(groups)
        self.assertEqual(len(skipped), 2)


class APITests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'tmp')
        def fake(record, source):
            with self.jobs.lock:
                record.update(state='completed', message='done', result={'frames': [], 'fake_score': .42}, assets={})
        self.jobs = Jobs(self.folder.name, runner=fake)
        self.server = make_server(0, ['https://demo.vercel.app'], self.jobs)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.jobs.executor.shutdown(wait=True)
        self.folder.cleanup()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        raw = response.read()
        value = json.loads(raw) if raw else None
        status, origin = response.status, response.getheader('Access-Control-Allow-Origin')
        connection.close()
        return status, value, origin

    def test_file_request_to_result_and_unknown_jobs_are_not_listed(self):
        body = b'\x00\x00\x00\x18ftypisom0000'
        status, result, _ = self.request('POST', '/jobs/file', body, {'Content-Type': 'application/octet-stream'})
        self.assertEqual(status, 202)
        for _ in range(20):
            status, state, _ = self.request('GET', '/jobs/' + result['id'])
            if state['state'] == 'completed':
                break
            time.sleep(.01)
        self.assertEqual(state['result']['fake_score'], .42)
        self.assertNotIn('folder', state)
        self.assertEqual(self.request('GET', '/jobs/unknown')[0], 404)

    def test_youtube_accepts_only_public_youtube_https_hosts(self):
        for url in ['https://127.0.0.1/video', 'http://www.youtube.com/shorts/a', 'https://youtube.com@evil.example/x']:
            status, _, _ = self.request('POST', '/jobs/youtube', json.dumps({'url': url}), {'Content-Type': 'application/json'})
            self.assertEqual(status, 400)
        status, _, _ = self.request('POST', '/jobs/youtube', json.dumps({'url': 'https://www.youtube.com/shorts/abcdefghijk'}), {'Content-Type': 'application/json'})
        self.assertEqual(status, 202)

    def test_cross_origin_boundary_and_upload_limit(self):
        self.assertEqual(self.request('GET', '/health', headers={'Origin': 'https://evil.example'})[0], 403)
        status, _, origin = self.request('OPTIONS', '/jobs/file', headers={'Origin': 'https://demo.vercel.app'})
        self.assertEqual((status, origin), (204, 'https://demo.vercel.app'))
        status, _, _ = self.request('POST', '/jobs/file', b'', {'Content-Length': str(91 * 1024 * 1024)})
        self.assertEqual(status, 413)

    def test_health_advertises_duration_policy(self):
        status, value, _ = self.request('GET', '/health')
        self.assertEqual((status, value['frame_policy']), (200, 'duration_v1'))

    def test_playlist_and_filesystem_paths_cannot_be_served(self):
        status, _, _ = self.request('POST', '/jobs/file', b'#EXTM3U\nfile:///secret')
        self.assertEqual(status, 400)
        self.assertEqual(self.request('GET', '/../.env')[0], 404)
        self.assertEqual(self.request('GET', '/jobs')[0], 404)


if __name__ == '__main__':
    unittest.main()
