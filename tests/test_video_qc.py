from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotter.steps import video_qc


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class VideoQCTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_video(
        self,
        filename: str,
        *,
        fps: float = 30.0,
        duration_sec: float = 10.0,
        size: tuple[int, int] = (640, 480),
        mode: str = "valid",
    ) -> Path:
        path = Path(self.temp_dir.name) / filename
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            size,
        )
        self.assertTrue(writer.isOpened())

        width, height = size
        for frame_index in range(max(1, int(fps * duration_sec))):
            if mode == "dark":
                frame = np.full((height, width, 3), 8, dtype=np.uint8)
            elif mode == "blurry":
                frame = np.full((height, width, 3), 150, dtype=np.uint8)
            else:
                frame = np.full((height, width, 3), 135, dtype=np.uint8)
                offset = frame_index % max(1, width // 4)
                cv2.rectangle(
                    frame, (40 + offset, 80), (220 + offset, 300), (245, 245, 245), -1
                )
                cv2.line(
                    frame,
                    (0, frame_index % height),
                    (width - 1, height - 1),
                    (10, 10, 10),
                    3,
                )
                cv2.putText(
                    frame,
                    str(frame_index),
                    (30, 420),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    (255, 255, 255),
                    4,
                )
            writer.write(frame)

        writer.release()
        return path

    def test_valid_video_produces_real_metadata(self) -> None:
        path = self._write_video("valid.mp4")

        manifest = video_qc.run(str(path))

        self.assertTrue(manifest.analysis_allowed)
        self.assertEqual(manifest.quality_warnings, [])
        self.assertEqual(manifest.width, 640)
        self.assertEqual(manifest.height, 480)
        self.assertEqual(manifest.total_frames, 300)
        self.assertAlmostEqual(manifest.fps, 30.0, places=1)
        self.assertAlmostEqual(manifest.duration_sec, 10.0, places=1)
        self.assertGreater(manifest.sampled_frames, 0)
        self.assertEqual(manifest.container, "mp4")
        self.assertIsNotNone(manifest.codec)
        self.assertIsNotNone(manifest.brightness_mean)
        self.assertIsNotNone(manifest.blur_laplacian_var)

    def test_invalid_video_sets_decode_failure_and_blocks_analysis(self) -> None:
        manifest = video_qc.run(str(Path(self.temp_dir.name) / "missing.mp4"))

        self.assertFalse(manifest.analysis_allowed)
        self.assertEqual(manifest.quality_warnings, ["video_decode_failed"])
        self.assertEqual(manifest.total_frames, 0)
        self.assertEqual(manifest.sampled_frames, 0)

    def test_short_low_resolution_low_fps_video_reports_warnings(self) -> None:
        path = self._write_video(
            "short_low.mp4", fps=10.0, duration_sec=2.0, size=(320, 240)
        )

        manifest = video_qc.run(str(path))

        self.assertFalse(manifest.analysis_allowed)
        self.assertIn("too_short", manifest.quality_warnings)
        self.assertIn("fps_too_low", manifest.quality_warnings)
        self.assertIn("resolution_too_low", manifest.quality_warnings)

    def test_long_video_reports_warning(self) -> None:
        original_min_duration = video_qc.MIN_DURATION_SEC
        original_max_duration = video_qc.MAX_DURATION_SEC
        video_qc.MIN_DURATION_SEC = 0.0
        video_qc.MAX_DURATION_SEC = 1.0
        try:
            path = self._write_video("long.mp4", duration_sec=2.0)

            manifest = video_qc.run(str(path))
        finally:
            video_qc.MIN_DURATION_SEC = original_min_duration
            video_qc.MAX_DURATION_SEC = original_max_duration

        self.assertFalse(manifest.analysis_allowed)
        self.assertIn("too_long", manifest.quality_warnings)

    def test_dark_video_reports_warning(self) -> None:
        path = self._write_video("dark.mp4", mode="dark")

        manifest = video_qc.run(str(path))

        self.assertTrue(manifest.analysis_allowed)
        self.assertIn("too_dark", manifest.quality_warnings)

    def test_blurry_video_reports_warning(self) -> None:
        path = self._write_video("blurry.mp4", mode="blurry")

        manifest = video_qc.run(str(path))

        self.assertTrue(manifest.analysis_allowed)
        self.assertIn("too_blurry", manifest.quality_warnings)

    def test_sample_frame_indices_are_bounded_and_ordered(self) -> None:
        indices = video_qc.sample_frame_indices(total_frames=100, sample_count=5)

        self.assertEqual(indices, sorted(indices))
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 99)
        self.assertEqual(len(indices), 5)


if __name__ == "__main__":
    unittest.main()
