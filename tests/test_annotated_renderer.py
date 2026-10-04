from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotter.contracts import (
    IssueMarker,
    IssueMarkers,
    PoseFrame,
    PoseSequence,
    Rep,
    Reps,
    VideoManifest,
)
from spotter.steps import annotated_renderer


def _frame(frame_index: int) -> PoseFrame:
    landmarks = {
        "left_shoulder": {"x": 0.35, "y": 0.3},
        "right_shoulder": {"x": 0.65, "y": 0.3},
        "left_elbow": {"x": 0.3, "y": 0.45},
        "right_elbow": {"x": 0.7, "y": 0.45},
        "left_wrist": {"x": 0.28, "y": 0.6},
        "right_wrist": {"x": 0.72, "y": 0.6},
        "left_hip": {"x": 0.42, "y": 0.55},
        "right_hip": {"x": 0.58, "y": 0.55},
        "left_knee": {"x": 0.42, "y": 0.75},
        "right_knee": {"x": 0.58, "y": 0.75},
        "left_ankle": {"x": 0.42, "y": 0.92},
        "right_ankle": {"x": 0.58, "y": 0.92},
    }
    return PoseFrame(
        frame_index=frame_index,
        timestamp_sec=round(frame_index / 30.0, 3),
        landmarks=landmarks,
        world_landmarks={},
        pose_quality={"source": "fake_pose", "mean_visibility": 0.95},
    )


class AnnotatedRendererTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_video(self, frame_count: int = 6) -> Path:
        path = Path(self.temp_dir.name) / "input.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
        self.assertTrue(writer.isOpened())
        for frame_index in range(frame_count):
            frame = np.full((240, 320, 3), 90 + frame_index, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return path

    def test_renderer_writes_annotated_video(self) -> None:
        video_path = self._write_video()
        manifest = VideoManifest(
            video_path=str(video_path),
            fps=30.0,
            duration_sec=0.2,
            total_frames=6,
            sampled_frames=6,
            width=320,
            height=240,
            codec="mp4v",
            container="mp4",
            brightness_mean=100.0,
            blur_laplacian_var=100.0,
            quality_warnings=[],
            analysis_allowed=True,
        )
        pose_sequence = PoseSequence(
            frames=[_frame(index) for index in range(6)],
            normalized=True,
            smoothing_method="exponential_smoothing",
            pose_valid_ratio=1.0,
        )
        reps = Reps(
            exercise="push_up",
            reps=[Rep(1, 0, 2, 5, 0.0, 0.067, 0.167)],
            partial_reps=[],
        )

        result = annotated_renderer.run(
            manifest,
            pose_sequence,
            reps,
            IssueMarkers(issues=[]),
            Path(self.temp_dir.name),
        )

        self.assertIsNotNone(result.annotated_video_path)
        self.assertTrue(Path(str(result.annotated_video_path)).exists())
        self.assertEqual(result.issue_thumbnail_paths, [])
        self.assertEqual(result.issue_clip_paths, [])

    def test_renderer_highlights_active_issue_and_writes_thumbnail(self) -> None:
        video_path = self._write_video()
        manifest = VideoManifest(
            video_path=str(video_path),
            fps=30.0,
            duration_sec=0.2,
            total_frames=6,
            sampled_frames=6,
            width=320,
            height=240,
            codec="mp4v",
            container="mp4",
            brightness_mean=100.0,
            blur_laplacian_var=100.0,
            quality_warnings=[],
            analysis_allowed=True,
        )
        pose_sequence = PoseSequence(
            frames=[_frame(index) for index in range(6)],
            normalized=True,
            smoothing_method="exponential_smoothing",
            pose_valid_ratio=1.0,
        )
        reps = Reps(
            exercise="push_up",
            reps=[Rep(1, 0, 2, 5, 0.0, 0.067, 0.167)],
            partial_reps=[],
        )
        issues = IssueMarkers(
            issues=[
                IssueMarker(
                    rep_id=1,
                    issue="hip_sag",
                    severity=0.82,
                    start_frame=2,
                    end_frame=4,
                    start_sec=0.067,
                    end_sec=0.133,
                    affected_joints=["left_hip"],
                    evidence={
                        "body_line_score": 0.42,
                        "threshold": 0.6,
                        "confidence": 0.54,
                        "peak_frame": 3,
                    },
                )
            ]
        )

        result = annotated_renderer.run(
            manifest,
            pose_sequence,
            reps,
            issues,
            Path(self.temp_dir.name),
        )

        self.assertIsNotNone(result.annotated_video_path)
        self.assertEqual(len(result.issue_thumbnail_paths), 1)
        self.assertTrue(Path(result.issue_thumbnail_paths[0]["path"]).exists())
        self.assertEqual(len(result.issue_clip_paths), 1)
        self.assertTrue(Path(result.issue_clip_paths[0]["path"]).exists())
        self.assertEqual(result.issue_clip_paths[0]["clip_start_sec"], 0.0)
        self.assertGreater(result.issue_clip_paths[0]["clip_end_sec"], 1.0)

        capture = cv2.VideoCapture(str(result.annotated_video_path))
        frames = []
        for _ in range(4):
            ok, frame = capture.read()
            self.assertTrue(ok)
            frames.append(frame)
        capture.release()

        left_hip_x, left_hip_y = 134, 132
        active_roi = frames[3][left_hip_y - 8 : left_hip_y + 9, left_hip_x - 8 : left_hip_x + 9]
        inactive_roi = frames[0][left_hip_y - 8 : left_hip_y + 9, left_hip_x - 8 : left_hip_x + 9]
        active_red_pixels = np.count_nonzero(
            (active_roi[:, :, 2] > 180) & (active_roi[:, :, 1] < 180) & (active_roi[:, :, 0] < 120)
        )
        inactive_red_pixels = np.count_nonzero(
            (inactive_roi[:, :, 2] > 180)
            & (inactive_roi[:, :, 1] < 180)
            & (inactive_roi[:, :, 0] < 120)
        )
        self.assertGreater(active_red_pixels, inactive_red_pixels * 4)

    def test_angle_label_uses_degree_evidence(self) -> None:
        issue = IssueMarker(
            rep_id=1,
            issue="incomplete_depth",
            severity=0.7,
            start_frame=2,
            end_frame=4,
            start_sec=0.067,
            end_sec=0.133,
            affected_joints=["left_elbow", "right_elbow"],
            evidence={
                "elbow_angle_deg": 123.4,
                "threshold": 115.0,
                "confidence": 0.74,
            },
        )

        self.assertEqual(
            annotated_renderer._issue_angle_label(issue),
            "elbow angle 123 deg",
        )

    def test_hdr_metadata_requires_sdr_conversion(self) -> None:
        self.assertTrue(
            annotated_renderer._needs_sdr_conversion(
                {
                    "color_space": "bt2020nc",
                    "color_transfer": "arib-std-b67",
                    "color_primaries": "bt2020",
                }
            )
        )

    def test_bt709_encode_uses_first_optional_audio_stream(self) -> None:
        command = annotated_renderer._encode_bt709_command(
            "ffmpeg",
            Path("raw.mp4"),
            Path("output.mp4"),
            Path("source.mov"),
        )

        self.assertIn("1:a:0?", command)
        self.assertNotIn("1:a?", command)
        self.assertNotIn("-an", command)
        self.assertFalse(
            annotated_renderer._needs_sdr_conversion(
                {
                    "color_space": "bt709",
                    "color_transfer": "bt709",
                    "color_primaries": "bt709",
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
