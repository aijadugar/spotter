from __future__ import annotations


class MMPoseBackend:
    source = "mmpose"

    def __init__(self) -> None:
        raise NotImplementedError(
            "The mmpose backend is reserved behind the PoseBackend interface. "
            "Install MMPose/MMCV and implement MMPoseBackend.detect() to map model keypoints "
            "into the shared landmark dictionary."
        )
