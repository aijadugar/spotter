# Custom Data Collection Guide

This guide describes how to collect custom clips that match the exercise-router dataset format used
by Spotter. The current router training pipeline starts from the Riccio dataset layout and converts
each video into the same 30-frame pose-window features used at inference time.

## Supported Labels

Use these router labels:

| Router label     | Folder names accepted by the normalizer                              | Notes                    |
| ---------------- | -------------------------------------------------------------------- | ------------------------ |
| `squat`          | `squat`                                                              | Supported exercise class |
| `push_up`        | `push-up`, `push_up`, `pushup`                                       | Supported exercise class |
| `shoulder_press` | `shoulder press`, `shoulder_press`, `overhead press`                 | Supported exercise class |
| `unknown`        | `unknown`, unsupported exercise folders, setup/idle/stretching clips | Rejection class          |

Unsupported exercises from the Riccio dataset, such as `barbell biceps curl` and `hammer curl`, are
mapped to `unknown`. Use the same convention for custom negative examples.

## Folder Layout

For supported custom exercise clips, mirror the Riccio class-folder structure:

```text
custom_router_dataset/
  push-up/
    push-up_001.mp4
    push-up_002.mp4
  squat/
    squat_001.mp4
  shoulder press/
    shoulder_press_001.mp4
  unknown/
    idle_001.mp4
    setup_motion_001.mp4
    bad_camera_angle_001.mp4
```

For Modal training, upload unknown-only additions to:

```text
/data/raw/custom_unknown/
```

Useful `unknown` clips include:

- standing idle
- walking into frame
- setup motion before the first rep
- stretching or mobility drills
- unsupported exercises
- partial reps
- severe occlusion
- bad camera angle

## Capture Protocol

| Setting     | Recommendation                                                                          |
| ----------- | --------------------------------------------------------------------------------------- |
| Clip length | 5-15 seconds for router clips; 10-30 seconds if also used for rep-count evaluation      |
| Frame rate  | 24-30 FPS preferred; minimum 15 FPS                                                     |
| Resolution  | 720p or better preferred; minimum 480x360                                               |
| Camera      | Static phone or webcam, no panning                                                      |
| Framing     | Full body visible for squats/presses; upper body and hips visible for push-ups          |
| View angle  | Side or 30-45 degree angle for squats/push-ups; front or slight-front angle for presses |
| Lighting    | Even lighting; avoid strong backlight and heavy shadows                                 |
| Reps        | 3-8 clean reps per supported clip when possible                                         |

Keep one athlete and one exercise per clip. Avoid transitions between exercises inside a single clip.

## File Naming

Use deterministic names:

```text
<label>_<subject_or_session>_<index>.mp4
```

Examples:

```text
push-up_s01_001.mp4
squat_s03_004.mp4
unknown_idle_s02_002.mp4
```

Do not include personal names in filenames.

## Metadata Manifest

Keep a sidecar manifest when collecting custom data:

```json
[
  {
    "file": "push-up/push-up_s01_001.mp4",
    "label": "push_up",
    "source_label": "push-up",
    "subject_id": "s01",
    "camera_angle": "side",
    "consent": "internal_demo",
    "notes": "clean reps, full body visible"
  }
]
```

Required fields:

- `file`
- `label`
- `source_label`
- `subject_id`
- `camera_angle`
- `consent`

Optional but useful fields:

- `notes`
- `equipment`
- `lighting`
- `rep_count_estimate`
- `split` (`train`, `validation`, `demo`)

## Consent And Privacy

- Collect only consented clips.
- Do not upload private, medical, or identifying context unless the subject explicitly agreed.
- Prefer synthetic subject IDs over names.
- Strip unrelated audio when possible.
- Do not include faces in demo assets unless the subject agreed to public display.
- Store raw consent records outside the repo.

## Quality Checks

Before adding clips to training:

- Verify the video decodes with OpenCV.
- Confirm the intended label and mapped router label.
- Check that the body is visible for most frames.
- Remove duplicates and near-identical retakes.
- Keep a small held-out set for evaluation and demo validation.

The current training report is in
[exercise-router-training-report.md](exercise-router-training-report.md).
