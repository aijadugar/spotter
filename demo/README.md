# Router Demo Videos

These clips are small examples pulled from the Riccio exercise-recognition dataset cache used by the
Modal training pipeline.

| File                                                   | Source label          | Router label | Duration |    FPS | Resolution | Source path                                                                                                                                |
| ------------------------------------------------------ | --------------------- | ------------ | -------: | -----: | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `videos/router_demo_push_up_1.mp4`                     | `push-up`             | `push_up`    |   5.005s | 29.970 | 1920x1080  | `/data/raw/riccio/Real-Time Exercise Recognition Dataset/final_kaggle_with_additional_video/push-up/push-up_1.mp4`                         |
| `videos/router_demo_unknown_barbell_biceps_curl_1.mp4` | `barbell biceps curl` | `unknown`    |   3.804s | 29.970 | 1920x1080  | `/data/raw/riccio/Real-Time Exercise Recognition Dataset/final_kaggle_with_additional_video/barbell biceps curl/barbell biceps curl_1.mp4` |

The second clip is an unsupported exercise class in the current router contract. It is intentionally
mapped to `unknown` so the demo includes both a supported exercise and a rejection-class example.

## Usage

Run either video through the app or API with `intended_exercise=auto` to inspect router behavior.
The expected labels are `push_up` for the push-up clip and `unknown` for the unsupported curl clip.
