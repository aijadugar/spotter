# Spotter Docs Index

This folder is ordered by purpose so the current workflow is easier to follow.

## Current Workflow

Read in this order if you want to understand and run the current project:

1. [../README.md](../README.md)
2. [02-technical-setup.md](02-technical-setup.md)
3. [10-overview-build-small-hackathon-report.md](10-overview-build-small-hackathon-report.md)
4. [20-router-training-report.md](20-router-training-report.md)
5. [21-router-huggingface-release.md](21-router-huggingface-release.md)
6. [30-coach-modal-training.md](30-coach-modal-training.md)
7. [31-coach-training-report.md](31-coach-training-report.md)
8. [40-data-custom-collection-guide.md](40-data-custom-collection-guide.md)

## Presentation / Demo

- [11-overview-demo-video-transcript.md](11-overview-demo-video-transcript.md)

## Companion

- [60-progress-plan-training-report.md](60-progress-plan-training-report.md)
- [superpowers/specs/2026-10-03-spotter-companion-design.md](superpowers/specs/2026-10-03-spotter-companion-design.md)
- [superpowers/plans/2026-10-03-spotter-companion.md](superpowers/plans/2026-10-03-spotter-companion.md)

## Team Process

- [50-codex-development-workflow.md](50-codex-development-workflow.md)

## Notes

- Current coach-summary runtime default: `build-small-hackathon/pozify-coach-summary1`
- The Hugging Face provider tries `chat_completion` first, then `text_generation` for non-chat
  model repos; local merged-model inference remains the most predictable fine-tuned path
- Current app entrypoint and runtime options are documented in [02-technical-setup.md](02-technical-setup.md)
