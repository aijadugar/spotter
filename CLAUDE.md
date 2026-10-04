# Rules for working on Spotter

- Never edit, run, or try to fix tests/test_speech_elevenlabs.py or tests/test_app_zero_gpu.py.
  They have known failures and are parked until the final cleanup. Do not mention them.
- Never run the full test suite. Only run the test file you just created for new code.
- Never ask me for API keys and never block on them. Build all code so it works without keys
  (mocks, dry-run, graceful fallback). I will add keys at the end for live checks.
- No long reports. Final reply max 15 lines. Do not re-read files you don't need.
- Small commits on branch feat/demo-and-voice. Never merge or push to main.
- No new heavy dependencies.