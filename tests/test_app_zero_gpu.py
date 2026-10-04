from __future__ import annotations

import importlib
import inspect
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SRC_DIR))


def _decorator_factory(_path: str | None = None, **_kwargs: object):
    def decorator(function):
        return function

    return decorator


class _ServerStub:
    def __init__(self, **_kwargs: object) -> None:
        pass

    def get(self, *_args: object, **_kwargs: object):
        return _decorator_factory()

    def post(self, *_args: object, **_kwargs: object):
        return _decorator_factory()

    def mount(self, *_args: object, **_kwargs: object) -> None:
        pass

    def launch(self, **_kwargs: object) -> None:
        pass


class _ObjectStub:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass


def _default_marker(*_args: object, default: object = None, **_kwargs: object) -> object:
    return default


def _app_import_stubs() -> dict[str, types.ModuleType]:
    gradio = types.ModuleType("gradio")
    gradio.Server = _ServerStub

    fastapi = types.ModuleType("fastapi")
    fastapi.File = _default_marker
    fastapi.Form = _default_marker
    fastapi.UploadFile = _ObjectStub
    fastapi.HTTPException = type(
        "HTTPException",
        (Exception,),
        {"__init__": _ObjectStub.__init__},
    )

    responses = types.ModuleType("fastapi.responses")
    responses.FileResponse = _ObjectStub
    responses.HTMLResponse = _ObjectStub
    responses.StreamingResponse = _ObjectStub
    responses.JSONResponse = _ObjectStub

    staticfiles = types.ModuleType("fastapi.staticfiles")
    staticfiles.StaticFiles = _ObjectStub

    exercise_catalog = types.ModuleType("spotter.exercise_catalog")
    exercise_catalog.USER_SELECTABLE_EXERCISES = ["squat"]
    exercise_catalog.EXERCISES = ["squat", "push_up", "shoulder_press"]
    exercise_catalog.INTENDED_EXERCISES = ["squat", "push_up", "shoulder_press"]

    pipeline = types.ModuleType("spotter.pipeline")

    def run_pipeline(**_kwargs: object) -> dict[str, object]:
        return {"source": "pipeline"}

    pipeline.run_pipeline = run_pipeline

    # Add stubs for modules used in app.py
    hf_spaces = types.ModuleType("spotter.hf_spaces")
    hf_spaces.zero_gpu_enabled = lambda: False
    hf_spaces.default_spaces_gpu_duration = lambda: 0
    hf_spaces.spaces_gpu = _decorator_factory
    hf_spaces.router_torch_device = lambda: "cpu"

    session_memory = types.ModuleType("spotter.steps.session_memory")
    session_memory.get_session_memory = lambda: None
    session_memory.DEFAULT_DB_PATH = None

    return {
        "gradio": gradio,
        "fastapi": fastapi,
        "fastapi.responses": responses,
        "fastapi.staticfiles": staticfiles,
        "spotter.exercise_catalog": exercise_catalog,
        "spotter.pipeline": pipeline,
        "spotter.hf_spaces": hf_spaces,
        "spotter.steps.session_memory": session_memory,
    }


def _import_app_module():
    sys.modules.pop("app", None)
    return importlib.import_module("app")


class AppZeroGpuProgressTests(unittest.TestCase):
    def tearDown(self) -> None:
        sys.modules.pop("app", None)

    def test_analysis_pipeline_is_not_wrapped_at_api_layer(self) -> None:
        with patch.dict(sys.modules, _app_import_stubs()):
            app = _import_app_module()

        signature = inspect.signature(app._run_analysis_pipeline)
        self.assertIn("progress", signature.parameters)
        self.assertEqual(app._run_analysis_pipeline.__name__, "_run_analysis_pipeline")

    def test_analysis_pipeline_forwards_progress_callback_inside_api_process(self) -> None:
        with patch.dict(sys.modules, _app_import_stubs()):
            app = _import_app_module()

        progress_events: list[dict[str, object]] = []
        progress_callback = progress_events.append

        def local_pipeline(**kwargs: object) -> dict[str, object]:
            self.assertIs(kwargs["progress"], progress_callback)
            return {"source": "local"}

        with patch.object(app, "run_pipeline", side_effect=local_pipeline):
            result = app._run_analysis_pipeline(
                "video.mp4",
                {"goal": "beginner_practice"},
                False,
                progress_callback,
            )

        self.assertEqual(result, {"source": "local"})


if __name__ == "__main__":
    unittest.main()
