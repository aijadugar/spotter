import React, {
  useEffect,
  useMemo,
  useState,
} from "https://esm.sh/react@18.2.0";
import { createRoot } from "https://esm.sh/react-dom@18.2.0/client";
import { defaults, h, label } from "./common.js?v=20260614-modular-app";
import {
  ReportPanel,
  ReplayReviewPanel,
  ReviewInsights,
} from "./report.js?v=20260614-modular-app";

function Field({ labelText, children, full = false }) {
  return h(
    "label",
    { className: `field${full ? " full" : ""}` },
    h("span", { className: "label" }, labelText),
    children,
  );
}

function SelectField({ labelText, value, onChange, options, name }) {
  return h(
    Field,
    { labelText },
    h(
      "select",
      { name, value, onChange: (event) => onChange(event.target.value) },
      options.map((option) =>
        h("option", { key: option, value: option }, label(option)),
      ),
    ),
  );
}

function KineticFigure({ compact = false }) {
  const imageUrl = compact
    ? "https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?auto=format&fit=crop&fm=jpg&w=900&q=84"
    : "https://images.unsplash.com/photo-1517836357463-d25dfeac3438?auto=format&fit=crop&fm=jpg&w=1100&q=84";
  return h(
    "div",
    { className: `kinetic-photo${compact ? " compact" : ""}`, "aria-hidden": "true" },
    h("img", {
      alt: "",
      className: "motion-photo",
      draggable: false,
      src: imageUrl,
    }),
    h("span", { className: "photo-vignette" }),
    h("span", { className: "photo-scanline" }),
  );
}

function SignalStack() {
  return h(
    "div",
    { className: "signal-stack", "aria-label": "Analysis pipeline" },
    ["Video QC", "Pose Map", "Rep Count", "Coach Notes"].map((item, index) =>
      h(
        "div",
        { className: "signal-row", key: item },
        h("span", null, `0${index + 1}`),
        h("strong", null, item),
      ),
    ),
  );
}

const runningProgressSteps = [
  {
    id: "quality",
    text: "First up, I am checking if the video is clear enough to coach from.",
    delayMs: 0,
  },
  {
    id: "pose",
    text: "Now I am mapping your posture and tracking the key body landmarks.",
    delayMs: 900,
  },
  {
    id: "exercise",
    text: "Let me figure out which exercise you are doing.",
    delayMs: 1800,
  },
  {
    id: "reps",
    text: "Counting your reps now. One clean rep at a time.",
    delayMs: 2800,
  },
  {
    id: "issues",
    text: "Almost there. I am checking the moments that may need a small fix.",
    delayMs: 3900,
  },
  {
    id: "render",
    text: "I am preparing your annotated video and issue clips.",
    delayMs: 4800,
  },
  {
    id: "coach",
    text: "I am turning the scan into coaching notes you can use right away.",
    delayMs: 5600,
  },
];

function pendingProgressState() {
  return runningProgressSteps.map((step, index) => ({
    id: step.id,
    text: step.text,
    status: index === 0 ? "active" : "pending",
  }));
}

function finalProgressState(result) {
  const report = result.report;
  const warnings = report.video_manifest?.quality_warnings || [];
  const exercise = label(report.exercise?.exercise || "movement");
  const repCount = report.reps?.reps?.length || 0;
  const issues = report.issue_markers?.issues || [];
  return [
    {
      id: "quality",
      status: "done",
      text: warnings.length
        ? `Quick note: the video has a few things to watch, like ${warnings.map(label).join(", ")}.`
        : "Nice, your video quality looks solid.",
    },
    {
      id: "pose",
      status: "done",
      text: "Posture tracking is done. I found the key landmarks I need.",
    },
    {
      id: "exercise",
      status: "done",
      text: `Looks like you are doing ${exercise}.`,
    },
    {
      id: "reps",
      status: "done",
      text: `I counted ${repCount} ${exercise} reps in this set.`,
    },
    {
      id: "issues",
      status: "done",
      text: issues.length
        ? `I found ${issues.length} coaching point${issues.length === 1 ? "" : "s"} worth reviewing.`
        : "Good news, I did not spot any clear form issues in this set.",
    },
    {
      id: "render",
      status: "done",
      text: result.annotated_video_url
        ? "Your annotated video is ready."
        : "I could not render an annotated video, but the report is ready.",
    },
    {
      id: "coach",
      status: "done",
      text: "Coach notes are ready.",
    },
  ];
}

function applyProgressEvent(currentSteps, event) {
  const baseSteps = currentSteps.length ? currentSteps : pendingProgressState();
  const knownStepIds = new Set(runningProgressSteps.map((step) => step.id));
  if (!knownStepIds.has(event.step)) return baseSteps;
  return baseSteps.map((step) =>
    step.id === event.step
      ? {
          ...step,
          status: event.status || "active",
          text: event.text || step.text,
        }
      : step,
  );
}

async function readAnalysisStream(response, onEvent) {
  if (!response.body)
    throw new Error("Streaming is not available in this browser.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line);
      if (event.type === "progress") onEvent(event);
      if (event.type === "complete") result = event.result;
      if (event.type === "error")
        throw new Error(event.detail || "Analysis failed.");
    }
  }

  if (buffer.trim()) {
    const event = JSON.parse(buffer);
    if (event.type === "progress") onEvent(event);
    if (event.type === "complete") result = event.result;
    if (event.type === "error")
      throw new Error(event.detail || "Analysis failed.");
  }

  if (!result) throw new Error("Analysis finished without a report.");
  return result;
}

function ProgressPanel({ steps }) {
  if (!steps.length) return null;
  const isComplete = steps.every((step) => step.status === "done");
  return h(
    "section",
    { className: "progress-panel", "aria-live": "polite" },
    h(
      "h3",
      null,
      isComplete ? "Scan results are ready" : "Your scan is moving",
    ),
    h(
      "ol",
      { className: "progress-list" },
      steps.map((step) =>
        h(
          "li",
          { className: `progress-step ${step.status}`, key: step.id },
          h("span", { className: "progress-dot", "aria-hidden": "true" }),
          h("span", null, step.text),
        ),
      ),
    ),
  );
}

function StageEmpty() {
  return h(
    "div",
    { className: "result-empty" },
    h(KineticFigure, { compact: true }),
    h("strong", null, "Your annotated replay will land here"),
    h(
      "span",
      null,
      "Upload a set and Spotter will paint the movement path, rep timing, and coaching moments on top of the video.",
    ),
  );
}

function App() {
  const [config, setConfig] = useState(defaults);
  const [file, setFile] = useState(null);
  const [goal, setGoal] = useState("beginner_practice");
  const [experience, setExperience] = useState("beginner");
  const [exercise, setExercise] = useState("auto");
  const [equipment, setEquipment] = useState("bodyweight");
  const [limitations, setLimitations] = useState([]);
  const [result, setResult] = useState(null);
  const [activeTab, setActiveTab] = useState("summary");
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [progressSteps, setProgressSteps] = useState([]);

  const previewUrl = useMemo(
    () => (file ? URL.createObjectURL(file) : ""),
    [file],
  );

  useEffect(() => {
    fetch("/api/config")
      .then((response) => response.json())
      .then(setConfig)
      .catch(() => setConfig(defaults));
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  function toggleLimitation(value) {
    setLimitations((current) =>
      current.includes(value)
        ? current.filter((item) => item !== value)
        : [...current, value],
    );
  }

  async function analyze(event) {
    event.preventDefault();
    setError("");
    setStatus("running");
    setResult(null);
    setProgressSteps(pendingProgressState());

    const payload = new FormData();
    if (file) payload.append("video", file);
    payload.append("goal", goal);
    payload.append("experience_level", experience);
    payload.append("intended_exercise", exercise);
    payload.append("intended_variation", "");
    payload.append("limitations", JSON.stringify(limitations));
    payload.append("equipment", equipment);
    payload.append("bypass_verifier", "true");

    try {
      const response = await fetch("/api/analyze/stream", {
        method: "POST",
        body: payload,
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || "Analysis failed.");
      }
      const body = await readAnalysisStream(response, (progressEvent) => {
        setProgressSteps((currentSteps) =>
          applyProgressEvent(currentSteps, progressEvent),
        );
      });
      setResult(body);
      setProgressSteps(finalProgressState(body));
      setStatus("complete");
    } catch (caught) {
      setError(caught.message || "Analysis failed.");
      setProgressSteps([
        {
          id: "error",
          status: "active",
          text: "The scan did not finish. Try another video or check the connection, and we can run it again.",
        },
      ]);
      setStatus("idle");
    }
  }

  return h(
    "main",
    { className: "app" },
    h(
      "header",
      { className: "hero" },
      h(
        "div",
        { className: "hero-content" },
        h(
          "p",
          { className: "eyebrow" },
          "Pose intelligence for coached training",
        ),
        h("h1", null, "Spotter"),
        h("p", null, config.description),
        h(
          "div",
          { className: "hero-actions", "aria-label": "Demo strengths" },
          h("span", null, "Realtime stream"),
          h("span", null, "Annotated replay"),
          h("span", null, "Grounded coach notes"),
        ),
      ),
      h(
        "aside",
        { className: "hero-lab", "aria-label": "Motion analysis preview" },
        h(KineticFigure, null),
        h(SignalStack, null),
        h(
          "div",
          { className: "hero-metrics", "aria-label": "Pipeline highlights" },
          h(
            "div",
            { className: "metric" },
            h("strong", null, "17"),
            h("span", null, "pose landmarks"),
          ),
          h(
            "div",
            { className: "metric" },
            h("strong", null, "60s"),
            h("span", null, "clip ceiling"),
          ),
          h(
            "div",
            { className: "metric" },
            h("strong", null, "JSON"),
            h("span", null, "audit trail"),
          ),
        ),
      ),
    ),
    h(
      "form",
      { className: "workspace", onSubmit: analyze },
      h(
        "section",
        { className: "panel" },
        h(
          "div",
          { className: "panel-head" },
          h(
            "div",
            null,
            h("h2", null, "Session setup"),
            h("p", null, "Movement context"),
          ),
          h(
            "span",
            { className: "status-pill" },
            status === "running" ? "Analyzing" : result ? "Complete" : "Ready",
          ),
        ),
        h(
          "label",
          { className: "dropzone" },
          h("input", {
            name: "video",
            type: "file",
            accept: "video/*",
            onChange: (event) => setFile(event.target.files?.[0] || null),
          }),
          previewUrl
            ? h("video", {
                className: "dropzone-preview",
                src: previewUrl,
                controls: true,
              })
            : h(
                "span",
                { className: "dropzone-empty" },
                h("span", { className: "upload-icon" }, "↑"),
                h("strong", null, "Drop a workout clip"),
                h("span", null, "or click to upload an MP4, MOV, or WebM file"),
              ),
        ),
        h(
          "div",
          { className: "form-grid" },
          h(SelectField, {
            labelText: "Goal",
            name: "goal",
            value: goal,
            onChange: setGoal,
            options: config.goals,
          }),
          h(SelectField, {
            labelText: "Experience",
            name: "experience_level",
            value: experience,
            onChange: setExperience,
            options: config.experience_levels,
          }),
          h(SelectField, {
            labelText: "Exercise",
            name: "intended_exercise",
            value: exercise,
            onChange: setExercise,
            options: config.exercises,
          }),
          h(SelectField, {
            labelText: "Equipment",
            name: "equipment",
            value: equipment,
            onChange: setEquipment,
            options: config.equipment,
          }),
          h(
            "div",
            { className: "field full" },
            h("span", { className: "label" }, "Known limitations"),
            h(
              "div",
              { className: "check-grid" },
              config.limitations.map((item) =>
                h(
                  "label",
                  { className: "check-chip", key: item },
                  h("input", {
                    name: "limitations",
                    type: "checkbox",
                    checked: limitations.includes(item),
                    onChange: () => toggleLimitation(item),
                  }),
                  h("span", null, label(item)),
                ),
              ),
            ),
          ),
        ),
        h(
          "button",
          { className: "primary", disabled: status === "running", type: "submit" },
          status === "running" ? "Analyzing…" : "Analyze Form",
        ),
        error ? h("div", { className: "error" }, error) : null,
      ),
      h(
        "section",
        { className: "panel" },
        h(
          "div",
          { className: "panel-head" },
          h(
            "div",
            null,
            h("h2", null, "Review output"),
            h("p", null, "Annotated movement"),
          ),
          result
            ? h("span", { className: "status-pill" }, result.run_id)
            : null,
        ),
        result
          ? h(ReplayReviewPanel, {
              result,
              videoSrc: result?.annotated_video_url || previewUrl,
              className: "review-output-replay",
            })
          : h(
              React.Fragment,
              null,
              h(
                "div",
                { className: "result-stage" },
                h(StageEmpty, null),
              ),
              h(ProgressPanel, { steps: progressSteps }),
            ),
        h(ReviewInsights, { result }),
      ),
    ),
    h(ReportPanel, {
      result,
      activeTab,
      onTabChange: setActiveTab,
    }),
  );
}

createRoot(document.getElementById("root")).render(h(App));
