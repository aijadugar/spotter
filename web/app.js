import React, {
  useEffect,
  useMemo,
  useState,
} from "https://esm.sh/react@18.2.0";
import { createRoot } from "https://esm.sh/react-dom@18.2.0/client";
import { h } from "./common.js";
import {
  ReportPanel,
  ReplayReviewPanel,
  ReviewInsights,
} from "./report.js";

// Main App component
function App() {
  const [result, setResult] = useState(null);
  const [activeTab, setActiveTab] = useState("summary");
  const [videoSrc, setVideoSrc] = useState(null);
  const [selectedRepId, setSelectedRepId] = useState(null);
  const [playbackTime, setPlaybackTime] = useState(0);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState(null);

  async function handleAnalyze(file) {
    setStatus("analyzing");
    setError(null);
    setResult(null);
    setVideoSrc(null);
    setSelectedRepId(null);
    setPlaybackTime(0);

    const formData = new FormData();
    formData.append("video", file);
    formData.append("goal", "beginner_practice");
    formData.append("experience_level", "beginner");
    formData.append("intended_exercise", "auto");
    formData.append("limitations", "[]");
    formData.append("equipment", "bodyweight");
    formData.append("bypass_verifier", "true");
    formData.append("voice_coach", "false");

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        setError(data.detail || "Analysis failed");
        setStatus("error");
        return;
      }
      if (data.annotated_video_url) {
        setVideoSrc(data.annotated_video_url);
      }
      setResult(data);
      setStatus("complete");
    } catch (err) {
      setError(err.message);
      setStatus("error");
    }
  }

  function handleFileSelect(event) {
    const file = event.target.files?.[0];
    if (file) {
      handleAnalyze(file);
    }
  }

  return h(
    "div",
    { className: "app" },
    h("header", { className: "app-header" },
      h("h1", null, "Spotter"),
      h("p", null, "Video-based workout form review")
    ),
    h("main", { className: "app-main" },
      h("section", { className: "upload-section" },
        h("input", {
          type: "file",
          accept: "video/*",
          onChange: handleFileSelect,
          disabled: status === "analyzing",
        }),
        status === "analyzing" && h("p", { className: "status" }, "Analyzing..."),
        error && h("p", { className: "error" }, error)
      ),
      result && h(
        "div",
        { className: "results" },
        h(ReviewInsights, { result }),
        h(ReportPanel, { result, activeTab, onTabChange: setActiveTab }),
        videoSrc && h(ReplayReviewPanel, { result, videoSrc })
      )
    )
  );
}

// Test component to verify React works
function TestComponent() {
  return h("div", { style: { padding: "20px", color: "#d4ff5b", background: "rgba(212,255,91,0.1)", border: "1px solid #d4ff5b", borderRadius: "8px", margin: "20px" } },
    h("h3", null, "✅ React is working!"),
    h("p", null, "If you see this, React mounted successfully."),
    h("p", null, "Now rendering main app...")
  );
}

console.log("app.js: Module loaded, creating root...");

// Clear root before mounting
const rootEl = document.getElementById("root");
if (rootEl) {
  rootEl.innerHTML = "";
  console.log("Root cleared");
}

try {
  console.log("Creating React root...");
  const root = createRoot(rootEl);
  console.log("Root created, rendering TestComponent...");

  // First render test component to verify React works
  root.render(h(TestComponent));
  console.log("TestComponent rendered successfully!");

  // Now render the real app after a short delay
  setTimeout(() => {
    console.log("Rendering main App...");
    root.render(h(App));
    console.log("App rendered!");
  }, 100);

} catch (err) {
  console.error("Failed to mount React:", err);
  if (rootEl) {
    rootEl.innerHTML = `
      <div style="padding: 20px; color: #ff8b74; background: rgba(255,139,116,0.1); border: 1px solid rgba(255,139,116,0.3); border-radius: 8px; margin: 20px;">
        <h3>React Mount Failed</h3>
        <pre style="white-space: pre-wrap;">${err.stack || err.toString()}</pre>
        <p>Check browser console for full stack trace.</p>
      </div>
    `;
  }
}

// Error boundary for debugging
window.addEventListener("error", (event) => {
  console.error("Global error:", event.error || event.message);
  document.getElementById("root").innerHTML = `
    <div style="padding: 20px; color: #ff8b74; background: rgba(255,139,116,0.1); border: 1px solid rgba(255,139,116,0.3); border-radius: 8px; margin: 20px;">
      <h3>JavaScript Error</h3>
      <pre style="white-space: pre-wrap;">${(event.error || event.message || "Unknown error").toString()}</pre>
      <p>Check browser console for full stack trace.</p>
    </div>
  `;
});

window.addEventListener("unhandledrejection", (event) => {
  console.error("Unhandled rejection:", event.reason);
});