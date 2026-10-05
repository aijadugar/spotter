import React, {
  useEffect,
  useMemo,
  useState,
} from "https://esm.sh/react@18.2.0";
import { createRoot } from "https://esm.sh/react-dom@18.2.0/client";
import { defaults, h, label } from "./common.js";
import {
  ReportPanel,
  ReplayReviewPanel,
  ReviewInsights,
} from "./report.js";

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