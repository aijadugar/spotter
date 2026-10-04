import React, {
  useEffect,
  useRef,
  useState,
} from "https://esm.sh/react@18.2.0";
import {
  formatValue,
  h,
  label,
  percent,
} from "./common.js?v=20260614-modular-app";

function ReviewInsights({ result }) {
  if (!result) return null;
  const report = result.report;
  const warnings = report.video_manifest?.quality_warnings || [];
  const exercise = label(report.exercise?.exercise || "movement");
  const repCount = report.reps?.reps?.length || 0;
  const issues = report.issue_markers?.issues || [];
  const confidence = percent(report.exercise?.confidence);
  const issueText = issues.length
    ? `${issues.length} coaching moment${issues.length === 1 ? "" : "s"}`
    : "No clear form issues";

  return h(
    "div",
    { className: "scan-insights", "aria-label": "Scan results" },
    h(
      "article",
      { className: "scan-insight" },
      h("span", null, "Movement"),
      h("strong", null, exercise),
      h("small", null, `confidence ${confidence}`),
    ),
    h(
      "article",
      { className: "scan-insight" },
      h("span", null, "Reps counted"),
      h("strong", null, String(repCount)),
      h("small", null, repCount === 1 ? "clean rep detected" : "reps detected"),
    ),
    h(
      "article",
      { className: "scan-insight" },
      h("span", null, "Coach review"),
      h("strong", null, issueText),
      h(
        "small",
        null,
        issues.length ? "tap Issues for clips" : "nothing major stood out",
      ),
    ),
    h(
      "article",
      { className: "scan-insight" },
      h("span", null, "Video quality"),
      h("strong", null, warnings.length ? "Needs a quick note" : "Looks good"),
      h(
        "small",
        null,
        warnings.length
          ? warnings.map(label).join(", ")
          : "clear enough to coach",
      ),
    ),
  );
}

function metricScore(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.max(0, Math.min(1, value))
    : null;
}

function metricScoreRows(report) {
  const aggregate = report.rep_analysis?.aggregate_metrics || {};
  return [
    ["ROM", aggregate.avg_rom_score],
    ["Stability", aggregate.avg_stability_score],
    ["Symmetry", aggregate.avg_symmetry_score],
    ["Pose coverage", aggregate.pose_valid_ratio],
  ]
    .map(([name, value]) => ({ name, value: metricScore(value) }))
    .filter((item) => item.value !== null);
}

function scoreTone(score) {
  if (score === null) return "neutral";
  if (score >= 0.8) return "strong";
  if (score >= 0.62) return "watch";
  return "risk";
}

function repScore(item) {
  if (!item) return null;
  const values = [
    item.range_of_motion_score,
    item.stability_score,
    item.symmetry_score,
  ]
    .map(metricScore)
    .filter((value) => value !== null);
  if (!values.length) return null;
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function repAnalysisById(report) {
  return new Map(
    (report.rep_analysis?.items || []).map((item) => [item.rep_id, item]),
  );
}

function issueList(report) {
  return report.issue_markers?.issues || [];
}

function issuesForRep(report, repId) {
  return issueList(report).filter((issue) => issue.rep_id === repId);
}

function topIssue(report) {
  return (
    [...issueList(report)].sort((a, b) => b.severity - a.severity)[0] || null
  );
}

function bestRep(report) {
  const analysisById = repAnalysisById(report);
  return (
    (report.reps?.reps || [])
      .map((rep) => {
        const score = repScore(analysisById.get(rep.rep_id));
        return { rep, score };
      })
      .filter((entry) => entry.score !== null)
      .sort((a, b) => b.score - a.score)[0] || null
  );
}

function reportDuration(report) {
  const manifestDuration = report.video_manifest?.duration_sec;
  const repEnd = Math.max(
    0,
    ...(report.reps?.reps || []).map((rep) => rep.end_sec || 0),
    ...issueList(report).map((issue) => issue.end_sec || 0),
  );
  return Math.max(manifestDuration || 0, repEnd, 1);
}

function repAtTime(report, playbackTime) {
  return (
    (report.reps?.reps || []).find(
      (rep) => playbackTime >= rep.start_sec && playbackTime <= rep.end_sec,
    ) || null
  );
}

function qualityVerdict(report) {
  const scores = metricScoreRows(report)
    .filter((row) => row.name !== "Pose coverage")
    .map((row) => row.value);
  const average = scores.length
    ? scores.reduce((total, score) => total + score, 0) / scores.length
    : null;
  const issue = topIssue(report);
  if (average === null) {
    return {
      title: "Report ready",
      detail: "Coach notes are grounded in the scan.",
      score: null,
    };
  }
  const adjusted = Math.max(0, average - (issue?.severity || 0) * 0.08);
  if (adjusted >= 0.82) {
    return {
      title: issue ? "Strong set, one watchpoint" : "Clean training set",
      detail: issue
        ? `${issueTitle(issue)} is the main limiter.`
        : "No clear form issue dominated the set.",
      score: adjusted,
    };
  }
  if (adjusted >= 0.66) {
    return {
      title: "Usable reps, focused fix",
      detail: issue
        ? `${issueTitle(issue)} should be fixed first.`
        : "Quality is workable, but keep reps controlled.",
      score: adjusted,
    };
  }
  return {
    title: "Needs cleaner reps",
    detail: issue
      ? `${issueTitle(issue)} is limiting this set.`
      : "Repeat with slower reps before progressing.",
    score: adjusted,
  };
}

function timeRange(start, end) {
  return `${Number(start || 0).toFixed(2)}s-${Number(end || 0).toFixed(2)}s`;
}

function metaChip(text, tone = "", key = undefined) {
  const props = { className: `meta-chip${tone ? ` ${tone}` : ""}` };
  if (key !== undefined) props.key = key;
  return h("span", props, text);
}

function ScoreBar({ labelText, value, detail }) {
  const score = metricScore(value);
  const width = score === null ? "0%" : `${Math.round(score * 100)}%`;
  return h(
    "div",
    { className: `score-card ${scoreTone(score)}` },
    h(
      "div",
      { className: "score-card-head" },
      h("span", null, labelText),
      h("strong", null, score === null ? "n/a" : percent(score)),
    ),
    h(
      "span",
      { className: "score-track", "aria-hidden": "true" },
      h("span", { className: "score-fill", style: { width } }),
    ),
    detail ? h("small", null, detail) : null,
  );
}

function pointPath(points) {
  return points
    .map(
      ([x, y], index) => `${index ? "L" : "M"} ${x.toFixed(1)} ${y.toFixed(1)}`,
    )
    .join(" ");
}

function pointList(points) {
  return points.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
}

function metricAverage(rows) {
  const values = rows.map((row) => row.value).filter((value) => value !== null);
  if (!values.length) return null;
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function MetricRadar({ rows }) {
  if (!rows.length) return null;
  const size = 220;
  const center = size / 2;
  const radius = 72;
  const rings = [0.33, 0.66, 1];
  const points = rows.map((row, index) => {
    const angle = -Math.PI / 2 + (index / rows.length) * Math.PI * 2;
    const score = row.value ?? 0;
    return [
      center + Math.cos(angle) * radius * score,
      center + Math.sin(angle) * radius * score,
    ];
  });
  const labelPoints = rows.map((row, index) => {
    const angle = -Math.PI / 2 + (index / rows.length) * Math.PI * 2;
    return {
      row,
      x: center + Math.cos(angle) * (radius + 28),
      y: center + Math.sin(angle) * (radius + 28),
      anchor:
        Math.cos(angle) > 0.25
          ? "start"
          : Math.cos(angle) < -0.25
            ? "end"
            : "middle",
    };
  });
  const average = metricAverage(rows);

  return h(
    "article",
    { className: "metric-radar-card" },
    h(
      "div",
      { className: "metric-chart-head" },
      h(
        "div",
        null,
        h("span", null, "Aggregate profile"),
        h("strong", null, average === null ? "n/a" : percent(average)),
      ),
      h("small", null, "set average"),
    ),
    h(
      "svg",
      {
        className: "metric-radar-svg",
        viewBox: `0 0 ${size} ${size}`,
        role: "img",
        "aria-label": "Aggregate movement metric radar chart",
      },
      h("title", null, "Aggregate movement metric radar chart"),
      rings.map((ring) => {
        const ringPoints = rows.map((_, index) => {
          const angle = -Math.PI / 2 + (index / rows.length) * Math.PI * 2;
          return [
            center + Math.cos(angle) * radius * ring,
            center + Math.sin(angle) * radius * ring,
          ];
        });
        return h("polygon", {
          key: ring,
          className: "radar-ring",
          points: pointList(ringPoints),
        });
      }),
      rows.map((_, index) => {
        const angle = -Math.PI / 2 + (index / rows.length) * Math.PI * 2;
        return h("line", {
          key: index,
          className: "radar-axis",
          x1: center,
          y1: center,
          x2: center + Math.cos(angle) * radius,
          y2: center + Math.sin(angle) * radius,
        });
      }),
      h("polygon", {
        className: "radar-shape",
        points: pointList(points),
      }),
      points.map(([x, y], index) =>
        h("circle", {
          key: `${rows[index].name}-dot`,
          className: `radar-dot ${scoreTone(rows[index].value)}`,
          cx: x,
          cy: y,
          r: 4,
        }),
      ),
      labelPoints.map(({ row, x, y, anchor }) =>
        h(
          "text",
          {
            key: row.name,
            className: "radar-label",
            x,
            y,
            textAnchor: anchor,
          },
          h("tspan", { x, dy: 0 }, row.name),
          h("tspan", { x, dy: 13 }, percent(row.value)),
        ),
      ),
    ),
  );
}

const repMetricSpecs = [
  { key: "range_of_motion_score", label: "ROM", className: "rom" },
  { key: "stability_score", label: "Stability", className: "stability" },
  { key: "symmetry_score", label: "Symmetry", className: "symmetry" },
];

function RepTrendChart({ reps, metricKey, labelText, className }) {
  const width = 360;
  const height = 150;
  const pad = 22;
  const innerWidth = width - pad * 2;
  const innerHeight = height - pad * 2;
  const trendPoints = reps
    .map((rep, index) => {
      const value = metricScore(rep[metricKey]);
      if (value === null) return null;
      const x =
        pad + (reps.length <= 1 ? 0.5 : index / (reps.length - 1)) * innerWidth;
      const y = pad + (1 - value) * innerHeight;
      return { rep, value, x, y };
    })
    .filter(Boolean);
  const values = trendPoints.map((point) => point.value);
  const average = values.length
    ? values.reduce((total, value) => total + value, 0) / values.length
    : null;
  const lowPoint = trendPoints.length
    ? [...trendPoints].sort((a, b) => a.value - b.value)[0]
    : null;
  const linePath = trendPoints.map((point) => [point.x, point.y]);
  const areaPath =
    trendPoints.length > 1
      ? `${pointPath(linePath)} L ${trendPoints[trendPoints.length - 1].x.toFixed(1)} ${(height - pad).toFixed(1)} L ${trendPoints[0].x.toFixed(1)} ${(height - pad).toFixed(1)} Z`
      : "";

  return h(
    "article",
    { className: `metric-trend-card ${className}` },
    h(
      "div",
      { className: "metric-chart-head" },
      h(
        "div",
        null,
        h("span", null, labelText),
        h("strong", null, average === null ? "n/a" : percent(average)),
      ),
      h("small", null, lowPoint ? `low R${lowPoint.rep.rep_id}` : "no data"),
    ),
    h(
      "svg",
      {
        className: "metric-trend-svg",
        viewBox: `0 0 ${width} ${height}`,
        role: "img",
        "aria-label": `${labelText} metric trend by rep`,
      },
      h("title", null, `${labelText} metric trend by rep`),
      [0, 0.5, 1].map((level) =>
        h("line", {
          key: level,
          className: "trend-grid-line",
          x1: pad,
          y1: pad + (1 - level) * innerHeight,
          x2: width - pad,
          y2: pad + (1 - level) * innerHeight,
        }),
      ),
      areaPath ? h("path", { className: "trend-area", d: areaPath }) : null,
      linePath.length
        ? h("path", {
            className: "trend-line",
            d: pointPath(linePath),
          })
        : null,
      trendPoints.map((point) =>
        h("circle", {
          key: point.rep.rep_id,
          className: `trend-dot ${scoreTone(point.value)}`,
          cx: point.x,
          cy: point.y,
          r: 4,
        }),
      ),
      h("text", { className: "trend-axis-label", x: pad, y: height - 4 }, "R1"),
      h(
        "text",
        {
          className: "trend-axis-label",
          x: width - pad,
          y: height - 4,
          textAnchor: "end",
        },
        reps.length ? `R${reps[reps.length - 1].rep_id}` : "R0",
      ),
    ),
  );
}

function RepQualityStrip({ reps }) {
  if (!reps.length) return null;
  const scores = reps.map((rep) => ({ rep, score: repScore(rep) }));
  const average = metricAverage(scores.map(({ score }) => ({ value: score })));
  const weakest = scores
    .filter(({ score }) => score !== null)
    .sort((a, b) => a.score - b.score)[0];

  return h(
    "article",
    { className: "rep-quality-card" },
    h(
      "div",
      { className: "metric-chart-head" },
      h(
        "div",
        null,
        h("span", null, "Rep quality strip"),
        h("strong", null, average === null ? "n/a" : percent(average)),
      ),
      h("small", null, weakest ? `review R${weakest.rep.rep_id}` : "no data"),
    ),
    h(
      "div",
      {
        className: "rep-quality-strip",
        "aria-label": "Average quality by rep",
      },
      scores.map(({ rep, score }) =>
        h(
          "div",
          {
            className: `rep-quality-bar ${scoreTone(score)}`,
            key: rep.rep_id,
            title: `Rep ${rep.rep_id}: ${score === null ? "n/a" : percent(score)}`,
          },
          h("span", {
            style: {
              height:
                score === null
                  ? "8%"
                  : `${Math.max(8, Math.round(score * 100))}%`,
            },
            "aria-hidden": "true",
          }),
          h("strong", null, `R${rep.rep_id}`),
        ),
      ),
    ),
  );
}

function BriefTile({ eyebrow, title, body, tone = "" }) {
  return h(
    "article",
    { className: `brief-tile${tone ? ` ${tone}` : ""}` },
    h("span", null, eyebrow),
    h("strong", null, title),
    h("p", null, body),
  );
}

function CoachOverview({ result }) {
  if (!result) return null;
  const summary = result.report.coach_summary;
  const coachArtifacts = result.report.artifacts || {};
  const coachSource = coachArtifacts.coach_summary_source || "";
  const coachModel = coachArtifacts.coach_summary_model || "n/a";
  const verifierBypassed = Boolean(
    coachArtifacts.coach_summary_verifier_bypassed,
  );
  const isFallback = coachSource.startsWith("fallback");
  const cueNow =
    summary.top_fixes?.[0] || summary.next_session_plan?.[0] || summary.summary;

  return h(
    "section",
    { className: "coach-spotlight" },
    h(
      "div",
      { className: "coach-command" },
      h(
        "div",
        { className: "coach-command-copy" },
        h("span", null, "Coach intelligence"),
        h("strong", null, cueNow),
        h("p", null, summary.summary),
      ),
      h(
        "div",
        { className: "coach-meta-stack" },
        metaChip(`Model: ${coachModel}`),
      ),
    ),
    isFallback || verifierBypassed
      ? h(
          "div",
          { className: "coach-system-row" },
          isFallback
            ? h(
                "p",
                { className: "system-note" },
                "A conservative fallback summary was used because the generated summary was unavailable or did not pass verification.",
              )
            : null,
        )
      : null,
    h(
      "div",
      { className: "coach-overview-grid" },
      h(NoteList, {
        title: "Fix first",
        items: summary.top_fixes,
        className: "coach-note priority",
      }),
      h(NoteList, {
        title: "Next session",
        items: summary.next_session_plan,
        className: "coach-note next",
      }),
      h(NoteList, {
        title: "Keep",
        items: summary.what_looked_good,
        className: "coach-note",
      }),
      h(NoteList, {
        title: "Variation vs issue",
        items: summary.valid_variation_vs_issue,
        className: "coach-note",
      }),
      h(NoteList, {
        title: "Confidence notes",
        items: summary.confidence_notes,
        className: "coach-note muted",
      }),
    ),
  );
}

function RepTimeline({
  report,
  compact = false,
  activeRepId = null,
  playbackTime = 0,
  onRepSelect = null,
}) {
  const reps = report.reps?.reps || [];
  const duration = reportDuration(report);
  const analysisById = repAnalysisById(report);
  const isPlayback = typeof onRepSelect === "function";
  const playheadLeft = `${Math.max(0, Math.min(100, (playbackTime / duration) * 100))}%`;
  if (!reps.length) {
    return h(
      "p",
      { className: "empty-copy" },
      "No complete reps were detected.",
    );
  }

  return h(
    "div",
    {
      className: `rep-timeline${compact ? " compact" : ""}${isPlayback ? " playback" : ""}`,
    },
    h(
      "div",
      { className: "timeline-ruler", "aria-hidden": "true" },
      h("span", null, "0s"),
      h("span", null, `${duration.toFixed(1)}s`),
    ),
    h(
      "div",
      {
        className: `rep-track${isPlayback ? " interactive" : ""}`,
        "aria-label": isPlayback ? "Interactive rep timeline" : "Rep timeline",
      },
      isPlayback
        ? h("span", {
            className: "timeline-playhead",
            style: { left: playheadLeft },
            "aria-hidden": "true",
          })
        : null,
      reps.map((rep) => {
        const left = `${Math.max(0, Math.min(98, (rep.start_sec / duration) * 100))}%`;
        const width = `${Math.max(6, Math.min(100, ((rep.end_sec - rep.start_sec) / duration) * 100))}%`;
        const score = repScore(analysisById.get(rep.rep_id));
        const repIssues = issuesForRep(report, rep.rep_id);
        const tagName = isPlayback ? "button" : "div";
        const segmentProps = {
          className: `rep-segment ${scoreTone(score)}${repIssues.length ? " has-issue" : ""}${activeRepId === rep.rep_id ? " active" : ""}`,
          key: rep.rep_id,
          style: { left, width },
          title: `Rep ${rep.rep_id} ${timeRange(rep.start_sec, rep.end_sec)}`,
        };
        if (isPlayback) {
          segmentProps.type = "button";
          segmentProps.onClick = () => onRepSelect(rep);
          segmentProps["aria-label"] =
            `Play from rep ${rep.rep_id}, ${timeRange(rep.start_sec, rep.end_sec)}`;
          segmentProps["aria-pressed"] = activeRepId === rep.rep_id;
        }
        return h(
          tagName,
          segmentProps,
          h("strong", null, `R${rep.rep_id}`),
          repIssues.map((issue, index) =>
            h("span", {
              className: `issue-pin ${severityLevel(issue)}`,
              key: `${issue.issue}-${index}`,
              style: {
                left: `${Math.max(8, Math.min(92, ((issue.start_sec - rep.start_sec) / Math.max(0.01, rep.end_sec - rep.start_sec)) * 100))}%`,
              },
              title: issueTitle(issue),
            }),
          ),
        );
      }),
    ),
  );
}

function ProgressPlanPanel({ result }) {
  const plan = result.report?.progress_plan;
  if (!plan) {
    return null;
  }
  const speech = result.report?.speech;
  return h(
    "div",
    { className: "summary progress-plan" },
    h("p", { className: "eyebrow" }, "Coach memory"),
    h("h3", null, "Next session plan"),
    h("p", { className: "report-lede" }, plan.focus),
    h(
      "div",
      { className: "note-grid report-notes" },
      h(NoteList, { title: "Targets", items: plan.targets || [] }),
      h(NoteList, { title: "Cues", items: plan.next_session_cues || [] }),
      h(NoteList, {
        title: "Encouragement",
        items: plan.encouragement ? [plan.encouragement] : [],
      }),
    ),
    speech && speech.lines && speech.lines.length
      ? h(NoteList, { title: "Spoken feedback", items: speech.lines })
      : null,
  );
}

function SummaryTab({ result }) {
  if (!result) {
    return h(
      "section",
      { className: "summary" },
      h("h2", null, "Ready for review"),
      h("p", null, "Choose the movement context and start the analyzer."),
    );
  }

  const report = result.report;
  const summary = report.coach_summary;
  const warnings = report.video_manifest.quality_warnings || [];
  const issues = issueList(report);
  const issue = topIssue(report);
  const best = bestRep(report);
  const verdict = qualityVerdict(report);
  const scores = metricScoreRows(report);
  const provider = report.artifacts?.coach_summary_provider || "n/a";
  const model = report.artifacts?.coach_summary_model || "n/a";
  const repCount = report.reps?.reps?.length || 0;
  const cueNow =
    summary.top_fixes?.[0] ||
    (issue
      ? issueEvidence(issue)
      : "Repeat the next set with the same controlled tempo.");
  const nextSession =
    summary.next_session_plan?.[0] || "Repeat the set with controlled reps.";

  return h(
    "section",
    { className: "summary overview-panel" },
    h(
      "div",
      { className: "report-hero" },
      h(
        "div",
        { className: "report-copy" },
        h("p", { className: "eyebrow" }, "Movement report"),
        h("h2", null, `${label(report.exercise.exercise)} review`),
        h("p", { className: "report-lede" }, summary.summary),
        h(
          "div",
          { className: "pill-row" },
          metaChip(`confidence ${percent(report.exercise.confidence)}`, "blue"),
          metaChip(`${repCount} rep${repCount === 1 ? "" : "s"}`),
          metaChip(
            issues.length
              ? `${issues.length} coaching moment${issues.length === 1 ? "" : "s"}`
              : "no clear issue",
            issues.length ? severityLevel(issue) : "strong",
          ),
          metaChip(
            provider === "local_transformers" ? "local coach" : provider,
          ),
        ),
      ),
      h(
        "aside",
        { className: `verdict-card ${scoreTone(verdict.score)}` },
        h("span", null, "Set verdict"),
        h("strong", null, verdict.title),
        h("p", null, verdict.detail),
        h("small", null, model),
      ),
    ),
    h(
      "div",
      { className: "brief-grid" },
      h(BriefTile, {
        eyebrow: "Best rep",
        title: best ? `Rep ${best.rep.rep_id}` : "Not enough data",
        body: best
          ? `Most consistent rep at ${timeRange(best.rep.start_sec, best.rep.end_sec)} with ${percent(best.score)} quality.`
          : "Complete reps are needed before the app can pick a best rep.",
        tone: "mint",
      }),
      h(BriefTile, {
        eyebrow: "Main limiter",
        title: issue ? issueTitle(issue) : "Nothing major",
        body: issue
          ? `${issueClipText(result, issue, 0)}. ${issueEvidence(issue)}.`
          : "No sustained threshold violations were found in this set.",
        tone: issue ? severityLevel(issue) : "mint",
      }),
      h(BriefTile, {
        eyebrow: "Cue now",
        title: "Try this first",
        body: cueNow,
        tone: "volt",
      }),
      h(BriefTile, {
        eyebrow: "Next session",
        title: "Keep it simple",
        body: nextSession,
      }),
    ),
    h(
      "div",
      { className: "score-board" },
      scores.length
        ? scores.map((row) =>
            h(ScoreBar, {
              key: row.name,
              labelText: row.name,
              value: row.value,
            }),
          )
        : h("p", { className: "empty-copy" }, "No movement scores available."),
    ),
    h(CoachOverview, { result }),
    h(ProgressPlanPanel, { result }),
    h(
      "div",
      { className: "note-grid report-notes" },
      h(NoteList, { title: "What you did", items: summary.what_you_did }),
      h(NoteList, {
        title: "What looked good",
        items: summary.what_looked_good,
      }),
      h(NoteList, {
        title: "What changed",
        items: summary.what_changed_across_reps,
      }),
    ),
    warnings.length
      ? h(
          "div",
          { className: "quality-list" },
          warnings.map((warning) =>
            h("span", { key: warning }, label(warning)),
          ),
        )
      : h(
          "div",
          { className: "quality-list" },
          h("span", null, "No quality warnings"),
        ),
  );
}

function MetricsTab({ result }) {
  if (!result)
    return h(
      "section",
      { className: "summary" },
      h("h2", null, "Movement metrics"),
      h("p", null, "Metrics appear after analysis."),
    );
  const report = result.report;
  const reps = report.rep_analysis?.items || [];
  const scoreRows = metricScoreRows(report);
  const trendSpecs = repMetricSpecs.filter((spec) =>
    reps.some((rep) => metricScore(rep[spec.key]) !== null),
  );
  return h(
    "section",
    { className: "summary metrics-panel" },
    h("h2", null, "Movement metrics"),
    h(
      "p",
      null,
      "Charts show which qualities shaped the coach note and where the set started to drift.",
    ),
    h(
      "div",
      { className: "metric-chart-hero" },
      h(MetricRadar, { rows: scoreRows }),
      h(
        "div",
        { className: "score-board metric-score-board" },
        scoreRows.length
          ? scoreRows.map((row) =>
              h(ScoreBar, {
                key: row.name,
                labelText: row.name,
                value: row.value,
              }),
            )
          : h(
              "p",
              { className: "empty-copy" },
              "No aggregate scores available.",
            ),
      ),
    ),
    trendSpecs.length
      ? h(
          "div",
          { className: "metric-trend-grid" },
          trendSpecs.map((spec) =>
            h(RepTrendChart, {
              key: spec.key,
              reps,
              metricKey: spec.key,
              labelText: spec.label,
              className: spec.className,
            }),
          ),
        )
      : h("p", { className: "empty-copy" }, "No per-rep metrics available."),
    h(RepQualityStrip, { reps }),
    h(
      "div",
      { className: "metric-table-head" },
      h("span", null, "Raw values"),
      h("small", null, "per rep detail"),
    ),
    h(
      "div",
      { className: "table-wrap" },
      h(
        "table",
        null,
        h(
          "thead",
          null,
          h(
            "tr",
            null,
            ["Rep", "Duration", "ROM", "Stability", "Symmetry"].map((heading) =>
              h("th", { key: heading }, heading),
            ),
          ),
        ),
        h(
          "tbody",
          null,
          reps.map((rep) =>
            h(
              "tr",
              { key: rep.rep_id },
              h("td", null, rep.rep_id),
              h("td", null, `${formatValue(rep.duration_sec)}s`),
              h("td", null, percent(rep.range_of_motion_score)),
              h("td", null, percent(rep.stability_score)),
              h("td", null, percent(rep.symmetry_score)),
            ),
          ),
        ),
      ),
    ),
  );
}

function RepDetailPanel({
  report,
  rep,
  result,
  playbackTime,
  compact = false,
}) {
  if (!rep) {
    return h(
      "aside",
      { className: `rep-detail-panel${compact ? " compact" : ""}` },
      h("span", { className: "rep-detail-kicker" }, "Current rep"),
      h("strong", null, "No rep selected"),
      h(
        "p",
        null,
        "Play the video or click a rep segment to inspect its metrics.",
      ),
    );
  }

  const analysis = repAnalysisById(report).get(rep.rep_id);
  const repIssues = issuesForRep(report, rep.rep_id);
  const score = repScore(analysis);
  const duration = Math.max(0, (rep.end_sec || 0) - (rep.start_sec || 0));
  const firstHalfDuration = Math.max(
    0,
    (rep.mid_sec || 0) - (rep.start_sec || 0),
  );
  const secondHalfDuration = Math.max(
    0,
    (rep.end_sec || 0) - (rep.mid_sec || 0),
  );
  const metricRows = [
    { name: "ROM", value: metricScore(analysis?.range_of_motion_score) },
    { name: "Stability", value: metricScore(analysis?.stability_score) },
    { name: "Symmetry", value: metricScore(analysis?.symmetry_score) },
  ].filter((item) => item.value !== null);
  const weakestMetric = metricRows.length
    ? [...metricRows].sort((a, b) => a.value - b.value)[0]
    : null;
  const repProgress =
    duration > 0
      ? Math.max(0, Math.min(1, (playbackTime - rep.start_sec) / duration))
      : 0;
  const phase =
    playbackTime < rep.start_sec
      ? "setup"
      : playbackTime > rep.end_sec
        ? "review"
        : playbackTime <= rep.mid_sec
          ? "half 1"
          : "half 2";
  const focusText = weakestMetric
    ? `${weakestMetric.name} is the main review cue at ${percent(weakestMetric.value)}.`
    : "Metric detail is not available for this rep.";
  const scoreText =
    score === null
      ? "Metric data is limited for this rep."
      : score >= 0.8
        ? "This rep is one of the steadier parts of the set."
        : score >= 0.62
          ? "This rep is usable, but one metric deserves attention."
          : "This rep is the kind to slow down and review before progressing.";
  const issueText = repIssues.length
    ? `${repIssues.length} issue marker${repIssues.length === 1 ? "" : "s"} attached to this rep.`
    : "No issue marker is attached, so use the lowest metric as the review cue.";
  const primaryIssue = repIssues[0] || null;

  if (compact) {
    return h(
      "aside",
      { className: `rep-detail-panel compact ${scoreTone(score)}` },
      h(
        "div",
        { className: "rep-detail-head" },
        h(
          "div",
          null,
          h("span", { className: "rep-detail-kicker" }, "Current rep"),
          h("strong", null, `Rep ${rep.rep_id}`),
          h("small", null, timeRange(rep.start_sec, rep.end_sec)),
        ),
        metaChip(
          score === null ? "no score" : percent(score),
          scoreTone(score),
        ),
      ),
      h(
        "div",
        { className: "rep-fact-grid compact" },
        h(
          "div",
          { className: "rep-fact" },
          h("span", null, "Duration"),
          h("strong", null, `${duration.toFixed(2)}s`),
        ),
        h(
          "div",
          { className: "rep-fact" },
          h("span", null, "Focus"),
          h(
            "strong",
            null,
            weakestMetric
              ? `${weakestMetric.name} ${percent(weakestMetric.value)}`
              : "n/a",
          ),
        ),
        h(
          "div",
          { className: "rep-fact" },
          h("span", null, "Frames"),
          h("strong", null, `${rep.start_frame}-${rep.end_frame}`),
        ),
        h(
          "div",
          { className: "rep-fact" },
          h("span", null, "Issues"),
          h(
            "strong",
            null,
            repIssues.length ? String(repIssues.length) : "clear",
          ),
        ),
      ),
      h(
        "div",
        { className: "rep-progress-card compact" },
        h(
          "div",
          { className: "rep-now" },
          h("span", null, "playhead"),
          h("strong", null, `${playbackTime.toFixed(2)}s`),
        ),
        h(
          "div",
          { className: "rep-progress-track", "aria-hidden": "true" },
          h("span", {
            className: "rep-progress-fill",
            style: { width: `${Math.round(repProgress * 100)}%` },
          }),
          h("span", { className: "rep-progress-mid" }),
        ),
        h(
          "div",
          { className: "rep-progress-labels" },
          h("span", null, `${rep.start_sec.toFixed(2)}s`),
          h("span", null, `${rep.mid_sec.toFixed(2)}s mid`),
          h("span", null, `${rep.end_sec.toFixed(2)}s`),
        ),
      ),
      primaryIssue
        ? h(
            "article",
            { className: `rep-issue-brief ${severityLevel(primaryIssue)}` },
            h(
              "div",
              { className: "rep-issue-brief-head" },
              h(
                "div",
                null,
                h("span", null, "Issue detail"),
                h("strong", null, issueTitle(primaryIssue)),
              ),
              h(
                "span",
                { className: `severity-chip ${severityLevel(primaryIssue)}` },
                severityText(primaryIssue),
              ),
            ),
            h("p", null, issueEvidence(primaryIssue)),
            h(
              "div",
              { className: "rep-issue-meta" },
              h("span", null, issueFocus(primaryIssue)),
              h("span", null, issueClipText(result, primaryIssue, 0)),
              h(
                "span",
                null,
                `confidence ${percent(primaryIssue.evidence?.confidence)}`,
              ),
              repIssues.length > 1
                ? h("span", null, `+${repIssues.length - 1} more`)
                : null,
            ),
          )
        : null,
    );
  }

  return h(
    "aside",
    { className: `rep-detail-panel ${scoreTone(score)}` },
    h(
      "div",
      { className: "rep-detail-head" },
      h(
        "div",
        null,
        h("span", { className: "rep-detail-kicker" }, "Current rep"),
        h("strong", null, `Rep ${rep.rep_id}`),
        h("small", null, timeRange(rep.start_sec, rep.end_sec)),
      ),
      metaChip(score === null ? "no score" : percent(score), scoreTone(score)),
    ),
    h(
      "div",
      { className: "rep-fact-grid" },
      h(
        "div",
        { className: "rep-fact" },
        h("span", null, "Duration"),
        h("strong", null, `${duration.toFixed(2)}s`),
      ),
      h(
        "div",
        { className: "rep-fact" },
        h("span", null, "Frames"),
        h("strong", null, `${rep.start_frame}-${rep.end_frame}`),
      ),
      h(
        "div",
        { className: "rep-fact" },
        h("span", null, "Tempo split"),
        h(
          "strong",
          null,
          `${firstHalfDuration.toFixed(2)}/${secondHalfDuration.toFixed(2)}s`,
        ),
      ),
      h(
        "div",
        { className: "rep-fact" },
        h("span", null, "Phase"),
        h("strong", null, phase),
      ),
      h(
        "div",
        { className: "rep-fact" },
        h("span", null, "Focus"),
        h(
          "strong",
          null,
          weakestMetric
            ? `${weakestMetric.name} ${percent(weakestMetric.value)}`
            : "n/a",
        ),
      ),
      h(
        "div",
        { className: "rep-fact" },
        h("span", null, "Issues"),
        h(
          "strong",
          null,
          repIssues.length ? String(repIssues.length) : "clear",
        ),
      ),
    ),
    h(
      "div",
      { className: "rep-progress-card" },
      h(
        "div",
        { className: "rep-now" },
        h("span", null, "playhead"),
        h("strong", null, `${playbackTime.toFixed(2)}s`),
      ),
      h(
        "div",
        { className: "rep-progress-track", "aria-hidden": "true" },
        h("span", {
          className: "rep-progress-fill",
          style: { width: `${Math.round(repProgress * 100)}%` },
        }),
        h("span", { className: "rep-progress-mid" }),
      ),
      h(
        "div",
        { className: "rep-progress-labels" },
        h("span", null, `${rep.start_sec.toFixed(2)}s`),
        h("span", null, `${rep.mid_sec.toFixed(2)}s mid`),
        h("span", null, `${rep.end_sec.toFixed(2)}s`),
      ),
    ),
    h(
      "article",
      { className: "rep-readout" },
      h("span", null, "Movement read"),
      h("strong", null, scoreText),
      h("p", null, `${focusText} ${issueText}`),
    ),
    h(
      "div",
      { className: "rep-mini-metrics" },
      h(ScoreBar, {
        labelText: "ROM",
        value: analysis?.range_of_motion_score,
      }),
      h(ScoreBar, {
        labelText: "Stability",
        value: analysis?.stability_score,
      }),
      h(ScoreBar, {
        labelText: "Symmetry",
        value: analysis?.symmetry_score,
      }),
    ),
    h(
      "div",
      { className: "issue-mini-list" },
      h("h3", null, repIssues.length ? "Issue in this rep" : "Issue check"),
      repIssues.length
        ? repIssues.map((issue, index) =>
            h(
              "article",
              {
                className: `issue-mini ${severityLevel(issue)}`,
                key: `${issue.issue}-${index}`,
              },
              h(
                "div",
                { className: "issue-mini-head" },
                h("strong", null, issueTitle(issue)),
                h(
                  "span",
                  { className: `severity-chip ${severityLevel(issue)}` },
                  severityText(issue),
                ),
              ),
              h("p", null, issueEvidence(issue)),
              h(
                "div",
                { className: "timeline-meta" },
                h("span", null, issueClipText(result, issue, index)),
                h(
                  "span",
                  null,
                  `evidence ${percent(issue.evidence?.confidence)}`,
                ),
              ),
            ),
          )
        : h("p", null, "No issue marker is attached to this rep."),
    ),
  );
}

function ReplayReviewPanel({ result, videoSrc, className = "" }) {
  const [playbackTime, setPlaybackTime] = useState(0);
  const [selectedRepId, setSelectedRepId] = useState(null);
  const replayVideoRef = useRef(null);

  useEffect(() => {
    setPlaybackTime(0);
    setSelectedRepId(null);
  }, [result?.run_id, videoSrc]);

  const report = result.report;
  const reps = report.reps?.reps || [];
  const timedRep = repAtTime(report, playbackTime);
  const selectedRep = reps.find((rep) => rep.rep_id === selectedRepId) || null;
  const activeRep = timedRep || selectedRep || reps[0] || null;

  function playFromRep(rep) {
    const startTime = Math.max(0, (rep.start_sec || 0) - 0.04);
    setSelectedRepId(rep.rep_id);
    setPlaybackTime(startTime);
    if (!replayVideoRef.current) return;
    replayVideoRef.current.currentTime = startTime;
    const playPromise = replayVideoRef.current.play();
    if (playPromise?.catch) playPromise.catch(() => undefined);
  }

  return h(
    "div",
    { className: `replay-review${className ? ` ${className}` : ""}` },
    h(
      "div",
      { className: "replay-main" },
      h(
        "div",
        { className: "replay-video-card" },
        videoSrc
          ? h("video", {
              className: "timeline-video",
              controls: true,
              muted: true,
              playsInline: true,
              preload: "metadata",
              ref: replayVideoRef,
              src: videoSrc,
              onLoadedMetadata: (event) => {
                setPlaybackTime(event.currentTarget.currentTime || 0);
              },
              onTimeUpdate: (event) => {
                setPlaybackTime(event.currentTarget.currentTime || 0);
              },
              onSeeked: (event) => {
                setPlaybackTime(event.currentTarget.currentTime || 0);
              },
            })
          : h(
              "div",
              { className: "timeline-video-placeholder" },
              h("strong", null, "No replay video available"),
              h(
                "span",
                null,
                "Upload a clip or use a run with an annotated video to sync the timeline.",
              ),
            ),
      ),
      h(RepDetailPanel, {
        report,
        rep: activeRep,
        result,
        playbackTime,
        compact: true,
      }),
    ),
    h(
      "div",
      { className: "replay-timeline-card" },
      h(RepTimeline, {
        report,
        activeRepId: activeRep?.rep_id || null,
        playbackTime,
        onRepSelect: playFromRep,
      }),
    ),
  );
}

function RepsTab({ result }) {
  if (!result)
    return h(
      "section",
      { className: "summary" },
      h("h2", null, "Rep list"),
      h("p", null, "Rep segments appear after analysis."),
    );
  const report = result.report;
  const reps = report.reps?.reps || [];
  const analysisById = repAnalysisById(report);

  return h(
    "section",
    { className: "summary reps-panel" },
    h("h2", null, "All reps"),
    h(
      "p",
      null,
      "A compact pass over every detected rep, with timing, frames, movement score, and issue markers.",
    ),
    reps.length
      ? h(
          "div",
          { className: "rep-grid" },
          reps.map((rep) => {
            const analysis = analysisById.get(rep.rep_id);
            const repIssues = issuesForRep(report, rep.rep_id);
            const score = repScore(analysis);
            return h(
              "article",
              { className: `rep-card ${scoreTone(score)}`, key: rep.rep_id },
              h(
                "div",
                { className: "rep-card-head" },
                h("strong", null, `Rep ${rep.rep_id}`),
                metaChip(
                  score === null ? "no score" : percent(score),
                  scoreTone(score),
                ),
              ),
              h("span", null, timeRange(rep.start_sec, rep.end_sec)),
              h("span", null, `frames ${rep.start_frame}-${rep.end_frame}`),
              h("span", null, `midpoint ${rep.mid_sec.toFixed(2)}s`),
              h(
                "div",
                { className: "pill-row compact" },
                repIssues.length
                  ? repIssues.map((issue, index) =>
                      metaChip(
                        issueTitle(issue),
                        `${severityLevel(issue)}`,
                        `${issue.issue}-${index}`,
                      ),
                    )
                  : metaChip("no issue marker", "strong"),
              ),
            );
          }),
        )
      : h("p", null, "No complete reps were detected."),
  );
}

function issueEvidence(issue) {
  const degreeEntry = Object.entries(issue.evidence || {}).find(
    ([key, value]) =>
      key.endsWith("_deg") && typeof value === "number" && !Number.isNaN(value),
  );
  if (degreeEntry) {
    return `${label(degreeEntry[0].replace("_deg", ""))} about ${Math.round(degreeEntry[1])} deg`;
  }

  const cues = {
    shallow_depth: "Lower the hips a little more before standing up",
    hip_sag: "Keep hips in line with shoulders and ankles",
    incomplete_depth: "Bend deeper at the bottom of the rep",
    knee_valgus: "Keep knees tracking over the toes",
    excessive_torso_lean: "Keep the chest taller through the bottom",
    incomplete_lockout: "Finish by straightening the elbows",
    asymmetry: "Keep both sides moving evenly",
  };
  return cues[issue.issue] || "Review this part of the rep";
}

function issueTitle(issue) {
  const titles = {
    shallow_depth: "Squat depth is shallow",
    hip_sag: "Hips are dropping",
    incomplete_depth: "Rep is not deep enough",
    knee_valgus: "Knees are caving inward",
    excessive_torso_lean: "Torso leans too far forward",
    incomplete_lockout: "Lockout is incomplete",
    asymmetry: "Left and right sides are uneven",
  };
  return titles[issue.issue] || label(issue.issue);
}

function issueFocus(issue) {
  const focus = {
    shallow_depth: "Focus on hips and knees",
    hip_sag: "Focus on trunk and hips",
    incomplete_depth: "Focus on shoulders, elbows, and wrists",
    knee_valgus: "Focus on knees and ankles",
    excessive_torso_lean: "Focus on chest and hips",
    incomplete_lockout: "Focus on elbows and wrists",
    asymmetry: "Focus on left-right balance",
  };
  if (focus[issue.issue]) return focus[issue.issue];
  return `Focus on ${issue.affected_joints.map(label).join(", ")}`;
}

function severityText(issue) {
  const severity = Math.round(issue.severity * 100);
  if (severity >= 70) return `high attention ${severity}%`;
  if (severity >= 40) return `moderate attention ${severity}%`;
  return `minor attention ${severity}%`;
}

function severityLevel(issue) {
  const severity = Math.round(issue.severity * 100);
  if (severity >= 70) return "high";
  if (severity >= 40) return "moderate";
  return "minor";
}

function thumbnailForIssue(result, issue, index) {
  const thumbnails = result?.issue_thumbnail_urls || [];
  return (
    thumbnails.find(
      (thumbnail) =>
        thumbnail.rep_id === issue.rep_id && thumbnail.issue === issue.issue,
    ) || thumbnails[index]
  );
}

function clipForIssue(result, issue, index) {
  const clips = result?.issue_clip_urls || [];
  return (
    clips.find(
      (clip) => clip.rep_id === issue.rep_id && clip.issue === issue.issue,
    ) || clips[index]
  );
}

function IssueMedia({ result, issue, index }) {
  const clip = clipForIssue(result, issue, index);
  if (clip?.url) {
    return h("video", {
      className: "issue-clip",
      src: clip.url,
      controls: true,
      muted: true,
      playsInline: true,
      preload: "metadata",
    });
  }

  const thumbnail = thumbnailForIssue(result, issue, index);
  if (thumbnail?.url) {
    return h("img", {
      className: "issue-thumb",
      src: thumbnail.url,
      alt: `${label(issue.issue)} thumbnail`,
    });
  }

  return h("div", { className: "issue-thumb empty" }, "No clip");
}

function issueClipText(result, issue, index) {
  const clip = clipForIssue(result, issue, index);
  if (
    !clip ||
    typeof clip.clip_start_sec !== "number" ||
    typeof clip.clip_end_sec !== "number"
  ) {
    return `Rep ${issue.rep_id} · ${issue.start_sec.toFixed(2)}s-${issue.end_sec.toFixed(2)}s`;
  }
  return `Rep ${issue.rep_id} · clip ${clip.clip_start_sec.toFixed(2)}s-${clip.clip_end_sec.toFixed(2)}s`;
}

function IssuesTab({ result }) {
  const issues = result?.report?.issue_markers?.issues || [];
  return h(
    "section",
    { className: "summary issue-board", "aria-label": "Issue timeline" },
    h(
      "div",
      { className: "timeline-head" },
      h("h3", null, "Coaching moments"),
      h(
        "span",
        null,
        issues.length
          ? `${issues.length} interval${issues.length === 1 ? "" : "s"}`
          : "clear",
      ),
    ),
    issues.length
      ? h(
          "div",
          { className: "issue-card-grid" },
          issues.map((issue, index) =>
            h(
              "article",
              {
                className: `issue-card issue-card-feature ${severityLevel(issue)}`,
                key: `${issue.rep_id}-${issue.issue}-${index}`,
              },
              h(IssueMedia, { result, issue, index }),
              h(
                "div",
                { className: "issue-body" },
                h(
                  "div",
                  { className: "issue-card-head" },
                  h(
                    "span",
                    { className: `severity-chip ${severityLevel(issue)}` },
                    severityText(issue),
                  ),
                  h("strong", null, issueTitle(issue)),
                  h("small", null, issueClipText(result, issue, index)),
                ),
                h("p", null, issueEvidence(issue)),
                h(
                  "div",
                  { className: "timeline-meta" },
                  h(
                    "span",
                    null,
                    `issue ${timeRange(issue.start_sec, issue.end_sec)}`,
                  ),
                  h(
                    "span",
                    null,
                    `evidence ${percent(issue.evidence?.confidence)}`,
                  ),
                  h("span", null, issueFocus(issue)),
                  issue.affected_joints.length
                    ? h(
                        "span",
                        null,
                        `joints ${issue.affected_joints.map(label).join(", ")}`,
                      )
                    : null,
                ),
              ),
            ),
          ),
        )
      : h("p", null, "No sustained threshold violations were found."),
  );
}

function JsonTab({ result }) {
  return h(
    "pre",
    { className: "json-block" },
    result ? JSON.stringify(result.report, null, 2) : "{}",
  );
}

function ArtifactsTab({ result }) {
  const links = result?.artifact_urls || [];
  return h(
    "section",
    { className: "artifact-panel" },
    h(
      "div",
      { className: "artifact-links" },
      links.length
        ? links.map((artifact) =>
            h(
              "a",
              {
                className: "artifact-link",
                key: artifact.url,
                href: artifact.url,
                download: artifact.name,
              },
              artifact.name,
              h("span", null, "download"),
            ),
          )
        : h("p", null, "Artifacts appear after analysis."),
    ),
    h(
      "details",
      { className: "json-details" },
      h("summary", null, "Raw report JSON"),
      h(JsonTab, { result }),
    ),
  );
}

const reportTabs = [
  ["summary", "Overview"],
  ["reps", "Reps"],
  ["metrics", "Metrics"],
  ["issues", "Issues"],
  ["artifacts", "Artifacts"],
];

function ReportPanel({ result, activeTab, onTabChange }) {
  const content = {
    summary: h(SummaryTab, { result }),
    metrics: h(MetricsTab, { result }),
    reps: h(RepsTab, { result }),
    issues: h(IssuesTab, { result }),
    artifacts: h(ArtifactsTab, { result }),
  }[activeTab];

  return h(
    React.Fragment,
    null,
    h(
      "div",
      { className: "tabs", role: "tablist", "aria-label": "Report sections" },
      reportTabs.map(([key, name]) =>
        h(
          "button",
          {
            className: `tab${activeTab === key ? " active" : ""}`,
            "aria-selected": activeTab === key,
            key,
            onClick: () => onTabChange(key),
            role: "tab",
            type: "button",
          },
          name,
        ),
      ),
    ),
    content,
  );
}

function NoteList({ title, items, className = "" }) {
  return h(
    "article",
    { className: `note${className ? ` ${className}` : ""}` },
    h("h3", null, title),
    h(
      "ul",
      null,
      items.map((item) => h("li", { key: item }, item)),
    ),
  );
}

export { ReportPanel, ReplayReviewPanel, ReviewInsights };
