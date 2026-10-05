/**
 * Progress insight utilities for frontend - generates SVG trend sparklines
 * Mirrors the Python implementation in src/spotter/steps/progress_insight.py
 */

export function generate_trend_svg(trendData, width = 120, height = 40) {
  const trend = trendData?.form_trend || "stable";

  const color = {
    "improving": "#22c55e",  // green
    "declining": "#ef4444",  // red
    "stable": "#f59e0b",     // amber
    "insufficient_data": "#6b7280",  // gray
  }[trend] || "#6b7280";

  const direction = {
    "improving": "▲",
    "declining": "▼",
    "stable": "●",
    "insufficient_data": "—",
  }[trend] || "—";

  return `
<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">
    <rect width="${width}" height="${height}" fill="transparent"/>
    <text x="${width/2}" y="${height/2 + 5}" text-anchor="middle"
          font-family="system-ui, sans-serif" font-size="14" fill="${color}">
        ${direction}
    </text>
    <text x="${width/2}" y="${height/2 + 20}" text-anchor="middle"
          font-family="system-ui, sans-serif" font-size="9" fill="#6b7280">
        ${trend}
    </text>
</svg>
`.trim();
}