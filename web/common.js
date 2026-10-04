import React from "https://esm.sh/react@18.2.0";

export const h = React.createElement;

export const defaults = {
  description:
    "Upload a short workout clip, tune the athlete context, and generate an annotated form-review report with structured artifacts.",
  goals: [
    "strength",
    "hypertrophy",
    "endurance",
    "mobility",
    "beginner_practice",
  ],
  experience_levels: ["beginner", "intermediate"],
  exercises: ["auto"],
  limitations: ["wrist_discomfort", "knee_discomfort", "shoulder_discomfort"],
  equipment: ["bodyweight", "dumbbell", "barbell", "unknown"],
};

export function label(value) {
  return value.replaceAll("_", " ");
}

export function formatValue(value) {
  if (typeof value === "number")
    return Number.isInteger(value) ? String(value) : value.toFixed(3);
  if (Array.isArray(value)) return value.map(formatValue).join(", ");
  if (value && typeof value === "object") return JSON.stringify(value);
  return value ?? "n/a";
}

export function percent(value) {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : "n/a";
}
