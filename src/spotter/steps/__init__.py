"""Pipeline step modules."""

from spotter.steps.progress_insight import (
    SessionComparison,
    compare_sessions,
    compute_trend_over_sessions,
    format_comparison_for_plan,
    generate_trend_svg,
)

__all__ = [
    "SessionComparison",
    "compare_sessions",
    "compute_trend_over_sessions",
    "format_comparison_for_plan",
    "generate_trend_svg",
]

