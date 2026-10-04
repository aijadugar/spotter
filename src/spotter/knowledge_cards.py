from __future__ import annotations

from dataclasses import dataclass
import json
import os
from functools import lru_cache
from pathlib import Path

from spotter.contracts import ExerciseClassification, GOALS, IssueMarkers, UserProfile, Variation


CARD_TYPE_ORDER = {
    "exercise": 0,
    "variation": 1,
    "issue": 2,
    "equipment": 3,
    "goal": 4,
    "goal_overlay": 5,
    "safety_rule": 6,
}


@dataclass(frozen=True)
class KnowledgeCard:
    card_id: str
    card_type: str
    labels: tuple[str, ...]
    title: str
    summary: str
    evidence_rules: tuple[str, ...]
    coaching_points: tuple[str, ...]
    allowed_interpretations: tuple[str, ...] = ()
    forbidden_claims: tuple[str, ...] = ()
    related_cards: tuple[str, ...] = ()
    source_kind: str = "builtin"
    source_path: str | None = None


@dataclass(frozen=True)
class KnowledgeCatalog:
    cards: tuple[KnowledgeCard, ...]
    cards_by_id: dict[str, KnowledgeCard]
    cards_by_label: dict[str, KnowledgeCard]
    loaded_pack_paths: tuple[str, ...]


@dataclass(frozen=True)
class KnowledgeRetrieval:
    cards: list[KnowledgeCard]
    loaded_pack_paths: tuple[str, ...]
    external_cards_loaded: int
    external_cards_retrieved: int


def _card(
    card_id: str,
    card_type: str,
    labels: tuple[str, ...],
    title: str,
    summary: str,
    evidence_rules: tuple[str, ...],
    coaching_points: tuple[str, ...],
    *,
    allowed_interpretations: tuple[str, ...] = (),
    forbidden_claims: tuple[str, ...] = (),
    related_cards: tuple[str, ...] = (),
    source_kind: str = "builtin",
    source_path: str | None = None,
) -> KnowledgeCard:
    return KnowledgeCard(
        card_id=card_id,
        card_type=card_type,
        labels=labels,
        title=title,
        summary=summary,
        evidence_rules=evidence_rules,
        coaching_points=coaching_points,
        allowed_interpretations=allowed_interpretations,
        forbidden_claims=forbidden_claims,
        related_cards=related_cards,
        source_kind=source_kind,
        source_path=source_path,
    )


def _equipment_card(
    equipment: str,
    title: str,
    summary: str,
    coaching_point: str,
) -> KnowledgeCard:
    return _card(
        f"equipment:{equipment}",
        "equipment",
        (equipment,),
        title,
        summary,
        (
            "Treat equipment as context for cue wording, not as a new issue.",
        ),
        (coaching_point,),
    )


def _goal_overlay_card(
    exercise: str,
    goal: str,
    title: str,
    summary: str,
    coaching_points: tuple[str, ...],
) -> KnowledgeCard:
    return _card(
        f"goal_overlay:{exercise}:{goal}",
        "goal_overlay",
        (f"{exercise}:{goal}",),
        title,
        summary,
        (
            "Use this overlay only as a prioritization hint for the detected exercise and goal.",
        ),
        coaching_points,
    )


CARD_REGISTRY: tuple[KnowledgeCard, ...] = (
    _card(
        "exercise:squat",
        "exercise",
        ("squat",),
        "Squat",
        "A squat summary should describe depth, balance, and torso position "
        "from structured rep evidence.",
        (
            "Use rep analysis and issue markers instead of inferring directly from the video.",
            "Valid stance variations should not be framed as faults by default.",
        ),
        (
            "Call out depth, stance, and torso control only when they appear "
            "in structured evidence.",
            "Keep fixes simple and specific to the detected issue labels.",
        ),
    ),
    _card(
        "exercise:push_up",
        "exercise",
        ("push_up",),
        "Push-up",
        "A push-up summary should focus on body line, depth, and rep consistency "
        "from structured evidence.",
        (
            "Treat hand placement or knee support as variation context when the "
            "variation detector marks them as not-issues.",
            "Do not infer shoulder or wrist pain from the movement.",
        ),
        (
            "Explain whether the set looked controlled before suggesting changes.",
            "Use issue labels such as `hip_sag` or `incomplete_depth` only "
            "when they are present in JSON.",
        ),
    ),
    _card(
        "exercise:shoulder_press",
        "exercise",
        ("shoulder_press",),
        "Shoulder Press",
        "A shoulder press summary should focus on lockout, symmetry, and rep "
        "consistency from structured evidence.",
        (
            "Partial range can be a valid variation context and should not be "
            "automatically overcorrected.",
            "Use rep analysis and issue markers instead of diagnosing shoulder limitations.",
        ),
        (
            "Separate partial range context from incomplete lockout issue markers.",
            "Use `asymmetry` only when it is explicitly present in JSON.",
        ),
    ),
    _card(
        "variation:wide_grip_push_up",
        "variation",
        ("wide_grip_push_up", "wide_hand_placement"),
        "Wide-Grip Push-up",
        "A wide-grip push-up is a valid push-up variation when detected by the "
        "variation step.",
        (
            "If `wide_hand_placement` appears in not_issues, treat hand width "
            "as context, not a fault.",
        ),
        (
            "Acknowledge the wide-grip setup without asking the athlete to "
            "normalize it unless another issue requires it.",
        ),
        allowed_interpretations=("Variation, not automatically an issue.",),
    ),
    _card(
        "variation:knee_push_up",
        "variation",
        ("knee_push_up", "knee_contact"),
        "Knee Push-up",
        "A knee push-up is a valid push-up variation when knee support is intentionally detected.",
        (
            "If `knee_contact` appears in not_issues, do not correct knee support as an error.",
        ),
        (
            "Explain the movement as a valid regression or variation rather than a mistake.",
        ),
        allowed_interpretations=("Variation, not automatically an issue.",),
    ),
    _card(
        "issue:shallow_depth",
        "issue",
        ("shallow_depth",),
        "Shallow Depth",
        "The squat bottom position stayed above the expected depth threshold in the issue markers.",
        (
            "Only mention this issue when `shallow_depth` exists in `issue_markers.json`.",
        ),
        (
            "Sit slightly deeper before standing up.",
            "Slow the bottom portion so depth stays consistent.",
        ),
    ),
    _card(
        "issue:hip_sag",
        "issue",
        ("hip_sag",),
        "Hip Sag",
        "The push-up body line dropped below the body-line threshold across a sustained interval.",
        (
            "Only mention this issue when `hip_sag` exists in `issue_markers.json`.",
        ),
        (
            "Keep shoulders, hips, and ankles moving as one line.",
            "Reduce speed if body line drops on later reps.",
        ),
    ),
    _card(
        "issue:incomplete_lockout",
        "issue",
        ("incomplete_lockout",),
        "Incomplete Lockout",
        "The elbows did not reach the lockout threshold at the top of the shoulder press.",
        (
            "Only mention this issue when `incomplete_lockout` exists in `issue_markers.json`.",
        ),
        (
            "Finish each rep by reaching a cleaner top position.",
            "Use a slower press so the top range stays consistent.",
        ),
    ),
    _card(
        "issue:incomplete_depth",
        "issue",
        ("incomplete_depth",),
        "Incomplete Depth",
        "The push-up bottom position stayed above the depth threshold at the bottom of the rep.",
        ("Only mention this issue when it exists in `issue_markers.json`.",),
        (
            "Lower a bit more at the bottom if control stays clean.",
            "Use slower reps to make bottom depth repeatable.",
        ),
    ),
    _card(
        "issue:knee_valgus",
        "issue",
        ("knee_valgus",),
        "Knee Valgus",
        "The knees tracked inward relative to the ankles beyond the configured threshold.",
        ("Only mention this issue when it exists in `issue_markers.json`.",),
        (
            "Keep the knees tracking more evenly over the feet.",
            "Use a slightly slower descent so knee path stays consistent.",
        ),
    ),
    _card(
        "issue:excessive_torso_lean",
        "issue",
        ("excessive_torso_lean",),
        "Excessive Torso Lean",
        "The torso lean exceeded the configured threshold near the squat bottom.",
        ("Only mention this issue when it exists in `issue_markers.json`.",),
        (
            "Keep the chest taller through the bottom.",
            "Use a controlled descent so the torso angle stays steadier.",
        ),
    ),
    _card(
        "issue:asymmetry",
        "issue",
        ("asymmetry",),
        "Asymmetry",
        "The left-right wrist height difference exceeded the configured symmetry threshold.",
        ("Only mention this issue when it exists in `issue_markers.json`.",),
        (
            "Try to finish both sides at a more even height.",
            "Use a slower tempo to keep both arms in sync.",
        ),
    ),
    _card(
        "goal:strength",
        "goal",
        ("strength",),
        "Strength Goal",
        "Strength-oriented coaching should prioritize a few high-value fixes over many cues.",
        ("Keep the plan focused and repeatable.",),
        ("Use 1 to 2 form priorities for the next session.",),
    ),
    _card(
        "goal:hypertrophy",
        "goal",
        ("hypertrophy",),
        "Hypertrophy Goal",
        "Hypertrophy-oriented coaching should emphasize repeatable reps and manageable fixes.",
        ("Keep cues practical for multi-rep sets.",),
        ("Prioritize consistency over perfect-looking single reps.",),
    ),
    _card(
        "goal:endurance",
        "goal",
        ("endurance",),
        "Endurance Goal",
        "Endurance-oriented coaching should emphasize repeatability across the full set.",
        ("Call out late-set drift when the rep analysis shows it.",),
        ("Use pacing and consistency cues for the next session.",),
    ),
    _card(
        "goal:mobility",
        "goal",
        ("mobility",),
        "Mobility Goal",
        "Mobility-oriented coaching should stay descriptive and avoid medical claims.",
        ("Describe range findings without diagnosing restrictions.",),
        ("Use easy controlled reps next session to compare range consistency.",),
    ),
    _card(
        "goal:beginner_practice",
        "goal",
        ("beginner_practice",),
        "Beginner Practice Goal",
        "Beginner practice coaching should stay simple, encouraging, and concrete.",
        ("Limit the number of corrections in a single summary.",),
        ("Pick the top one or two form priorities for next time.",),
    ),
    _equipment_card(
        "bodyweight",
        "Bodyweight Context",
        "Bodyweight sessions should keep cues simple and focused on repeatable control.",
        "Prefer control and repeatability cues that do not assume external load adjustments.",
    ),
    _equipment_card(
        "dumbbell",
        "Dumbbell Context",
        "Dumbbell sessions can use load-management cues, but only as context rather than a requirement.",
        "If range or symmetry drifts, favor tempo or setup cues before suggesting heavier changes.",
    ),
    _equipment_card(
        "barbell",
        "Barbell Context",
        "Barbell sessions may benefit from setup and path-consistency cues, while still staying grounded in the evidence.",
        "Keep the coaching focused on bar path, balance, and repeatable setup when the evidence supports it.",
    ),
    _goal_overlay_card(
        "squat",
        "strength",
        "Squat Strength Overlay",
        "For strength-focused squats, prioritize the highest-value rep-quality cue rather than many small changes.",
        (
            "Lead with the one squat fix that most affects repeatable depth or balance.",
            "Keep the next-session plan short and repeatable.",
        ),
    ),
    _goal_overlay_card(
        "squat",
        "beginner_practice",
        "Squat Beginner Overlay",
        "For beginner-practice squats, keep the tone encouraging and corrections minimal.",
        (
            "Explain what looked stable before introducing the top squat fix.",
            "Use simple depth or balance language rather than stacking many cues.",
        ),
    ),
    _goal_overlay_card(
        "push_up",
        "beginner_practice",
        "Push-up Beginner Overlay",
        "For beginner-practice push-ups, prioritize body-line control and simple repeatable cues.",
        (
            "Keep push-up corrections to one or two cues that the athlete can repeat next set.",
            "Treat valid regressions or grip variations as context, not failure.",
        ),
    ),
    _goal_overlay_card(
        "push_up",
        "endurance",
        "Push-up Endurance Overlay",
        "For endurance-oriented push-ups, explain what changed later in the set before suggesting fixes.",
        (
            "Call out late-set body-line or depth drift when the rep evidence shows it.",
            "Use pacing and consistency cues for the next session plan.",
        ),
    ),
    _goal_overlay_card(
        "shoulder_press",
        "strength",
        "Shoulder Press Strength Overlay",
        "For strength-focused presses, prioritize clean top position and left-right consistency.",
        (
            "Lead with the top-position or symmetry cue that most affected repeatable reps.",
            "Keep the next-session plan focused on one or two overhead pressing priorities.",
        ),
    ),
    _goal_overlay_card(
        "shoulder_press",
        "mobility",
        "Shoulder Press Mobility Overlay",
        "For mobility-oriented presses, stay descriptive about range consistency without diagnosing restriction.",
        (
            "Describe the observed top-range consistency without speculating about limitations.",
            "Favor smooth controlled reps for the next-session comparison.",
        ),
    ),
    _card(
        "safety:no_diagnosis",
        "safety_rule",
        ("no_diagnosis",),
        "No Diagnosis",
        "The summary must not diagnose pain, injury, imbalance, mobility deficits, or pathology.",
        ("Do not use diagnostic language.",),
        ("Use uncertainty language when evidence is limited.",),
        forbidden_claims=("diagnosis", "injury", "pathology", "medical assessment"),
    ),
    _card(
        "safety:no_injury_prevention_claim",
        "safety_rule",
        ("no_injury_prevention_claim",),
        "No Injury Prevention Claim",
        "The summary must not claim that a cue will prevent injury.",
        ("Do not promise injury prevention.",),
        ("Keep coaching language descriptive and performance-focused.",),
        forbidden_claims=("injury prevention", "prevent injury"),
    ),
    _card(
        "safety:grounded_only",
        "safety_rule",
        ("grounded_only",),
        "Grounded Only",
        "The summary must explain only the structured evidence and retrieved knowledge cards.",
        ("Do not infer new issues that are absent from JSON.",),
        ("State confidence limits when the evidence is thin.",),
    ),
    _card(
        "safety:variation_not_issue",
        "safety_rule",
        ("variation_not_issue",),
        "Variation Is Not Automatically An Issue",
        "A detected variation or listed not-issue should not be overcorrected as a mistake.",
        ("Keep valid variation context separate from issue language.",),
        ("Explain why the variation is treated as context when needed.",),
    ),
)


DEFAULT_CARD_PACK_PATHS = (
    Path(__file__).resolve().parents[2] / "data/knowledge_cards/grounded_exercise_expansion.json",
)
CARD_PACKS_ENV = "SPOTTER_KNOWLEDGE_CARD_PACKS"


def _candidate_card_pack_paths() -> tuple[str, ...]:
    candidates: list[str] = []
    for path in DEFAULT_CARD_PACK_PATHS:
        if path.is_file():
            candidates.append(str(path.resolve()))

    configured = os.getenv(CARD_PACKS_ENV, "").strip()
    if not configured:
        return tuple(candidates)

    for raw_path in configured.split(os.pathsep):
        cleaned = raw_path.strip()
        if not cleaned:
            continue
        resolved = str(Path(cleaned).expanduser().resolve())
        if resolved not in candidates:
            candidates.append(resolved)
    return tuple(candidates)


def _normalize_string_tuple(values: object, field_name: str, source_path: str) -> tuple[str, ...]:
    if not isinstance(values, list):
        raise ValueError(f"{source_path}: {field_name} must be a list of strings")
    normalized: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{source_path}: {field_name}[{index}] must be a non-empty string"
            )
        normalized.append(value.strip())
    return tuple(normalized)


def _load_pack_card(payload: object, source_path: str) -> KnowledgeCard:
    if not isinstance(payload, dict):
        raise ValueError(f"{source_path}: each card must be an object")

    required = {
        "card_id",
        "card_type",
        "labels",
        "title",
        "summary",
        "evidence_rules",
        "coaching_points",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(
            f"{source_path}: card missing required field(s): {', '.join(missing)}"
        )

    card_id = payload["card_id"]
    card_type = payload["card_type"]
    title = payload["title"]
    summary = payload["summary"]
    if not isinstance(card_id, str) or not card_id.strip():
        raise ValueError(f"{source_path}: card_id must be a non-empty string")
    if not isinstance(card_type, str) or not card_type.strip():
        raise ValueError(f"{source_path}: card_type must be a non-empty string")
    if not isinstance(title, str) or not title.strip():
        raise ValueError(f"{source_path}: title must be a non-empty string")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError(f"{source_path}: summary must be a non-empty string")

    return _card(
        card_id=card_id.strip(),
        card_type=card_type.strip(),
        labels=_normalize_string_tuple(payload["labels"], "labels", source_path),
        title=title.strip(),
        summary=summary.strip(),
        evidence_rules=_normalize_string_tuple(
            payload["evidence_rules"], "evidence_rules", source_path
        ),
        coaching_points=_normalize_string_tuple(
            payload["coaching_points"], "coaching_points", source_path
        ),
        allowed_interpretations=_normalize_string_tuple(
            payload.get("allowed_interpretations", []),
            "allowed_interpretations",
            source_path,
        ),
        forbidden_claims=_normalize_string_tuple(
            payload.get("forbidden_claims", []),
            "forbidden_claims",
            source_path,
        ),
        related_cards=_normalize_string_tuple(
            payload.get("related_cards", []),
            "related_cards",
            source_path,
        ),
        source_kind="external",
        source_path=source_path,
    )


def _load_external_cards(path: str) -> tuple[KnowledgeCard, ...]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Knowledge card pack not found: {resolved}")

    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{resolved}: card pack must be a JSON object")
    cards_payload = payload.get("cards")
    if not isinstance(cards_payload, list):
        raise ValueError(f"{resolved}: `cards` must be a list")

    cards = [_load_pack_card(card_payload, str(resolved)) for card_payload in cards_payload]
    seen_ids: set[str] = set()
    for card in cards:
        if card.card_id in seen_ids:
            raise ValueError(f"{resolved}: duplicate card_id {card.card_id!r}")
        seen_ids.add(card.card_id)
    return tuple(cards)


def _build_cards_by_label(cards: tuple[KnowledgeCard, ...]) -> dict[str, KnowledgeCard]:
    cards_by_label: dict[str, KnowledgeCard] = {}
    for card in cards:
        for label in card.labels:
            existing = cards_by_label.get(label)
            if existing is not None and existing.card_id != card.card_id:
                raise ValueError(
                    f"Knowledge label conflict for {label!r}: "
                    f"{existing.card_id!r} vs {card.card_id!r}"
                )
            cards_by_label[label] = card
    return cards_by_label


@lru_cache(maxsize=8)
def _catalog_for_paths(pack_paths: tuple[str, ...]) -> KnowledgeCatalog:
    cards_by_id = {card.card_id: card for card in CARD_REGISTRY}
    for path in pack_paths:
        for card in _load_external_cards(path):
            cards_by_id[card.card_id] = card

    cards = tuple(
        sorted(
            cards_by_id.values(),
            key=lambda card: (CARD_TYPE_ORDER.get(card.card_type, 99), card.card_id),
        )
    )
    return KnowledgeCatalog(
        cards=cards,
        cards_by_id={card.card_id: card for card in cards},
        cards_by_label=_build_cards_by_label(cards),
        loaded_pack_paths=pack_paths,
    )


def get_catalog(pack_paths: tuple[str, ...] | None = None) -> KnowledgeCatalog:
    selected_paths = _candidate_card_pack_paths() if pack_paths is None else pack_paths
    return _catalog_for_paths(tuple(selected_paths))


def configured_card_pack_paths() -> tuple[str, ...]:
    return get_catalog().loaded_pack_paths


def clear_catalog_cache() -> None:
    _catalog_for_paths.cache_clear()


def _labels_for_card_type(card_type: str, *, pack_paths: tuple[str, ...] | None = None) -> frozenset[str]:
    catalog = get_catalog(pack_paths)
    return frozenset(
        label
        for card in catalog.cards
        if card.card_type == card_type
        for label in card.labels
    )


KNOWN_ISSUE_LABELS = _labels_for_card_type("issue")
KNOWN_VARIATION_LABELS = _labels_for_card_type("variation")


def known_issue_labels() -> frozenset[str]:
    return _labels_for_card_type("issue")


def known_variation_labels() -> frozenset[str]:
    return _labels_for_card_type("variation")


def get_card_by_label(label: str) -> KnowledgeCard | None:
    return get_catalog().cards_by_label.get(label)


def cards_for_labels(labels: list[str] | tuple[str, ...]) -> list[KnowledgeCard]:
    cards = {
        card.card_id: card
        for label in labels
        for card in [get_card_by_label(label)]
        if card is not None
    }
    return sorted(
        cards.values(),
        key=lambda card: (CARD_TYPE_ORDER.get(card.card_type, 99), card.card_id),
    )


def prioritized_coaching_points(cards: list[KnowledgeCard], *, limit: int = 6) -> list[str]:
    priority_order = {
        "issue": 0,
        "variation": 1,
        "exercise": 2,
        "equipment": 3,
        "goal_overlay": 4,
        "goal": 5,
        "safety_rule": 6,
    }
    ordered_cards = sorted(
        cards,
        key=lambda card: (
            priority_order.get(card.card_type, 99),
            0 if card.source_kind == "external" else 1,
            card.card_id,
        ),
    )
    points: list[str] = []
    seen: set[str] = set()
    for card in ordered_cards:
        for point in card.coaching_points:
            normalized = point.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            points.append(normalized)
            if len(points) >= limit:
                return points
    return points


def retrieve_cards_with_metadata(
    *,
    profile: UserProfile,
    classification: ExerciseClassification,
    variation: Variation,
    issues: IssueMarkers,
) -> KnowledgeRetrieval:
    cards = retrieve_cards(
        profile=profile,
        classification=classification,
        variation=variation,
        issues=issues,
    )
    catalog = get_catalog()
    return KnowledgeRetrieval(
        cards=cards,
        loaded_pack_paths=catalog.loaded_pack_paths,
        external_cards_loaded=sum(
            1 for card in catalog.cards if card.source_kind == "external"
        ),
        external_cards_retrieved=sum(
            1 for card in cards if card.source_kind == "external"
        ),
    )


def retrieve_cards(
    *,
    profile: UserProfile,
    classification: ExerciseClassification,
    variation: Variation,
    issues: IssueMarkers,
) -> list[KnowledgeCard]:
    labels = [
        classification.exercise,
        variation.detected_variation,
        *variation.not_issues,
        *[issue.issue for issue in issues.issues],
        profile.equipment,
        profile.goal,
        f"{classification.exercise}:{profile.goal}",
        f"{classification.exercise}:{profile.equipment}",
        "no_diagnosis",
        "no_injury_prevention_claim",
        "grounded_only",
        "variation_not_issue",
    ]
    return cards_for_labels(labels)


def supported_goal_labels() -> frozenset[str]:
    return frozenset(GOALS)
