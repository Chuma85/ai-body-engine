from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import re
from typing import Any


GENDER_MEASUREMENT_SCHEMA_VERSION = "gender_measurement_schema_v1"


class ProfileType(str, Enum):
    MALE = "male"
    FEMALE = "female"
    UNISEX = "unisex"
    UNSPECIFIED = "unspecified"


SHARED_PROFILE_TYPES = (
    ProfileType.MALE.value,
    ProfileType.FEMALE.value,
    ProfileType.UNISEX.value,
    ProfileType.UNSPECIFIED.value,
)


@dataclass(frozen=True)
class MeasurementTargetDefinition:
    name: str
    label: str
    profile_types: tuple[str, ...]
    category: str
    required_for_profile: bool = False

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def _humanize_target(target: str) -> str:
    token = target.removesuffix("_cm").removesuffix("_kg")
    return token.replace("_", " ").title()


SHARED_MEASUREMENT_TARGETS = (
    "height_cm",
    "weight_kg",
    "neck_cm",
    "shoulder_cm",
    "shoulder_width_cm",
    "chest_cm",
    "waist_cm",
    "abdomen_cm",
    "stomach_cm",
    "hip_cm",
    "inseam_cm",
    "outseam_cm",
    "sleeve_cm",
    "sleeve_shoulder_to_wrist_cm",
    "bicep_cm",
    "forearm_cm",
    "wrist_cm",
    "thigh_cm",
    "knee_cm",
    "calf_cm",
    "ankle_cm",
)

MALE_MEASUREMENT_TARGETS = (
    "jacket_length_cm",
    "trouser_rise_cm",
    "across_back_cm",
)

FEMALE_MEASUREMENT_TARGETS = (
    "bust_cm",
    "high_bust_cm",
    "underbust_cm",
    "bust_point_to_bust_point_cm",
    "shoulder_to_bust_cm",
    "waist_to_hip_cm",
)

DEFAULT_TRAINING_TARGET_COLUMNS = (
    "chest_cm",
    "waist_cm",
    "hip_cm",
    "shoulder_cm",
    "sleeve_cm",
    "sleeve_shoulder_to_wrist_cm",
    "inseam_cm",
    "neck_cm",
    "abdomen_cm",
    "stomach_cm",
    "wrist_cm",
    "thigh_cm",
    "knee_cm",
    "calf_cm",
    "ankle_cm",
)

TARGET_DEFINITIONS: dict[str, MeasurementTargetDefinition] = {
    **{
        target: MeasurementTargetDefinition(
            name=target,
            label=_humanize_target(target),
            profile_types=SHARED_PROFILE_TYPES,
            category="shared",
            required_for_profile=target in DEFAULT_TRAINING_TARGET_COLUMNS,
        )
        for target in SHARED_MEASUREMENT_TARGETS
    },
    **{
        target: MeasurementTargetDefinition(
            name=target,
            label=_humanize_target(target),
            profile_types=(ProfileType.MALE.value,),
            category="male_specific",
        )
        for target in MALE_MEASUREMENT_TARGETS
    },
    **{
        target: MeasurementTargetDefinition(
            name=target,
            label=_humanize_target(target),
            profile_types=(ProfileType.FEMALE.value,),
            category="female_specific",
        )
        for target in FEMALE_MEASUREMENT_TARGETS
    },
}

ALL_MEASUREMENT_TARGETS = tuple(TARGET_DEFINITIONS)
SUPPORTED_TARGETS = ALL_MEASUREMENT_TARGETS


def normalize_profile_type(value: Any) -> str:
    if isinstance(value, ProfileType):
        return value.value
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "m": ProfileType.MALE.value,
        "man": ProfileType.MALE.value,
        "men": ProfileType.MALE.value,
        "male": ProfileType.MALE.value,
        "masculine": ProfileType.MALE.value,
        "f": ProfileType.FEMALE.value,
        "woman": ProfileType.FEMALE.value,
        "women": ProfileType.FEMALE.value,
        "female": ProfileType.FEMALE.value,
        "feminine": ProfileType.FEMALE.value,
        "unisex": ProfileType.UNISEX.value,
        "unspecified": ProfileType.UNSPECIFIED.value,
        "unknown": ProfileType.UNSPECIFIED.value,
        "not_specified": ProfileType.UNSPECIFIED.value,
        "prefer_not_to_say": ProfileType.UNSPECIFIED.value,
        "": ProfileType.UNSPECIFIED.value,
    }
    if token not in aliases:
        raise ValueError(f"Unsupported profileType '{value}'. Expected male, female, unisex, or unspecified.")
    return aliases[token]


def profile_type_from_payload(payload: dict[str, Any] | None) -> str:
    if not payload:
        return ProfileType.UNSPECIFIED.value
    for key in ("profile_type", "profileType", "body_profile_type", "bodyProfileType", "gender", "sex"):
        value = payload.get(key)
        if value not in ("", None):
            return normalize_profile_type(value)
    return ProfileType.UNSPECIFIED.value


def normalize_target_name(target: str) -> str:
    token = re.sub(r"(?<!^)(?=[A-Z])", "_", str(target).strip())
    token = token.replace("-", "_").replace(" ", "_").lower()
    if token in TARGET_DEFINITIONS:
        return token
    if not token.endswith("_cm") and f"{token}_cm" in TARGET_DEFINITIONS:
        return f"{token}_cm"
    return token


def target_available_for_profile(target: str, profile_type: str | ProfileType | None) -> bool:
    normalized_target = normalize_target_name(target)
    definition = TARGET_DEFINITIONS.get(normalized_target)
    if definition is None:
        return True
    normalized_profile = normalize_profile_type(profile_type)
    if normalized_profile == ProfileType.UNSPECIFIED.value:
        return definition.category == "shared"
    return normalized_profile in definition.profile_types


def targets_for_profile(profile_type: str | ProfileType | None) -> list[str]:
    return [
        target
        for target in ALL_MEASUREMENT_TARGETS
        if target_available_for_profile(target, profile_type)
    ]


def unavailable_targets_for_profile(targets: list[str] | tuple[str, ...], profile_type: str | ProfileType | None) -> list[str]:
    return [
        normalize_target_name(target)
        for target in targets
        if not target_available_for_profile(target, profile_type)
    ]


def target_availability_payload(profile_type: str | ProfileType | None) -> dict[str, Any]:
    normalized_profile = normalize_profile_type(profile_type)
    available = targets_for_profile(normalized_profile)
    return {
        "schemaVersion": GENDER_MEASUREMENT_SCHEMA_VERSION,
        "profileType": normalized_profile,
        "availableTargets": available,
        "unavailableTargets": [target for target in ALL_MEASUREMENT_TARGETS if target not in available],
        "definitions": [TARGET_DEFINITIONS[target].to_payload() for target in ALL_MEASUREMENT_TARGETS],
    }
