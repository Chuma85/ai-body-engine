from dataclasses import asdict, dataclass
import random

from training.measurements.measurement_targets import GENDER_MEASUREMENT_SCHEMA_VERSION, ProfileType


BODY_SHAPES = ("slim", "average", "athletic", "curvy", "broad", "plus")


@dataclass(frozen=True)
class BodyProfile:
    sample_id: str
    profile_type: str
    dataset_schema_version: str
    height_cm: float
    weight_kg: float
    chest_cm: float
    waist_cm: float
    hip_cm: float
    shoulder_cm: float
    shoulder_width_cm: float
    inseam_cm: float
    sleeve_cm: float
    sleeve_shoulder_to_wrist_cm: float
    neck_cm: float
    abdomen_cm: float
    stomach_cm: float
    outseam_cm: float
    bicep_cm: float
    forearm_cm: float
    wrist_cm: float
    thigh_cm: float
    knee_cm: float
    calf_cm: float
    ankle_cm: float
    jacket_length_cm: float | None
    trouser_rise_cm: float | None
    across_back_cm: float | None
    bust_cm: float | None
    high_bust_cm: float | None
    underbust_cm: float | None
    bust_point_to_bust_point_cm: float | None
    shoulder_to_bust_cm: float | None
    waist_to_hip_cm: float | None
    body_shape: str
    camera_angle_degrees: float
    camera_distance_m: float
    body_rotation_degrees: float
    phone_framing_offset_x: float
    phone_framing_offset_y: float
    phone_framing_scale: float

    def to_dict(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def generate_body_profile(index: int, rng: random.Random) -> BodyProfile:
    profile_type = ProfileType.MALE.value if index % 2 else ProfileType.FEMALE.value
    body_shape = rng.choice(BODY_SHAPES)
    shape_bias = {
        "slim": -0.12,
        "average": 0.0,
        "athletic": 0.06,
        "curvy": 0.09,
        "broad": 0.12,
        "plus": 0.18,
    }[body_shape]

    height_cm = _rounded(rng.uniform(150, 205))
    weight_kg = _rounded(_bounded(rng.uniform(45, 130) * (1 + shape_bias * 0.45), 45, 130))
    chest_base = rng.uniform(82, 128) if profile_type == ProfileType.MALE.value else rng.uniform(74, 118)
    chest_cm = _rounded(_bounded(chest_base * (1 + shape_bias * 0.35), 74, 130))
    waist_cm = _rounded(_bounded(rng.uniform(55, 125) * (1 + shape_bias * 0.45), 55, 125))
    hip_cm = _rounded(_bounded(rng.uniform(75, 135) * (1 + shape_bias * 0.4), 75, 135))
    shoulder_cm = _rounded(_bounded(rng.uniform(35, 60) * (1 + shape_bias * 0.18), 35, 60))
    inseam_cm = _rounded(_bounded(height_cm * rng.uniform(0.42, 0.49), 65, 95))
    sleeve_cm = _rounded(_bounded(height_cm * rng.uniform(0.31, 0.38), 50, 75))
    sleeve_shoulder_to_wrist_cm = _rounded(_bounded(height_cm * rng.uniform(0.34, 0.42), 52, 82))
    neck_cm = _rounded(_bounded(rng.uniform(30, 50) * (1 + shape_bias * 0.18), 30, 50))
    abdomen_cm = _rounded(_bounded((waist_cm * rng.uniform(1.02, 1.14)) + shape_bias * 8, 58, 140))
    stomach_cm = _rounded(_bounded((waist_cm * rng.uniform(1.00, 1.18)) + shape_bias * 9, 58, 145))
    outseam_cm = _rounded(_bounded(inseam_cm + rng.uniform(18, 30), 85, 125))
    bicep_cm = _rounded(_bounded(rng.uniform(24, 45) * (1 + shape_bias * 0.24), 22, 48))
    forearm_cm = _rounded(_bounded(rng.uniform(20, 36) * (1 + shape_bias * 0.16), 18, 38))
    wrist_cm = _rounded(_bounded(rng.uniform(14, 22) * (1 + shape_bias * 0.08), 13, 24))
    thigh_cm = _rounded(_bounded(rng.uniform(40, 80) * (1 + shape_bias * 0.32), 40, 80))
    knee_cm = _rounded(_bounded(rng.uniform(32, 50) * (1 + shape_bias * 0.14), 30, 55))
    calf_cm = _rounded(_bounded(rng.uniform(28, 55) * (1 + shape_bias * 0.24), 28, 55))
    ankle_cm = _rounded(_bounded(rng.uniform(19, 31) * (1 + shape_bias * 0.10), 18, 33))

    male_fields = {
        "jacket_length_cm": _rounded(_bounded(height_cm * rng.uniform(0.39, 0.47), 58, 88)),
        "trouser_rise_cm": _rounded(_bounded(rng.uniform(24, 36) * (1 + shape_bias * 0.08), 22, 40)),
        "across_back_cm": _rounded(_bounded((shoulder_cm * rng.uniform(0.72, 0.84)), 30, 54)),
    } if profile_type == ProfileType.MALE.value else {"jacket_length_cm": None, "trouser_rise_cm": None, "across_back_cm": None}
    bust_cm = _rounded(_bounded(chest_cm + rng.uniform(4, 16), 78, 142)) if profile_type == ProfileType.FEMALE.value else None
    female_fields = {
        "bust_cm": bust_cm,
        "high_bust_cm": _rounded(_bounded((bust_cm or chest_cm) - rng.uniform(2, 8), 74, 136)) if bust_cm is not None else None,
        "underbust_cm": _rounded(_bounded((bust_cm or chest_cm) - rng.uniform(8, 18), 62, 124)) if bust_cm is not None else None,
        "bust_point_to_bust_point_cm": _rounded(_bounded(rng.uniform(16, 28) * (1 + shape_bias * 0.12), 14, 32)) if bust_cm is not None else None,
        "shoulder_to_bust_cm": _rounded(_bounded(height_cm * rng.uniform(0.13, 0.18), 19, 34)) if bust_cm is not None else None,
        "waist_to_hip_cm": _rounded(_bounded(height_cm * rng.uniform(0.10, 0.16), 15, 30)) if bust_cm is not None else None,
    }

    return BodyProfile(
        sample_id=f"sample_{index:06d}",
        profile_type=profile_type,
        dataset_schema_version=GENDER_MEASUREMENT_SCHEMA_VERSION,
        height_cm=height_cm,
        weight_kg=weight_kg,
        chest_cm=chest_cm,
        waist_cm=waist_cm,
        hip_cm=hip_cm,
        shoulder_cm=shoulder_cm,
        shoulder_width_cm=shoulder_cm,
        inseam_cm=inseam_cm,
        sleeve_cm=sleeve_cm,
        sleeve_shoulder_to_wrist_cm=sleeve_shoulder_to_wrist_cm,
        neck_cm=neck_cm,
        abdomen_cm=abdomen_cm,
        stomach_cm=stomach_cm,
        outseam_cm=outseam_cm,
        bicep_cm=bicep_cm,
        forearm_cm=forearm_cm,
        wrist_cm=wrist_cm,
        thigh_cm=thigh_cm,
        knee_cm=knee_cm,
        calf_cm=calf_cm,
        ankle_cm=ankle_cm,
        **male_fields,
        **female_fields,
        body_shape=body_shape,
        camera_angle_degrees=_rounded(rng.uniform(-7.5, 7.5)),
        camera_distance_m=_rounded(rng.uniform(1.6, 2.8)),
        body_rotation_degrees=_rounded(rng.uniform(-12, 12)),
        phone_framing_offset_x=_rounded(rng.uniform(-0.10, 0.10)),
        phone_framing_offset_y=_rounded(rng.uniform(-0.08, 0.08)),
        phone_framing_scale=_rounded(rng.uniform(0.92, 1.08)),
    )


def _bounded(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _rounded(value: float) -> float:
    return round(value, 1)
