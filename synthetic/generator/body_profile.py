from dataclasses import asdict, dataclass
import random


BODY_SHAPES = ("slim", "average", "athletic", "curvy", "broad", "plus")


@dataclass(frozen=True)
class BodyProfile:
    sample_id: str
    height_cm: float
    weight_kg: float
    chest_cm: float
    waist_cm: float
    hip_cm: float
    shoulder_cm: float
    inseam_cm: float
    sleeve_cm: float
    neck_cm: float
    thigh_cm: float
    calf_cm: float
    body_shape: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def generate_body_profile(index: int, rng: random.Random) -> BodyProfile:
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
    chest_cm = _rounded(_bounded(rng.uniform(75, 130) * (1 + shape_bias * 0.35), 75, 130))
    waist_cm = _rounded(_bounded(rng.uniform(55, 125) * (1 + shape_bias * 0.45), 55, 125))
    hip_cm = _rounded(_bounded(rng.uniform(75, 135) * (1 + shape_bias * 0.4), 75, 135))
    shoulder_cm = _rounded(_bounded(rng.uniform(35, 60) * (1 + shape_bias * 0.18), 35, 60))
    inseam_cm = _rounded(_bounded(height_cm * rng.uniform(0.42, 0.49), 65, 95))
    sleeve_cm = _rounded(_bounded(height_cm * rng.uniform(0.31, 0.38), 50, 75))
    neck_cm = _rounded(_bounded(rng.uniform(30, 50) * (1 + shape_bias * 0.18), 30, 50))
    thigh_cm = _rounded(_bounded(rng.uniform(40, 80) * (1 + shape_bias * 0.32), 40, 80))
    calf_cm = _rounded(_bounded(rng.uniform(28, 55) * (1 + shape_bias * 0.24), 28, 55))

    return BodyProfile(
        sample_id=f"sample_{index:06d}",
        height_cm=height_cm,
        weight_kg=weight_kg,
        chest_cm=chest_cm,
        waist_cm=waist_cm,
        hip_cm=hip_cm,
        shoulder_cm=shoulder_cm,
        inseam_cm=inseam_cm,
        sleeve_cm=sleeve_cm,
        neck_cm=neck_cm,
        thigh_cm=thigh_cm,
        calf_cm=calf_cm,
        body_shape=body_shape,
    )


def _bounded(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _rounded(value: float) -> float:
    return round(value, 1)
