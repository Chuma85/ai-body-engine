def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def normalize_measurement_to_unit(value: float, reference: float, min_value: float, max_value: float) -> float:
    if max_value <= min_value:
        raise ValueError("max_value must be greater than min_value")

    normalized = (value - min_value) / (max_value - min_value)
    reference_normalized = (reference - min_value) / (max_value - min_value)
    centered = 0.5 + (normalized - reference_normalized)
    return clamp(centered, 0.0, 1.0)


def compute_shape_key_targets(params: dict, shape_key_mapping: dict[str, list[str]]) -> dict[str, float]:
    targets = {
        "height": normalize_measurement_to_unit(params.get("height_cm", 175), 175, 150, 205),
        "weight": normalize_measurement_to_unit(params.get("weight_kg", 75), 75, 45, 130),
        "chest": normalize_measurement_to_unit(params.get("chest_cm", 100), 100, 75, 130),
        "waist": normalize_measurement_to_unit(params.get("waist_cm", 82), 82, 55, 125),
        "hips": normalize_measurement_to_unit(params.get("hip_cm", 98), 98, 75, 135),
        "shoulders": normalize_measurement_to_unit(params.get("shoulder_cm", 45), 45, 35, 60),
        "muscle": 0.8 if params.get("body_shape") == "athletic" else 0.35,
        "body_fat": normalize_measurement_to_unit(params.get("weight_kg", 75), 75, 45, 130),
    }

    return {key: clamp(targets[key], 0.0, 1.0) for key in shape_key_mapping if key in targets}
