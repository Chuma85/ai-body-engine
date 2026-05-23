REQUIRED_MEASUREMENT_COLUMNS = [
    "sample_id",
    "front_image_path",
    "side_image_path",
    "height_cm",
    "weight_kg",
    "chest_cm",
    "waist_cm",
    "hip_cm",
    "shoulder_cm",
    "inseam_cm",
    "sleeve_cm",
    "neck_cm",
    "thigh_cm",
    "calf_cm",
    "body_shape",
    "generator_version",
]

OPTIONAL_METADATA_COLUMNS = [
    "skin_tone_id",
    "pose_variation_degrees",
    "camera_distance",
    "camera_focal_length",
    "render_width",
    "render_height",
    "anatomy_version",
    "renderer_mode",
    "base_mesh_asset",
    "mesh_deformation_enabled",
    "fallback_used",
    "deformation_mode",
    "deformation_strength",
    "preserve_height",
    "auto_frame_body",
    "camera_margin",
]


def validate_measurement_row(row: dict) -> bool:
    return all(column in row and row[column] not in ("", None) for column in REQUIRED_MEASUREMENT_COLUMNS)
