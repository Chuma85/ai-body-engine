from pathlib import Path

from PIL import Image, ImageDraw

from synthetic.generator.body_profile import BodyProfile


def render_front_silhouette(profile: BodyProfile, output_path: str, width: int, height: int) -> None:
    image, draw = _canvas(width, height)
    cx = width // 2
    top = int(height * 0.07)
    bottom = int(height * 0.94)
    body_height = bottom - top

    shoulder_w = _scale(profile.shoulder_cm, 35, 60, width * 0.20, width * 0.38)
    chest_w = _scale(profile.chest_cm, 75, 130, width * 0.18, width * 0.36)
    waist_w = _scale(profile.waist_cm, 55, 125, width * 0.12, width * 0.31)
    hip_w = _scale(profile.hip_cm, 75, 135, width * 0.18, width * 0.37)
    thigh_w = _scale(profile.thigh_cm, 40, 80, width * 0.07, width * 0.14)
    calf_w = _scale(profile.calf_cm, 28, 55, width * 0.045, width * 0.095)

    head_r = int(body_height * 0.055)
    neck_w = int(_scale(profile.neck_cm, 30, 50, width * 0.04, width * 0.08))
    head_cy = top + head_r
    neck_y = top + int(body_height * 0.13)
    shoulder_y = top + int(body_height * 0.18)
    chest_y = top + int(body_height * 0.31)
    waist_y = top + int(body_height * 0.46)
    hip_y = top + int(body_height * 0.58)
    knee_y = top + int(body_height * 0.79)
    ankle_y = bottom - int(body_height * 0.03)

    draw.ellipse((cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r), fill="black")
    draw.rounded_rectangle((cx - neck_w // 2, neck_y, cx + neck_w // 2, shoulder_y + 8), radius=8, fill="black")

    torso = [
        (cx - shoulder_w / 2, shoulder_y),
        (cx - chest_w / 2, chest_y),
        (cx - waist_w / 2, waist_y),
        (cx - hip_w / 2, hip_y),
        (cx + hip_w / 2, hip_y),
        (cx + waist_w / 2, waist_y),
        (cx + chest_w / 2, chest_y),
        (cx + shoulder_w / 2, shoulder_y),
    ]
    draw.polygon(torso, fill="black")

    arm_top_y = shoulder_y + 8
    wrist_y = top + int(body_height * 0.58)
    arm_w = max(12, int(width * 0.035))
    for side in (-1, 1):
        shoulder_x = cx + side * (shoulder_w / 2)
        wrist_x = cx + side * (hip_w / 2 + width * 0.09)
        draw.line((shoulder_x, arm_top_y, wrist_x, wrist_y), fill="black", width=arm_w)
        draw.ellipse((wrist_x - arm_w / 2, wrist_y - arm_w / 2, wrist_x + arm_w / 2, wrist_y + arm_w / 2), fill="black")

    gap = int(width * 0.025)
    for side in (-1, 1):
        hip_inner = cx + side * gap
        hip_outer = cx + side * (hip_w / 2 * 0.78)
        knee_inner = cx + side * gap
        knee_outer = cx + side * thigh_w
        ankle_inner = cx + side * (gap * 0.7)
        ankle_outer = cx + side * calf_w
        leg = [(hip_inner, hip_y), (hip_outer, hip_y), (knee_outer, knee_y), (ankle_outer, ankle_y), (ankle_inner, ankle_y), (knee_inner, knee_y)]
        draw.polygon(leg, fill="black")
        foot_w = int(width * 0.08)
        draw.ellipse((ankle_inner - foot_w / 2, ankle_y - 5, ankle_inner + foot_w / 2, ankle_y + 14), fill="black")

    _save(image, output_path)


def render_side_silhouette(profile: BodyProfile, output_path: str, width: int, height: int) -> None:
    image, draw = _canvas(width, height)
    cx = width // 2
    top = int(height * 0.07)
    bottom = int(height * 0.94)
    body_height = bottom - top

    depth = _scale((profile.chest_cm + profile.waist_cm + profile.hip_cm) / 3, 68, 130, width * 0.11, width * 0.23)
    hip_depth = _scale(profile.hip_cm, 75, 135, width * 0.13, width * 0.26)
    shoulder_depth = _scale(profile.shoulder_cm, 35, 60, width * 0.09, width * 0.17)
    leg_depth = _scale(profile.thigh_cm, 40, 80, width * 0.07, width * 0.13)

    head_r = int(body_height * 0.055)
    head_cy = top + head_r
    neck_y = top + int(body_height * 0.13)
    shoulder_y = top + int(body_height * 0.19)
    chest_y = top + int(body_height * 0.32)
    waist_y = top + int(body_height * 0.47)
    hip_y = top + int(body_height * 0.59)
    knee_y = top + int(body_height * 0.79)
    ankle_y = bottom - int(body_height * 0.03)
    back_x = cx - depth * 0.42

    draw.ellipse((cx - head_r * 0.75, head_cy - head_r, cx + head_r * 0.95, head_cy + head_r), fill="black")
    draw.rounded_rectangle((cx - depth * 0.18, neck_y, cx + depth * 0.1, shoulder_y + 10), radius=8, fill="black")

    torso = [
        (back_x, shoulder_y),
        (cx + shoulder_depth * 0.8, shoulder_y + 10),
        (cx + depth, chest_y),
        (cx + depth * 0.72, waist_y),
        (cx + hip_depth, hip_y),
        (back_x - hip_depth * 0.15, hip_y),
        (back_x - depth * 0.12, waist_y),
        (back_x - depth * 0.08, chest_y),
    ]
    draw.polygon(torso, fill="black")

    arm_x = cx + depth * 0.65
    draw.line((arm_x, shoulder_y + 18, arm_x + width * 0.035, hip_y + 5), fill="black", width=max(12, int(width * 0.032)))

    leg = [
        (cx - leg_depth * 0.55, hip_y),
        (cx + leg_depth * 0.85, hip_y),
        (cx + leg_depth * 0.6, knee_y),
        (cx + leg_depth * 0.42, ankle_y),
        (cx - leg_depth * 0.35, ankle_y),
        (cx - leg_depth * 0.28, knee_y),
    ]
    draw.polygon(leg, fill="black")
    draw.ellipse((cx - leg_depth * 0.25, ankle_y - 5, cx + leg_depth * 1.25, ankle_y + 14), fill="black")

    _save(image, output_path)


def render_back_silhouette(profile: BodyProfile, output_path: str, width: int, height: int) -> None:
    image, draw = _canvas(width, height)
    cx = width // 2
    top = int(height * 0.07)
    bottom = int(height * 0.94)
    body_height = bottom - top

    shoulder_w = _scale(profile.shoulder_cm, 35, 60, width * 0.22, width * 0.40)
    across_back_w = _scale((profile.shoulder_cm * 0.72) + (profile.chest_cm * 0.18), 39, 66, width * 0.19, width * 0.36)
    waist_w = _scale(profile.waist_cm, 55, 125, width * 0.12, width * 0.30)
    hip_w = _scale(profile.hip_cm, 75, 135, width * 0.18, width * 0.37)
    thigh_w = _scale(profile.thigh_cm, 40, 80, width * 0.07, width * 0.14)
    calf_w = _scale(profile.calf_cm, 28, 55, width * 0.045, width * 0.095)

    head_r = int(body_height * 0.055)
    neck_w = int(_scale(profile.neck_cm, 30, 50, width * 0.04, width * 0.08))
    head_cy = top + head_r
    neck_y = top + int(body_height * 0.13)
    shoulder_y = top + int(body_height * 0.18)
    upper_back_y = top + int(body_height * 0.28)
    waist_y = top + int(body_height * 0.46)
    hip_y = top + int(body_height * 0.58)
    knee_y = top + int(body_height * 0.79)
    ankle_y = bottom - int(body_height * 0.03)

    draw.ellipse((cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r), fill="black")
    draw.rounded_rectangle((cx - neck_w // 2, neck_y, cx + neck_w // 2, shoulder_y + 8), radius=8, fill="black")
    torso = [
        (cx - shoulder_w / 2, shoulder_y),
        (cx - across_back_w / 2, upper_back_y),
        (cx - waist_w / 2, waist_y),
        (cx - hip_w / 2, hip_y),
        (cx + hip_w / 2, hip_y),
        (cx + waist_w / 2, waist_y),
        (cx + across_back_w / 2, upper_back_y),
        (cx + shoulder_w / 2, shoulder_y),
    ]
    draw.polygon(torso, fill="black")

    arm_w = max(12, int(width * 0.035))
    for side in (-1, 1):
        shoulder_x = cx + side * (shoulder_w / 2)
        wrist_x = cx + side * (hip_w / 2 + width * 0.075)
        wrist_y = top + int(body_height * 0.58)
        draw.line((shoulder_x, shoulder_y + 8, wrist_x, wrist_y), fill="black", width=arm_w)
        draw.ellipse((wrist_x - arm_w / 2, wrist_y - arm_w / 2, wrist_x + arm_w / 2, wrist_y + arm_w / 2), fill="black")

    gap = int(width * 0.025)
    for side in (-1, 1):
        hip_inner = cx + side * gap
        hip_outer = cx + side * (hip_w / 2 * 0.78)
        knee_inner = cx + side * gap
        knee_outer = cx + side * thigh_w
        ankle_inner = cx + side * (gap * 0.7)
        ankle_outer = cx + side * calf_w
        leg = [(hip_inner, hip_y), (hip_outer, hip_y), (knee_outer, knee_y), (ankle_outer, ankle_y), (ankle_inner, ankle_y), (knee_inner, knee_y)]
        draw.polygon(leg, fill="black")

    _save(image, output_path)


def _canvas(width: int, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), "white")
    return image, ImageDraw.Draw(image)


def _scale(value: float, source_min: float, source_max: float, target_min: float, target_max: float) -> float:
    ratio = (value - source_min) / (source_max - source_min)
    ratio = max(0, min(1, ratio))
    return target_min + ratio * (target_max - target_min)


def _save(image: Image.Image, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
