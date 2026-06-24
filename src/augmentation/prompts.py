"""Prompt presets for diffusion-based UAV augmentation."""

PROMPT_PRESETS = {
    "night_lowlight": (
        "realistic aerial image at night, low-light conditions, same scene layout, "
        "preserve object positions, preserve the drone appearance"
    ),
    "night_cloudy": (
        "realistic aerial drone-detection image at night under cloudy sky, dim ambient light, "
        "same camera viewpoint, preserve object positions and drone shape"
    ),
    "foggy_morning": (
        "realistic aerial image in foggy early morning light, soft haze, same scene layout, "
        "preserve object positions, preserve the drone appearance"
    ),
}

NEGATIVE_PROMPT = (
    "extra drone, duplicated drone, missing drone, deformed drone, changed object position, "
    "text, watermark, logo, strong blur on the main drone object, distorted bounding box, "
    "cartoon, painting"
)


def build_positive_prompt(preset: str, extra_prompt: str | None = None) -> str:
    """Return a final positive prompt from a preset plus optional extra wording."""
    if preset not in PROMPT_PRESETS:
        valid = ", ".join(sorted(PROMPT_PRESETS))
        raise KeyError(f"Unknown prompt preset '{preset}'. Valid presets: {valid}")

    prompt = PROMPT_PRESETS[preset]
    if extra_prompt:
        prompt = f"{prompt}, {extra_prompt.strip()}"
    return prompt
