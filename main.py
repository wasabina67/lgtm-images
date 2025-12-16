import base64
import json
import os
import random
from pathlib import Path

from openai import OpenAI


def generate_lgtm_image(client: OpenAI, prompt: str, output_path: str) -> None:
    response = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        n=1,
        size="1024x1024",
    )
    b64_json = response.data[0].b64_json
    image_data = base64.b64decode(b64_json)
    with open(output_path, "wb") as f:
        f.write(image_data)


def generate_output_path() -> Path:
    images_dir = Path(__file__).parent / "docs" / "images"
    existing_files = list(images_dir.glob("*.png"))

    if not existing_files:
        return images_dir / "1.png"

    numbers = []
    for file in existing_files:
        if file.stem.isdigit():
            numbers.append(int(file.stem))

    next_number = max(numbers) + 1 if numbers else 1
    return images_dir / f"{next_number}.png"


def load_prompt() -> str:
    variations_path = Path(__file__).parent / "variations.json"
    with open(variations_path, encoding="utf-8") as f:
        variations = json.load(f)

    character = random.choice(variations["characters"])
    background = random.choice(variations["backgrounds"])

    return f"""High-quality square-format anime illustration.
Soft lighting, clear details, natural shading.
Warm and gentle color harmony.

Character:
{character}

Background:
{background}

Camera framing:
medium-full-body shot, centered composition.
Ensure the character remains clearly visible and unobstructed.

Large, bold, sans-serif white text "LGTM" centered across the image.
Thick, modern, minimalistic font style.
Below it, small thin-weight white text "Looks Good To Me"
with wide letter spacing, centered horizontally.
Clean, flat overlay design that doesn't block key character features."""


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set")
        return

    client = OpenAI(api_key=api_key)
    prompt = load_prompt()
    output_path = generate_output_path()

    generate_lgtm_image(client, prompt, str(output_path))


if __name__ == "__main__":
    main()
