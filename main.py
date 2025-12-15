import os
from pathlib import Path

import httpx
from openai import OpenAI


def generate_lgtm_image(client: OpenAI, prompt: str, output_path: str) -> None:
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    image_url = response.data[0].url
    image_data = httpx.get(image_url).content
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
    prompt_path = Path(__file__).parent / "PROMPT.md"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


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
