from pathlib import Path
from PIL import Image
import numpy as np

from scaleio import Upscaler


def upscale_single_image():
    upscaler = Upscaler(scale=4, model="general", device="auto")

    result = upscaler.upscale("input.jpg", "output_upscaled.png")
    print(f"Upscaled image saved to output_upscaled.png")
    return result


def upscale_with_pil_image():
    upscaler = Upscaler(scale=2, model="anime", device="auto")

    img = Image.open("input.jpg")

    result = upscaler.upscale(img)

    result.save("anime_upscaled.png")
    print(f"Anime model upscaled image saved to anime_upscaled.png")


def upscale_numpy_array():
    upscaler = Upscaler(scale=4, device="cpu")

    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    result = upscaler.upscale(arr)

    result.save("random_array_upscaled.png")
    print(f"NumPy array upscaled saved to random_array_upscaled.png")


def batch_upscale():
    upscaler = Upscaler(scale=4, model="general", tile=512)

    input_dir = Path("images")
    output_dir = Path("upscaled")

    output_paths = upscaler.upscale_batch(list(input_dir.glob("*.jpg")), output_dir, suffix="_4x")

    print(f"Upscaled {len(output_paths)} images")


def main():
    print("Scaleio - AI Image Upscaler Examples")
    print("=" * 40)

    print("\n1. Upscale single image from file...")
    try:
        upscale_single_image()
    except Exception as e:
        print(f"  Skipped: {e}")

    print("\n2. Upscale with PIL Image...")
    try:
        upscale_with_pil_image()
    except Exception as e:
        print(f"  Skipped: {e}")

    print("\n3. Upscale NumPy array...")
    try:
        upscale_numpy_array()
    except Exception as e:
        print(f"  Skipped: {e}")

    print("\n4. Batch upscale...")
    try:
        batch_upscale()
    except Exception as e:
        print(f"  Skipped: {e}")


if __name__ == "__main__":
    main()
