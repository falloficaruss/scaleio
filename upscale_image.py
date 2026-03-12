#!/usr/bin/env python3
"""
Local Image Upscaler using RealESRGAN (NCNN version)
A completely offline image upscaling tool that runs locally on your machine.
Uses pre-compiled RealESRGAN-ncnn-vulkan binaries.
"""

import os
import sys
import platform
import argparse
import subprocess
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm
import urllib.request
import tempfile


BINARY_URLS = {
    "windows": {
        "url": "https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/v20220728/realesrgan-ncnn-vulkan-20220728-windows.zip",
        "binary": "realesrgan-ncnn-vulkan.exe",
    },
    "linux": {
        "url": "https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/v20220728/realesrgan-ncnn-vulkan-20220728-linux.zip",
        "binary": "realesrgan-ncnn-vulkan",
    },
    "darwin": {
        "url": "https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/v20220728/realesrgan-ncnn-vulkan-20220728-mac.zip",
        "binary": "realesrgan-ncnn-vulkan",
    },
}


class RealESRGANUpscaler:
    def __init__(self, model_name="realesrgan-x4plus", scale_factor=4, gpuid=0):
        self.model_name = model_name
        self.scale_factor = scale_factor
        self.gpuid = gpuid
        self.binary_path = None
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)

    def get_platform(self):
        system = platform.system().lower()
        if system == "windows":
            return "windows"
        elif system == "linux":
            return "linux"
        elif system == "darwin":
            return "darwin"
        raise RuntimeError(f"Unsupported platform: {system}")

    def download_binary(self):
        platform_info = self.get_platform()
        binary_info = BINARY_URLS[platform_info]

        binary_dir = Path("bin")
        binary_dir.mkdir(exist_ok=True)

        binary_path = binary_dir / binary_info["binary"]

        if binary_path.exists():
            print(f"Binary already exists at {binary_path}")
            self.binary_path = binary_path
            return

        zip_path = (
            Path(tempfile.gettempdir()) / f"realesrgan-ncnn-vulkan-{platform_info}.zip"
        )

        if not zip_path.exists():
            print(f"Downloading RealESRGAN-ncnn-vulkan binary...")
            urllib.request.urlretrieve(binary_info["url"], zip_path)

        print("Extracting binary...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for member in zip_ref.namelist():
                if binary_info["binary"] in member:
                    source = zip_ref.open(member)
                    target = open(binary_path, "wb")
                    with source, target:
                        shutil.copyfileobj(source, target)
                    break

        binary_path.chmod(binary_path.stat().st_mode | 0o111)
        self.binary_path = binary_path
        print(f"Binary extracted to: {binary_path}")

    def download_models(self):
        models_dir = self.model_dir / "ncnn-models"
        models_dir.mkdir(exist_ok=True)

        base_url = "https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/models"

        models = [
            ("realesrgan-x4plus", "realesrgan-x4plus", "4x"),
            ("realesrgan-x4plus-anime", "realesrgan-x4plus-anime", "4x"),
            ("realesr-animevideov3-x2", "realesr-animevideov3-x2", "2x"),
            ("realesr-animevideov3-x3", "realesr-animevideov3-x3", "3x"),
            ("realesr-animevideov3-x4", "realesr-animevideov3-x4", "4x"),
        ]

        for model_id, model_file, scale in models:
            param_path = models_dir / f"{model_file}.param"
            bin_path = models_dir / f"{model_file}.bin"

            if param_path.exists() and bin_path.exists():
                continue

            print(f"Downloading model: {model_id}...")

            param_url = f"{base_url}/{model_file}.param"
            bin_url = f"{base_url}/{model_file}.bin"

            try:
                urllib.request.urlretrieve(param_url, param_path)
                urllib.request.urlretrieve(bin_url, bin_path)
            except Exception as e:
                if param_path.exists():
                    param_path.unlink()
                if bin_path.exists():
                    bin_path.unlink()
                print(f"Warning: Could not download {model_id}: {e}")

    def load_model(self):
        self.download_binary()
        self.download_models()
        print(f"Model {self.model_name} ready")

    def upscale_image(self, input_path, output_path=None):
        if self.binary_path is None:
            self.load_model()

        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {input_path}")

        if output_path is None:
            output_path = (
                input_path.parent
                / f"{input_path.stem}_upscaled_{self.scale_factor}x{input_path.suffix}"
            )
        else:
            output_path = Path(output_path)

        print(f"Upscaling {input_path.name}...")

        try:
            cmd = [
                str(self.binary_path),
                "-i",
                str(input_path),
                "-o",
                str(output_path),
                "-n",
                self.model_name,
                "-s",
                str(self.scale_factor),
                "-g",
                str(self.gpuid),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"Upscaling failed: {result.stderr}")

            print(f"Upscaled image saved to: {output_path}")
            return str(output_path)

        except Exception as e:
            print(f"Error upscaling image: {e}")
            raise

    def upscale_batch(
        self,
        input_dir,
        output_dir=None,
        file_extensions=(".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
    ):
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        if output_dir is None:
            output_dir = input_dir.parent / f"{input_dir.name}_upscaled"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(exist_ok=True)

        image_files = []
        for ext in file_extensions:
            image_files.extend(input_dir.glob(f"*{ext}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))

        if not image_files:
            print(f"No images found in {input_dir}")
            return

        print(f"Found {len(image_files)} images to upscale")

        for img_file in tqdm(image_files, desc="Upscaling images"):
            output_path = (
                output_dir
                / f"{img_file.stem}_upscaled_{self.scale_factor}x{img_file.suffix}"
            )
            try:
                self.upscale_image(img_file, output_path)
            except Exception as e:
                print(f"Error processing {img_file.name}: {e}")


MODEL_CHOICES = [
    "realesrgan-x4plus",
    "realesrgan-x4plus-anime",
    "realesr-animevideov3-x2",
    "realesr-animevideov3-x3",
    "realesr-animevideov3-x4",
]

MODEL_SCALES = {
    "realesrgan-x4plus": 4,
    "realesrgan-x4plus-anime": 4,
    "realesr-animevideov3-x2": 2,
    "realesr-animevideov3-x3": 3,
    "realesr-animevideov3-x4": 4,
}


def main():
    parser = argparse.ArgumentParser(
        description="Local Image Upscaler using RealESRGAN (NCNN)"
    )
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory (optional)")
    parser.add_argument(
        "-m",
        "--model",
        default="realesrgan-x4plus",
        choices=MODEL_CHOICES,
        help="Model to use for upscaling",
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=int,
        default=4,
        choices=[2, 3, 4, 8],
        help="Upscaling factor",
    )
    parser.add_argument(
        "-g", "--gpu", type=int, default=0, help="GPU device ID (default: 0)"
    )
    parser.add_argument(
        "--batch", action="store_true", help="Process all images in directory"
    )

    args = parser.parse_args()

    model_scale = MODEL_SCALES.get(args.model, 4)
    if args.scale != model_scale:
        print(f"Warning: Scale factor {args.scale} doesn't match model {args.model}")
        print(f"Using scale factor from model: {model_scale}")
        args.scale = model_scale

    upscaler = RealESRGANUpscaler(
        model_name=args.model, scale_factor=args.scale, gpuid=args.gpu
    )

    try:
        if args.batch or os.path.isdir(args.input):
            upscaler.upscale_batch(args.input, args.output)
        else:
            upscaler.upscale_image(args.input, args.output)

        print("Upscaling completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
