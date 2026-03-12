#!/usr/bin/env python3
"""
Local Image Upscaler using RealESRGAN
A completely offline image upscaling tool that runs locally on your machine.
"""

import os
import sys
import re
import argparse
import torch
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import requests
from tqdm import tqdm
import tempfile


class RealESRGANUpscaler:
    def __init__(self, model_name="RealESRGAN_x4plus", scale_factor=4):
        """
        Initialize the RealESRGAN upscaler

        Args:
            model_name (str): Name of the model to use
            scale_factor (int): Upscaling factor (2, 4, or 8)
        """
        self.model_name = model_name
        self.scale_factor = scale_factor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)

    def download_model(self):
        """Download the RealESRGAN model if not already present"""
        model_urls = {
            "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "RealESRGAN_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRGAN_x2plus.pth",
            "RealESRGAN_x8plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x8plus.pth",
        }

        if self.model_name not in model_urls:
            raise ValueError(f"Model {self.model_name} not supported")

        model_path = self.model_dir / f"{self.model_name}.pth"

        if model_path.exists():
            print(f"Model already exists at {model_path}")
            return model_path

        print(f"Downloading {self.model_name} model...")
        url = model_urls[self.model_name]

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            with open(model_path, "wb") as f:
                with tqdm(
                    total=total_size, unit="B", unit_scale=True, desc="Downloading"
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            print(f"Model downloaded to {model_path}")
            return model_path

        except Exception as e:
            print(f"Error downloading model: {e}")
            raise

    def load_model(self):
        """Load the RealESRGAN model"""
        try:
            # Import realesrgan here to avoid import errors if not installed
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from basicsr.utils.download_util import load_file_from_url
            from realesrgan import RealESRGANer

            model_path = self.download_model()

            # Initialize model based on the model name
            if self.model_name == "RealESRGAN_x4plus":
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=4,
                )
                upsampler = RealESRGANer(
                    scale=4,
                    model_path=str(model_path),
                    model=model,
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=False if self.device.type == "cpu" else True,
                    device=self.device,
                )
            elif self.model_name == "RealESRGAN_x2plus":
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=2,
                )
                upsampler = RealESRGANer(
                    scale=2,
                    model_path=str(model_path),
                    model=model,
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=False if self.device.type == "cpu" else True,
                    device=self.device,
                )
            elif self.model_name == "RealESRGAN_x8plus":
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=8,
                )
                upsampler = RealESRGANer(
                    scale=8,
                    model_path=str(model_path),
                    model=model,
                    tile=0,
                    tile_pad=10,
                    pre_pad=0,
                    half=False if self.device.type == "cpu" else True,
                    device=self.device,
                )

            self.upsampler = upsampler
            print(f"Model loaded successfully on {self.device}")

        except ImportError as e:
            print(f"Missing dependencies: {e}")
            print("Please install realesrgan: pip install realesrgan")
            sys.exit(1)

    def upscale_image(self, input_path, output_path=None):
        """
        Upscale a single image

        Args:
            input_path (str): Path to input image
            output_path (str): Path to save upscaled image (optional)

        Returns:
            str: Path to the upscaled image
        """
        if self.upsampler is None:
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

        # Read image
        img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        print(f"Upscaling {input_path.name}...")

        try:
            # Upscale the image
            output, _ = self.upsampler.enhance(img, outscale=self.scale_factor)

            # Convert back to PIL and save
            if output.shape[2] == 3:
                output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

            output_img = Image.fromarray(output)
            output_img.save(output_path)
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
        """
        Upscale all images in a directory

        Args:
            input_dir (str): Input directory containing images
            output_dir (str): Output directory for upscaled images (optional)
            file_extensions (tuple): Supported file extensions
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        if output_dir is None:
            output_dir = input_dir.parent / f"{input_dir.name}_upscaled"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(exist_ok=True)

        # Find all image files
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


def main():
    parser = argparse.ArgumentParser(
        description="Local Image Upscaler using RealESRGAN"
    )
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory (optional)")
    parser.add_argument(
        "-m",
        "--model",
        default="RealESRGAN_x4plus",
        choices=["RealESRGAN_x4plus", "RealESRGAN_x2plus", "RealESRGAN_x8plus"],
        help="Model to use for upscaling",
    )
    parser.add_argument(
        "-s", "--scale", type=int, default=4, choices=[2, 4, 8], help="Upscaling factor"
    )
    parser.add_argument(
        "--batch", action="store_true", help="Process all images in directory"
    )

    args = parser.parse_args()

    # Validate scale factor matches model
    model_scale = int(re.search(r"x(\d+)", args.model).group(1))
    if args.scale != model_scale:
        print(f"Warning: Scale factor {args.scale} doesn't match model {args.model}")
        print(f"Using scale factor from model: {model_scale}")
        args.scale = model_scale

    # Initialize upscaler
    upscaler = RealESRGANUpscaler(model_name=args.model, scale_factor=args.scale)

    try:
        if args.batch or os.path.isdir(args.input):
            # Batch processing
            upscaler.upscale_batch(args.input, args.output)
        else:
            # Single image processing
            upscaler.upscale_image(args.input, args.output)

        print("Upscaling completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
