#!/usr/bin/env python3
"""
Local Image Upscaler using RealESRGAN
Supports both NCNN (Vulkan) and PyTorch backends with automatic fallback.
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
        "url": "https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/v0.2.0/realesrgan-ncnn-vulkan-v0.2.0-windows.zip",
        "binary": "realesrgan-ncnn-vulkan.exe",
    },
    "linux": {
        "url": "https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/v0.2.0/realesrgan-ncnn-vulkan-v0.2.0-ubuntu.zip",
        "binary": "realesrgan-ncnn-vulkan",
    },
    "darwin": {
        "url": "https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/v0.2.0/realesrgan-ncnn-vulkan-v0.2.0-macos.zip",
        "binary": "realesrgan-ncnn-vulkan",
    },
}


class NCNNBackend:
    def __init__(self, model_name="realesrgan-x4plus", scale_factor=4, gpuid=0):
        self.model_name = model_name
        self.scale_factor = scale_factor
        self.gpuid = gpuid
        self.binary_path = None
        self.models_dir = None
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

        models_dir = self.model_dir / "ncnn-models"
        models_dir.mkdir(exist_ok=True)

        binary_path = binary_dir / binary_info["binary"]

        zip_path = (
            Path(tempfile.gettempdir()) / f"realesrgan-ncnn-vulkan-{platform_info}.zip"
        )

        if not zip_path.exists():
            print(f"Downloading RealESRGAN-ncnn-vulkan binary...")
            urllib.request.urlretrieve(binary_info["url"], zip_path)

        print("Extracting binary and models...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for member in zip_ref.namelist():
                if binary_info["binary"] in member and not member.endswith("/"):
                    source = zip_ref.open(member)
                    target = open(binary_path, "wb")
                    with source, target:
                        shutil.copyfileobj(source, target)
                    break

            for member in zip_ref.namelist():
                if member.endswith(".param") or member.endswith(".bin"):
                    filename = Path(member).name
                    target_path = models_dir / filename
                    if not target_path.exists():
                        with zip_ref.open(member) as source:
                            with open(target_path, "wb") as target:
                                shutil.copyfileobj(source, target)
                    model_in_bin_dir = binary_dir / filename
                    if not model_in_bin_dir.exists():
                        shutil.copy2(target_path, model_in_bin_dir)

        binary_path.chmod(0o755)
        self.binary_path = binary_path
        self.models_dir = models_dir
        print(f"Binary extracted to: {binary_path}")
        print(f"Models extracted to: {models_dir}")

    def load(self):
        self.download_binary()
        print(f"NCNN backend loaded")

    def upscale(self, input_path, output_path):
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
            raise RuntimeError(result.stderr)

        return output_path


class PyTorchBackend:
    def __init__(self, model_name="RealESRGAN_x4plus", scale_factor=4, device=None):
        self.model_name = model_name
        self.scale_factor = scale_factor
        self.device = device or ("cuda" if sys.platform != "darwin" else "cpu")
        self.model = None
        self.upsampler = None
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)

    def get_pytorch_model_name(self):
        mapping = {
            "realesrgan-x4plus": "RealESRGAN_x4plus",
            "realesrgan-x4plus-anime": "RealESRGAN_x4plus_anime",
            "realesr-animevideov3-x2": "RealESRGAN_x2plus",
            "realesr-animevideov3-x3": "RealESRGAN_x4plus",
            "realesr-animevideov3-x4": "RealESRGAN_x4plus",
        }
        return mapping.get(self.model_name, "RealESRGAN_x4plus")

    def download_model(self):
        model_urls = {
            "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "RealESRGAN_x4plus_anime": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus_anime.pth",
            "RealESRGAN_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRGAN_x2plus.pth",
        }

        pytorch_model_name = self.get_pytorch_model_name()
        if pytorch_model_name not in model_urls:
            pytorch_model_name = "RealESRGAN_x4plus"

        model_path = self.model_dir / f"{pytorch_model_name}.pth"

        if not model_path.exists():
            print(f"Downloading {pytorch_model_name} model...")
            url = model_urls[pytorch_model_name]
            urllib.request.urlretrieve(url, model_path)

        return model_path, pytorch_model_name

    def load(self):
        try:
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError as e:
            raise ImportError(f"PyTorch dependencies not installed: {e}")

        model_path, pytorch_model_name = self.download_model()

        import torch

        if pytorch_model_name == "RealESRGAN_x4plus":
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )
            scale = 4
        elif pytorch_model_name == "RealESRGAN_x4plus_anime":
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )
            scale = 4
        elif pytorch_model_name == "RealESRGAN_x2plus":
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2,
            )
            scale = 2
        else:
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )
            scale = 4

        half = self.device == "cuda"

        self.upsampler = RealESRGANer(
            scale=scale,
            model_path=str(model_path),
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=half,
            device=self.device,
        )
        print(f"PyTorch backend loaded (device: {self.device})")

    def upscale(self, input_path, output_path):
        import cv2
        from PIL import Image

        img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read image: {input_path}")

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        output, _ = self.upsampler.enhance(img, outscale=self.scale_factor)

        if output.shape[2] == 3:
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

        output_img = Image.fromarray(output)
        output_img.save(output_path)

        return output_path


class RealESRGANUpscaler:
    def __init__(
        self,
        model_name="realesrgan-x4plus",
        scale_factor=4,
        gpuid=0,
        prefer_backend="auto",
    ):
        self.model_name = model_name
        self.scale_factor = scale_factor
        self.gpuid = gpuid
        self.prefer_backend = prefer_backend

        self.ncnn_backend = NCNNBackend(model_name, scale_factor, gpuid)
        self.pytorch_backend = PyTorchBackend(model_name, scale_factor)

        self.backend = None
        self.backend_name = None

    def load(self, backend=None):
        if backend == "ncnn":
            self.ncnn_backend.load()
            self.backend = self.ncnn_backend
            self.backend_name = "ncnn"
        elif backend == "pytorch":
            self.pytorch_backend.load()
            self.backend = self.pytorch_backend
            self.backend_name = "pytorch"
        else:
            try:
                print("Trying NCNN backend (Vulkan)...")
                self.ncnn_backend.load()
                self.backend = self.ncnn_backend
                self.backend_name = "ncnn"
                print("Using NCNN backend")
            except Exception as e:
                print(f"NCNN failed: {e}")
                print("Falling back to PyTorch backend...")
                try:
                    self.pytorch_backend.load()
                    self.backend = self.pytorch_backend
                    self.backend_name = "pytorch"
                    print("Using PyTorch backend")
                except Exception as e2:
                    raise RuntimeError(
                        f"Both backends failed. NCNN: {e}, PyTorch: {e2}"
                    )

    def upscale_image(self, input_path, output_path=None):
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

        if self.backend is None:
            self.load()

        print(f"Upscaling {input_path.name} using {self.backend_name} backend...")

        try:
            result = self.backend.upscale(input_path, output_path)
            print(f"Upscaled image saved to: {output_path}")
            return str(result)
        except Exception as e:
            if self.backend_name == "ncnn":
                print(f"NCNN failed: {e}")
                print("Falling back to PyTorch backend...")
                self.backend = None
                self.pytorch_backend.gpuid = -1
                self.load("pytorch")
                result = self.backend.upscale(input_path, output_path)
                print(f"Upscaled image saved to: {output_path}")
                return str(result)
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
        description="Local Image Upscaler using RealESRGAN (NCNN + PyTorch)"
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
        "-g",
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID (default: 0, use -1 for CPU-only)",
    )
    parser.add_argument(
        "-b",
        "--backend",
        default="auto",
        choices=["auto", "ncnn", "pytorch"],
        help="Backend to use: auto (fallback), ncnn, or pytorch",
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

    if args.gpu == -1:
        args.backend = "pytorch"

    upscaler = RealESRGANUpscaler(
        model_name=args.model,
        scale_factor=args.scale,
        gpuid=args.gpu,
        prefer_backend=args.backend,
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
