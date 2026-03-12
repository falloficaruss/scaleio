import argparse
import sys
from pathlib import Path

from scaleio.upscaler import Upscaler
from scaleio.models import MODEL_CONFIGS


def parse_args():
    parser = argparse.ArgumentParser(description="AI Image Upscaler using Real-ESRGAN")
    parser.add_argument(
        "input",
        help="Input image file or directory (for batch mode)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        choices=[2, 4, 8],
        help="Upscaling factor (default: 4)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="general",
        choices=list(MODEL_CONFIGS.keys()),
        help=f"Model to use (default: general). Available: {list(MODEL_CONFIGS.keys())}",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (for single image)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (for batch mode)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Enable batch mode (process all images in directory)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_upscaled",
        help="Suffix for output files in batch mode (default: _upscaled)",
    )
    parser.add_argument(
        "--tile",
        type=int,
        default=0,
        help="Tile size for large images (0 = disabled, default: 0)",
    )
    parser.add_argument(
        "--tile-pad",
        type=int,
        default=10,
        help="Tile padding (default: 10)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use (default: auto)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)

    if args.batch:
        if not input_path.is_dir():
            print(f"Error: Batch mode requires a directory, got: {input_path}")
            sys.exit(1)

        if not args.output_dir:
            print("Error: --output-dir is required for batch mode")
            sys.exit(1)

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]

        if not image_files:
            print(f"No image files found in {input_path}")
            sys.exit(1)

        upscaler = Upscaler(
            scale=args.scale,
            model=args.model,
            tile=args.tile,
            tile_pad=args.tile_pad,
            device=args.device,
        )

        output_paths = upscaler.upscale_batch(image_files, args.output_dir, suffix=args.suffix)

        print(f"Upscaled {len(output_paths)} images to {args.output_dir}")

    else:
        if not input_path.is_file():
            print(f"Error: Input file not found: {input_path}")
            sys.exit(1)

        if not args.output:
            args.output = input_path.parent / f"{input_path.stem}_upscaled{input_path.suffix}"

        upscaler = Upscaler(
            scale=args.scale,
            model=args.model,
            tile=args.tile,
            tile_pad=args.tile_pad,
            device=args.device,
        )

        result = upscaler.upscale(input_path, args.output)
        print(f"Upscaled image saved to {args.output}")


if __name__ == "__main__":
    main()
