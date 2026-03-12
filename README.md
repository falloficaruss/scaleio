# Scaleio

AI Image Upscaler using Real-ESRGAN.

## Installation

```bash
pip install scaleio
```

Or install from source:

```bash
pip install -e .
```

### Dependencies

- `realesrgan` - Real-ESRGAN inference
- `Pillow` - Image handling
- `numpy` - Array operations
- `opencv-python-headless` - Computer vision
- `torch` - Deep learning backend
- `basicsr` - Basic Super Resolution library
- `tqdm` - Progress bars

## Quick Start

```python
from scaleio import Upscaler

upscaler = Upscaler(scale=4, model="general")

# From file
result = upscaler.upscale("input.jpg", "output.png")

# From PIL Image
from PIL import Image
img = Image.open("input.jpg")
result = upscaler.upscale(img)

# Batch processing
output_paths = upscaler.upscale_batch(
    ["img1.jpg", "img2.jpg", "img3.jpg"],
    output_dir="upscaled/"
)
```

## CLI Usage

```bash
# Single image
scaleio input.jpg --scale 4 --model general --output output.png

# Batch processing
scaleio ./images/ --batch --output-dir ./upscaled/

# With custom settings
scaleio input.jpg --scale 2 --model anime --tile 512
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--scale` | Upscaling factor (2, 4, 8) | 4 |
| `--model` | Model type | general |
| `--tile` | Tile size for large images | 0 (disabled) |
| `--device` | Device (auto, cuda, mps, cpu) | auto |
| `--output` | Output file (single mode) | auto-generated |
| `--output-dir` | Output directory (batch mode) | required for batch |
| `--batch` | Enable batch mode | false |
| `--suffix` | Output suffix for batch | _upscaled |

## Available Models

- `general` - General purpose upscaling (RealESRGAN_x4plus)
- `anime` - Optimized for anime/illustrations (RealESRGAN_x4plus_anime_6B)
- `general-denoise` - General with denoising
- `anime-denoise` - Anime with denoising

## API Reference

### Upscaler

```python
Upscaler(
    scale: int = 4,           # 2, 4, or 8
    model: str = "general",   # model name
    tile: int = 0,            # tile size (0 = disabled)
    tile_pad: int = 10,       # tile padding
    device: str = "auto"      # auto, cuda, mps, cpu
)
```

### Methods

- `upscale(input, output=None)` - Upscale single image
  - `input`: str, Path, PIL.Image, or np.ndarray
  - `output`: Optional output path
  - Returns: PIL.Image

- `upscale_batch(inputs, output_dir, suffix="_upscaled")` - Batch process
  - `inputs`: List of file paths
  - `output_dir`: Output directory
  - `suffix`: Suffix for output filenames

## Error Handling

```python
from scaleio import (
    Upscaler,
    ModelNotFoundError,
    UnsupportedScaleError,
    ImageLoadError
)

try:
    upscaler = Upscaler(model="invalid")
except ModelNotFoundError as e:
    print(f"Invalid model: {e}")

try:
    upscaler = Upscaler(scale=3)
except UnsupportedScaleError as e:
    print(f"Invalid scale: {e}")
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black scaleio/
ruff check scaleio/
```

## License

MIT
