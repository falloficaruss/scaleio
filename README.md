# Local Image Upscaler

A completely offline image upscaling tool using RealESRGAN that runs locally on your machine. No cloud services, no data leaving your computer.

## Features

- 🖼️ **Local Processing**: All processing happens on your machine
- 🚀 **High Quality**: Uses RealESRGAN for superior upscaling results
- 🔧 **Easy Setup**: Simple Python installation
- 📁 **Batch Processing**: Upscale multiple images at once
- 💾 **Multiple Models**: Support for 2x, 3x, and 4x upscaling
- 🎯 **Free & Open Source**: No costs, no subscriptions

## Requirements

- Python 3.8 or higher
- A computer with GPU (recommended) or CPU
- For Linux: Vulkan runtime libraries

## Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd scaleio
   ```

2. **Install system dependencies** (Linux only)
   ```bash
   # Debian/Ubuntu
   sudo apt install -y libvulkan-dev vulkan-tools
   
   # Fedora/RHEL
   sudo dnf install vulkan-loader
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python upscale_image.py --help
   ```

## Usage

### Single Image Upscaling

```bash
# Basic usage (4x upscaling with default model)
python upscale_image.py input_image.jpg

# Specify output file
python upscale_image.py input_image.jpg -o output_image.jpg

# Use different model (2x upscaling)
python upscale_image.py input_image.jpg -m realesr-animevideov3-x2

# Use anime-optimized model
python upscale_image.py input_image.jpg -m realesrgan-x4plus-anime
```

### Batch Processing

```bash
# Upscale all images in a directory
python upscale_image.py /path/to/images --batch

# Specify output directory
python upscale_image.py /path/to/images --batch -o /path/to/output
```

### Command Line Options

- `input`: Input image file or directory
- `-o, --output`: Output file or directory (optional)
- `-m, --model`: Model to use (see Models section)
- `-s, --scale`: Upscaling factor (2, 3, 4)
- `-g, --gpu`: GPU device ID (default: 0, use -1 for CPU)
- `--batch`: Process all images in directory

## Models

| Model | Scale Factor | Best For |
|-------|-------------|----------|
| realesr-animevideov3-x2 | 2x | Anime video, faster processing |
| realesr-animevideov3-x3 | 3x | Anime video, 3x upscaling |
| realesr-animevideov3-x4 | 4x | Anime video |
| realesrgan-x4plus | 4x | General purpose upscaling (default) |
| realesrgan-x4plus-anime | 4x | Anime-style images |

## Supported Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)
- WebP (.webp)

## How It Works

1. **First Run**: The tool automatically downloads the required binary and models
2. **Local Processing**: Images are processed entirely on your machine using NCNN + Vulkan
3. **Smart Upscaling**: RealESRGAN uses deep learning to intelligently enhance image details
4. **Privacy**: No data is sent to any external service

## Performance Tips

- **GPU Acceleration**: Uses Vulkan for GPU acceleration. Works with Intel, AMD, and NVIDIA GPUs
- **CPU Fallback**: Use `-g -1` for CPU-only processing
- **Memory Usage**: Efficient memory usage with tile-based processing
- **Batch Processing**: Process multiple images efficiently with the `--batch` option

## Troubleshooting

### Common Issues

1. **"Vulkan not found" error**
   - Linux: `sudo apt install -y libvulkan-dev vulkan-tools`
   - Windows: Install Vulkan runtime from https://vulkan.lunarg.com/
   - macOS: Limited GPU support, try CPU mode with `-g -1`

2. **Binary download fails**
   - Check your internet connection
   - Try downloading manually from GitHub releases

3. **No GPU detected**
   - Verify Vulkan is working: `vulkaninfo`
   - Try CPU mode: `-g -1`

### Getting Help

If you encounter issues:
1. Check that all dependencies are installed
2. Verify the input file exists and is a supported format
3. Run with `--help` to see all available options

## Technical Details

- **Framework**: NCNN (Tencent's neural network inference framework)
- **Backend**: Vulkan for GPU acceleration
- **Model**: RealESRGAN (Real-World Super-Resolution via Synthetic Data)
- **Processing**: Efficient tile-based processing for large images

## License

This project uses RealESRGAN, which is licensed under the BSD 2-Clause License.

## Privacy Notice

- ✅ All processing happens locally on your machine
- ✅ No images are uploaded to any cloud service
- ✅ No data is collected or transmitted
- ✅ Models are downloaded once and cached locally
