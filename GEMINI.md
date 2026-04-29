# ScaleIO: Continuous-to-Discrete Image Super-Resolution (C2D-ISR)

ScaleIO is a framework for Image Super-Resolution (ISR) that combines continuous-scale representation learning with discrete-scale fine-tuning. It utilizes a Hierarchical Encoding Transformer (HiET) backbone and a Hyper-Implicit Image Function (HIIF-L) for high-quality upsampling.

## Project Architecture

### 1. Model Components (`models/`)
- **`C2DISR` (`models/c2d_isr.py`)**: The main model wrapper. It can operate in two stages:
    - **Stage 1 (Continuous)**: Uses `HIIFL` for upsampling, supporting arbitrary scale factors.
    - **Stage 2 (Discrete)**: Uses `SubPixelUpsampler` for fixed-scale upsampling (e.g., 2x, 3x, 4x).
- **`HiETBlock` (`models/hiet_block.py`)**: The core deep feature extractor. It follows a hierarchical encoder-decoder structure with skip connections, using `HiETLayer` as the building block.
- **`HIIFL` (`models/hiif_l.py`)**: Hyper-Implicit Image Function Layer. It performs continuous-scale upsampling by combining deep features with hierarchical coordinate encodings and applying linear attention.
- **Backbones**: The project supports using `SwinIR` as an alternative backbone (`models/backbones/swinir_l.py`).

### 2. Training Workflow (`training/`)
The training is divided into two primary stages:
- **Stage 1: Continuous-scale Pre-training** (`training/stage1_continuous.py`)
    - **Goal**: Learn a continuous representation of images.
    - **Input**: LR images downsampled by random scales (e.g., 1.0x to 4.0x).
    - **Model**: `C2DISR` with `HIIFL` upsampler.
    - **Optimizer**: Adam with `WarmupCosineScheduler`.
- **Stage 2: Discrete-scale Fine-tuning** (`training/stage2_discrete.py`)
    - **Goal**: Fine-tune for a specific target scale (e.g., 4x) to achieve maximum performance.
    - **Input**: LR images downsampled by the target scale factor.
    - **Model**: `C2DISR` with `SubPixelUpsampler`, initialized with Stage 1 weights (backbone only).
    - **Loss**: L1 Loss (default).

### 3. Data & Evaluation
- **Datasets (`data/datasets.py`)**:
    - `ContinuousScaleData`: For Stage 1 training, applies random resizing.
    - `SRDataset`: For Stage 2 training, applies fixed resizing.
- **Metrics (`evaluation/metrics.py`)**: Implements PSNR and SSIM for performance evaluation.

## Directory Structure

```text
/home/falloficaruss/scaleio/
├── data/               # Data loading and preprocessing
│   └── datasets.py     # SR and Continuous scale datasets
├── evaluation/         # Performance metrics and evaluation scripts
│   └── metrics.py      # PSNR, SSIM implementation
├── models/             # Model architectures
│   ├── backbones/      # Alternative backbone architectures
│   ├── c2d_isr.py      # Main C2D-ISR model and factory
│   ├── hiet_block.py   # Hierarchical Encoding Transformer block
│   ├── hiet_layer.py   # Basic HiET layer
│   └── hiif_l.py       # Hyper-Implicit Image Function Layer
└── training/           # Training scripts and loss functions
    ├── losses.py       # L1, MSE, Charbonnier, Perceptual, Gradient losses
    ├── stage1_continuous.py # Stage 1 training script
    └── stage2_discrete.py   # Stage 2 training script
```

## Tech Stack
- **Framework**: PyTorch
- **Image Processing**: Kornia, Pillow, Scikit-image
- **Logging**: TensorBoard, Tqdm
- **Utilities**: Timm, Einops

## Engineering Standards

- **Conventions**:
    - Use `C2DISRFactory` for model instantiation.
    - Adhere to the two-stage training workflow.
    - Metrics should be calculated on the [0, 1] range.
- **Types**: Use Python type hints for better readability and maintainability.
- **Style**: Follow standard PyTorch practices (e.g., `model.train()`, `model.eval()`, `optimizer.zero_grad()`).

## Maintenance History

- [x] **Missing File**: `training/scheduler.py` has been implemented with `WarmupCosineScheduler`.
- [x] **Typo**: `training/stage2_discrete.py` fixed to call `L1Loss()`.
- [x] **Import Mismatch**: `training/stage1_continuous.py` updated to import `ContinuousScaleData`.
- [x] **Factory Argument Mismatch**: `C2DISRFactory.create_model_from_stage1` argument name fixed in `stage2_discrete.py`.
- [x] **Import Style**: Converted relative imports in `training/` to absolute imports for better root-level execution compatibility.
