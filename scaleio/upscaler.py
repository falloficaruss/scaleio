from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from PIL import Image
from tqdm import tqdm

from scaleio.models import ModelManager, MODEL_CONFIGS
from scaleio.utils import (
    detect_device,
    load_image,
    pil_to_cv2,
    cv2_to_pil,
    validate_scale,
)


class Upscaler:
    def __init__(
        self,
        scale: int = 4,
        model: str = "general",
        tile: int = 0,
        tile_pad: int = 10,
        device: str = "auto",
    ):
        validate_scale(scale)
        if model not in MODEL_CONFIGS:
            from scaleio.models import ModelNotFoundError

            raise ModelNotFoundError(
                f"Unknown model: {model}. Available: {list(MODEL_CONFIGS.keys())}"
            )

        self.scale = scale
        self.model_name = model
        self.tile = tile
        self.tile_pad = tile_pad

        if device == "auto":
            self.device = detect_device()
        else:
            self.device = device

        if self.device == "cpu":
            import warnings

            warnings.warn("CUDA not available, using CPU (slow)")

        self.model_manager = ModelManager()
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        try:
            from realesrgan import RealESRGAN
        except ImportError:
            raise ImportError("realesrgan package not found. Install with: pip install realesrgan")

        model_config = self.model_manager.get_model_config(self.model_name)

        if self.device == "cuda":
            use_gpu = 0
        elif self.device == "mps":
            use_gpu = -1
        else:
            use_gpu = -1

        self._model = RealESRGAN(
            scale=self.scale,
            model=model_config["name"],
            tile=self.tile,
            tile_pad=self.tile_pad,
            gpu_id=use_gpu,
        )

    def upscale(
        self,
        input: Union[str, Path, Image.Image, np.ndarray],
        output: Optional[Union[str, Path]] = None,
    ) -> Image.Image:
        img = load_image(input)
        input_array = np.array(img)

        result = self.model.process(input_array)

        result_img = Image.fromarray(result)

        if output is not None:
            output_path = Path(output)
            result_img.save(output_path)
            return result_img

        return result_img

    def upscale_batch(
        self,
        inputs: List[Union[str, Path]],
        output_dir: Union[str, Path],
        suffix: str = "_upscaled",
    ) -> List[Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_paths = []
        for input_path in tqdm(inputs, desc="Upscaling"):
            input_path = Path(input_path)
            output_path = output_dir / f"{input_path.stem}{suffix}{input_path.suffix}"

            self.upscale(input_path, output_path)
            output_paths.append(output_path)

        return output_paths
