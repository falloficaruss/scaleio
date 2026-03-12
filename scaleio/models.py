from pathlib import Path
from typing import Dict, Optional

from tqdm import tqdm


class ModelNotFoundError(ValueError):
    pass


MODEL_CONFIGS: Dict[str, dict] = {
    "general": {
        "name": "RealESRGAN_x4plus",
        "scale": 4,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    },
    "anime": {
        "name": "RealESRGAN_x4plus_anime_6B",
        "scale": 4,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.2/RealESRGAN_x4plus_anime_6B.pth",
    },
    "general-denoise": {
        "name": "RealESRGAN_x4plus_denoise_3x",
        "scale": 4,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRGAN_x4plus_denoise_3x.pth",
    },
    "anime-denoise": {
        "name": "RealESRGAN_x4plus_anime_denoise_3x",
        "scale": 4,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRGAN_x4plus_anime_denoise_3x.pth",
    },
}


class ModelManager:
    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "ai_upscaler"
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, model_name: str) -> Path:
        if model_name not in MODEL_CONFIGS:
            raise ModelNotFoundError(
                f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}"
            )
        config = MODEL_CONFIGS[model_name]
        model_path = self.cache_dir / f"{config['name']}.pth"
        if not model_path.exists():
            self._download_model(config["url"], model_path)
        return model_path

    def _download_model(self, url: str, dest: Path) -> None:
        import requests

        print(f"Downloading model to {dest}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(dest, "wb") as f, tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    def get_model_config(self, model_name: str) -> dict:
        if model_name not in MODEL_CONFIGS:
            raise ModelNotFoundError(
                f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}"
            )
        return MODEL_CONFIGS[model_name].copy()
