import numpy as np
import pytest
from pathlib import Path
from PIL import Image
from unittest.mock import patch, MagicMock

from scaleio.upscaler import Upscaler
from scaleio.models import MODEL_CONFIGS, ModelNotFoundError
from scaleio.utils import UnsupportedScaleError


class TestUpscaler:
    def test_init_defaults(self):
        upscaler = Upscaler()
        assert upscaler.scale == 4
        assert upscaler.model_name == "general"
        assert upscaler.tile == 0
        assert upscaler.tile_pad == 10

    def test_init_custom_values(self):
        upscaler = Upscaler(scale=2, model="anime", tile=512, device="cpu")
        assert upscaler.scale == 2
        assert upscaler.model_name == "anime"
        assert upscaler.tile == 512
        assert upscaler.device == "cpu"

    def test_init_invalid_scale(self):
        with pytest.raises(UnsupportedScaleError):
            Upscaler(scale=3)

    def test_init_invalid_model(self):
        with pytest.raises(ModelNotFoundError):
            Upscaler(model="nonexistent_model")

    def test_init_device_auto(self):
        upscaler = Upscaler(device="auto")
        assert upscaler.device in ["cuda", "mps", "cpu"]

    def test_model_configs_exist(self):
        expected_models = ["general", "anime", "general-denoise", "anime-denoise"]
        for model in expected_models:
            assert model in MODEL_CONFIGS
            assert "name" in MODEL_CONFIGS[model]
            assert "scale" in MODEL_CONFIGS[model]
            assert "url" in MODEL_CONFIGS[model]

    @patch("scaleio.upscaler.RealESRGAN")
    def test_upscale_with_mock(self, mock_realesrgan):
        mock_model = MagicMock()
        mock_model.process.return_value = np.zeros((40, 40, 3), dtype=np.uint8)
        mock_realesrgan.return_value = mock_model

        upscaler = Upscaler(scale=4, model="general", device="cpu")
        upscaler._model = mock_model

        input_img = Image.new("RGB", (10, 10), color="red")
        result = upscaler.upscale(input_img)

        assert isinstance(result, Image.Image)
        assert result.size == (40, 40)
        mock_model.process.assert_called_once()

    @patch("scaleio.upscaler.RealESRGAN")
    def test_upscale_with_path(self, mock_realesrgan, tmp_path):
        mock_model = MagicMock()
        mock_model.process.return_value = np.zeros((40, 40, 3), dtype=np.uint8)
        mock_realesrgan.return_value = mock_model

        input_path = tmp_path / "input.png"
        img = Image.new("RGB", (10, 10), color="blue")
        img.save(input_path)

        upscaler = Upscaler(scale=4, model="general", device="cpu")
        upscaler._model = mock_model

        output_path = tmp_path / "output.png"
        result = upscaler.upscale(input_path, output_path)

        assert output_path.exists()

    @patch("scaleio.upscaler.RealESRGAN")
    def test_upscale_output_dimensions(self, mock_realesrgan):
        mock_model = MagicMock()
        mock_model.process.return_value = np.zeros((80, 80, 3), dtype=np.uint8)
        mock_realesrgan.return_value = mock_model

        upscaler = Upscaler(scale=4, model="general", device="cpu")
        upscaler._model = mock_model

        input_img = Image.new("RGB", (20, 20), color="green")
        result = upscaler.upscale(input_img)

        assert result.size == (80, 80)

    @patch("scaleio.upscaler.RealESRGAN")
    def test_upscale_batch(self, mock_realesrgan, tmp_path):
        mock_model = MagicMock()
        mock_model.process.return_value = np.zeros((40, 40, 3), dtype=np.uint8)
        mock_realesrgan.return_value = mock_model

        input_dir = tmp_path / "inputs"
        input_dir.mkdir()

        for i in range(3):
            img_path = input_dir / f"test{i}.png"
            Image.new("RGB", (10, 10)).save(img_path)

        output_dir = tmp_path / "outputs"

        upscaler = Upscaler(scale=4, model="general", device="cpu")
        upscaler._model = mock_model

        output_paths = upscaler.upscale_batch(list(input_dir.glob("*.png")), output_dir)

        assert len(output_paths) == 3
        assert all(p.exists() for p in output_paths)
