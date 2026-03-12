import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from scaleio.utils import (
    pil_to_cv2,
    cv2_to_pil,
    validate_scale,
    detect_device,
    load_image,
    UnsupportedScaleError,
    ImageLoadError,
)


class TestUtils:
    def test_pil_to_cv2(self):
        pil_img = Image.new("RGB", (10, 10), color="red")
        cv2_img = pil_to_cv2(pil_img)
        assert cv2_img.shape == (10, 10, 3)

    def test_cv2_to_pil(self):
        cv2_img = np.zeros((10, 10, 3), dtype=np.uint8)
        pil_img = cv2_to_pil(cv2_img)
        assert isinstance(pil_img, Image.Image)
        assert pil_img.size == (10, 10)

    def test_validate_scale_valid(self):
        for scale in [2, 4, 8]:
            validate_scale(scale)

    def test_validate_scale_invalid(self):
        with pytest.raises(UnsupportedScaleError):
            validate_scale(3)
        with pytest.raises(UnsupportedScaleError):
            validate_scale(1)
        with pytest.raises(UnsupportedScaleError):
            validate_scale(16)

    def test_detect_device(self):
        device = detect_device()
        assert device in ["cuda", "mps", "cpu"]

    def test_load_image_from_pil(self):
        pil_img = Image.new("RGB", (10, 10), color="blue")
        result = load_image(pil_img)
        assert isinstance(result, Image.Image)

    def test_load_image_from_array(self):
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        result = load_image(arr)
        assert isinstance(result, Image.Image)

    def test_load_image_from_path(self, tmp_path):
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (10, 10), color="green")
        img.save(img_path)
        result = load_image(img_path)
        assert isinstance(result, Image.Image)

    def test_load_image_invalid_path(self):
        with pytest.raises(ImageLoadError):
            load_image("/nonexistent/path/to/image.png")

    def test_load_image_invalid_type(self):
        with pytest.raises(ImageLoadError):
            load_image(12345)
