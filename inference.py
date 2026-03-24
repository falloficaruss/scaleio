import cv2
import numpy as np
import torch
from basicsr.arch.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

model = RRDBNet(
    num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
)

model_path = "4x-UltraSharp.pth"
upscaler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=model,
    title=400,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

img = cv2.imread("input.jpg")
output, _ = upscaler.enhance(img, outscale=4)
cv2.imwrite("output.png", output)
