import torch as t
from torchvision import utils
import albumentations as A
from albumentations.pytorch import ToTensorV2

import os
import numpy as np
from PIL import Image

from unet import UNet
from train_utils import load_checkpoint
from config import *

# transform for inference
infer_transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ]
)

# load model
model = UNet(in_ch=3, out_ch=1)
model = load_checkpoint(model, filepath=MODEL_RESTORE_PATH).to(DEVICE)
model.eval()

for image_file in os.listdir(INFERENCE_IMAGE_PATH):
    image_path = os.path.join(INFERENCE_IMAGE_PATH, image_file)

    image = np.array(Image.open(image_path).convert('RGB'))
    image = infer_transform(image=image)['image']
    image = image.unsqueeze(0).to(DEVICE)

    with t.no_grad():
        pred = t.sigmoid(model(image))
        pred = (pred > 0.5).float()

        pred = pred.repeat(1, 3, 1, 1)  # extend to all channels
        pred[:, 2][pred[:, 2] == 1.0] = 0.4  # apply yellow on car pixels

        pred[:, 1][pred[:, 1] == 0.0] = 0.4  # apply blue on non-car pixels
        pred[:, 2][pred[:, 2] == 0.0] = 0.8

        segmap = image * pred

    utils.save_image(segmap, f"{INFERENCE_SEG_PATH}/{image_file}.png")
