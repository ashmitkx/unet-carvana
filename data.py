from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import os
import numpy as np
from PIL import Image


class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '_mask.gif'))

        image = np.array(Image.open(image_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)

        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        return image, mask

    @staticmethod
    def get_dataloaders(
        img_height,
        img_width,
        train_img_dir,
        train_mask_dir,
        val_img_dir,
        val_mask_dir,
        batch_size=32,
        pin_memory=True,
        num_workers=2
    ):
        train_transform = A.Compose(
            [
                A.Resize(height=img_height, width=img_width),
                A.Rotate(limit=35, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2()
            ]
        )

        train_ds = CarvanaDataset(
            image_dir=train_img_dir,
            mask_dir=train_mask_dir,
            transform=train_transform,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True,
        )

        val_transform = A.Compose(
            [
                A.Resize(height=img_height, width=img_width),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2()
            ]
        )

        val_ds = CarvanaDataset(
            image_dir=val_img_dir,
            mask_dir=val_mask_dir,
            transform=val_transform,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
        )

        return train_loader, val_loader
