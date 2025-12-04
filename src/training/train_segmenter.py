import os
import argparse
from typing import Tuple

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from torchvision import transforms


# ------------------------
# Dataset
# ------------------------

class SegmentationDataset(Dataset):
    """
    Expects a CSV with at least:
      - rooftop_image : RGB image filename
      - mask_image    : corresponding mask filename (binary or 0/255)
    """

    def __init__(self, csv_path: str, img_root: str, mask_root: str, transform=None, mask_transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.mask_root = mask_root
        self.transform = transform
        self.mask_transform = mask_transform

        required_cols = ["rooftop_image", "mask_image"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"CSV must contain column '{col}'")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        img_path = os.path.join(self.img_root, row["rooftop_image"])
        mask_path = os.path.join(self.mask_root, row["mask_image"])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")   # grayscale

        if self.transform:
            img = self.transform(img)

        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = transforms.ToTensor()(mask)

        # Ensure mask is 0/1
        mask = (mask > 0.5).float()
        return img, mask


# ------------------------
# Simple U-Net-like model
# ------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 32)
        self.down2 = DoubleConv(32, 64)
        self.down3 = DoubleConv(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv1 = DoubleConv(64, 32)

        self.out_conv = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        # Encoder
        x1 = self.down1(x)       # -> 32
        x2 = self.pool(x1)

        x2 = self.down2(x2)      # -> 64
        x3 = self.pool(x2)

        x3 = self.down3(x3)      # -> 128

        # Decoder
        x = self.up2(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv1(x)

        logits = self.out_conv(x)
        return logits


# ------------------------
# Loss / Metrics
# ------------------------

def dice_loss(pred, target, eps=1e-6):
    """
    pred, target: (B,1,H,W), values in [0,1]
    """
    num = 2 * (pred * target).sum(dim=(2, 3))
    den = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    dice = 1 - (num / den)
    return dice.mean()


# ------------------------
# Training Loop
# ------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = SegmentationDataset(
        csv_path=args.train_csv,
        img_root=args.img_root,
        mask_root=args.mask_root,
        transform=img_transform,
        mask_transform=mask_transform,
    )

    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = UNetSmall(in_channels=3, out_channels=1).to(device)
    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_dice = 0.0
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            prob = torch.sigmoid(logits)

            loss_bce = bce(logits, masks)
            loss_dice = dice_loss(prob, masks)
            loss = loss_bce + loss_dice

            loss.backw
