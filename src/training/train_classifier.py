import os
import argparse
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from sklearn.metrics import f1_score


class SolarDataset(Dataset):
    def __init__(self, csv_path, img_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.transform = transform

        # Expecting column 'has_solar' as 0/1 and 'rooftop_image' as image filename
        if 'has_solar' not in self.df.columns:
            raise ValueError("CSV must contain a 'has_solar' column.")
        if 'rooftop_image' not in self.df.columns:
            raise ValueError("CSV must contain a 'rooftop_image' column.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_root, row['rooftop_image'])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(float(row['has_solar']), dtype=torch.float32)
        return img, label


def get_model():
    # Simple ResNet18 binary classifier
    model = models.resnet18(weights=None)  # or weights=models.ResNet18_Weights.DEFAULT
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)   # single logit for BCEWithLogitsLoss
    return model


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # you can add Normalize here if you want
    ])

    dataset = SolarDataset(args.train_csv, args.img_root, transform=transform)

    # Train/val split
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = get_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_f1 = 0.0
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device).unsqueeze(1)  # (B,) -> (B,1)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)

        # ---- VALIDATION ----
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device).unsqueeze(1)

                logits = model(imgs)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                val_preds.extend(preds.cpu().numpy().flatten().tolist())
                val_targets.extend(labels.cpu().numpy().flatten().tolist())

        f1 = f1_score(val_targets, val_preds) if len(set(val_targets)) > 1 else 0.0

        print(f"Epoch [{epoch}/{args.epochs}] "
              f"Train Loss: {avg_train_loss:.4f} "
              f"Val F1: {f1:.4f}")

        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), args.out_path)
            print(f"[INFO] New best model saved with F1={best_val_f1:.4f} at {args.out_path}")

    print("[DONE] Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True, help="Path to train_rooftop_data.csv")
    parser.add_argument("--img_root", required=True, help="Folder with rooftop images")
    parser.add_argument("--out_path", default="weights/trained/classifier_best.pth")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()
    train(args)
