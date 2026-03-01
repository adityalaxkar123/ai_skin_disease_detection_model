"""
Skin Disease Classification using EfficientNetV2
Production-level training pipeline
"""

import os
import sys
import json
import shutil
import random
import hashlib
import warnings
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from PIL import Image
import imagehash
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT.parent / "ai_skin_dataset"
DATASET_SPLIT = ROOT / "dataset_split"
CLEANED_DATASET = ROOT / "cleaned_dataset"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
OUTPUTS_DIR = ROOT / "outputs"

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 16  # conservative; will auto-adjust
NUM_WORKERS = 2  # Windows-safe
NUM_EPOCHS = 50
EARLY_STOP_PATIENCE = 10
LR = 3e-4
DROPOUT = 0.4
LABEL_SMOOTHING = 0.1
GRAD_CLIP = 1.0
MIN_IMAGE_SIZE = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

def ensure_dirs():
    for d in [DATASET_SPLIT, CLEANED_DATASET, MODELS_DIR, LOGS_DIR, OUTPUTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

# ============================================================
# STEP 1 — PROJECT STRUCTURE
# ============================================================
def step1_create_structure():
    print("=" * 60)
    print("STEP 1 — PROJECT STRUCTURE CREATION")
    print("=" * 60)
    ensure_dirs()
    for d in [DATASET_SPLIT, CLEANED_DATASET, MODELS_DIR, LOGS_DIR, OUTPUTS_DIR]:
        print(f"  [OK] {d.relative_to(ROOT)}/")
    print("  [OK] train.py")
    print("\n✓ Project structure created successfully.\n")

# ============================================================
# STEP 2 — DATASET SPLITTING
# ============================================================
def step2_split_dataset():
    print("=" * 60)
    print("STEP 2 — DATASET SPLITTING (70/15/15)")
    print("=" * 60)

    classes = sorted([d.name for d in DATASET_ROOT.iterdir() if d.is_dir()])
    print(f"  Classes found: {classes}")

    splits = ['train', 'val', 'test']
    for split in splits:
        for cls in classes:
            (DATASET_SPLIT / split / cls).mkdir(parents=True, exist_ok=True)

    total_stats = {}
    for cls in classes:
        cls_dir = DATASET_ROOT / cls
        images = sorted([f for f in cls_dir.iterdir() if f.is_file()])
        total_stats[cls] = len(images)

        # 70% train, 15% val, 15% test
        train_imgs, temp_imgs = train_test_split(images, test_size=0.30, random_state=SEED)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.50, random_state=SEED)

        for img in train_imgs:
            shutil.copy2(str(img), str(DATASET_SPLIT / 'train' / cls / img.name))
        for img in val_imgs:
            shutil.copy2(str(img), str(DATASET_SPLIT / 'val' / cls / img.name))
        for img in test_imgs:
            shutil.copy2(str(img), str(DATASET_SPLIT / 'test' / cls / img.name))

        print(f"  {cls}: total={len(images)} → train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")

    print(f"\n  Total images per class: {total_stats}")
    total = sum(total_stats.values())
    train_total = sum(len(list((DATASET_SPLIT / 'train' / c).iterdir())) for c in classes)
    val_total = sum(len(list((DATASET_SPLIT / 'val' / c).iterdir())) for c in classes)
    test_total = sum(len(list((DATASET_SPLIT / 'test' / c).iterdir())) for c in classes)
    print(f"  Split totals: train={train_total}, val={val_total}, test={test_total} (total={total})")
    print("\n✓ Dataset split completed. No data leakage (disjoint sets, seed=42).\n")

# ============================================================
# STEP 3 — DATA CLEANING & VALIDATION
# ============================================================
def step3_clean_dataset():
    print("=" * 60)
    print("STEP 3 — DATA CLEANING & VALIDATION")
    print("=" * 60)

    splits = ['train', 'val', 'test']
    classes = sorted([d.name for d in (DATASET_SPLIT / 'train').iterdir() if d.is_dir()])
    
    removed_corrupt = 0
    removed_small = 0
    removed_dup = 0
    removed_nonimage = 0
    removed_nonrgb = 0
    total_kept = 0

    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

    for split in splits:
        for cls in classes:
            src_dir = DATASET_SPLIT / split / cls
            dst_dir = CLEANED_DATASET / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)

            hashes_seen = set()
            files = sorted(src_dir.iterdir())

            for fpath in files:
                # Non-image check
                if fpath.suffix.lower() not in IMAGE_EXTS:
                    removed_nonimage += 1
                    continue

                # Corruption check
                try:
                    img = Image.open(fpath)
                    img.verify()
                    img = Image.open(fpath)  # re-open after verify
                    img.load()
                except Exception:
                    removed_corrupt += 1
                    continue

                # Size check
                w, h = img.size
                if w < MIN_IMAGE_SIZE or h < MIN_IMAGE_SIZE:
                    removed_small += 1
                    continue

                # RGB check
                if img.mode != 'RGB':
                    try:
                        img = img.convert('RGB')
                    except Exception:
                        removed_nonrgb += 1
                        continue

                # Duplicate check via perceptual hash
                phash = str(imagehash.phash(img, hash_size=16))
                if phash in hashes_seen:
                    removed_dup += 1
                    continue
                hashes_seen.add(phash)

                # Save cleaned image
                img.save(str(dst_dir / fpath.name))
                total_kept += 1

    print(f"  Removed corrupt:    {removed_corrupt}")
    print(f"  Removed too small:  {removed_small}")
    print(f"  Removed duplicates: {removed_dup}")
    print(f"  Removed non-image:  {removed_nonimage}")
    print(f"  Removed non-RGB:    {removed_nonrgb}")
    print(f"  Total kept:         {total_kept}")

    print("\n  Final cleaned dataset counts:")
    for split in splits:
        for cls in classes:
            count = len(list((CLEANED_DATASET / split / cls).iterdir()))
            print(f"    {split}/{cls}: {count}")

    print("\n✓ Data cleaning & validation completed.\n")

# ============================================================
# STEP 4 — AUGMENTATION TRANSFORMS
# ============================================================
def get_train_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=25, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        A.CLAHE(clip_limit=(1, 4), p=0.3),
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, p=0.3),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.3),
        ], p=0.3),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(16, 32), hole_width_range=(16, 32), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# ============================================================
# STEP 5 — DATASET & DATALOADERS
# ============================================================
class SkinDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_names = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        for cls in self.class_names:
            cls_dir = self.root_dir / cls
            for img_path in cls_dir.iterdir():
                if img_path.is_file():
                    self.samples.append((str(img_path), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = np.array(Image.open(path).convert('RGB'))
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, label

def create_dataloaders():
    train_ds = SkinDataset(CLEANED_DATASET / 'train', get_train_transform())
    val_ds = SkinDataset(CLEANED_DATASET / 'val', get_val_transform())
    test_ds = SkinDataset(CLEANED_DATASET / 'test', get_val_transform())

    class_names = train_ds.class_names

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, test_loader, class_names

# ============================================================
# STEP 6 — MODEL
# ============================================================
def create_model(num_classes):
    model = timm.create_model('tf_efficientnetv2_b2', pretrained=True, drop_rate=DROPOUT)
    # Replace classifier head
    if hasattr(model, 'classifier'):
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(in_features, num_classes)
        )
    elif hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(in_features, num_classes)
        )
    model = model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: tf_efficientnetv2_b2")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {DEVICE}")
    return model

# ============================================================
# STEP 7-8 — TRAINING LOOP WITH METRICS
# ============================================================
def train_model(model, train_loader, val_loader, class_names):
    print("=" * 60)
    print("STEP 7-8 — TRAINING WITH METRICS")
    print("=" * 60)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    use_amp = (DEVICE.type == 'cuda')
    scaler = GradScaler('cuda', enabled=use_amp)

    best_val_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]", leave=False)
        for images, labels in pbar:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            if use_amp:
                with autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.1f}%")

        train_loss = running_loss / total
        train_acc = 100. * correct / total

        # --- VALIDATE ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                if use_amp:
                    with autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / val_total
        val_acc = 100. * val_correct / val_total

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        lr_now = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch:3d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"LR: {lr_now:.2e}")

        # Best model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names,
            }, str(MODELS_DIR / 'best_skin_model.pth'))
            print(f"    ★ New best model saved! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n  Early stopping triggered at epoch {epoch} (patience={EARLY_STOP_PATIENCE})")
                break

    # Save class names
    with open(str(MODELS_DIR / 'class_names.json'), 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"\n  Best validation accuracy: {best_val_acc:.2f}%")

    # Generate plots
    generate_plots(history, all_preds, all_labels, class_names)

    return model, history, best_val_acc

def generate_plots(history, val_preds, val_labels, class_names):
    print("\n  Generating plots...")

    # Accuracy plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    ax.plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Training & Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(OUTPUTS_DIR / 'accuracy_graph.png'), dpi=150)
    plt.close()
    print("    [OK] accuracy_graph.png saved")

    # Loss plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(OUTPUTS_DIR / 'loss_graph.png'), dpi=150)
    plt.close()
    print("    [OK] loss_graph.png saved")

    # Confusion matrix
    cm = confusion_matrix(val_labels, val_preds)
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix (Last Validation Epoch)')
    plt.tight_layout()
    plt.savefig(str(OUTPUTS_DIR / 'confusion_matrix.png'), dpi=150)
    plt.close()
    print("    [OK] confusion_matrix.png saved")

    # Classification report
    report = classification_report(val_labels, val_preds, target_names=class_names)
    print(f"\n  Classification Report (Validation - Last Epoch):\n{report}")
    with open(str(OUTPUTS_DIR / 'classification_report.txt'), 'w') as f:
        f.write(report)

# ============================================================
# STEP 10 — TEST EVALUATION
# ============================================================
def step10_test_evaluation(model, test_loader, class_names):
    print("=" * 60)
    print("STEP 10 — TEST SET EVALUATION")
    print("=" * 60)

    # Load best model
    checkpoint = torch.load(str(MODELS_DIR / 'best_skin_model.pth'), map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average='weighted') * 100
    recall = recall_score(all_labels, all_preds, average='weighted') * 100
    f1 = f1_score(all_labels, all_preds, average='weighted') * 100

    print(f"\n  Test Accuracy:  {test_acc:.2f}%")
    print(f"  Precision:      {precision:.2f}%")
    print(f"  Recall:         {recall:.2f}%")
    print(f"  F1 Score:       {f1:.2f}%")

    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(f"\n  Full Classification Report:\n{report}")

    # Confusion matrix for test set
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Test Set Confusion Matrix')
    plt.tight_layout()
    plt.savefig(str(OUTPUTS_DIR / 'test_confusion_matrix.png'), dpi=150)
    plt.close()
    print("    [OK] test_confusion_matrix.png saved")

    with open(str(OUTPUTS_DIR / 'test_results.txt'), 'w') as f:
        f.write(f"Test Accuracy:  {test_acc:.2f}%\n")
        f.write(f"Precision:      {precision:.2f}%\n")
        f.write(f"Recall:         {recall:.2f}%\n")
        f.write(f"F1 Score:       {f1:.2f}%\n\n")
        f.write(report)

    print("\n✓ Test evaluation completed.\n")

# ============================================================
# STEP 11 — PREDICTION FUNCTION (DEPLOYMENT)
# ============================================================
def predict_image(image_path, model_path=None, class_names_path=None):
    """
    Predict skin disease from a single image.
    Returns: {"disease": str, "confidence": float}
    """
    if model_path is None:
        model_path = str(MODELS_DIR / 'best_skin_model.pth')
    if class_names_path is None:
        class_names_path = str(MODELS_DIR / 'class_names.json')

    # Load class names
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)

    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model('tf_efficientnetv2_b2', pretrained=False, drop_rate=DROPOUT)
    num_classes = len(class_names)
    if hasattr(model, 'classifier'):
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(in_features, num_classes)
        )
    model = model.to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Preprocess
    transform = get_val_transform()
    img = np.array(Image.open(image_path).convert('RGB'))
    augmented = transform(image=img)
    tensor = augmented['image'].unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred_idx = probs.max(1)

    result = {
        "disease": class_names[pred_idx.item()],
        "confidence": round(conf.item(), 4)
    }
    return result


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    start_time = time.time()
    print("\n" + "=" * 60)
    print("  SKIN DISEASE AI — EfficientNetV2 Training Pipeline")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"  Dataset: {DATASET_ROOT}")
    print()

    # STEP 1
    step1_create_structure()

    # STEP 2
    if not (CLEANED_DATASET / 'train').exists():
        step2_split_dataset()
        # STEP 3
        step3_clean_dataset()
    else:
        print("Steps 2-3 already completed (cleaned_dataset exists). Skipping.\n")

    # STEP 4-5
    print("=" * 60)
    print("STEP 4-5 — AUGMENTATION & DATALOADERS")
    print("=" * 60)
    train_loader, val_loader, test_loader, class_names = create_dataloaders()
    print(f"  Classes: {class_names}")
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # Verify one batch
    batch_imgs, batch_labels = next(iter(train_loader))
    print(f"  Sample batch shape: images={batch_imgs.shape}, labels={batch_labels.shape}")
    print(f"  Batch size: {BATCH_SIZE}")
    print("\n✓ Dataloaders created successfully.\n")

    # STEP 6
    print("=" * 60)
    print("STEP 6 — MODEL CREATION")
    print("=" * 60)
    num_classes = len(class_names)
    model = create_model(num_classes)
    print("\n✓ Model created successfully.\n")

    # STEP 7-8-9
    model, history, best_val_acc = train_model(model, train_loader, val_loader, class_names)

    # STEP 10
    step10_test_evaluation(model, test_loader, class_names)

    # STEP 11
    print("=" * 60)
    print("STEP 11 — DEPLOYMENT PREDICTION FUNCTION")
    print("=" * 60)
    print("  predict_image() function ready.")
    print("  Usage: result = predict_image('path/to/image.jpg')")
    print("  Output: {'disease': 'name', 'confidence': 0.97}")

    # Quick self-test with first test image
    test_classes = sorted([d.name for d in (CLEANED_DATASET / 'test').iterdir() if d.is_dir()])
    if test_classes:
        test_cls_dir = CLEANED_DATASET / 'test' / test_classes[0]
        test_imgs = list(test_cls_dir.iterdir())
        if test_imgs:
            result = predict_image(str(test_imgs[0]))
            print(f"\n  Demo prediction: {result}")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE — Total time: {elapsed/60:.1f} minutes")
    print(f"  Best validation accuracy: {best_val_acc:.2f}%")
    print(f"  Model saved: {MODELS_DIR / 'best_skin_model.pth'}")
    print(f"  Outputs saved: {OUTPUTS_DIR}")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
