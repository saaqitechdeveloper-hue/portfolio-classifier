from pathlib import Path
import json
import random
import hashlib

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset

from torchvision import datasets, transforms
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIG
# ============================================================

DATASET_DIR = Path("dataset")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = 224

BATCH_SIZE = 16

EPOCHS = 15

LEARNING_RATE = 0.001

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

NUM_WORKERS = 0


# ============================================================
# SEED
# ============================================================

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cpu")

print("=" * 70)
print("PORTFOLIO IMAGE CLASSIFIER")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Dataset: {DATASET_DIR}")
print()


# ============================================================
# IMAGE HASH
# ============================================================

def get_file_hash(path):

    sha = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            sha.update(chunk)

    return sha.hexdigest()


# ============================================================
# TRANSFORMS
# ============================================================

weights = MobileNet_V3_Small_Weights.DEFAULT


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.75, 1.0)
    ),

    transforms.RandomRotation(
        degrees=5
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


eval_transform = transforms.Compose([
    transforms.Resize(256),

    transforms.CenterCrop(
        IMAGE_SIZE
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading dataset...")

base_dataset = datasets.ImageFolder(
    DATASET_DIR
)

classes = base_dataset.classes

num_classes = len(classes)


print()
print(f"Classes ({num_classes}):")

for i, class_name in enumerate(classes):

    print(f"  {i}: {class_name}")

print()


# ============================================================
# REMOVE EXACT DUPLICATES FROM SPLIT GROUPS
# ============================================================

print("Grouping duplicate images...")

hash_groups = {}

for index, (path, label) in enumerate(
    base_dataset.samples
):

    file_hash = get_file_hash(
        path
    )

    if file_hash not in hash_groups:

        hash_groups[file_hash] = []

    hash_groups[file_hash].append(
        index
    )


groups = list(
    hash_groups.values()
)

print(
    f"Unique image groups: {len(groups)}"
)

print(
    f"Duplicate groups: "
    f"{sum(1 for g in groups if len(g) > 1)}"
)

print()


# ============================================================
# CREATE GROUP LABELS
# ============================================================

group_labels = []

for group in groups:

    labels = [
        base_dataset.samples[i][1]
        for i in group
    ]

    # Exact duplicates should belong to same class.
    if len(set(labels)) != 1:

        raise RuntimeError(
            "Found identical image content "
            "inside different categories."
        )

    group_labels.append(
        labels[0]
    )


group_indices = list(
    range(len(groups))
)


# ============================================================
# TRAIN / VAL / TEST SPLIT
# ============================================================

train_groups, temp_groups, train_labels, temp_labels = (
    train_test_split(
        group_indices,
        group_labels,
        test_size=0.30,
        stratify=group_labels,
        random_state=RANDOM_SEED
    )
)


val_groups, test_groups = (
    train_test_split(
        temp_groups,
        test_size=0.50,
        stratify=temp_labels,
        random_state=RANDOM_SEED
    )
)


def expand_groups(group_ids):

    result = []

    for group_id in group_ids:

        result.extend(
            groups[group_id]
        )

    return result


train_indices = expand_groups(
    train_groups
)

val_indices = expand_groups(
    val_groups
)

test_indices = expand_groups(
    test_groups
)


print("=" * 70)
print("DATASET SPLIT")
print("=" * 70)

print(
    f"Train images:      {len(train_indices)}"
)

print(
    f"Validation images: {len(val_indices)}"
)

print(
    f"Test images:       {len(test_indices)}"
)

print()


# ============================================================
# DATASETS
# ============================================================

train_dataset_full = datasets.ImageFolder(
    DATASET_DIR,
    transform=train_transform
)

eval_dataset_full = datasets.ImageFolder(
    DATASET_DIR,
    transform=eval_transform
)


train_dataset = Subset(
    train_dataset_full,
    train_indices
)

val_dataset = Subset(
    eval_dataset_full,
    val_indices
)

test_dataset = Subset(
    eval_dataset_full,
    test_indices
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# MODEL
# ============================================================

print("Loading MobileNetV3 Small...")

model = mobilenet_v3_small(
    weights=weights
)


# Freeze pretrained features

for parameter in model.features.parameters():

    parameter.requires_grad = False


# Replace final classifier

in_features = (
    model.classifier[-1].in_features
)

model.classifier[-1] = nn.Linear(
    in_features,
    num_classes
)


model = model.to(
    DEVICE
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in train_loader:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )


        optimizer.zero_grad()


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()

        optimizer.step()


        running_loss += (
            loss.item()
        )


        predictions = outputs.argmax(
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)


    return (
        running_loss / len(train_loader),
        correct / total
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(loader):

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0


    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels
            )


            running_loss += (
                loss.item()
            )


            predictions = outputs.argmax(
                dim=1
            )


            correct += (
                predictions == labels
            ).sum().item()


            total += labels.size(0)


    return (
        running_loss / len(loader),
        correct / total
    )


# ============================================================
# TRAINING LOOP
# ============================================================

print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)
print()


best_val_accuracy = 0.0

best_model_path = (
    MODEL_DIR / "best_model.pth"
)


for epoch in range(
    1,
    EPOCHS + 1
):

    train_loss, train_accuracy = (
        train_one_epoch()
    )


    val_loss, val_accuracy = (
        evaluate(
            val_loader
        )
    )


    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy * 100:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy * 100:.2f}%"
    )


    if val_accuracy > best_val_accuracy:

        best_val_accuracy = (
            val_accuracy
        )


        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "classes":
                    classes,

                "image_size":
                    IMAGE_SIZE,

                "best_val_accuracy":
                    best_val_accuracy,
            },

            best_model_path
        )


        print(
            f"  ✓ Best model saved: "
            f"{best_val_accuracy * 100:.2f}%"
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("Loading best model...")


checkpoint = torch.load(
    best_model_path,
    map_location=DEVICE,
    weights_only=True
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


# ============================================================
# FINAL TEST
# ============================================================

test_loss, test_accuracy = (
    evaluate(
        test_loader
    )
)


print()
print("=" * 70)
print("FINAL TEST RESULT")
print("=" * 70)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# PER CLASS ACCURACY
# ============================================================

class_correct = [
    0
    for _ in range(num_classes)
]

class_total = [
    0
    for _ in range(num_classes)
]


model.eval()


with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(
            images.to(DEVICE)
        )

        predictions = (
            outputs.argmax(
                dim=1
            ).cpu()
        )


        for label, prediction in zip(
            labels,
            predictions
        ):

            class_total[
                label.item()
            ] += 1


            if (
                label.item()
                == prediction.item()
            ):

                class_correct[
                    label.item()
                ] += 1


print()
print("Per-class accuracy:")

for class_id, class_name in enumerate(
    classes
):

    if class_total[class_id] == 0:
        continue


    accuracy = (
        class_correct[class_id]
        / class_total[class_id]
    )


    print(
        f"  {class_name:25} "
        f"{accuracy * 100:.2f}% "
        f"({class_correct[class_id]}/"
        f"{class_total[class_id]})"
    )


# ============================================================
# SAVE CLASS NAMES
# ============================================================

classes_path = (
    MODEL_DIR / "classes.json"
)


with open(
    classes_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        classes,
        f,
        indent=2,
        ensure_ascii=False
    )


print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Model saved at: "
    f"{best_model_path}"
)

print(
    f"Classes saved at: "
    f"{classes_path}"
)
