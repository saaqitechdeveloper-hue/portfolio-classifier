from pathlib import Path
import sys

import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = Path("models/best_model.pth")
IMAGE_SIZE = 224
DEVICE = torch.device("cpu")

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# ARGUMENTS
# ============================================================

if len(sys.argv) != 2:
    print()
    print("Usage:")
    print("  ./.venv/bin/python3 bulk_predict.py uploads/")
    print()
    sys.exit(1)

input_path = Path(sys.argv[1])

if not input_path.exists():
    print(f"ERROR: Path not found: {input_path}")
    sys.exit(1)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading AI model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

classes = checkpoint["classes"]

model = mobilenet_v3_small(weights=None)

in_features = model.classifier[-1].in_features

model.classifier[-1] = torch.nn.Linear(
    in_features,
    len(classes)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# FIND IMAGES
# ============================================================

if input_path.is_file():

    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(
            f"ERROR: Unsupported image type: "
            f"{input_path.suffix}"
        )
        sys.exit(1)

    image_files = [input_path]

else:

    image_files = sorted([
        path
        for path in input_path.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ])


if not image_files:

    print("No supported images found.")

    sys.exit(1)


# ============================================================
# PREDICT
# ============================================================

print()
print("=" * 100)
print("BULK IMAGE CLASSIFICATION")
print("=" * 100)
print()

results = []

for image_path in image_files:

    try:

        image = Image.open(image_path).convert("RGB")

        image_tensor = (
            transform(image)
            .unsqueeze(0)
            .to(DEVICE)
        )

        with torch.no_grad():

            outputs = model(image_tensor)

            probabilities = F.softmax(
                outputs,
                dim=1
            )[0]

        top_k = min(3, len(classes))

        top_probabilities, top_indices = torch.topk(
            probabilities,
            top_k
        )

        best_index = top_indices[0].item()

        best_class = classes[best_index]

        best_confidence = (
            top_probabilities[0].item() * 100
        )

        results.append({
            "path": image_path,
            "class": best_class,
            "confidence": best_confidence,
        })

        print(
            f"{image_path.name:30} "
            f"→ {best_class:20} "
            f"{best_confidence:6.2f}%"
        )

    except Exception as e:

        print(
            f"{image_path.name:30} "
            f"→ ERROR: {e}"
        )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)

print()

print(
    f"Images processed: {len(results)}"
)

print()

category_counts = {}

for result in results:

    category = result["class"]

    category_counts[category] = (
        category_counts.get(category, 0) + 1
    )


for category in classes:

    count = category_counts.get(category, 0)

    print(
        f"{category:25} {count}"
    )

print()

print("=" * 100)
print("DONE")
print("=" * 100)
