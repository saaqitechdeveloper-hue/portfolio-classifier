from pathlib import Path
import json
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


# ============================================================
# CHECK ARGUMENT
# ============================================================

if len(sys.argv) != 2:
    print("Usage:")
    print("  ./.venv/bin/python3 predict.py path/to/image.jpg")
    sys.exit(1)


image_path = Path(sys.argv[1])


if not image_path.exists():
    print(f"ERROR: Image not found: {image_path}")
    sys.exit(1)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

classes = checkpoint["classes"]


# ============================================================
# MODEL
# ============================================================

model = mobilenet_v3_small(
    weights=None
)

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
# TRANSFORM
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
# LOAD IMAGE
# ============================================================

try:

    image = Image.open(image_path).convert("RGB")

except Exception as e:

    print(f"ERROR: Could not open image.")
    print(e)

    sys.exit(1)


image_tensor = transform(
    image
).unsqueeze(0).to(DEVICE)


# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    outputs = model(
        image_tensor
    )

    probabilities = F.softmax(
        outputs,
        dim=1
    )[0]


# ============================================================
# TOP 3
# ============================================================

top_k = min(
    3,
    len(classes)
)

top_probabilities, top_indices = (
    torch.topk(
        probabilities,
        top_k
    )
)


# ============================================================
# RESULT
# ============================================================

best_index = top_indices[0].item()

best_class = classes[
    best_index
]

best_confidence = (
    top_probabilities[0].item()
)


print()
print("=" * 60)
print("IMAGE CLASSIFICATION")
print("=" * 60)

print(
    f"Image:      {image_path}"
)

print(
    f"Prediction: {best_class}"
)

print(
    f"Confidence: {best_confidence * 100:.2f}%"
)

print()

print("Top predictions:")

for rank in range(top_k):

    index = top_indices[
        rank
    ].item()

    probability = (
        top_probabilities[
            rank
        ].item()
    )

    print(
        f"{rank + 1}. "
        f"{classes[index]:20} "
        f"{probability * 100:.2f}%"
    )

print(
    "=" * 60
)
