from pathlib import Path
import random
import json

import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small


MODEL_PATH = Path("models/best_model.pth")
DATASET_PATH = Path("dataset")
IMAGE_SIZE = 224
DEVICE = torch.device("cpu")

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

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

print()
print("=" * 80)
print("TESTING ALL CATEGORIES")
print("=" * 80)

total = 0
correct = 0

for category in classes:

    category_path = DATASET_PATH / category

    images = [
        image
        for image in category_path.iterdir()
        if image.is_file()
        and image.suffix.lower() in extensions
    ]

    if not images:
        continue

    selected_images = random.sample(
        images,
        min(3, len(images))
    )

    print()
    print(f"--- {category} ---")

    for image_path in selected_images:

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            print(f"{image_path.name}: could not open")
            continue

        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]

        best_index = probabilities.argmax().item()

        prediction = classes[best_index]
        confidence = probabilities[best_index].item() * 100

        is_correct = prediction == category

        if is_correct:
            correct += 1

        total += 1

        result = "OK" if is_correct else "WRONG"

        print(
            f"{result:5} | "
            f"{image_path.name:25} | "
            f"Actual: {category:20} | "
            f"Predicted: {prediction:20} | "
            f"{confidence:.2f}%"
        )

print()
print("=" * 80)

if total:
    accuracy = correct / total * 100

    print(
        f"Sample accuracy: "
        f"{correct}/{total} = {accuracy:.2f}%"
    )

print("=" * 80)
