from pathlib import Path

import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small


MODEL_PATH = Path("models/best_model.pth")
DEVICE = torch.device("cpu")
IMAGE_SIZE = 224

CATEGORIES = [
    "Banner",
    "Social-Media",
]

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
    ".webp",
}

total = 0
correct = 0

print()
print("=" * 90)
print("BANNER vs SOCIAL-MEDIA ANALYSIS")
print("=" * 90)

for category in CATEGORIES:

    folder = Path("dataset") / category

    print()
    print(f"--- {category} ---")

    for image_path in sorted(folder.iterdir()):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in extensions:
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue

        tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(tensor)
            probabilities = F.softmax(output, dim=1)[0]

        index = probabilities.argmax().item()

        prediction = classes[index]
        confidence = probabilities[index].item() * 100

        total += 1

        if prediction == category:
            correct += 1

        if prediction != category:
            print(
                f"WRONG | "
                f"{image_path.name:25} | "
                f"Actual: {category:15} | "
                f"Predicted: {prediction:15} | "
                f"{confidence:.2f}%"
            )

print()
print("=" * 90)

accuracy = correct / total * 100

print(
    f"Accuracy: {correct}/{total} = {accuracy:.2f}%"
)

print("=" * 90)
