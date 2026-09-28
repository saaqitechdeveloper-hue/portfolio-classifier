from pathlib import Path
import sys
import shutil

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

MODEL_PATH = Path("models/best_model_with_others.pth")
OUTPUT_DIR = Path("organized")
DEVICE = torch.device("cpu")

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if len(sys.argv) != 2:
    print("Usage: ./.venv/bin/python3 organize_images.py uploads/")
    sys.exit(1)

input_path = Path(sys.argv[1])

if not input_path.exists():
    print(f"Path not found: {input_path}")
    sys.exit(1)

checkpoint = torch.load(
    MODEL_PATH, map_location=DEVICE, weights_only=True
)
classes = checkpoint["classes"]

model = mobilenet_v3_small(weights=None)
model.classifier[-1] = torch.nn.Linear(
    model.classifier[-1].in_features, len(classes)
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

if input_path.is_file():
    files = [input_path]
else:
    files = sorted(
        p for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

if not files:
    print("No supported images found.")
    sys.exit(1)

OUTPUT_DIR.mkdir(exist_ok=True)
counts = {name: 0 for name in classes}

for path in files:
    try:
        with Image.open(path) as im:
            image = im.convert("RGB")

        tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probs = F.softmax(model(tensor), dim=1)[0]

        idx = probs.argmax().item()
        category = classes[idx]
        confidence = probs[idx].item() * 100

        dest_dir = OUTPUT_DIR / category
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest = dest_dir / path.name
        if dest.exists():
            dest = dest_dir / f"{path.stem}_{counts[category]+1}{path.suffix}"

        shutil.copy2(path, dest)
        counts[category] += 1

        print(f"{path.name} -> {category} ({confidence:.2f}%)")

    except Exception as e:
        print(f"ERROR {path.name}: {e}")

print("\nSUMMARY")
print(f"Processed: {sum(counts.values())}")
for category, count in counts.items():
    print(f"{category:22} {count}")

print(f"\nCopies saved in: {OUTPUT_DIR.resolve()}")
print("Original images were not moved or deleted.")
