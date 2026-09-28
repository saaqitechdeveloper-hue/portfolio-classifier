from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

MODEL_PATH = Path("models/best_model.pth")
OTHERS_DIR = Path("dataset/Others")

IMAGE_SIZE = 224
DEVICE = torch.device("cpu")

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

# -----------------------------
# Load model
# -----------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

classes = checkpoint["classes"]

model = mobilenet_v3_small(weights=None)

in_features = model.classifier[-1].in_features

model.classifier[-1] = torch.nn.Linear(
    in_features,
    len(classes)
)

model.load_state_dict(checkpoint["model_state_dict"])

model.to(DEVICE)
model.eval()

# -----------------------------
# Image transform
# -----------------------------

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Collect Others images
# -----------------------------

images = sorted([
    p for p in OTHERS_DIR.rglob("*")
    if p.is_file()
    and p.suffix.lower() in SUPPORTED_EXTENSIONS
])

print(f"\nOthers images: {len(images)}")
print("=" * 100)

results = []

# -----------------------------
# Predict
# -----------------------------

for image_path in images:

    try:
        image = Image.open(image_path).convert("RGB")

        tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(tensor)
            probabilities = F.softmax(output, dim=1)[0]

        top_probs, top_indices = torch.topk(
            probabilities,
            k=min(2, len(classes))
        )

        top1_class = classes[top_indices[0].item()]
        top1_conf = top_probs[0].item()

        top2_class = classes[top_indices[1].item()]
        top2_conf = top_probs[1].item()

        margin = top1_conf - top2_conf

        results.append({
            "file": image_path.name,
            "class": top1_class,
            "confidence": top1_conf,
            "margin": margin
        })

        print(
            f"{image_path.name:<40} "
            f"{top1_class:<20} "
            f"conf={top1_conf * 100:6.2f}% "
            f"margin={margin * 100:6.2f}%"
        )

    except Exception as e:
        print(f"ERROR: {image_path.name} -> {e}")

# -----------------------------
# Statistics
# -----------------------------

print("\n")
print("=" * 100)
print("SUMMARY")
print("=" * 100)

confidences = [
    r["confidence"]
    for r in results
]

margins = [
    r["margin"]
    for r in results
]

if confidences:

    confidences_sorted = sorted(confidences)
    margins_sorted = sorted(margins)

    def percentile(values, p):
        index = int((len(values) - 1) * p)
        return values[index]

    print(f"\nTotal Others images: {len(results)}")

    print("\nCONFIDENCE")
    print(f"Min    : {min(confidences) * 100:.2f}%")
    print(f"25%    : {percentile(confidences_sorted, 0.25) * 100:.2f}%")
    print(f"Median : {percentile(confidences_sorted, 0.50) * 100:.2f}%")
    print(f"75%    : {percentile(confidences_sorted, 0.75) * 100:.2f}%")
    print(f"Max    : {max(confidences) * 100:.2f}%")

    print("\nTOP-1 / TOP-2 MARGIN")
    print(f"Min    : {min(margins) * 100:.2f}%")
    print(f"25%    : {percentile(margins_sorted, 0.25) * 100:.2f}%")
    print(f"Median : {percentile(margins_sorted, 0.50) * 100:.2f}%")
    print(f"75%    : {percentile(margins_sorted, 0.75) * 100:.2f}%")
    print(f"Max    : {max(margins) * 100:.2f}%")

    print("\n")
    print("OTHERS PREDICTION DISTRIBUTION")
    print("-" * 50)

    counts = {}

    for r in results:
        counts[r["class"]] = counts.get(r["class"], 0) + 1

    for class_name in classes:
        print(
            f"{class_name:<20} "
            f"{counts.get(class_name, 0)}"
        )

print("\nDone.")
