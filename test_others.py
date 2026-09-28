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

# Load checkpoint
checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

classes = checkpoint["classes"]

# Build model
model = mobilenet_v3_small(weights=None)

in_features = model.classifier[-1].in_features
model.classifier[-1] = torch.nn.Linear(
    in_features,
    len(classes)
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

images = sorted([
    p for p in OTHERS_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
])

print(f"\nTesting {len(images)} Others images...\n")
print("=" * 80)

for image_path in images:

    try:
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(tensor)
            probabilities = F.softmax(output, dim=1)[0]

        top_probs, top_indices = torch.topk(
            probabilities,
            k=min(3, len(classes))
        )

        top1_class = classes[top_indices[0].item()]
        top1_conf = top_probs[0].item() * 100

        top2_class = classes[top_indices[1].item()]
        top2_conf = top_probs[1].item() * 100

        margin = top1_conf - top2_conf

        print(f"\n{image_path.name}")
        print(f"  Top 1 : {top1_class:<20} {top1_conf:6.2f}%")
        print(f"  Top 2 : {top2_class:<20} {top2_conf:6.2f}%")
        print(f"  Margin: {margin:6.2f}%")

        print("  Top 3:")
        for prob, idx in zip(top_probs, top_indices):
            print(
                f"    {classes[idx.item()]:<20}"
                f"{prob.item() * 100:6.2f}%"
            )

    except Exception as e:
        print(f"\nERROR: {image_path.name}")
        print(f"  {e}")

print("\n" + "=" * 80)
print("Done.")
