
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small
from torchvision import transforms
from PIL import Image
from collections import defaultdict

MODEL_PATH = Path("models/best_model.pth")
DATASET_DIR = Path("dataset")
OTHERS_DIR = DATASET_DIR / "Others"

DEVICE = torch.device("cpu")
IMAGE_SIZE = 224

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------------------------
# Load trained model
# ---------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

classes = checkpoint["classes"]

model = mobilenet_v3_small(weights=None)
in_features = model.classifier[-1].in_features
model.classifier[-1] = torch.nn.Linear(
    in_features, len(classes)
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

# ---------------------------------
# Extract MobileNet feature vectors
# ---------------------------------

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def get_feature(path):
    image = Image.open(path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # MobileNet feature extractor
        x = model.features(tensor)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)

        # Normalize for cosine distance
        x = F.normalize(x, p=2, dim=1)

    return x.squeeze(0).cpu()

# ---------------------------------
# Load known-category embeddings
# ---------------------------------

known = defaultdict(list)
errors = 0

print("\nExtracting known category features...")

for class_name in classes:
    folder = DATASET_DIR / class_name

    if not folder.exists():
        print(f"Missing folder: {folder}")
        continue

    files = sorted([
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in EXTENSIONS
    ])

    for path in files:
        try:
            known[class_name].append(
                (path.name, get_feature(path))
            )
        except Exception as e:
            errors += 1
            print(f"ERROR {path}: {e}")

    print(f"{class_name}: {len(known[class_name])} images")

# ---------------------------------
# Build normalized class prototypes
# ---------------------------------

prototypes = {}

for class_name, items in known.items():
    if not items:
        continue

    vectors = torch.stack([item[1] for item in items])
    prototype = vectors.mean(dim=0)
    prototype = F.normalize(prototype, p=2, dim=0)

    prototypes[class_name] = prototype

# ---------------------------------
# Cosine distance helper
# ---------------------------------

def nearest_distance(vector, prototype_map):
    distances = {}

    for name, prototype in prototype_map.items():
        # cosine distance = 1 - cosine similarity
        distance = 1 - torch.dot(vector, prototype).item()
        distances[name] = distance

    best_class = min(distances, key=distances.get)
    return best_class, distances[best_class]

# ---------------------------------
# Known images: leave-one-out prototype
# ---------------------------------

known_distances = []

print("\nEvaluating known images (leave-one-out)...")

for class_name, items in known.items():
    if len(items) < 2:
        continue

    vectors = torch.stack([item[1] for item in items])

    for filename, vector in items:
        # Prototype excludes this image
        other_vectors = torch.stack([
            v for name, v in items if name != filename
        ])

        own_prototype = F.normalize(
            other_vectors.mean(dim=0), p=2, dim=0
        )

        other_prototypes = dict(prototypes)
        other_prototypes[class_name] = own_prototype

        predicted_class, distance = nearest_distance(
            vector, other_prototypes
        )

        known_distances.append(distance)

# ---------------------------------
# Evaluate Others
# ---------------------------------

other_files = sorted([
    p for p in OTHERS_DIR.rglob("*")
    if p.is_file()
    and p.suffix.lower() in EXTENSIONS
])

others_distances = []

print(f"\nEvaluating Others: {len(other_files)} images")

for path in other_files:
    try:
        vector = get_feature(path)

        predicted_class, distance = nearest_distance(
            vector, prototypes
        )

        others_distances.append(distance)

        print(
            f"{path.name:<42} "
            f"nearest={predicted_class:<20} "
            f"distance={distance:.4f}"
        )

    except Exception as e:
        errors += 1
        print(f"ERROR {path}: {e}")

# ---------------------------------
# Statistics
# ---------------------------------

def percentile(values, p):
    values = sorted(values)
    if not values:
        return float("nan")
    index = int((len(values) - 1) * p)
    return values[index]

def report(label, values):
    print(f"\n{label} ({len(values)} images)")
    if not values:
        print("No data")
        return

    print(f"Min    : {min(values):.4f}")
    print(f"25%    : {percentile(values, 0.25):.4f}")
    print(f"Median : {percentile(values, 0.50):.4f}")
    print(f"75%    : {percentile(values, 0.75):.4f}")
    print(f"Max    : {max(values):.4f}")

print("\n" + "=" * 70)
print("FEATURE DISTANCE SUMMARY")
print("=" * 70)

report("KNOWN (leave-one-out)", known_distances)
report("OTHERS", others_distances)

if known_distances and others_distances:
    known_95 = percentile(known_distances, 0.95)
    others_25 = percentile(others_distances, 0.25)

    print("\nCOMPARISON")
    print(f"Known 95th percentile: {known_95:.4f}")
    print(f"Others 25th percentile: {others_25:.4f}")

    if known_95 < others_25:
        print("There is a gap between these sampled distributions.")
    else:
        print("The distributions overlap; one distance threshold")
        print("may reject some known images or accept some Others.")

print(f"\nImage errors: {errors}")
print("Done.")

