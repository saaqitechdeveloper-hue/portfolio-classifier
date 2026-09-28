from pathlib import Path
from collections import defaultdict, Counter

import torch
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small
from torchvision import transforms
from PIL import Image

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

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------------------------
# Feature extraction
# ---------------------------------

def get_feature(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        x = model.features(tensor)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.normalize(x, p=2, dim=1)

    return x.squeeze(0).cpu()


def cosine_distance(vector, prototype):
    return 1.0 - torch.dot(vector, prototype).item()


def make_prototype(vectors):
    prototype = torch.stack(vectors).mean(dim=0)
    return F.normalize(prototype, p=2, dim=0)


def nearest_distance(vector, prototype_map):
    distances = {
        name: cosine_distance(vector, prototype)
        for name, prototype in prototype_map.items()
    }

    best_class = min(distances, key=distances.get)
    return best_class, distances[best_class], distances


# ---------------------------------
# Load known images
# Store each image separately with a unique index
# ---------------------------------

known = defaultdict(list)
errors = 0

print("\nExtracting known category features...")

for class_name in classes:
    folder = DATASET_DIR / class_name

    if not folder.exists():
        print(f"Missing folder: {folder}")
        continue

    files = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    for path in files:
        try:
            known[class_name].append({
                "path": path,
                "feature": get_feature(path)
            })
        except Exception as e:
            errors += 1
            print(f"ERROR {path}: {e}")

    print(f"{class_name}: {len(known[class_name])} images")


# ---------------------------------
# Build full class prototypes
# ---------------------------------

prototypes = {}

for class_name, items in known.items():
    if items:
        vectors = [item["feature"] for item in items]
        prototypes[class_name] = make_prototype(vectors)


# ---------------------------------
# Statistics helpers
# ---------------------------------

def percentile(values, p):
    values = sorted(values)

    if not values:
        return float("nan")

    index = int((len(values) - 1) * p)
    return values[index]


def report_stats(label, values):
    print(f"\n{label} (n={len(values)})")

    if not values:
        print("No data")
        return

    print(f"Min    : {min(values):.4f}")
    print(f"25%    : {percentile(values, 0.25):.4f}")
    print(f"Median : {percentile(values, 0.50):.4f}")
    print(f"75%    : {percentile(values, 0.75):.4f}")
    print(f"95%    : {percentile(values, 0.95):.4f}")
    print(f"Max    : {max(values):.4f}")


# ---------------------------------
# Known images: leave-one-out
# ---------------------------------

known_results = defaultdict(list)

print("\nEvaluating known images with leave-one-out...")

for class_name, items in known.items():
    if len(items) < 2:
        print(f"Skipping {class_name}: fewer than 2 images")
        continue

    for i, item in enumerate(items):
        # Exclude this exact image by index, not filename.
        other_vectors = [
            other["feature"]
            for j, other in enumerate(items)
            if j != i
        ]

        own_prototype = make_prototype(other_vectors)

        # Other categories retain their full prototypes.
        comparison_prototypes = dict(prototypes)
        comparison_prototypes[class_name] = own_prototype

        predicted, nearest_dist, all_distances = nearest_distance(
            item["feature"],
            comparison_prototypes
        )

        own_distance = all_distances[class_name]

        known_results[class_name].append({
            "path": item["path"],
            "predicted": predicted,
            "own_distance": own_distance,
            "nearest_distance": nearest_dist,
            "correct": predicted == class_name
        })


# ---------------------------------
# Report known classes separately
# ---------------------------------

print("\n" + "=" * 76)
print("KNOWN CATEGORY RESULTS")
print("=" * 76)

for class_name in classes:
    results = known_results.get(class_name, [])

    if not results:
        continue

    own_distances = [r["own_distance"] for r in results]
    nearest_distances = [r["nearest_distance"] for r in results]
    correct_count = sum(r["correct"] for r in results)

    print(f"\n### {class_name}")
    print(
        f"Nearest-class prediction accuracy: "
        f"{correct_count}/{len(results)} "
        f"({100 * correct_count / len(results):.1f}%)"
    )

    report_stats("Distance to own class prototype", own_distances)
    report_stats("Distance to nearest prototype", nearest_distances)

    mistakes = Counter(
        r["predicted"]
        for r in results
        if not r["correct"]
    )

    if mistakes:
        print("Misclassified as:")
        for target, count in mistakes.most_common():
            print(f"  {target}: {count}")


# ---------------------------------
# Evaluate Others
# ---------------------------------

other_files = sorted(
    p for p in OTHERS_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in EXTENSIONS
)

others_results = []
others_by_nearest_class = defaultdict(list)
errors_others = 0

print("\n" + "=" * 76)
print(f"OTHERS RESULTS ({len(other_files)} images)")
print("=" * 76)

for path in other_files:
    try:
        vector = get_feature(path)

        predicted, distance, distances = nearest_distance(
            vector, prototypes
        )

        others_results.append({
            "path": path,
            "nearest": predicted,
            "distance": distance
        })

        others_by_nearest_class[predicted].append(distance)

        print(
            f"{path.name:<38} "
            f"nearest={predicted:<20} "
            f"distance={distance:.4f}"
        )

    except Exception as e:
        errors += 1
        errors_others += 1
        print(f"ERROR {path}: {e}")


# ---------------------------------
# Others grouped by nearest class
# ---------------------------------

print("\n" + "=" * 76)
print("OTHERS GROUPED BY NEAREST CATEGORY")
print("=" * 76)

for class_name, distances in sorted(
    others_by_nearest_class.items(),
    key=lambda item: len(item[1]),
    reverse=True
):
    print(f"\n### {class_name}: {len(distances)} Others images")
    report_stats("Nearest-prototype distance", distances)


# ---------------------------------
# Compare known vs Others per category
# ---------------------------------

print("\n" + "=" * 76)
print("PER-CATEGORY DISTANCE COMPARISON")
print("=" * 76)

for class_name in classes:
    known_items = known_results.get(class_name, [])
    others_items = others_by_nearest_class.get(class_name, [])

    known_values = [
        item["nearest_distance"]
        for item in known_items
    ]

    print(f"\n### {class_name}")
    print(f"Known images: {len(known_values)}")
    print(f"Others nearest to this class: {len(others_items)}")

    if known_values:
        print(
            f"Known nearest-distance median: "
            f"{percentile(known_values, 0.50):.4f}"
        )
        print(
            f"Known nearest-distance 95th percentile: "
            f"{percentile(known_values, 0.95):.4f}"
        )

    if others_items:
        print(
            f"Others nearest-distance median: "
            f"{percentile(others_items, 0.50):.4f}"
        )
        print(
            f"Others nearest-distance 25th percentile: "
            f"{percentile(others_items, 0.25):.4f}"
        )


# ---------------------------------
# Summary
# ---------------------------------

all_known_nearest = [
    r["nearest_distance"]
    for results in known_results.values()
    for r in results
]

all_others_nearest = [
    r["distance"] for r in others_results
]

print("\n" + "=" * 76)
print("OVERALL SUMMARY")
print("=" * 76)

report_stats("Known leave-one-out nearest distance", all_known_nearest)
report_stats("Others nearest distance", all_others_nearest)

print(f"\nImage errors: {errors}")
print(f"Others image errors: {errors_others}")
print("Done.")
