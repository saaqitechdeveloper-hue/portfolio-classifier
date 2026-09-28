from pathlib import Path
from PIL import Image
import hashlib
from collections import defaultdict

DATASET_DIR = Path("dataset")

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

MIN_WIDTH = 32
MIN_HEIGHT = 32


def sha256_file(path):
    sha = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            sha.update(chunk)

    return sha.hexdigest()


def main():
    if not DATASET_DIR.exists():
        print("ERROR: dataset/ folder not found.")
        return

    categories = sorted(
        [
            folder
            for folder in DATASET_DIR.iterdir()
            if folder.is_dir()
        ],
        key=lambda x: x.name.lower()
    )

    if not categories:
        print("ERROR: No category folders found.")
        return

    print("=" * 70)
    print("DATASET VALIDATION")
    print("=" * 70)

    print("\nCategories found:")

    for category in categories:
        print(f"  - {category.name}")

    print()

    total_images = 0
    total_invalid = 0

    hashes = defaultdict(list)

    category_counts = {}

    for category in categories:
        valid_count = 0
        invalid_count = 0

        print(f"\nChecking: {category.name}")

        files = sorted(category.rglob("*"))

        for path in files:

            if not path.is_file():
                continue

            if path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            total_images += 1

            try:
                with Image.open(path) as img:

                    width, height = img.size

                    if width < MIN_WIDTH or height < MIN_HEIGHT:
                        print(
                            f"  INVALID SIZE: {path} "
                            f"({width}x{height})"
                        )

                        invalid_count += 1
                        total_invalid += 1
                        continue

                    # Force image decoding to catch corrupted files
                    img.verify()

                file_hash = sha256_file(path)

                hashes[file_hash].append(path)

                valid_count += 1

            except Exception as e:
                print(f"  CORRUPTED: {path}")
                print(f"  Reason: {e}")

                invalid_count += 1
                total_invalid += 1

        category_counts[category.name] = valid_count

        print(f"  Valid:   {valid_count}")
        print(f"  Invalid: {invalid_count}")

    print("\n" + "=" * 70)
    print("CATEGORY SUMMARY")
    print("=" * 70)

    for category, count in category_counts.items():
        print(f"{category:25} {count:6}")

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(f"Total image files: {total_images}")
    print(f"Invalid images:    {total_invalid}")

    duplicate_groups = []

    for file_hash, paths in hashes.items():
        if len(paths) > 1:
            duplicate_groups.append(paths)

    print(f"Duplicate groups:  {len(duplicate_groups)}")

    if duplicate_groups:

        print("\nDuplicate images:")

        for group in duplicate_groups[:20]:
            print("\nGROUP:")

            for path in group:
                print(f"  {path}")

        if len(duplicate_groups) > 20:
            print(
                f"\n... and "
                f"{len(duplicate_groups) - 20} more groups."
            )

    print("\nValidation complete.")


if __name__ == "__main__":
    main()
