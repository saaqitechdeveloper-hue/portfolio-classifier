import os
import json
import hmac
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from flask import Flask, request, jsonify

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.onnx"
CLASSES_PATH = BASE_DIR / "models" / "classes_with_others.json"

API_KEY = os.getenv("API_KEY", "").strip()

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB per image
MAX_BATCH_IMAGES = 20

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}

ALLOWED_CATEGORIES = [
    "Logo",
    "Banner",
    "UI/UX",
    "Color Separation",
    "Flyer",
    "Poster",
    "Social Media",
    "Other",
]

CATEGORY_MAPPING = {
    "Logo": "Logo",
    "Banner": "Banner",
    "UI-UX": "UI/UX",
    "Color-Separation": "Color Separation",
    "Flyer": "Flyer",
    "Poster": "Poster",
    "Social-Media": "Social Media",
    "Others": "Other",
}

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

# Load classes
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    classes = json.load(f)

# Load ONNX model once
session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
model_ready = True
print("ONNX AI model loaded successfully in Flask.")


# ============================================================
# HELPERS
# ============================================================

def check_auth():
    if not API_KEY:
        return None  # No key configured, allow open access

    auth = request.headers.get("Authorization", "")
    parts = auth.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return jsonify({"success": False, "error": "Authorization must use Bearer token"}), 401

    if not hmac.compare_digest(parts[1].strip(), API_KEY):
        return jsonify({"success": False, "error": "Invalid API key"}), 401
    return None


def parse_allowed(val):
    if not val:
        return ALLOWED_CATEGORIES.copy()
    try:
        parsed = json.loads(val)
        if isinstance(parsed, list) and parsed:
            return [x for x in parsed if x in ALLOWED_CATEGORIES]
    except Exception:
        pass
    return ALLOWED_CATEGORIES.copy()


def preprocess_image(image: Image.Image) -> np.ndarray:
    w, h = image.size
    if w < h:
        new_w = 256
        new_h = int(256 * h / w)
    else:
        new_h = 256
        new_w = int(256 * w / h)

    im = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    left = (new_w - 224) // 2
    top = (new_h - 224) // 2
    im = im.crop((left, top, left + 224, top + 224))

    arr = np.array(im, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return np.expand_dims(np.transpose(arr, (2, 0, 1)), axis=0)


def predict(image: Image.Image, allowed_cats):
    inp = preprocess_image(image)
    out = session.run(["output"], {"input": inp})[0][0]
    exp = np.exp(out - np.max(out))
    probs = exp / np.sum(exp)

    allowed_indices = [
        i for i, c in enumerate(classes)
        if CATEGORY_MAPPING[c] in allowed_cats
    ]
    if not allowed_indices:
        allowed_indices = list(range(len(classes)))

    allowed_probs = probs[allowed_indices]
    best_pos = int(np.argmax(allowed_probs))
    best_idx = allowed_indices[best_pos]

    return {
        "category": CATEGORY_MAPPING[classes[best_idx]],
        "confidence": round(float(probs[best_idx]), 4),
    }


# ============================================================
# ROUTES
# ============================================================

@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "status": "healthy" if model_ready else "not_ready",
        "model_loaded": model_ready,
        "model_version": "portfolio-v1-onnx",
    })


@app.route("/api/classify-image", methods=["POST"])
def classify_image():
    err = check_auth()
    if err:
        return err

    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"success": False, "error": "Empty filename"}), 400

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return jsonify({"success": False, "error": "Unsupported image format"}), 400

    allowed = parse_allowed(request.form.get("allowed_categories"))

    try:
        with Image.open(file.stream) as original:
            img = ImageOps.exif_transpose(original).convert("RGB")
        res = predict(img, allowed)
        return jsonify({
            "success": True,
            "filename": file.filename,
            "category": res["category"],
            "confidence": res["confidence"],
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/classify-images-batch", methods=["POST"])
def classify_images_batch():
    err = check_auth()
    if err:
        return err

    files = request.files.getlist("images")
    if not files:
        return jsonify({"success": False, "error": "No images provided"}), 400

    if len(files) > MAX_BATCH_IMAGES:
        return jsonify({"success": False, "error": f"Max {MAX_BATCH_IMAGES} images allowed"}), 400

    allowed = parse_allowed(request.form.get("allowed_categories"))
    results = []
    processed = 0
    failed = 0

    for file in files:
        try:
            with Image.open(file.stream) as original:
                img = ImageOps.exif_transpose(original).convert("RGB")
            res = predict(img, allowed)
            results.append({
                "success": True,
                "filename": file.filename or "image",
                "category": res["category"],
                "confidence": res["confidence"],
            })
            processed += 1
        except Exception as e:
            failed += 1
            results.append({
                "success": False,
                "filename": file.filename or "image",
                "error": str(e),
            })

    return jsonify({
        "success": failed == 0,
        "total": len(files),
        "processed": processed,
        "failed": failed,
        "results": results,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)
