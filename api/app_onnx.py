import os
import json
import hmac
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.onnx"
CLASSES_PATH = BASE_DIR / "models" / "classes_with_others.json"

load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(BASE_DIR / ".env")

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
# APP
# ============================================================

app = FastAPI(
    title="Portfolio AI Image Classification API (Ultra-Light ONNX)",
    description="AI-powered portfolio image categorization.",
    version="1.0.0",
)

session = None
classes = []
model_ready = False


def init_onnx_model():
    global session, classes, model_ready

    if not MODEL_PATH.exists():
        raise RuntimeError(f"ONNX Model file not found: {MODEL_PATH}")

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        classes = json.load(f)

    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    model_ready = True
    print("ONNX AI model loaded successfully.")


# Load immediately on module load so WSGI/FastAPI can serve right away
try:
    init_onnx_model()
except Exception as e:
    print("Warning: Model init during import failed:", e)


@app.on_event("startup")
def startup_event():
    if not model_ready:
        init_onnx_model()


# ============================================================
# AUTHENTICATION
# ============================================================

def verify_api_key(authorization: Optional[str]):
    if not API_KEY:
        return  # If no API_KEY configured, allow open access

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header",
        )

    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Authorization must use Bearer token",
        )

    provided_key = parts[1].strip()
    if not provided_key or not hmac.compare_digest(provided_key, API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )


# ============================================================
# PREPROCESSING & INFERENCE
# ============================================================

def parse_allowed_categories(value: Optional[str]):
    if not value:
        return ALLOWED_CATEGORIES.copy()

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="allowed_categories must be a JSON array",
        )

    if not isinstance(parsed, list) or not parsed:
        raise HTTPException(
            status_code=400,
            detail="allowed_categories must be a non-empty JSON array",
        )

    invalid = [item for item in parsed if item not in ALLOWED_CATEGORIES]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail={"message": "Unsupported category", "invalid_categories": invalid},
        )
    return parsed


async def read_image(upload: UploadFile):
    filename = upload.filename or "image"
    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported image extension. Use JPG, JPEG, PNG, WEBP, or BMP.")

    content = await upload.read(MAX_FILE_SIZE + 1)
    if not content:
        raise ValueError("Uploaded image is empty")

    if len(content) > MAX_FILE_SIZE:
        raise ValueError("Image exceeds the 10 MB size limit")

    import io
    try:
        with Image.open(io.BytesIO(content)) as original:
            original.verify()
        with Image.open(io.BytesIO(content)) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
    except Exception:
        raise ValueError("Uploaded file is not a valid image")

    return filename, image


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


def predict_image(image: Image.Image, allowed_categories):
    if not model_ready or session is None:
        raise RuntimeError("AI model is not ready")

    inp = preprocess_image(image)
    out = session.run(["output"], {"input": inp})[0][0]

    exp = np.exp(out - np.max(out))
    probabilities = exp / np.sum(exp)

    allowed_indices = [
        index
        for index, internal_class in enumerate(classes)
        if CATEGORY_MAPPING[internal_class] in allowed_categories
    ]

    if not allowed_indices:
        raise ValueError("No model classes match allowed_categories")

    allowed_probs = probabilities[allowed_indices]
    best_position = int(np.argmax(allowed_probs))
    best_index = allowed_indices[best_position]

    internal_category = classes[best_index]
    category = CATEGORY_MAPPING[internal_category]
    confidence = float(probabilities[best_index])

    return {
        "category": category,
        "confidence": round(confidence, 4),
    }


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health")
def health():
    return {
        "success": True,
        "status": "healthy" if model_ready else "not_ready",
        "model_loaded": model_ready,
        "model_version": "portfolio-v1-onnx",
    }


@app.post("/api/classify-image")
async def classify_image(
    image: UploadFile = File(...),
    allowed_categories: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None),
):
    verify_api_key(authorization)
    allowed = parse_allowed_categories(allowed_categories)

    try:
        filename, pil_image = await read_image(image)
        result = predict_image(pil_image, allowed)
        return {
            "success": True,
            "filename": filename,
            "category": result["category"],
            "confidence": result["confidence"],
        }
    except HTTPException:
        raise
    except ValueError as e:
        return JSONResponse(status_code=400, content={"success": False, "error": str(e)})
    except Exception as e:
        print("Classification error:", repr(e))
        return JSONResponse(status_code=500, content={"success": False, "error": "Image classification failed"})
    finally:
        await image.close()


@app.post("/api/classify-images-batch")
async def classify_images_batch(
    images: list[UploadFile] = File(...),
    allowed_categories: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None),
):
    verify_api_key(authorization)
    allowed = parse_allowed_categories(allowed_categories)

    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required")
    if len(images) > MAX_BATCH_IMAGES:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_BATCH_IMAGES} images per request")

    results = []
    processed = 0
    failed = 0

    for upload in images:
        try:
            filename, pil_image = await read_image(upload)
            result = predict_image(pil_image, allowed)
            results.append({
                "success": True,
                "filename": filename,
                "category": result["category"],
                "confidence": result["confidence"],
            })
            processed += 1
        except Exception as e:
            failed += 1
            results.append({
                "success": False,
                "filename": upload.filename or "unknown",
                "error": str(e),
            })
        finally:
            await upload.close()

    return {
        "success": failed == 0,
        "total": len(images),
        "processed": processed,
        "failed": failed,
        "results": results,
    }
