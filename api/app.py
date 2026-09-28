import os
import json
import hmac
import secrets
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import (
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.responses import JSONResponse
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model_with_others.pth"

load_dotenv(Path(__file__).resolve().parent / ".env")

API_KEY = os.getenv("API_KEY", "").strip()

DEVICE = torch.device("cpu")

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

# Exact mapping from training classes to database categories.
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
    title="Portfolio AI Image Classification API",
    description=(
        "AI-powered portfolio image categorization "
        "for the portfolio management system."
    ),
    version="1.0.0",
)


# ============================================================
# LOAD MODEL ONCE AT STARTUP
# ============================================================

model = None
classes = []
transform = None
model_ready = False


@app.on_event("startup")
def load_model():

    global model, classes, transform, model_ready

    if not API_KEY:
        raise RuntimeError(
            "API_KEY is missing. Configure it in api/.env"
        )

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    classes = checkpoint["classes"]

    # Ensure the checkpoint contains exactly the expected
    # internal class labels.
    if set(classes) != set(CATEGORY_MAPPING.keys()):
        raise RuntimeError(
            "Model classes do not match CATEGORY_MAPPING. "
            f"Found: {classes}"
        )

    model_instance = mobilenet_v3_small(weights=None)

    in_features = model_instance.classifier[-1].in_features

    model_instance.classifier[-1] = torch.nn.Linear(
        in_features,
        len(classes),
    )

    model_instance.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model_instance.to(DEVICE)
    model_instance.eval()

    model = model_instance

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    model_ready = True

    print("AI model loaded successfully.")
    print("Model:", MODEL_PATH.name)
    print("Categories:", ALLOWED_CATEGORIES)


# ============================================================
# AUTHENTICATION
# ============================================================

def verify_api_key(authorization: Optional[str]):

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

    if not provided_key or not hmac.compare_digest(
        provided_key,
        API_KEY,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )


# ============================================================
# CATEGORY VALIDATION
# ============================================================

def parse_allowed_categories(value: Optional[str]):

    if not value:
        return ALLOWED_CATEGORIES.copy()

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail=(
                "allowed_categories must be a JSON array "
                "encoded as a multipart form string"
            ),
        )

    if not isinstance(parsed, list):
        raise HTTPException(
            status_code=400,
            detail="allowed_categories must be a JSON array",
        )

    if not parsed:
        raise HTTPException(
            status_code=400,
            detail="allowed_categories cannot be empty",
        )

    if any(not isinstance(item, str) for item in parsed):
        raise HTTPException(
            status_code=400,
            detail="Each category must be a string",
        )

    invalid = [
        item for item in parsed
        if item not in ALLOWED_CATEGORIES
    ]

    if invalid:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unsupported category provided",
                "invalid_categories": invalid,
                "allowed_categories": ALLOWED_CATEGORIES,
            },
        )

    return parsed


# ============================================================
# IMAGE VALIDATION
# ============================================================

async def read_image(upload: UploadFile):

    filename = upload.filename or "image"

    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported image extension. "
            "Use JPG, JPEG, PNG, WEBP, or BMP."
        )

    content = await upload.read(MAX_FILE_SIZE + 1)

    if not content:
        raise ValueError("Uploaded image is empty")

    if len(content) > MAX_FILE_SIZE:
        raise ValueError("Image exceeds the 10 MB size limit")

    try:
        import io

        with Image.open(io.BytesIO(content)) as original:

            # Protect against malformed or decompression-bomb images.
            original.verify()

        with Image.open(io.BytesIO(content)) as original:

            image = ImageOps.exif_transpose(original).convert("RGB")

    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise ValueError("Uploaded file is not a valid image")

    return filename, image


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_image(image, allowed_categories):

    if not model_ready or model is None:
        raise RuntimeError("AI model is not ready")

    tensor = (
        transform(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.inference_mode():

        outputs = model(tensor)

        probabilities = F.softmax(
            outputs,
            dim=1,
        )[0]

    # Only categories allowed by the PHP caller can be returned.
    # Restrict the output classes before selecting the prediction.
    allowed_indices = [
        index
        for index, internal_class in enumerate(classes)
        if CATEGORY_MAPPING[internal_class] in allowed_categories
    ]

    if not allowed_indices:
        raise ValueError(
            "No model classes match allowed_categories"
        )

    allowed_probs = probabilities[allowed_indices]

    best_position = allowed_probs.argmax().item()

    best_index = allowed_indices[best_position]

    internal_category = classes[best_index]

    category = CATEGORY_MAPPING[internal_category]

    confidence = float(probabilities[best_index].item())

    return {
        "category": category,
        "confidence": round(confidence, 4),
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "success": True,
        "status": "healthy" if model_ready else "not_ready",
        "model_loaded": model_ready,
        "model_version": "portfolio-v1",
    }


# ============================================================
# SINGLE IMAGE ENDPOINT
# POST /api/classify-image
# ============================================================

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

        result = predict_image(
            pil_image,
            allowed,
        )

        return {
            "success": True,
            "filename": filename,
            "category": result["category"],
            "confidence": result["confidence"],
        }

    except HTTPException:
        raise

    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": str(e),
            },
        )

    except Exception as e:
        print("Classification error:", repr(e))

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Image classification failed",
            },
        )

    finally:
        await image.close()


# ============================================================
# BATCH IMAGE ENDPOINT
# POST /api/classify-images-batch
# ============================================================

@app.post("/api/classify-images-batch")
async def classify_images_batch(
    images: list[UploadFile] = File(...),
    allowed_categories: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None),
):

    verify_api_key(authorization)

    allowed = parse_allowed_categories(allowed_categories)

    if not images:
        raise HTTPException(
            status_code=400,
            detail="At least one image is required",
        )

    if len(images) > MAX_BATCH_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_BATCH_IMAGES} images per request",
        )

    results = []

    try:

        for upload in images:

            filename = upload.filename or "image"

            try:

                filename, pil_image = await read_image(upload)

                result = predict_image(
                    pil_image,
                    allowed,
                )

                results.append({
                    "success": True,
                    "filename": filename,
                    "category": result["category"],
                    "confidence": result["confidence"],
                })

            except ValueError as e:

                results.append({
                    "success": False,
                    "filename": filename,
                    "error": str(e),
                })

            except Exception as e:

                print(
                    f"Batch classification error for {filename}:",
                    repr(e),
                )

                results.append({
                    "success": False,
                    "filename": filename,
                    "error": "Image classification failed",
                })

            finally:
                await upload.close()

        successful = sum(
            1 for item in results
            if item["success"]
        )

        failed = len(results) - successful

        return {
            "success": True,
            "total": len(results),
            "processed": successful,
            "failed": failed,
            "results": results,
        }

    except HTTPException:
        raise

    except Exception as e:

        print("Batch error:", repr(e))

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Batch classification failed",
            },
        )
