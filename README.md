# Portfolio AI Image Classification API

FastAPI backend that classifies portfolio images into one of eight
categories using a locally trained PyTorch image-classification model.

## Features

-   Classify one image at a time.
-   Classify multiple images in a single batch (up to 20 files per
    request).
-   Restrict predictions to categories provided by the client.
-   Bearer-token authentication.
-   Interactive Swagger documentation.
-   Health-check endpoint.

## Categories

The API returns one of these database-facing category names:

-   `Logo`
-   `Banner`
-   `UI/UX`
-   `Color Separation`
-   `Flyer`
-   `Poster`
-   `Social Media`
-   `Other`

The model's internal class names are mapped to the API/database names.

## Project layout

Expected relevant files:

``` text
portfolio-ai/
├── api/
│   └── app.py
├── models/
│   ├── best_model_with_others.pth
│   └── classes_with_others.json
├── .env
└── .venv/
```

The API expects the model at `models/best_model_with_others.pth` and the
class list at `models/classes_with_others.json`, relative to the project
root. Keep both model files in place.

## Requirements

-   Python installed
-   The trained model and class JSON file
-   A virtual environment with the project's dependencies
-   Optional: an ngrok account and CLI for temporary public access

## 1. Activate / create the Python environment

From the project root:

``` bash
cd ~/Hassan/personal-work/portfolio-ai
```

If the existing virtual environment is present, use it:

``` bash
source .venv/bin/activate
```

If you need to create a new environment:

``` bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the API dependencies:

``` bash
pip install fastapi "uvicorn[standard]" python-multipart python-dotenv torch torchvision pillow
```

If PyTorch is already installed in the environment and working with the
trained model, do not reinstall it unnecessarily. Use the PyTorch
installation appropriate for your machine.

## 2. Configure the API key

Create a `.env` file in the project root (same level as `api/` and
`models/`):

``` env
API_KEY=replace_with_a_long_random_secret
```

Replace the example value with a strong secret. The value must match the
Bearer token sent by authorized clients.

Do not commit `.env` to Git, put the API key in frontend JavaScript, or
expose it in a public repository. The PHP/backend server should store
the key securely and call this API server-to-server.

If the application uses a different environment-variable name in
`api/app.py`, use the exact name expected by that code.

## 3. Run the API locally

From the project root:

``` bash
./.venv/bin/python3 -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Expected startup messages include:

``` text
AI model loaded successfully.
Model: best_model_with_others.pth
Categories: ['Logo', 'Banner', 'UI/UX', 'Color Separation', 'Flyer', 'Poster', 'Social Media', 'Other']
Uvicorn running on http://127.0.0.1:8000
```

Keep this terminal open while using the API. Stop the server with
`Ctrl+C`.

## 4. Open the interactive API interface

With the local server running, open:

-   Swagger UI: <http://127.0.0.1:8000/docs>
-   ReDoc: <http://127.0.0.1:8000/redoc>
-   OpenAPI schema: <http://127.0.0.1:8000/openapi.json>

Swagger UI lets you expand an endpoint, select **Try it out**, fill in
the request, and click **Execute**.

For secured endpoints, use the Authorize button and enter the token in
the format requested by the Swagger security scheme. If it asks for a
Bearer token, provide the API key as the token value.

## API reference

Base URL for local development:

``` text
http://127.0.0.1:8000
```

### Health check

``` http
GET /health
```

Use this to check whether the service is responding. The exact response
fields are defined by the running application.

### Classify one image

``` http
POST /api/classify-image
Authorization: Bearer YOUR_API_KEY
Content-Type: multipart/form-data
```

Form fields:

  ------------------------------------------------------------------------------
  Field                  Type                          Required Description
  ---------------------- ---------------- --------------------- ----------------
  `image`                File                               Yes One image file

  `allowed_categories`   Text containing                     No Categories the
                         a JSON array                           prediction is
                                                                allowed to
                                                                return
  ------------------------------------------------------------------------------

Example `allowed_categories` value:

``` json
["Logo","Banner","UI/UX","Color Separation","Flyer","Poster","Social Media","Other"]
```

Example cURL request:

``` bash
curl -X POST "http://127.0.0.1:8000/api/classify-image" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "image=@/absolute/path/to/image.jpg" \
  -F 'allowed_categories=["Logo","Banner","UI/UX","Color Separation","Flyer","Poster","Social Media","Other"]'
```

Example success response:

``` json
{
  "success": true,
  "filename": "image.jpg",
  "category": "Banner",
  "confidence": 0.91
}
```

The category and confidence above are illustrative; actual results
depend on the image and model prediction.

### Classify multiple images (batch)

``` http
POST /api/classify-images-batch
Authorization: Bearer YOUR_API_KEY
Content-Type: multipart/form-data
```

The batch endpoint expects the multipart field name **`images`**
(plural). Add one `images` part for every file. Do not use `image` or
`images[]` unless the server code is explicitly changed to accept those
names.

Form fields:

  ------------------------------------------------------------------------------
  Field                  Type                          Required Description
  ---------------------- ---------------- --------------------- ----------------
  `images`               File, repeated                     Yes Up to 20 image
                                                                files; repeat
                                                                this same field
                                                                name for each
                                                                file

  `allowed_categories`   Text containing                     No Categories the
                         a JSON array                           prediction is
                                                                allowed to
                                                                return
  ------------------------------------------------------------------------------

Example cURL request with three images:

``` bash
curl -X POST "http://127.0.0.1:8000/api/classify-images-batch" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "images=@/absolute/path/to/1.jpg" \
  -F "images=@/absolute/path/to/2.jpg" \
  -F "images=@/absolute/path/to/3.jpg" \
  -F 'allowed_categories=["Logo","Banner","UI/UX","Color Separation","Flyer","Poster","Social Media","Other"]'
```

Example success response shape:

``` json
{
  "success": true,
  "results": [
    {
      "success": true,
      "filename": "1.jpg",
      "category": "Banner",
      "confidence": 0.91
    },
    {
      "success": true,
      "filename": "2.jpg",
      "category": "Logo",
      "confidence": 0.88
    },
    {
      "success": true,
      "filename": "3.jpg",
      "category": "Poster",
      "confidence": 0.94
    }
  ]
}
```

This is an example of the response structure, not a prediction
guarantee. Check the actual response from your running API for exact
per-file results.

## Postman setup

### Single image

1.  Create a `POST` request to
    `http://127.0.0.1:8000/api/classify-image`.
2.  Under **Authorization**, select **Bearer Token** and enter the API
    key.
3.  Under **Body → form-data**, add `image` as type **File** and select
    one image.
4.  Add `allowed_categories` as type **Text** with the JSON array shown
    above.
5.  Click **Send**.

### Batch images

1.  Create a `POST` request to
    `http://127.0.0.1:8000/api/classify-images-batch`.
2.  Set **Authorization → Bearer Token**.
3.  Under **Body → form-data**, add `images` as type **File** and select
    a file.
4.  Add more rows with the exact same key `images`, each set to type
    **File**, one file per row (up to 20).
5.  Add `allowed_categories` as type **Text** with the JSON array.
6.  Click **Send**.

Important: Do not manually set the `Content-Type` header in Postman. Let
Postman generate the multipart content type and boundary automatically.

## 5. Expose the local API using ngrok (temporary testing)

Ngrok creates a public HTTPS tunnel to your local server. This is useful
for testing with a remote PHP backend, but it is not a permanent
production deployment.

### Install and authenticate

Install ngrok using the method recommended for your Ubuntu version. If
Snap is available:

``` bash
sudo snap install ngrok
```

Create or sign in to an ngrok account, copy your authtoken from the
ngrok dashboard, then configure it:

``` bash
ngrok config add-authtoken YOUR_NGROK_AUTHTOKEN
```

Keep the FastAPI/Uvicorn server running in the first terminal. In a
second terminal, run:

``` bash
ngrok http 8000
```

Copy the HTTPS forwarding URL shown by ngrok, for example:

``` text
https://your-assigned-name.ngrok-free.app
```

Use the actual URL printed by ngrok, not the example above.

Public endpoint examples:

``` text
https://YOUR-NGROK-URL.ngrok-free.app/docs
https://YOUR-NGROK-URL.ngrok-free.app/health
https://YOUR-NGROK-URL.ngrok-free.app/api/classify-image
https://YOUR-NGROK-URL.ngrok-free.app/api/classify-images-batch
```

Both the local Uvicorn process and ngrok process must remain running. If
ngrok restarts, the free URL may change. Do not treat a temporary ngrok
URL as a stable production URL.

## Integration notes for the PHP developer

-   Make API requests from the PHP backend, not directly from browser
    JavaScript, so the API key remains secret.
-   Send requests as `multipart/form-data`.
-   For one image, use the multipart field `image`.
-   For batch requests, repeat the multipart field `images` for each
    uploaded file.
-   Send the API key in the HTTP header:
    `Authorization: Bearer YOUR_API_KEY`.
-   Parse the returned JSON and use the `category` field to populate the
    relevant database category.
-   Handle non-2xx responses and per-image errors; do not assume every
    file in a batch will necessarily classify successfully.
-   The API's allowed-category values must match the category names
    exactly, including spaces, slash, and capitalization.

## Troubleshooting

### `422 Unprocessable Entity` with missing `body.images`

The batch request did not include the required `images` form field.
Check that the key is `images` (plural), that each row is set to
**File**, and that files are selected. The single-image endpoint instead
expects `image` (singular).

### Only one prediction is returned

You may be calling `/api/classify-image`, which is the single-image
endpoint. Use `/api/classify-images-batch` and repeat the `images`
multipart field for each file.

### `401` or `403` authentication error

Check that the `Authorization` header is present and that the Bearer
token matches the `API_KEY` configured for the server. Do not include
`Bearer` twice if the client library adds it automatically.

### Cannot connect to `127.0.0.1:8000`

Confirm Uvicorn is running and listening on port 8000. The address
`127.0.0.1` is local to the machine making the request. A remote PHP
server cannot reach your laptop's localhost; use the ngrok HTTPS URL for
temporary remote testing or deploy the API to a server.

### Ngrok URL is unavailable

Confirm both Uvicorn and ngrok are running. Check the forwarding URL in
the ngrok terminal and use the current URL. A free ngrok URL may change
after restarting the tunnel.

### Model or class file not found

Run Uvicorn from the project root and verify these files exist:

``` text
models/best_model_with_others.pth
models/classes_with_others.json
```

## Security and deployment notes

-   Do not commit `.env` or share the API key in public.
-   Rotate the API key if it is exposed.
-   Ngrok is for temporary testing. For production, deploy to a secured
    server with HTTPS, process supervision, appropriate request limits,
    monitoring, and a stable domain.
-   Do not expose an unauthenticated model endpoint to the public
    internet.
-   Model confidence is a model score, not a guarantee that the
    classification is correct.

## Quick start

Terminal --- start FastAPI:

``` bash
cd ~/Hassan/personal-work/portfolio-ai
./.venv/bin/python3 -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

