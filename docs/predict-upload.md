# Face authentication upload

`POST /predict` requires `multipart/form-data` with one binary field named `file`.
The former JSON `{ "image": "data:image/...;base64,..." }` request is no longer
accepted (422). Enrollment at `/users` and `/admins` still uses Base64 JSON.
`python-multipart` is already declared in `pyproject.toml` and the lockfile.

## Swagger

Open `http://localhost:8000/docs`, expand `POST /predict`, select **Try it out**,
then **Choose File**, select a JPEG or PNG and execute. OpenAPI declares required
`file` as `type: string`, `format: binary` under `multipart/form-data`.

## curl

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@face.jpg;type=image/jpeg"

curl -X POST "http://localhost:8000/predict" \
  -F "file=@face.png;type=image/png"
```

## Frontend

```js
const body = new FormData();
body.append('file', fileInput.files[0]);
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  body,
});
const result = await response.json();
```

Let the browser set `Content-Type` and its multipart boundary. The login page
uses `canvas.toBlob(..., 'image/jpeg', 0.9)` and appends it as `face.jpg`.

## Validation and decisions

- Encoded file limit: 8 MiB; decoded resolution limit: 20 million pixels.
- Total request limit: 8 MiB plus 64 KiB for multipart framing, checked before
  parsing, including requests without a reliable `Content-Length`.
- Actual content must be JPEG or PNG and match declared `image/jpeg` or
  `image/png`. Filenames do not determine format. Generic or missing MIME is
  rejected. Empty, malformed, truncated and oversized images are rejected before
  authentication with HTTP 400 and `code: INVALID_IMAGE`.
- Missing `file`, including the former JSON body, returns the existing sanitized
  validation response (422). Invalid image contents are never echoed or logged.
- Decoding produces a BGR `uint8` NumPy array. Uploaded files are closed on
  successful decoding and validation failures; the framework also cleans up form
  resources. Responses retain `Cache-Control: no-store`.

After decoding, the existing authentication pipeline runs unchanged: exactly one
YOLO face, preprocessing, PAD, recognition, UNKNOWN/AMBIGUOUS decisions, current
account validation, session creation and audit logging. A valid image is not a
successful login by itself. Existing response bodies, decision status codes,
request ID and client source handling remain in place.
