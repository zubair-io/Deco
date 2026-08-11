"""FastAPI app: `POST /identify` and `GET /healthz`.

`POST /identify` accepts either a multipart file field named `image` or a
JSON body `{"image_b64": "..."}`. Both forms are handled off the raw
`Request` (rather than separate FastAPI parameter styles) because a single
endpoint can't declare both a `File(...)` parameter and a Pydantic JSON body
parameter at once.
"""

import base64
import binascii
import io

from fastapi import FastAPI, HTTPException, Request
from PIL import Image, UnidentifiedImageError

from deco.api.pipeline import IdentifyPipeline

_NO_IMAGE_DETAIL = (
    "no image provided: send a multipart file field 'image' or a JSON body {'image_b64': '...'}"
)
_BAD_IMAGE_DETAIL = "could not decode image"
_BAD_B64_DETAIL = "image_b64 is not valid base64"


async def _extract_image_bytes(request: Request) -> bytes | None:
    content_type = request.headers.get("content-type", "")

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("image")
        if upload is None or not hasattr(upload, "read"):
            return None
        return await upload.read()

    if content_type.startswith("application/json"):
        payload = await request.json()
        b64 = payload.get("image_b64") if isinstance(payload, dict) else None
        if not b64:
            return None
        try:
            return base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=422, detail=_BAD_B64_DETAIL) from exc

    return None


def create_app(pipeline: IdentifyPipeline) -> FastAPI:
    app = FastAPI(title="Deco identify service")

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/identify")
    async def identify(request: Request) -> dict:
        image_bytes = await _extract_image_bytes(request)
        if not image_bytes:
            raise HTTPException(status_code=422, detail=_NO_IMAGE_DETAIL)

        try:
            with Image.open(io.BytesIO(image_bytes)) as raw:
                raw.load()
                image = raw.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=422, detail=_BAD_IMAGE_DETAIL) from exc

        result = pipeline.identify(image)

        matches = (
            []
            if not result.confident
            else [
                {
                    "record": match.record.model_dump(),
                    "score": match.cosine,
                    "inliers": match.inliers,
                }
                for match in result.matches
            ]
        )
        return {
            "confident": result.confident,
            "no_confident_match": not result.confident,
            "matches": matches,
        }

    return app
