from fastapi import Depends, FastAPI, HTTPException

from src.api.inference_service import (
    InferenceUnavailableError,
    get_inference_service,
)
from src.api.schemas import PredictRequest, PredictResponse


app = FastAPI(
    title="Vietnamese News Topic Discovery API",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return the health status of the API service."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_article(
    payload: PredictRequest,
    service = Depends(get_inference_service),
) -> PredictResponse:
    """Gán topic cho bài mới bằng model đã lưu, không fit lại model."""
    if not payload.title.strip() and not payload.content.strip():
        raise HTTPException(
            status_code=422,
            detail="Tiêu đề và nội dung không được đồng thời để trống.",
        )
    try:
        return service.predict(payload.title, payload.content, payload.model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except InferenceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/topics")
def list_topics(model: str = "bertopic", service = Depends(get_inference_service)) -> dict:
    """Liệt kê topic từ artifact của mô hình được chọn."""
    try:
        return service.topics(model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except InferenceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
