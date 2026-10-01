from fastapi import FastAPI


app = FastAPI(
    title="Vietnamese News Topic Discovery API",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return the health status of the API service."""
    return {"status": "ok"}