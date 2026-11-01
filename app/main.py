from fastapi import FastAPI

from app.core.config import settings

app = FastAPI(title=settings.project_name, debug=settings.debug)


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "project": settings.project_name,
    }