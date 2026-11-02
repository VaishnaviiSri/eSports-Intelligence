from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.chat import router as chat_router
from app.core.config import settings

app = FastAPI(title=settings.project_name, debug=settings.debug)
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
app.include_router(chat_router)
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    """Serve the single-page chat interface."""
    return FileResponse(frontend_dir / "index.html")


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "project": settings.project_name,
    }