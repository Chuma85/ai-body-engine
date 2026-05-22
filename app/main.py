from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.core.config import settings

app = FastAPI(title=settings.app_name, version=settings.app_version)

app.include_router(health_router)


@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "service": settings.app_name,
        "status": "running",
        "version": settings.app_version,
    }
