"""HTTP API (FastAPI). Run it with ``hyrag serve`` or ``uvicorn hyrag.api:create_app --factory``."""

from hyrag.api.app import create_app

__all__ = ["create_app"]
