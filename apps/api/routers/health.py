from __future__ import annotations

import os
from fastapi import APIRouter
from core.schemas.models import HealthResponse
from apps.api.config import settings

router = APIRouter(tags=["health"])


# @router.get("/health", response_model=HealthResponse)
# def health() -> HealthResponse:
#     return HealthResponse(service=settings.app_name, version=settings.app_version)

@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        service=settings.app_name,
        version=settings.app_version,
        provider=settings.model_provider,
        parser=settings.parser,
    )
