from fastapi import APIRouter, HTTPException, status

from app.schemas.design_agent_virtual_fitting import (
    DesignApprovalRequest,
    DesignGenerationRequest,
    DesignRefinementRequest,
    DesignSession,
    DesignSessionCreateRequest,
    ProductionBrief,
    VirtualFittingRequest,
)
from app.services.design_agent_virtual_fitting import DesignSessionError, DesignSessionService

router = APIRouter(prefix="/v1/body-ai/design-sessions", tags=["body-ai-design-sessions"])

design_session_service = DesignSessionService()


@router.post("", response_model=DesignSession, status_code=status.HTTP_201_CREATED)
def create_design_session(request: DesignSessionCreateRequest) -> DesignSession:
    return run_or_400(lambda: design_session_service.create_session(request))


@router.post("/{design_session_id}/generate", response_model=DesignSession)
def generate_design_options(design_session_id: str, request: DesignGenerationRequest) -> DesignSession:
    return run_or_400(lambda: design_session_service.generate_options(design_session_id, request))


@router.post("/{design_session_id}/refine", response_model=DesignSession)
def refine_design_options(design_session_id: str, request: DesignRefinementRequest) -> DesignSession:
    return run_or_400(lambda: design_session_service.refine_options(design_session_id, request))


@router.post("/{design_session_id}/fitting-preview", response_model=DesignSession)
def create_fitting_preview(design_session_id: str, request: VirtualFittingRequest) -> DesignSession:
    return run_or_400(lambda: design_session_service.create_fitting_preview(design_session_id, request))


@router.post("/{design_session_id}/approve", response_model=DesignSession)
def approve_design(design_session_id: str, request: DesignApprovalRequest) -> DesignSession:
    return run_or_400(lambda: design_session_service.approve_design(design_session_id, request))


@router.get("/{design_session_id}", response_model=DesignSession)
def get_design_session(design_session_id: str) -> DesignSession:
    return run_or_400(lambda: design_session_service.get_session(design_session_id))


@router.get("/{design_session_id}/production-brief", response_model=ProductionBrief)
def get_production_brief(design_session_id: str) -> ProductionBrief:
    return run_or_400(lambda: design_session_service.production_brief(design_session_id))


def run_or_400(operation):
    try:
        return operation()
    except DesignSessionError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

