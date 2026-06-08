from fastapi import APIRouter, HTTPException, status

from app.schemas.three_view_measurements import ThreeViewMeasurementRequest, ThreeViewMeasurementResponse
from app.services.three_view_measurements import ThreeViewMeasurementService
from training.measurements.body_ai_inference import BodyAIInferenceError


router = APIRouter(prefix="/v1/body-ai/measurements", tags=["body-ai-measurements"])

three_view_measurement_service = ThreeViewMeasurementService()


@router.post("/three-view", response_model=ThreeViewMeasurementResponse)
def generate_three_view_measurements(request: ThreeViewMeasurementRequest) -> ThreeViewMeasurementResponse:
    try:
        return three_view_measurement_service.generate(request)
    except (BodyAIInferenceError, FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
