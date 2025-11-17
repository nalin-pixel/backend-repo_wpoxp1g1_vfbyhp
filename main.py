import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, conint, confloat
from typing import Literal
import math

app = FastAPI(title="GLOF Early Warning API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    lake_area_km2: confloat(gt=0, le=500) = Field(..., description="Surface area of the glacial lake in km^2")
    slope_deg: confloat(ge=0, le=90) = Field(..., description="Mean upstream slope in degrees")
    rainfall_mm_24h: confloat(ge=0, le=1000) = Field(..., description="Accumulated rainfall over last 24h in mm")
    temperature_c: confloat(ge=-40, le=50) = Field(..., description="Current temperature in °C")
    glacier_melt_index: confloat(ge=0, le=1) = Field(..., description="Normalized melt index from remote sensing (0-1)")
    distance_to_settlement_km: confloat(ge=0, le=200) = Field(..., description="Nearest settlement distance in km")
    dam_moraine_integrity: confloat(ge=0, le=1) = Field(..., description="Estimated moraine dam integrity (1=excellent, 0=failed)")


class PredictResponse(BaseModel):
    risk_score: float
    risk_level: Literal["Low", "Moderate", "High", "Critical"]
    recommendation: str
    contributing_factors: dict


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Basic health endpoint and environment probe."""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Used",
        "database_url": "❌ Not Set",
        "database_name": "❌ Not Set",
        "connection_status": "No DB for this demo",
        "collections": [],
    }
    return response


@app.post("/api/predict", response_model=PredictResponse)
def predict_glof(payload: PredictRequest):
    """Heuristic ML-like risk scoring as a stand-in for a trained model.

    This function mimics a logistic regression using domain-inspired weights.
    """
    try:
        # Normalize inputs and apply weights (domain-informed placeholders)
        w = {
            "lake_area_km2": 1.2,
            "slope_deg": 0.8,
            "rainfall_mm_24h": 1.5,
            "temperature_c": 0.3,
            "glacier_melt_index": 2.0,
            "distance_to_settlement_km": -0.6,
            "dam_moraine_integrity": -2.2,
        }

        # Feature scaling
        f = payload
        features = {
            "lake_area_km2": f.lake_area_km2 / 5.0,           # scale by 5 km^2
            "slope_deg": f.slope_deg / 45.0,                  # 0..2 range
            "rainfall_mm_24h": f.rainfall_mm_24h / 100.0,     # per 100 mm
            "temperature_c": max(f.temperature_c, 0) / 20.0,  # only positive temps drive melt
            "glacier_melt_index": f.glacier_melt_index,       # already 0..1
            "distance_to_settlement_km": f.distance_to_settlement_km / 10.0,  # per 10 km
            "dam_moraine_integrity": f.dam_moraine_integrity, # 0..1 (higher reduces risk)
        }

        linear = sum(features[k] * w[k] for k in features) - 1.0  # bias
        risk_score = 1.0 / (1.0 + math.exp(-linear))  # sigmoid 0..1

        if risk_score < 0.25:
            level = "Low"
            rec = "Maintain routine monitoring and update observations weekly."
        elif risk_score < 0.5:
            level = "Moderate"
            rec = "Increase monitoring frequency, verify dam integrity on-site if feasible."
        elif risk_score < 0.75:
            level = "High"
            rec = "Prepare early warning communications and evacuation drills for downstream communities."
        else:
            level = "Critical"
            rec = "Issue alerts, restrict access, and activate emergency response protocols immediately."

        return PredictResponse(
            risk_score=round(risk_score, 3),
            risk_level=level,
            recommendation=rec,
            contributing_factors={k: round(features[k] * w[k], 3) for k in features},
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
