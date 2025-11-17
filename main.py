from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conint, confloat
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import math

app = FastAPI(title="GLOF Early Warning API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictInput(BaseModel):
    water_level: confloat(ge=0, le=1000) = Field(..., description="Lake water level (cm)")
    temperature: confloat(ge=-50, le=60) = Field(..., description="Air temperature (C)")
    rainfall: confloat(ge=0, le=500) = Field(..., description="Rainfall last 24h (mm)")
    melt_rate: confloat(ge=0, le=100) = Field(..., description="Snow/ice melt rate (mm/day)")
    lake_area: confloat(ge=0, le=1000) = Field(..., description="Lake surface area (ha)")
    slope: confloat(ge=0, le=90) = Field(..., description="Downstream average slope (deg)")


class PredictOutput(BaseModel):
    risk_score: conint(ge=0, le=100)
    probability: confloat(ge=0, le=1)
    level: str
    details: dict


@app.get("/")
async def root():
    return {"message": "GLOF Early Warning API running"}


@app.get("/api/hello")
async def hello():
    return {"hello": "world"}


@app.get("/test")
async def test():
    # Lightweight healthcheck
    return {"status": "ok"}


@app.post("/api/predict", response_model=PredictOutput)
async def predict(payload: PredictInput):
    # Simple logistic-style risk computation (heuristic only)
    x = 0.0
    x += 0.004 * payload.water_level
    x += 0.08 * max(payload.temperature, 0)
    x += 0.01 * payload.rainfall
    x += 0.03 * payload.melt_rate
    x += 0.002 * payload.lake_area
    x += 0.01 * max(0, 30 - payload.slope)  # flatter terrain riskier
    # Normalize and squash
    prob = 1 / (1 + math.exp(-(x - 2.5)))
    score = int(round(prob * 100))
    if score <= 30:
        level = "Low"
    elif score <= 60:
        level = "Medium"
    else:
        level = "High"

    return {
        "risk_score": score,
        "probability": prob,
        "level": level,
        "details": {
            "inputs": payload.model_dump(),
            "explanation": "Heuristic combination of hydrometeorological indicators",
        },
    }
