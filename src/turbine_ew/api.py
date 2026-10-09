from datetime import date
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.turbine_ew.serving import has_turbine, operating_ok, predict_turbine, select_alerts


class Reading(BaseModel):
    turbine: int = Field(ge=1)
    wind_speed: float = Field(ge=0, le=60)
    power: float = Field(ge=-100, le=3000)
    rotor_speed: float = Field(ge=0, le=40)
    gearbox_speed: float = Field(ge=0, le=3000)
    ambient_temp: float = Field(ge=-50, le=60)
    actual: Optional[dict[str, float]] = None


def create_app(artifacts, thresholds, alerts, cleaning):
    app = FastAPI(title="Turbine early warning", version="0.1.0")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": artifacts["model_name"],
            "turbines": sorted({t for t, _ in artifacts["models"]}),
            "targets": artifacts["targets"],
        }

    @app.post("/predict")
    def predict(reading: Reading):
        if not has_turbine(artifacts, reading.turbine):
            raise HTTPException(status_code=404, detail=f"no models for turbine {reading.turbine}")
        actual = reading.actual or {}
        unknown = sorted(set(actual) - set(artifacts["targets"]))
        if unknown:
            raise HTTPException(status_code=422, detail=f"unknown targets in actual: {unknown}")
        row = reading.model_dump(exclude={"turbine", "actual"})
        expected = predict_turbine(artifacts, reading.turbine, row)
        in_range = operating_ok(row, cleaning)
        residual = None
        if in_range and actual:
            residual = {t: round(actual[t] - expected[t], 2) for t in actual}
        return {
            "turbine": reading.turbine,
            "model": artifacts["model_name"],
            "in_operating_range": in_range,
            "expected": {t: round(v, 2) for t, v in expected.items()},
            "residual": residual,
            "note": (
                "Residual is actual minus expected (positive means hotter than expected). "
                "It is only given when the reading is in the operating range. "
                "One 10-minute residual is not an alert; see /alerts."
            ),
        }

    @app.get("/alerts")
    def get_alerts(
        turbine: Optional[int] = None,
        target: Optional[str] = None,
        since: Optional[date] = None,
        until: Optional[date] = None,
    ):
        rows = select_alerts(alerts, thresholds, turbine, target, since, until)
        return {"count": len(rows), "alerts": rows}

    return app