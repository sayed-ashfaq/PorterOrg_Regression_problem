from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.pipeline import predict
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


app = FastAPI(
    title="Porter Delivery Time Predictor",
    description="Predicts food delivery time in minutes",
    version="1.0.0"
)

# ── Request schema — validates input automatically ─────────────────────
class OrderInput(BaseModel):
    created_at: str
    market_id: int
    store_primary_category: Optional[str] = 'unknown'
    order_protocol: int
    total_items: int
    subtotal: float
    num_distinct_items: int
    min_item_price: float
    max_item_price: float
    total_onshift_partners: float
    total_busy_partners: float
    total_outstanding_orders: float

class PredictionOutput(BaseModel):
    predicted_delivery_minutes: float
    optimistic_minutes: float
    pessimistic_minutes: float
    message: str

# ── Health check endpoint ──────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "healthy", "model": "porter_v2"}

# ── Prediction endpoint ────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionOutput)
def predict_delivery(order: OrderInput):
    try:
        result = predict(order.dict())
        mins = result['predicted_delivery_minutes']
        
        return PredictionOutput(
            predicted_delivery_minutes=mins,
            optimistic_minutes=result['optimistic_minutes'],
            pessimistic_minutes=result['pessimistic_minutes'],
            message=f"Estimated delivery in {mins} minutes "
                   f"({result['optimistic_minutes']}-"
                   f"{result['pessimistic_minutes']} min range)"
        )
    except Exception as e:
        raise HTTPException(status_code=500, 
                          detail=f"Prediction failed: {str(e)}")
