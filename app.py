from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
from transformers import AutoTokenizer
from traffy_predict.model import TraffyBertRegressor
import logging
from typing import Optional
from datetime import datetime
import cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MAX_LEN = 512
DEVICE = torch.device("cpu")

# Initialize FastAPI app
app = FastAPI(
    title="Traffy Fondue Duration Prediction API",
    description="API for predicting resolution time for Traffy Fondue complaints using BERT-based model",
    version="2.0.0"
)

# Global variables for model and tokenizer
model = None
tokenizer = None

class PredictionRequest(BaseModel):
    """Request model for single prediction"""
    comment: str = Field(..., description="Complaint comment text", example="หมาจรในซอยเฉลิมพระเกียรติร.9 42 นี้ดุ วิ่งไล่กัดคนเดินผ่าน")
    type: str = Field(default="{}", description="Complaint type", example="{ร้องเรียน,สัตว์จรจัด}")
    organization: str = Field(default="", description="Organization responsible", example="เขตประเวศ")
    district: str = Field(default="", description="District", example="ประเวศ")
    subdistrict: str = Field(default="", description="Subdistrict", example="หนองบอน")
    timestamp: Optional[str] = Field(default=None, description="Timestamp in ISO format", example="2025-01-15T11:27:20+00:00")

    class Config:
        json_schema_extra = {
            "example": {
                "comment": "หมาจรในซอยเฉลิมพระเกียรติร.9 42 นี้ดุ วิ่งไล่กัดคนเดินผ่าน เดือดร้อนคนแถวนั้น",
                "type": "{ร้องเรียน,สัตว์จรจัด}",
                "organization": "เขตประเวศ,สำนักงานสัตวแพทย์สาธารณสุข สำนักอนามัย กทม.",
                "district": "ประเวศ",
                "subdistrict": "หนองบอน",
                "timestamp": "2025-01-15T11:27:20+00:00"
            }
        }

class PredictionResponse(BaseModel):
    """Response model for prediction"""
    predicted_days: float = Field(..., description="Predicted resolution time in days")
    predicted_hours: float = Field(..., description="Predicted resolution time in hours")
    comment: str = Field(..., description="Original comment")
    features: dict = Field(..., description="Extracted features used for prediction")

def load_model_on_startup():
    """Load model and tokenizer on startup"""
    global model, tokenizer

    try:
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("./traffy_predict/tokenizer")

        logger.info("Loading model...")
        model = TraffyBertRegressor(model_name="airesearch/wangchanberta-base-att-spm-uncased")

        # Resize embeddings to match the saved model (249262 tokens)
        model.bert.resize_token_embeddings(249262)

        model.load_state_dict(
            torch.load("./traffy_predict/best_bert_regressor.pt", map_location=DEVICE)
        )
        model.eval()

        logger.info("Model and tokenizer loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def extract_features(comment, type_str, timestamp_str):
    """Extract features from complaint data"""
    from datetime import time
    import pandas as pd

    # Parse timestamp
    if timestamp_str:
        try:
            timestamp = pd.to_datetime(timestamp_str, errors="coerce")
        except:
            timestamp = pd.Timestamp.now()
    else:
        timestamp = pd.Timestamp.now()

    # Extract features
    is_type = 1.0 if type_str != "{}" else 0.0
    comment_length = len(comment)
    is_weekend = 1.0 if timestamp.dayofweek > 4 else 0.0

    # Working hours: 8:30 - 16:30
    start_time = time(8, 30)
    end_time = time(16, 30)
    working_hours = 1.0 if start_time <= timestamp.time() <= end_time else 0.0
    month = timestamp.month

    return {
        "is_type": is_type,
        "comment_length": comment_length,
        "is_weekend": is_weekend,
        "working_hours": working_hours,
        "month": month,
        "timestamp": str(timestamp)
    }

def predict_duration(text, is_type, comment_length, is_weekend, working_hours, month):
    """Make a prediction using the model"""
    model.eval()

    # Tokenize text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN
    )

    # Move to device
    ids = inputs["input_ids"].to(DEVICE)
    mask = inputs["attention_mask"].to(DEVICE)
    is_type_t = torch.tensor([float(is_type)], dtype=torch.float).to(DEVICE)
    comment_len_t = torch.tensor([float(comment_length)], dtype=torch.float).to(DEVICE)
    is_weekend_t = torch.tensor([float(is_weekend)], dtype=torch.float).to(DEVICE)
    working_hr_t = torch.tensor([float(working_hours)], dtype=torch.float).to(DEVICE)
    month_t = torch.tensor([float(month)], dtype=torch.float).to(DEVICE)

    with torch.no_grad():
        output = model(ids, mask, is_type_t, comment_len_t, is_weekend_t, working_hr_t, month_t)
        pred_days = output.cpu().item()

    return pred_days

@app.on_event("startup")
async def startup_event():
    """Load model and initialize cache on startup"""
    load_model_on_startup()
    cache.init_redis()

@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "message": "Traffy Fondue Duration Prediction API",
        "version": "2.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": True,
        "tokenizer_loaded": True,
        "device": str(DEVICE),
        "cache": cache.get_cache_stats()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a prediction for a single Traffy complaint

    Returns the predicted resolution time in days and hours.
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Generate cache key
        cache_key = cache.generate_cache_key(request.model_dump())

        # Check cache first
        cached_result = cache.get_cached_prediction(cache_key)
        if cached_result:
            return PredictionResponse(**cached_result)

        # Extract features
        features = extract_features(
            comment=request.comment,
            type_str=request.type,
            timestamp_str=request.timestamp
        )

        # Combine text like the training format
        text = (
            'ปัญหา:' + request.comment.strip() +
            ' ประเภท:' + request.type +
            ' หน่วยงาน:' + request.organization +
            ' เขต:' + request.district +
            ' แขวง:' + request.subdistrict
        )

        # Make prediction
        pred_days = predict_duration(
            text=text,
            is_type=features["is_type"],
            comment_length=features["comment_length"],
            is_weekend=features["is_weekend"],
            working_hours=features["working_hours"],
            month=features["month"]
        )

        # Prepare response
        result = {
            "predicted_days": round(pred_days, 2),
            "predicted_hours": round(pred_days * 24, 1),
            "comment": request.comment,
            "features": features
        }

        # Cache the result
        cache.set_cached_prediction(cache_key, result)

        return PredictionResponse(**result)

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch_predict")
async def batch_predict(requests: list[PredictionRequest]):
    """
    Make predictions for multiple complaints at once

    Returns a list of predictions.
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []

    for req in requests:
        try:
            # Generate cache key
            cache_key = cache.generate_cache_key(req.model_dump())

            # Check cache first
            cached_result = cache.get_cached_prediction(cache_key)
            if cached_result:
                results.append(cached_result)
                continue

            # Extract features
            features = extract_features(
                comment=req.comment,
                type_str=req.type,
                timestamp_str=req.timestamp
            )

            # Combine text
            text = (
                'ปัญหา:' + req.comment.strip() +
                ' ประเภท:' + req.type +
                ' หน่วยงาน:' + req.organization +
                ' เขต:' + req.district +
                ' แขวง:' + req.subdistrict
            )

            # Make prediction
            pred_days = predict_duration(
                text=text,
                is_type=features["is_type"],
                comment_length=features["comment_length"],
                is_weekend=features["is_weekend"],
                working_hours=features["working_hours"],
                month=features["month"]
            )

            result = {
                "predicted_days": round(pred_days, 2),
                "predicted_hours": round(pred_days * 24, 1),
                "comment": req.comment,
                "features": features
            }

            # Cache the result
            cache.set_cached_prediction(cache_key, result)

            results.append(result)

        except Exception as e:
            logger.error(f"Error during batch prediction: {str(e)}")
            results.append({
                "error": str(e),
                "comment": req.comment
            })

    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
