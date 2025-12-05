# Traffy Fondue Duration Prediction API Documentation

## Overview

FastAPI-based REST API for predicting resolution time of Traffy Fondue complaints using a BERT-based regression model (WangchanBERTa).

**Version:** 2.0.0
**Base URL:** `http://localhost:8000`

---

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [API Endpoints](#api-endpoints)
  - [GET /](#get-)
  - [GET /health](#get-health)
  - [POST /predict](#post-predict)
  - [POST /batch_predict](#post-batch_predict)
- [Request/Response Examples](#requestresponse-examples)
- [Error Handling](#error-handling)
- [Model Details](#model-details)

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Server

```bash
python app.py
```

### 3. Access API Documentation

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.2.0+
- Transformers 4.37.2+
- FastAPI 0.109.0+

### Setup

```bash
# Clone repository
git clone <repository-url>
cd project_model

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run server
python app.py
```

The API will be available at `http://localhost:8000`.

---

## API Endpoints

### GET `/`

**Description:** API information and available endpoints

**Response:**

```json
{
  "message": "Traffy Fondue Duration Prediction API",
  "version": "2.0.0",
  "status": "running",
  "model_loaded": true,
  "endpoints": {
    "docs": "/docs",
    "health": "/health",
    "predict": "/predict",
    "batch_predict": "/batch_predict"
  }
}
```

---

### GET `/health`

**Description:** Health check endpoint to verify model is loaded

**Response (Success - 200):**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "tokenizer_loaded": true,
  "device": "cpu"
}
```

**Response (Error - 503):**

```json
{
  "detail": "Model not loaded"
}
```

---

### POST `/predict`

**Description:** Predict resolution time for a single complaint

#### Request Body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `comment` | string | ✅ Yes | - | Complaint description in Thai |
| `type` | string | No | `"{}"` | Complaint type (e.g., `"{ร้องเรียน,สัตว์จรจัด}"`) |
| `organization` | string | No | `""` | Responsible organization |
| `district` | string | No | `""` | District name |
| `subdistrict` | string | No | `""` | Subdistrict name |
| `timestamp` | string | No | `null` | ISO format timestamp (e.g., `"2025-01-15T11:27:20+00:00"`) |

#### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "comment": "หมาจรในซอยเฉลิมพระเกียรติร.9 42 นี้ดุ วิ่งไล่กัดคนเดินผ่าน เดือดร้อนคนแถวนั้น",
    "type": "{ร้องเรียน,สัตว์จรจัด}",
    "organization": "เขตประเวศ,สำนักงานสัตวแพทย์สาธารณสุข สำนักอนามัย กทม.",
    "district": "ประเวศ",
    "subdistrict": "หนองบอน",
    "timestamp": "2025-01-15T11:27:20+00:00"
  }'
```

#### Response

```json
{
  "predicted_days": 1.50,
  "predicted_hours": 36.0,
  "comment": "หมาจรในซอยเฉลิมพระเกียรติร.9 42 นี้ดุ วิ่งไล่กัดคนเดินผ่าน เดือดร้อนคนแถวนั้น",
  "features": {
    "is_type": 1.0,
    "comment_length": 62,
    "is_weekend": 0.0,
    "working_hours": 1.0,
    "month": 1,
    "timestamp": "2025-01-15 11:27:20+00:00"
  }
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `predicted_days` | float | Predicted resolution time in days |
| `predicted_hours` | float | Predicted resolution time in hours |
| `comment` | string | Original complaint comment |
| `features` | object | Extracted features used for prediction |
| `features.is_type` | float | Type indicator (1.0 if type exists, 0.0 otherwise) |
| `features.comment_length` | int | Length of comment in characters |
| `features.is_weekend` | float | Weekend indicator (1.0 = weekend, 0.0 = weekday) |
| `features.working_hours` | float | Working hours indicator (1.0 = 8:30-16:30, 0.0 = outside) |
| `features.month` | int | Month (1-12) |
| `features.timestamp` | string | Parsed timestamp |

---

### POST `/batch_predict`

**Description:** Predict resolution time for multiple complaints at once

#### Request Body

Array of complaint objects (same format as `/predict`)

#### Example Request

```bash
curl -X POST "http://localhost:8000/batch_predict" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "comment": "ถนนมีหลุมบ่อมาก ต้องการซ่อมแซม",
      "type": "{ร้องเรียน,ถนน}",
      "district": "ประเวศ",
      "timestamp": "2025-01-15T10:00:00+00:00"
    },
    {
      "comment": "ขอแจ้งเรื่องไฟฟ้าขัดข้อง",
      "type": "{ร้องเรียน,ไฟฟ้า}",
      "district": "บางกอกน้อย",
      "timestamp": "2025-01-16T14:30:00+00:00"
    }
  ]'
```

#### Response

```json
[
  {
    "predicted_days": 2.34,
    "predicted_hours": 56.2,
    "comment": "ถนนมีหลุมบ่อมาก ต้องการซ่อมแซม",
    "features": {
      "is_type": 1.0,
      "comment_length": 28,
      "is_weekend": 0.0,
      "working_hours": 1.0,
      "month": 1,
      "timestamp": "2025-01-15 10:00:00+00:00"
    }
  },
  {
    "predicted_days": 1.87,
    "predicted_hours": 44.9,
    "comment": "ขอแจ้งเรื่องไฟฟ้าขัดข้อง",
    "features": {
      "is_type": 1.0,
      "comment_length": 24,
      "is_weekend": 0.0,
      "working_hours": 1.0,
      "month": 1,
      "timestamp": "2025-01-16 14:30:00+00:00"
    }
  }
]
```

---

## Request/Response Examples

### Python Example

```python
import requests

# Single prediction
url = "http://localhost:8000/predict"
payload = {
    "comment": "หมาจรในซอยเฉลิมพระเกียรติร.9 42 นี้ดุ",
    "type": "{ร้องเรียน,สัตว์จรจัด}",
    "organization": "เขตประเวศ",
    "district": "ประเวศ",
    "subdistrict": "หนองบอน",
    "timestamp": "2025-01-15T11:27:20+00:00"
}

response = requests.post(url, json=payload)
result = response.json()

print(f"Predicted resolution time: {result['predicted_days']:.2f} days")
print(f"Predicted resolution time: {result['predicted_hours']:.1f} hours")
```

### JavaScript Example

```javascript
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    comment: 'หมาจรในซอยเฉลิมพระเกียรติร.9 42 นี้ดุ',
    type: '{ร้องเรียน,สัตว์จรจัด}',
    organization: 'เขตประเวศ',
    district: 'ประเวศ',
    subdistrict: 'หนองบอน',
    timestamp: '2025-01-15T11:27:20+00:00'
  })
});

const result = await response.json();
console.log(`Predicted: ${result.predicted_days} days`);
```

---

## Error Handling

### Common Error Codes

| Status Code | Description | Possible Cause |
|-------------|-------------|----------------|
| 400 | Bad Request | Invalid JSON or missing required fields |
| 422 | Unprocessable Entity | Invalid data types or validation errors |
| 500 | Internal Server Error | Model prediction failed |
| 503 | Service Unavailable | Model not loaded |

### Error Response Format

```json
{
  "detail": "Error message here"
}
```

### Example Error Response

```json
{
  "detail": "Model not loaded"
}
```

---

## Model Details

### Architecture

- **Base Model:** WangchanBERTa (airesearch/wangchanberta-base-att-spm-uncased)
- **Task:** Regression
- **Input:** Thai text + 5 numeric features
- **Output:** Resolution time in days (continuous value)

### Input Features

1. **Text Features:**
   - Combined text: `ปัญหา:{comment} ประเภท:{type} หน่วยงาน:{organization} เขต:{district} แขวง:{subdistrict}`
   - Tokenized using WangchanBERTa tokenizer
   - Max length: 512 tokens

2. **Numeric Features:**
   - `is_type`: 1.0 if complaint type is specified, 0.0 otherwise
   - `comment_length`: Number of characters in comment
   - `is_weekend`: 1.0 if Saturday/Sunday, 0.0 otherwise
   - `working_hours`: 1.0 if 8:30-16:30, 0.0 otherwise
   - `month`: Month number (1-12)

### Model Performance

The model predicts resolution time in actual days (not normalized). Typical prediction range: 0.5 - 5.0 days.

### Limitations

- Model trained specifically for Thai Traffy Fondue complaints
- Best performance on complaint types seen during training
- Timestamp defaults to current time if not provided
- Requires Thai language input for best results

---

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest httpx

# Run tests (if test file exists)
pytest test_app.py
```

### Docker Deployment

```bash
# Build image
docker build -t traffy-api .

# Run container
docker run -p 8000:8000 traffy-api
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `./traffy_predict/best_bert_regressor.pt` | Path to model weights |
| `TOKENIZER_PATH` | `./traffy_predict/tokenizer` | Path to tokenizer |
| `PORT` | `8000` | Server port |
| `HOST` | `0.0.0.0` | Server host |

---

## Support & Contact

For issues or questions:
- Check the interactive API docs: http://localhost:8000/docs
- Review model architecture in `traffy_predict/model.py`
- Example usage in `traffy_predict/use.py`

---

## License

[Add your license information here]

---

## Changelog

### Version 2.0.0
- Complete rewrite based on working `use.py` implementation
- Fixed model loading with proper embedding resize
- Added text combining feature (ปัญหา + ประเภท + หน่วยงาน + เขต + แขวง)
- Returns predictions in both days and hours
- Includes extracted features in response
- Proper feature extraction (working hours, weekend detection)
- Improved error handling and logging

### Version 1.0.0
- Initial release
