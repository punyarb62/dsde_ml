# Traffy Prediction API

FastAPI microservice for predicting resolution time for Traffy Fondue complaints using a BERT-based regression model.

## Model Architecture

The model uses **WangchanBERTa** (Thai BERT) combined with 5 numeric features:
- `is_type`: Type indicator
- `comment_len`: Length of comment
- `is_weekend`: Weekend indicator (0.0 or 1.0)
- `working_hr`: Working hour indicator (0.0 or 1.0)
- `month`: Month (1-12)

## Quick Start

### 1. Test the Model Locally

```bash
cd traffy_predict
python use.py
```

### 2. Run FastAPI Server

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python app.py
```

The API will be available at `http://localhost:8000`

### 3. Run with Docker

```bash
# Build image
docker build -t traffy-api .

# Run container
docker run -p 8000:8000 traffy-api
```

## API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Health Check

```bash
GET /
GET /health
```

### Single Prediction

```bash
POST /predict
```

**Request Body:**
```json
{
  "text": "ถนนมีหลุมบ่อมาก ต้องการซ่อมแซม",
  "is_type": 1.0,
  "comment_len": 50.0,
  "is_weekend": 0.0,
  "working_hr": 1.0,
  "month": 5.0
}
```

**Response:**
```json
{
  "prediction": 123.45,
  "input_text": "ถนนมีหลุมบ่อมาก ต้องการซ่อมแซม"
}
```

### Batch Prediction

```bash
POST /batch_predict
```

**Request Body:**
```json
[
  {
    "text": "ถนนมีหลุมบ่อมาก",
    "is_type": 1.0,
    "comment_len": 30.0,
    "is_weekend": 0.0,
    "working_hr": 1.0,
    "month": 5.0
  },
  {
    "text": "ขอแจ้งเรื่องไฟฟ้าขัดข้อง",
    "is_type": 2.0,
    "comment_len": 40.0,
    "is_weekend": 1.0,
    "working_hr": 0.0,
    "month": 6.0
  }
]
```

## Example Usage with cURL

```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "ถนนมีหลุมบ่อมาก ต้องการซ่อมแซม",
    "is_type": 1.0,
    "comment_len": 50.0,
    "is_weekend": 0.0,
    "working_hr": 1.0,
    "month": 5.0
  }'
```

## Example Usage with Python

```python
import requests

url = "http://localhost:8000/predict"
payload = {
    "text": "ถนนมีหลุมบ่อมาก ต้องการซ่อมแซม",
    "is_type": 1.0,
    "comment_len": 50.0,
    "is_weekend": 0.0,
    "working_hr": 1.0,
    "month": 5.0
}

response = requests.post(url, json=payload)
print(response.json())
```

## Project Structure

```
project_model/
├── app.py                      # FastAPI application
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── README.md                   # This file
└── traffy_predict/
    ├── model.py               # Model architecture
    ├── use.py                 # Standalone usage example
    ├── best_bert_regressor.pt # Trained model weights
    └── tokenizer/             # WangchanBERTa tokenizer
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install pytest httpx

# Run tests (create test_app.py first)
pytest
```

### Environment Variables

You can configure the following environment variables:

- `MODEL_PATH`: Path to model weights (default: `./traffy_predict/best_bert_regressor.pt`)
- `TOKENIZER_PATH`: Path to tokenizer (default: `./traffy_predict/tokenizer`)
- `PORT`: Server port (default: `8000`)

## Requirements

- Python 3.10+
- PyTorch
- Transformers
- FastAPI
- See [requirements.txt](requirements.txt) for full list

## Model Details

- **Base Model**: airesearch/wangchanberta-base-att-spm-uncased
- **Task**: Regression (predicting resolution time)
- **Input**: Thai text + 5 numeric features
- **Output**: Predicted resolution time (continuous value)

## License

[Add your license here]
