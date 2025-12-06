# Traffy Fondue Duration Prediction API

FastAPI microservice for predicting resolution time for Traffy Fondue complaints using a BERT-based regression model (WangchanBERTa).

## Features

- 🚀 FastAPI-based REST API
- 🤖 Thai language BERT model (WangchanBERTa)
- 📊 Predicts complaint resolution time in days
- 🔄 Single and batch prediction endpoints
- ⚡ Redis caching for faster repeated predictions
- 📝 Complete API documentation
- 🐳 Docker support
- 📦 Git LFS for large model files

## Prerequisites

Before cloning this repository, install **Git LFS** to handle large model files (1.1GB):

### Install Git LFS

**Windows:**
```bash
# Download from https://git-lfs.github.com/
# Or using Chocolatey:
choco install git-lfs
```

**macOS:**
```bash
brew install git-lfs
```

**Linux:**
```bash
sudo apt-get install git-lfs
# or
sudo yum install git-lfs
```

**Initialize Git LFS:**
```bash
git lfs install
```

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd project_model

# Git LFS will automatically download large files
# If not, run:
git lfs pull
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Optional: Start Redis with Docker Compose (for caching)

**Why Redis?** Enables automatic caching of predictions for faster repeated queries.

```bash
docker-compose up -d
```

This starts Redis in the background. The API will automatically connect to it.

**Alternative:** Install Redis locally (see docs) or skip Redis entirely - the API works without it.

## Model Architecture

The model uses **WangchanBERTa** (Thai BERT) combined with 5 numeric features:
- `is_type`: Type indicator
- `comment_len`: Length of comment
- `is_weekend`: Weekend indicator (0.0 or 1.0)
- `working_hr`: Working hour indicator (0.0 or 1.0)
- `month`: Month (1-12)

## Quick Start

### 1. Start Redis (Optional but Recommended)

```bash
docker-compose up -d
```

### 2. Run FastAPI Server

```bash
python app.py
```

The API will be available at `http://localhost:8000`

**Access Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Full API Docs: See [API_DOCUMENTATION.md](API_DOCUMENTATION.md)

### 3. Stop Redis When Done

```bash
docker-compose down
```

### Test the Model Locally

```bash
cd traffy_predict
python use.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/predict` | POST | Single complaint prediction |
| `/batch_predict` | POST | Batch predictions |

### Example: Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "comment": "หมาจรในซอยเฉลิมพระเกียรติร.9 42 นี้ดุ วิ่งไล่กัดคนเดินผ่าน",
    "type": "{ร้องเรียน,สัตว์จรจัด}",
    "organization": "เขตประเวศ",
    "district": "ประเวศ",
    "subdistrict": "หนองบอน",
    "timestamp": "2025-01-15T11:27:20+00:00"
  }'
```

**Response:**
```json
{
  "predicted_days": 1.50,
  "predicted_hours": 36.0,
  "comment": "หมาจรในซอยเฉลิมพระเกียรติร.9 42 นี้ดุ วิ่งไล่กัดคนเดินผ่าน",
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

For complete API documentation, see [API_DOCUMENTATION.md](API_DOCUMENTATION.md)

## Performance & Caching

The API includes built-in Redis caching for optimal performance:

- **Cache Strategy:** Identical requests return cached results instantly
- **Cache Duration:** 20 minutes by default (configurable via `CACHE_TTL`)
- **Automatic Fallback:** If Redis is unavailable, predictions work normally without caching
- **Cache Key:** Based on MD5 hash of all input fields (comment, type, organization, district, subdistrict, timestamp)

**Performance Comparison:**
- Without cache: ~200-500ms per prediction (model inference)
- With cache hit: ~5-10ms per prediction (Redis lookup)

**Check cache status:**
```bash
curl http://localhost:8000/health
```

The `/health` endpoint shows cache statistics including connection status, TTL, and total cached keys.

## Project Structure

```
project_model/
├── app.py                          # FastAPI application
├── cache.py                        # Redis caching utilities
├── requirements.txt                # Python dependencies
├── docker-compose.yml              # Redis Docker setup
├── API_DOCUMENTATION.md            # Complete API documentation
├── README.md                       # This file
├── Dockerfile                      # Docker configuration (optional)
├── example_simple.py               # Simple usage example
├── .gitignore                      # Git ignore rules
├── .gitattributes                  # Git LFS configuration
└── traffy_predict/
    ├── model.py                   # Model architecture
    ├── use.py                     # Standalone prediction script
    ├── best_bert_regressor.pt     # Trained model weights (Git LFS)
    ├── model.pt                   # Alternative model weights (Git LFS)
    └── tokenizer/                 # WangchanBERTa tokenizer files
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

**Model Configuration:**
- `MODEL_PATH`: Path to model weights (default: `./traffy_predict/best_bert_regressor.pt`)
- `TOKENIZER_PATH`: Path to tokenizer (default: `./traffy_predict/tokenizer`)
- `PORT`: Server port (default: `8000`)

**Redis Cache Configuration (Optional):**
- `REDIS_HOST`: Redis server host (default: `localhost`)
- `REDIS_PORT`: Redis server port (default: `6379`)
- `REDIS_DB`: Redis database number (default: `0`)
- `REDIS_PASSWORD`: Redis password (default: `None`)
- `CACHE_TTL`: Cache expiration time in seconds (default: `1200` = 20 minutes)

**Example with custom cache settings:**
```bash
export REDIS_HOST=localhost
export REDIS_PORT=6379
export CACHE_TTL=1800  # 30 minutes
python app.py
```

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
