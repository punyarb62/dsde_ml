# Traffy Fondue Duration Prediction API

FastAPI microservice for predicting resolution time for Traffy Fondue complaints using a BERT-based regression model (WangchanBERTa).

## Features

- 🚀 FastAPI-based REST API
- 🤖 Thai language BERT model (WangchanBERTa)
- 📊 Predicts complaint resolution time in days
- 🔄 Single and batch prediction endpoints
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

## Model Architecture

The model uses **WangchanBERTa** (Thai BERT) combined with 5 numeric features:
- `is_type`: Type indicator
- `comment_len`: Length of comment
- `is_weekend`: Weekend indicator (0.0 or 1.0)
- `working_hr`: Working hour indicator (0.0 or 1.0)
- `month`: Month (1-12)

## Quick Start

### Run FastAPI Server

```bash
python app.py
```

The API will be available at `http://localhost:8000`

**Access Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Full API Docs: See [API_DOCUMENTATION.md](API_DOCUMENTATION.md)

### Test the Model Locally

```bash
cd traffy_predict
python use.py
```

### Run with Docker

```bash
# Build image
docker build -t traffy-api .

# Run container
docker run -p 8000:8000 traffy-api
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

## Project Structure

```
project_model/
├── app.py                          # FastAPI application
├── requirements.txt                # Python dependencies
├── API_DOCUMENTATION.md            # Complete API documentation
├── README.md                       # This file
├── Dockerfile                      # Docker configuration
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
