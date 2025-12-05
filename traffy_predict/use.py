import torch
from transformers import AutoTokenizer
from model import TraffyBertRegressor
from datetime import datetime, time
import pandas as pd

# Configuration
MAX_LEN = 512
DEVICE = torch.device("cpu")

def load_model(model_path="best_bert_regressor.pt", tokenizer_path="./tokenizer"):
    """Load the trained model and tokenizer"""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Initialize model
    model = TraffyBertRegressor(model_name="airesearch/wangchanberta-base-att-spm-uncased")

    # Resize embeddings to match the saved model (249262 tokens)
    model.bert.resize_token_embeddings(249262)

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    return model, tokenizer

def predict_duration(model, tokenizer, text, is_type, comment_length, is_weekend, working_hours, month):
    """
    Make a prediction using the model

    Args:
        model: Trained TraffyBertRegressor model
        tokenizer: BERT tokenizer
        text: Input text (complaint description with metadata)
        is_type: Type indicator (1.0 if type is not empty, 0.0 otherwise)
        comment_length: Length of comment (float)
        is_weekend: Weekend indicator (0.0 or 1.0)
        working_hours: Working hour indicator (0.0 or 1.0)
        month: Month (1-12)

    Returns:
        Predicted duration in days (float) - already in actual days, no denormalization needed
    """
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

if __name__ == "__main__":
    # Example problem from owner
    problem = {
        "ticket_id": "2025-F2F33Y",
        "type": "หาบเร่แผงลอย",
        "organization": "กรุงเทพมหานคร, เขตธนบุรี, ฝ่ายเทศกิจ เขตธนบุรี",
        "comment": "แม่ค้าลักลอบขายของกีดขวางถ.เลียบทางรถไฟตรงสุเหร่าสวนพลูโดยไม่ได้รีบอนุญาติ ตรงนี้นอกเขตผ่อนผันขายของไม่ได้เทศกิจมากวดขันให้หยุดขายของบริเวณนี้เดี๋ยวนี้",
        "photo": "https://storage.googleapis.com/traffy_public_bucket/attachment/2025-01/91c231182ee1160cdb974f6dc9db589f30664673.jpg",
        "photo_after": "https://storage.googleapis.com/traffy_public_bucket/attachment/2025-01/5a032ac5379fda18f181d27ed3784095.jpg",
        "coords": "100.67579,13.69392",
        "address": "31 ซอย เฉลิมพระเกียรติ42 แขวงหนองบอน เขตประเวศ กรุงเทพมหานคร 10250 ประเทศไทย",
        "subdistrict": "บางยี่เรือ",
        "district": "ธนบุรี",
        "province": "กรุงเทพมหานคร",
        "timestamp": "2025-01-15 11:27:20.109721+00",
        "state": "เสร็จสิ้น",
        "star": "null",
        "count_reopen": 0,
        "last_activity": "2025-01-16 02:33:20.132654+00"
    }

    # Load model
    print("Loading model...")
    model, tokenizer = load_model()

    # Parse timestamp
    timestamp = pd.to_datetime(problem["timestamp"], errors="coerce")

    # Combine text like the owner does
    text = (
        'ปัญหา:' + str(problem['comment']).strip() +
        ' ประเภท:' + problem['type'] +
        ' หน่วยงาน:' + problem['organization'] +
        ' เขต:' + problem['district'] +
        ' แขวง:' + problem['subdistrict']
    )

    # Extract features
    is_type = 1.0 if problem["type"] != "{}" else 0.0
    comment_length = len(problem['comment'])
    is_weekend = 1.0 if timestamp.dayofweek > 4 else 0.0

    # Working hours: 8:30 - 16:30
    start_time = time(8, 30)
    end_time = time(16, 30)
    working_hours = 1.0 if start_time <= timestamp.time() <= end_time else 0.0
    month = timestamp.month

    # Predict
    print("\n=== Prediction ===")
    print(f"Ticket ID: {problem['ticket_id']}")
    print(f"Comment: {problem['comment']}")
    print(f"Type: {problem['type']}")
    print(f"Organization: {problem['organization']}")
    print(f"Timestamp: {timestamp}")
    print(f"\nFeatures:")
    print(f"  - is_type: {is_type}")
    print(f"  - comment_length: {comment_length}")
    print(f"  - is_weekend: {is_weekend}")
    print(f"  - working_hours: {working_hours}")
    print(f"  - month: {month}")

    pred_days = predict_duration(
        model=model,
        tokenizer=tokenizer,
        text=text,
        is_type=is_type,
        comment_length=comment_length,
        is_weekend=is_weekend,
        working_hours=working_hours,
        month=month
    )

    print(f"\nPredicted resolution time: {pred_days:.2f} days")

    # Show in different units for clarity
    print(f"  ≈ {pred_days * 24:.1f} hours")
    if pred_days < 1:
        print(f"  ≈ {pred_days * 24 * 60:.0f} minutes")
