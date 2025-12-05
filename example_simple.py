"""
Simple example script to use the Traffy model
"""
import torch
from transformers import AutoTokenizer
from traffy_predict.model import TraffyBertRegressor

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("./traffy_predict/tokenizer")

# Load model
print("Loading model...")
model = TraffyBertRegressor(model_name="airesearch/wangchanberta-base-att-spm-uncased")
# Resize embeddings to match the saved model (249262 tokens)
model.bert.resize_token_embeddings(249262)
model.load_state_dict(torch.load("./traffy_predict/best_bert_regressor.pt", map_location=torch.device('cpu')))
model.eval()

# Example 1: Simple prediction
print("\n=== Example 1 ===")
text = "ถนนมีหลุมบ่อมาก ต้องการซ่อมแซม"

# Tokenize
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Prepare features
is_type = torch.tensor([1.0])
comment_len = torch.tensor([50.0])
is_weekend = torch.tensor([0.0])
working_hr = torch.tensor([1.0])
month = torch.tensor([5.0])

# Predict
with torch.no_grad():
    prediction = model(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        is_type=is_type,
        comment_len=comment_len,
        is_weekend=is_weekend,
        working_hr=working_hr,
        month=month
    )

print(f"Text: {text}")
print(f"Prediction: {prediction.item():.2f}")

# Example 2: Another prediction
print("\n=== Example 2 ===")
text2 = "ขอแจ้งเรื่องไฟฟ้าขัดข้อง"

inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True, max_length=512)

with torch.no_grad():
    prediction2 = model(
        input_ids=inputs2['input_ids'],
        attention_mask=inputs2['attention_mask'],
        is_type=torch.tensor([2.0]),
        comment_len=torch.tensor([40.0]),
        is_weekend=torch.tensor([1.0]),
        working_hr=torch.tensor([0.0]),
        month=torch.tensor([6.0])
    )

print(f"Text: {text2}")
print(f"Prediction: {prediction2.item():.2f}")
