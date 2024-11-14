import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Загрузка модели и токенизатора из директории saved_model
model_path = "saved_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Переводим модель в режим оценки
model.eval()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

def predict_comment(comment):
    # Токенизация текста
    inputs = tokenizer(comment, truncation=True, padding=True, max_length=128, return_tensors="pt")
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
    
    label = "Хороший" if prediction == 1 else "Плохой"
    return label

# comment = "Доставка быстрая, товар хорошего качества, всем доволен!"
# comment = "Продавец не отвечал на звонки, товар так и не отправили"
comment = "Я не ожидал, что доставка будет настолько быстрой."

print(f"Комментарий: {comment}")
print("Результат предсказания:", predict_comment(comment))
