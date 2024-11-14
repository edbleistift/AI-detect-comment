import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.preprocessing import LabelEncoder

# Загружаем и предобрабатываем данные
file_path = 'cleaned_kaspi_reviews.csv'
df = pd.read_csv(file_path, delimiter=",", encoding="ISO-8859-1", on_bad_lines="skip")

# Фильтрация и метки
df = df[df['rating'].isin([1, 2, 4, 5])]  # Убираем нейтральные оценки
df['label'] = df['rating'].apply(lambda x: 0 if x <= 2 else 1)  # 0 - плохие, 1 - хорошие

# Разделение данных на обучающую и тестовую выборки
train_texts, val_texts, train_labels, val_labels = train_test_split(df['combined_text'], df['label'], test_size=0.2)

# Токенизация и преобразование текста
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)

class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ReviewDataset(train_encodings, train_labels.tolist())
val_dataset = ReviewDataset(val_encodings, val_labels.tolist())

# Инициализация модели
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Настройка тренировки
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
optimizer = AdamW(model.parameters(), lr=1e-5)

# Функция для тренировки
def train_model():
    model.train()
    for epoch in range(3):  # 3 эпохи обучения
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f'Epoch {epoch + 1} loss: {loss.item()}')

# Тренировка модели
train_model()

# Оценка модели
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        predictions = torch.argmax(outputs.logits, dim=-1)
        correct += (predictions == batch['labels']).sum().item()
        total += batch['labels'].size(0)

print(f'Validation Accuracy: {correct / total:.2f}')
