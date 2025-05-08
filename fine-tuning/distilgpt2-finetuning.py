import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from torch.utils.data import Dataset
import torch

# 1. Path ke dataset CSV Anda
csv_path = "/mnt/d/ai-chat-bot/dataset/dataset-bandung.csv"
model_name = "distilgpt2"  # Mengganti gpt2 dengan distilgpt2
output_dir = "./distilgpt2-bandung-hf"
max_length = 512  # Perpanjang max_length

# 2. Muat tokenizer dan model untuk generasi
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

# 3. Baca dataset CSV
try:
    df = pd.read_csv(csv_path, delimiter=';', header=None, names=['pertanyaan', 'jawaban'])
    questions = df['pertanyaan'].tolist()
    answers = df['jawaban'].tolist()
except Exception as e:
    print(f"Error membaca CSV: {e}")
    exit()

# 4. Buat Dataset untuk Question Answering (Generatif)
class QADataset(Dataset):
    def __init__(self, questions, answers, tokenizer, max_length):
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]

        # Format prompt untuk QA generatif
        prompt = question + tokenizer.eos_token
        inputs = self.tokenizer(prompt, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')

        # Tokenisasi jawaban sebagai target
        labels = self.tokenizer(answer, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')['input_ids']

        # Penting: Ganti padding token di labels dengan -100 agar tidak dihitung dalam loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

dataset = QADataset(questions, answers, tokenizer, max_length)

# 5. Training Arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,  # Sesuaikan dengan GPU Anda
    gradient_accumulation_steps=8,
    learning_rate=1e-4,  # Sesuaikan learning rate
    num_train_epochs=5,  # Tambah jumlah epoch
    fp16=True,
    logging_dir="./logs-qa",
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    push_to_hub=False,
)

# 6. Trainer
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    tokenizer=tokenizer,
)

# 7. Train
trainer.train()

# 8. Simpan model
trainer.save_model(f"{output_dir}-final")
tokenizer.save_pretrained(f"{output_dir}-final")
print(f"\nFine-tuning selesai! Model disimpan di {output_dir}-final")