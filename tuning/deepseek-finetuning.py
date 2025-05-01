from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch
import pandas as pd

# 1. Tentukan nama model pre-trained DeepSeek yang ingin Anda fine-tune
model_name = "deepseek-ai/deepseek-llm-67B-base"  # Ganti dengan versi yang Anda gunakan di Ollama

# 2. Tentukan path ke file CSV dataset Anda yang berisi pengetahuan baru
csv_file_path = "dataset/data_pengetahuan_baru.csv"  # Ganti dengan path file CSV Anda

# 3. Konfigurasi LoRA (sesuaikan parameter sesuai kebutuhan dan sumber daya)
lora_config = LoraConfig(
    r=16,  # Rank dari matriks adaptasi rendah
    lora_alpha=32, # Skala untuk matriks adaptasi
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM" # Sesuaikan dengan tugas Anda (misalnya, QUESTION_ANS)
)

# 4. Konfigurasi Quantisasi (opsional, tapi sangat disarankan untuk model besar)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
)

# 5. Muat tokenizer yang sesuai dengan DeepSeek
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 6. Muat model pre-trained DeepSeek
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto" # Secara otomatis menempatkan model di GPU jika tersedia
)

# 7. Tambahkan adaptor LoRA ke model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 8. Muat dataset Anda dari CSV
df = pd.read_csv(csv_file_path)
# Asumsikan CSV memiliki kolom 'text' yang berisi data untuk fine-tuning
dataset = load_dataset("pandas", data_files=df)

# 9. Fungsi tokenisasi untuk memproses dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512) # Sesuaikan max_length

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 10. Konfigurasi Argumen Pelatihan
training_args = TrainingArguments(
    output_dir="./deepseek-addon-permanent", # Direktori untuk menyimpan model addon
    per_device_train_batch_size=4, # Sesuaikan dengan kapasitas GPU Anda
    gradient_accumulation_steps=4, # Untuk batch size efektif yang lebih besar
    learning_rate=3e-4, # Sesuaikan learning rate
    num_train_epochs=3, # Sesuaikan jumlah epoch
    optim="adamw_torch",
    fp16=True, # Gunakan mixed precision untuk efisiensi memori dan kecepatan
    logging_dir="./logs-permanent",
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="no", # Anda bisa menambahkan validasi jika punya dataset validasi
    report_to="tensorboard" # Aktifkan jika Anda ingin menggunakan TensorBoard untuk memantau pelatihan
)

# 11. Inisialisasi Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=lambda data: {
        "input_ids": torch.stack([f["input_ids"] for f in data]),
        "attention_mask": torch.stack([f["attention_mask"] for f in data]),
        "labels": torch.stack([f["input_ids"] for f in data]), # Untuk causal language modeling, targetnya adalah input itu sendiri
    },
)

# 12. Mulai proses fine-tuning
trainer.train()

# 13. Simpan model "addon" yang telah di-fine-tune (adaptor LoRA)
trainer.save_model("./deepseek-addon-permanent-final")

print(f"Fine-tuning selesai. Model addon (adaptor LoRA) telah disimpan di ./deepseek-addon-permanent-final.")
print("Selanjutnya, Anda dapat menggunakan adaptor ini dengan model DeepSeek di aplikasi web Anda untuk mendapatkan pengetahuan yang lebih permanen.")