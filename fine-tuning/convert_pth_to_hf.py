from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

model_path = "distilgpt2-bandung.pth"  # Ganti dengan path file .pth Anda
base_model_name = "distilgpt2"  # Asumsi model dasar adalah distilgpt2
output_path = "distilgpt2-bandung-pth-to-hf"  # Direktori untuk menyimpan format HF

try:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model.load_state_dict(torch.load(model_path))
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Model berhasil disimpan dalam format Hugging Face Transformers di: {output_path}")
except Exception as e:
    print(f"Gagal menyimpan model dalam format Hugging Face Transformers: {e}")
    print("Pastikan path file .pth benar dan model kompatibel dengan arsitektur DistilGPT-2.")
    print("Error: {e}")
    exit(