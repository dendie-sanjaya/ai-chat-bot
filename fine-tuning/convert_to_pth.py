from transformers import AutoModelForCausalLM
import torch

model_path = "./distilgpt2-bandung-hf-final"  # Path ke model yang Anda simpan
output_path = "./distilgpt2-bandung.pth"

model = AutoModelForCausalLM.from_pretrained(model_path)
torch.save(model.state_dict(), output_path)

print(f"Model saved to {output_path}")