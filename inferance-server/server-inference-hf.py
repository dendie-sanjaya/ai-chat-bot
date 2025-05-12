from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from fastapi.middleware.cors import CORSMiddleware # Import CORSMiddleware

app = FastAPI()

# Tambahkan middleware CORS di sini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Konfigurasi Model Hugging Face ---
MODEL_NAME = "/mnt/d/ai-chat-bot/fine-tuning/distilgpt2-bandung-hf-final/"  # Ganti dengan nama model Hugging Face yang ingin Anda gunakan
TOKENIZER = None
MODEL = None

@app.on_event("startup")
async def startup_event():
    global TOKENIZER, MODEL
    try:
        print(f"Memuat tokenizer untuk: {MODEL_NAME}")
        TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
        print(f"Memuat model untuk: {MODEL_NAME}")
        MODEL = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        MODEL.eval()  # Set model ke mode evaluasi
        if torch.cuda.is_available():
            MODEL.to("cuda")
            print("Model dipindahkan ke GPU.")
        else:
            print("Menggunakan CPU untuk inferensi.")
        print("Model dan tokenizer berhasil dimuat.")
    except Exception as e:
        raise RuntimeError(f"Gagal memuat model atau tokenizer: {e}")

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 50
    temperature: float = 1.0
    top_p: float = 1.0

class GenerationResponse(BaseModel):
    text: str

@app.post("/generate")
async def generate_text(request: GenerationRequest):
    if MODEL is None or TOKENIZER is None:
        raise HTTPException(status_code=503, detail="Model atau tokenizer belum dimuat.")
    try:
        input_ids = TOKENIZER.encode(request.prompt, return_tensors="pt")
        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")

        with torch.no_grad():
            output = MODEL.generate(
                input_ids,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=TOKENIZER.eos_token_id
            )

        generated_text = TOKENIZER.decode(output[0], skip_special_tokens=True)
        return GenerationResponse(text=generated_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
