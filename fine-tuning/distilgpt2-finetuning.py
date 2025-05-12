from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama
from transformers import pipeline
import asyncio

app = FastAPI()

# Tambahkan middleware CORS di sini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ganti dengan jalur sebenarnya ke model GGUF Anda
MODEL_PATH = "/mnt/d/ai-chat-bot/inferance-server/distilgpt2-bandung.gguf"

# Load model Llama untuk endpoint /generate
try:
    llm = Llama(model_path=MODEL_PATH)
except Exception as e:
    print(f"Gagal memuat model Llama: {e}")

# Load model DistilGPT2
try:
    generator = pipeline('text-generation', model='distilgpt2')
except Exception as e:
    print(f"Gagal memuat model DistilGPT2: {e}")

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.8
    top_p: float = 0.9

async def generate_tokens(request: GenerationRequest, model_type: str = "llama"):
    try:
        if model_type == "llama":
            output = llm.create_completion(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stream=True,
                echo=False
            )
            for item in output:
                token = item["choices"][0]["text"]
                yield f"data: {token}\n\n"
                await asyncio.sleep(0.001)
        elif model_type == "distilgpt2":
            generated_text = generator(
                request.prompt,
                max_length=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                num_return_sequences=1,
            )[0]['generated_text']
            for char in generated_text:
                yield f"data: {char}\n\n"
                await asyncio.sleep(0.001)
        else:
            raise ValueError(f"Model type '{model_type}' not supported.")
    except Exception as e:
        yield f"data: error: {str(e)}\n\n"

@app.post("/generate_stream")
async def generate_text_stream(request: GenerationRequest, model: str = "llama"):
    """
    Endpoint ini menggunakan StreamingResponse untuk menghasilkan teks secara bertahap.
    Argumen 'model' menentukan model mana yang akan digunakan ("llama" atau "distilgpt2").
    """
    if model not in ["llama", "distilgpt2"]:
        raise HTTPException(status_code=400, detail="Parameter 'model' harus 'llama' atau 'distilgpt2'.")
    return StreamingResponse(generate_tokens(request, model_type=model), media_type="text/event-stream")

@app.post("/generate")
async def generate_text(request: GenerationRequest):
    """
    Endpoint ini menghasilkan teks menggunakan model Llama dan mengembalikan respons JSON.
    """
    try:
        output = llm.create_completion(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=False,
            echo=False,
        )
        generated_text = output["choices"][0]["text"]
        return JSONResponse({"generated_text": generated_text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {e}")

@app.post("/generate_transformers")
async def generate_text_transformers(request: GenerationRequest):
    """
    Endpoint ini menghasilkan teks menggunakan model DistilGPT2 dan mengembalikan respons JSON.
    """
    return StreamingResponse(generate_tokens(request, model_type="distilgpt2"), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8181)

