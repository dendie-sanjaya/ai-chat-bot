from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llama_cpp import Llama
import asyncio

app = FastAPI()

# Ganti dengan jalur sebenarnya ke model GGUF Anda
# MODEL_PATH = "deepseek-r11.5b.gguf"
MODEL_PATH = "/mnt/d/ai/ai-chat-bot/inferance-server/distilgpt2-bandung.gguf"

try:
    llm = Llama(model_path=MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Gagal memuat model: {e}")

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.8
    top_p: float = 0.9

async def generate_tokens(request: GenerationRequest):
    try:
        for output in llm.create_completion(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=True,
            echo=False
        ):
            token = output["choices"][0]["text"]
            yield f"data: {token}\n\n"
            await asyncio.sleep(0.001)  # Optional: Add a small delay
    except Exception as e:
        yield f"data: error: {str(e)}\n\n"

@app.post("/generate_stream")
async def generate_text_stream(request: GenerationRequest):
    return StreamingResponse(generate_tokens(request), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8181)