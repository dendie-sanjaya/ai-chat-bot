import requests
import json

# --- Konfigurasi ---
ollama_model_name = "deepseek-llm-67b-base"
ollama_api_url = "http://localhost:11434/api/generate"

# --- Simulasi Pengetahuan Addon ---
addon_knowledge = "Topik X sangat penting karena Y dan Z."
user_query = "Jelaskan mengapa Topik X penting."

# --- Formulasikan Prompt untuk Ollama ---
final_prompt = f"Berdasarkan informasi berikut: '{addon_knowledge}', jawab pertanyaan ini: '{user_query}'"

# --- Fungsi untuk Berinteraksi dengan Ollama ---
def query_ollama(model_name, prompt):
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(ollama_api_url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        return response.json()['response']
    except requests.exceptions.RequestException as e:
        print(f"Error querying Ollama: {e}")
        return None

# --- Kirim Prompt dan Cetak Hasil ---
ollama_response = query_ollama(ollama_model_name, final_prompt)

if ollama_response:
    print(f"Pertanyaan Pengguna: {user_query}")
    print(f"Pengetahuan dari Model Addon: {addon_knowledge}")
    print(f"Jawaban dari Ollama ({ollama_model_name}): {ollama_response}")