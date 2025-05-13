# 1. Machine Learning - LLM 

Machine Learning (ML) adalah cabang dari kecerdasan buatan (AI) yang mempelajari pola dan membuat prediksi 
berdasarkan data. 

LLM, atau Large Language Model, adalah jenis model ML yang dilatih menggunakan sejumlah besar teks untuk 
memahami dan menghasilkan bahasa alami. 

Contohnya adalah GPT-3 dan Llama2, Gemini, Copilot, Meta AI,  yang digunakan untuk tugas seperti penerjemahan, penulisan, dan percakapan. 
LLM bekerja dengan cara mempelajari korelasi dan struktur bahasa dari data pelatihan, 
sehingga dapat menghasilkan teks yang koheren dan relevan dengan konteks. 

LLM, atau Large Language Model, adalah jenis model ML yang dilatih menggunakan sejumlah besar teks untuk memahami dan menghasilkan bahasa alami.

Contoh untuk deploy LLM di server private adalah dapat melihat contoh seperti ini 

https://github.com/dendie-sanjaya/ai-ml-llm-ollama

Pada source code ini akan dijelaskan cara membuat chatbot dengan backen AI-ML-LLM 

# 2. Arsitekur AI Chat 

Berikut ini adalah arsitekur AI Chatbot

![ss](./design/architecture.png)


# 3. Install Inferance Ollama Platform 

Sebagai server AI-MI-LLM dapat menggunakan ollama sebagai inferance AI-MML, ollama akan berperaan sebaga inferance dan menyediakan openAPI yang bisa 
di akses frontend 

cara installasi bila melihat disini -> https://github.com/dendie-sanjaya/ai-ml-llm-ollama

Apabila instalasi ollama berhasil dan dapat di run, makaakan tampak sperti ini 

<pre><code>ollama start</code>pre></pre>code> 

![ss](./screenshoot/1.png)


# 4. Import model LLM

Setelah Ollama dapa melakuan import model LLM yang mengambil langsung dari repository model ollama , dalam contoh kali ini model yg digunakan menggunakan deepseek-r1

![ss](./screenshoot/2.png)

![ss](./screenshoot/3.png)

![ss](./screenshoot/4.png)


# 5. Run Deepseek Model LLM

Untuk mengaktifkan Model Deepseek, dapat menggunakan command seperti ini 

<pre><code>ollama run deepseek-r1:1.5b</code></pre>

![ss](./screenshoot/5.png)

# 6. Test Prompt ke Ollama via API

Untuk melakukan test untuk memberikan  Prompt AI ke Deepseek dapat lakukan via postman, apabila API nya running hasilnya akan seperti ini 

![ss](./screenshoot/6.png)

![ss](./screenshoot/7.png)

# 7. Siapkan Dataset 

Dataset adalah data yang sudah bersih atau benar, semakin banyak datanya dengan kualitas yang bagus, maka akan semakin baik untuk training machine learning yang akan menghasil jawaban yg baik, berikut ini adalah contoh dalam membuat 
dataset di format csv untuk keperluan tujuan membuat chatbot

![ss](./screenshoot/8.png)


# 8. Fine Tuning 

Fine tuning adalah proses training mechine learning dengan memberikan pengetahuan baru yang hasil akan digabungkan dengan model induk nya 


# 9. Install Python3 

Untuk melakukan Fine tuning adalah proses training mechine learning dengan memberikan pengetahuan baru yang hasil akan digabungkan dengan model induk nya 

<pre><code>apt install python3.10-venv
python3 -m venv venv
source /venv/bin/activate
pip install datasets transformers peft accelerate
pip install peft</code></pre> 













Mendapatkan file GGUF biasanya melibatkan satu atau beberapa langkah berikut, tergantung pada dari mana model LLM Anda berasal:

    Mengunduh Model yang Sudah Dikonversi ke GGUF: Cara termudah adalah jika model LLM yang ingin Anda gunakan (termasuk versi yang sudah Anda fine-tune) sudah tersedia dalam format GGUF.
        Hugging Face Hub: Platform Hugging Face Hub (https://huggingface.co/) adalah sumber utama untuk model LLM. Seringkali, komunitas membuat dan mengunggah versi model dalam format GGUF. Anda bisa mencari model DeepSeek-R1:1.5B yang sudah di-fine-tune dan melihat apakah ada file dengan ekstensi .gguf di bagian "Files and versions".
        Repositori llama.cpp: Proyek llama.cpp (https://github.com/ggerganov/llama.cpp) adalah alat populer untuk bekerja dengan model LLM di CPU. Repositori ini sering menyediakan skrip dan bahkan file GGUF yang sudah dikonversi untuk berbagai model.

Tips saat mencari file GGUF:

    Cari model dengan nama yang mirip dengan model yang Anda fine-tune, dan perhatikan ekstensi filenya (.gguf).
    Perhatikan informasi tentang kuantisasi (misalnya, Q4_K_M, Q8_0). Kuantisasi yang lebih rendah menghasilkan ukuran file yang lebih kecil tetapi mungkin dengan sedikit penurunan kualitas. Pilih yang sesuai dengan kebutuhan dan sumber daya Anda.
    Baca deskripsi dan komentar untuk memastikan file GGUF tersebut memang berasal dari model yang benar dan berfungsi dengan baik.




root@Dev01:/mnt/d/ai-chat-bot/tuning# python3 --version
Python 3.10.12            


apt install python3.10-venv
python3 -m venv venv
source /venv/bin/activate
pip install datasets transformers peft accelerate
pip install peft 

python3


Clone Repositori llama.cpp (jika belum):
Buka terminal Anda dan jalankan perintah berikut untuk mengunduh kode sumber llama.cpp dari GitHub:
Bash


Compile LLama CPP
------------------
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp-master
mkdir build
cd build
cmake ../
make


python3 convert_hf_to_gguf.py --model-type gpt2 --model-dir /mnt/d/ai-chat-bot/tuning/distilgpt2-bandung.pth --outfile /mnt/d/ai-chat-bot/tuning/distilgpt2-bandung.gguf

python3 convert_hf_to_gguf_update.py --model-type gpt2 --model-dir /mnt/d/ai-chat-bot/tuning/distilgpt2-bandung-generation-final --outfile /mnt/d/ai-chat-bot/tuning/distilgpt2-bandung.gguf


convert pth to huging face 
---------------------------
pip install torch transformers optimum
python3 convert_pth_to_hf.py


convert to hf go gguf 
----------------------
cd /mnt/d/ai-chat-bot/tuning/llama.cpp-master
/mnt/d/ai-chat-bot/tuning/llama.cpp-master# 
python3 convert_hf_to_gguf.py ../distilgpt2-hf --outfile ../distilgpt2-bandung.gguf --model gpt2

import gguf ke ollama 
----------------------
ollama create distilgpt2-bandung -f /mnt/d/ai-chat-bot/tuning/Modelfile


menggunakan pythonserver 
--------------------------
pip install llama-cpp-python
pip install fastapi
pip install uvicorn


uvicorn server-inference-read-hf:app --reload


