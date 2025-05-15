# Membuat Chactbot AI

Chatbot AI ini  program komputer yang dirancang untuk mensimulasikan percakapan dengan manusia melalui teks atau suara. dengan memanfaatkan teknologi kecerdasan buatan (AI), terutama pemrosesan bahasa alami (NLP) dan machine learning (ML) d, untuk memahami pertanyaan pengguna, memberikan jawaban yang releva dengan backendnya menggunakan  AI-ML-LLM (seperti chatgpt, deepseek, gemini) 

#  Demo Video 

1. Video Demo Chatbox dengan satu model LLM menggunakan DeepSeek-R1:1.5B
[![Video Demo Chatbot](URL_GAMBAR_PRATINJAU)](./screenshoot/video-recording-chat-bot.mp4)

2. Video Demo Chatbox dengan tiga model LLM menggunakan DeepSeek-R1:1.5B, distilgpt2, distilgpt2-bandung (model hasil finetuning)
![Video Demo Chatbot Multi Model LLM](./screenshoot/video-recording-chat-bot-multiple-model.mp4)


# Daftar Isi

1.  [Machine Learning - LLM](#1-machine-learning---llm)
2.  [Arsitekur AI Chat](#2-arsitektur-ai-chat)
3.  [Install Inferance Ollama Platform](#3-install-inferance-ollama-platform)
4.  [Import model LLM](#4-import-model-llm)
5.  [Run Deepseek Model LLM](#5-run-deepseek-model-llm)
6.  [Test Prompt ke Ollama via API](#6-test-prompt-ke-ollama-via-api)
7.  [Siapkan Dataset](#7-siapkan-dataset)
8.  [Fine Tuning](#8-fine-tuning)
9.  [Install Python3](#9-install-python3)
10. [Download Model LLM](#10-download-model-llm)
11. [Fine Tuning Training](#11-fine-tuning-training)
12. [Run Model Hasil Fine Tuning di Inferance Server Python](#12-run-model-hasil-fine-tuning-di-inferance-server-python)
13. [Run Model format GGUF di Inferance Server Ollama](#13-run-model-format-gguf-di-inferance-server-ollama)
14. [Demo Video](#-demo-video)
    1. [Video Demo Chatbox dengan satu model LLM menggunakan DeepSeek-R1:1.5B](#1-video-demo-chatbox-dengan-satu-model-llm-menggunakan-deepseek-r115b)
    2. [Video Demo Chatbox dengan tiga model LLM menggunakan DeepSeek-R1:1.5B, distilgpt2, distilgpt2-bandung (model hasil finetuning)](#2-video-demo-chatbox-dengan-tiga-model-llm-menggunakan-deepseek-r115b-distilgpt2-distilgpt2-bandung-model-hasil-finetuning)





# 1. Machine Learning - LLM 

Machine Learning (ML) adalah cabang dari kecerdasan buatan (AI) yang mempelajari pola dan membuat prediksi berdasarkan data. 

LLM, atau Large Language Model, adalah jenis model ML yang dilatih menggunakan sejumlah besar teks untuk memahami dan menghasilkan bahasa alami. 

Contohnya adalah GPT-3 dan Llama2, Gemini, Copilot, Meta AI,  yang digunakan untuk tugas seperti penerjemahan, penulisan, dan percakapan. 
LLM bekerja dengan cara mempelajari korelasi dan struktur bahasa dari data pelatihan, sehingga dapat menghasilkan teks yang koheren dan relevan dengan konteks. 

LLM, atau Large Language Model, adalah jenis model ML yang dilatih menggunakan sejumlah besar teks untuk memahami dan menghasilkan bahasa alami.

Contoh untuk deploy LLM di server private adalah dapat melihat contoh seperti ini 

https://github.com/dendie-sanjaya/ai-ml-llm-ollama

Pada source code ini akan dijelaskan cara membuat chatbot dengan backen AI-ML-LLM 

# 2. Arsitekur AI Chat 

Berikut ini adalah arsitekur AI Chatbot

![ss](./design/architecture.png)


# 3. Install Inferance Ollama Platform 

Sebagai server AI-MI-LLM dapat menggunakan ollama sebagai inferance AI-MML, ollama akan berperan sebaga inferance dan menyediakan openAPI yang bisa 
di akses frontend 

cara installasi bila melihat disini -> https://github.com/dendie-sanjaya/ai-ml-llm-ollama

Apabila instalasi ollama berhasil dan dapat di run, makaakan tampak sperti ini 

<pre><code>ollama start</pre></code> 

![ss](./screenshoot/1.png)


# 4. Import model LLM

Ollama dapa melakuan import model LLM yang mengambil langsung dari repository model ollama , dalam contoh kali ini model yg digunakan menggunakan deepseek-r1

![ss](./screenshoot/2.png)

![ss](./screenshoot/3.png)

![ss](./screenshoot/4.png)


# 5. Run Deepseek Model LLM

Untuk mengaktifkan Model Deepseek, dapat menggunakan command seperti ini 

<pre><code>ollama run deepseek-r1:1.5b</code></pre>

![ss](./screenshoot/5.png)

# 6. Test Prompt ke Ollama via API

Untuk melakukan test untuk memberikan Prompt AI ke Deepseek dapat lakukan via postman, apabila API nya running hasilnya akan seperti ini 

![ss](./screenshoot/6.png)

![ss](./screenshoot/7.png)

# 7. Siapkan Dataset 

Dataset adalah data yang sudah bersih atau benar, semakin banyak datanya dengan kualitas yang bagus, maka akan semakin baik untuk training machine learning yang akan menghasil jawaban yg baik, berikut ini adalah contoh dalam membuat dataset di format csv untuk keperluan tujuan membuat chatbot

![ss](./screenshoot/8.png)


# 8. Fine Tuning 

Fine tuning adalah proses training mechine learning dengan memberikan pengetahuan baru yang hasil nya akan digabungkan dengan model llm induk nya 


# 9. Install Python3 

Untuk melakukan Fine tuning adalah proses training mechine learning dengan memberikan pengetahuan baru yang hasil nya akan digabungkan dengan model induk nya 

<pre><code>apt install python3.10-venv
python3 -m venv venv
source /venv/bin/activate
pip install datasets transformers peft accelerate
pip install peft</code></pre> 

![ss](./screenshoot/11.png)

<pre><code>root@Dev01:/mnt/d/ai-chat-bot/tuning# python3 --version
Python 3.10.12</code></pre>            


# 10. Download Model LLM  

Cari model dengan nama yang mirip dengan model LLM yg akan menjadi induk model fine-tune, dan perhatikan ekstensi filenya (.gguf),  misalkan Hugging Face Hub  Platform Hugging Face Hub (https://huggingface.co/) adalah sumber utama untuk model LLM. Seringkali, komunitas membuat dan mengunggah versi model dalam format GGUF, sebagai contoh  model DeepSeek-R1:1.5B, distilgpt2  

![ss](./screenshoot/9.png)

![ss](./screenshoot/10.png)


# 11. Fine Tuning Training 

Lakukan finetuning dengan menggabungkan model LLM induk sebagai contoh di kode program ini menggunakan distilgpt2.gguf dan menggunakan dataset file csv, seperti di contoh ini menggunakan dataset-bandung
![Dataset Kota Bandung](./dataset/dataset-bandung.csv) 

File script fine-tuning nya dapat diakses di ![distilgpt2-finetuning.py](./fine-tuning/distilgpt2-finetuning.py)

<pre><code>python3 distilgpt2-finetuning.py</code></pre> 

![ss](./screenshoot/14.png)

Hasil dari proses finetuning ini akan menghasilkan sebuah model baru dengan format safetensors (format model LLM dari Hugging Face)


# 12. Run Model Hasil Fine Tuning di Inferance Server Python

Server inference adalah adalah sebuah server yg menjadi penghubung antar frontend untuk memberikan prompt atau pertanyaan dan menampikan jawaban, server inference biasanya sudah dilengkapi 
dengan JSON API untuk sebagai cara untuk bertukar data atau informasi

Pada contoh ini yang digunakan adalah membuat server inference menggunakan python, install lebih dulu depedensinya  

<pre><code>pip install llama-cpp-python
pip install fastapi
pip install uvicorn</code></pre> 

Server Inference Python dapat membaca model format huge face
<pre><code>uvicorn server-inference-read-hf:app --reload</code></pre> 

Server Inference Python dapat membaca model fromat gguf
<pre><code>uvicorn server-inference-gguf:app --reload</code></pre> 

Apabila API server inference di akses via Postmant akan tampak seperti dibawah ini 

![ss](./screenshoot/12. python-server.png)


# 13. Run Model format GGUF di Inferance Server Ollama

Server inference Ollama dapat di import banyak model LLM, dan dari sisi frontend dapat memilih  untuk untuk menggunakan LLM yang mana, berikut ini adalah contoh import model.gguf 

<pre><code>ollama create distilgpt2-bandung -f /mnt/d/ai-chat-bot/tuning/Modelfile</code></pre> 

![ss](./screenshoot/18-import-to-ollama.png)

Berikut ini adalah daftar model 

![ss](./screenshoot/19-import-to-ollama-2.png)

