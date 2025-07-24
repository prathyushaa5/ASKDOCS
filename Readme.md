A **Gemini 1.5 Flash** powered Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDFs and ask contextual questions using Google's powerful generative models.


## 🧰 Tech Stack

- Python
- Google Generative AI API (`gemini-1.5-flash`)
- FAISS for vector storage
- LangChain for retrieval pipeline
- Sentence Transformers (`all-MiniLM-L6-v2`)
- Streamlit (optional for UI)



## 🔑 Prerequisites

1. **Python 3.9+**  
2. A **Google API Key** for Gemini:  
   - Sign up at: https://aistudio.google.com/app/apikey  
   - Copy your API key


## 🛠️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/prathyushaa5/ASKDOCS.git
cd <repo-name>
```

### 2. Create & Activate Environment
Using conda:
```bash 
conda create -n gemini-rag python=3.9 -y
conda activate gemini-rag
```

### 3. Install Dependencies
 ```bash
 pip install -r requirements.txt
```


### 4. Set Your Gemini API Key
Create a .env file in the project root:

```bash 
GOOGLE_API_KEY=your_api_key_here
```

### 5. Run the App

```bash
streamlit run app.py
```
