**Company Policy Q&A Bot**

A Streamlit-based Question & Answer application that allows users to upload company policy PDFs and ask questions using semantic search and a local LLM powered by Ollama.

**Features**

Upload and process company policy PDFs
Intelligent text chunking with metadata
Semantic search using FAISS vector database
Reranking using cosine similarity
Local LLM answer generation via Ollama (deepseek-r1:1.5b)
Clean and interactive Streamlit UI
Fully offline – no external API keys required

**How It Works**

PDF Upload
Users upload a policy document in PDF format.
The PDF is loaded and split into overlapping text chunks.
Embedding & Indexing
Each chunk is converted into vector embeddings using SentenceTransformer.
Vectors are stored in a FAISS index for fast similarity search.
Retrieval & Reranking
User questions are embedded and matched against indexed chunks.
Results are filtered by metadata and reranked by similarity.
Answer Generation
Relevant context is passed to a local LLM using Ollama.
The model generates a concise answer strictly from the provided context.

**Tech Stack**

Frontend: Streamlit
Vector Store: FAISS
Embeddings: SentenceTransformers (all-MiniLM-L6-v2)
LLM: Ollama (deepseek-r1:1.5b)
PDF Processing: LangChain + PyPDF

**Installation**

1️⃣ Clone the Repository
git clone https://github.com/your-username/company-policy-qa-bot.git
cd company-policy-qa-bot

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Install & Run Ollama
Download Ollama from: https://ollama.com
Pull the required model: ollama pull deepseek-r1:1.5b
Ensure Ollama is running: ollama serve

▶️ Run the App
streamlit run app.py
Open your browser at: http://localhost:8501
