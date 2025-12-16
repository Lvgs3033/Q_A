import streamlit as st
import ollama
import faiss
import numpy as np
import time
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# SESSION STATE INITIALIZATION
def init_session_state():
    defaults = {
        "index": None,
        "documents": [],
        "metadatas": [],
        "embedder": None,
        "processing": False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# PDF LOADING
def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    return loader.load()


# TEXT SPLITTING WITH METADATA
def split_documents(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(pages)

    # Attach metadata
    for chunk in chunks:
        chunk.metadata["source"] = "uploaded.pdf"
        chunk.metadata["department"] = "HR"

    return chunks



# VECTOR STORE CREATION
def create_vector_store(docs):
    embedder = SentenceTransformer(
        "all-MiniLM-L6-v2",
        device="cpu"   # avoids meta tensor error
    )

    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]

    embeddings = embedder.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype("float32"))

    return index, texts, metadatas, embedder


# VECTOR SEARCH + METADATA FILTER
def retrieve_candidates(question: str, k: int = 8) -> List[str]:
    query_vec = st.session_state.embedder.encode(
        [question],
        convert_to_numpy=True
    )

    _, indices = st.session_state.index.search(
        query_vec.astype("float32"),
        k
    )

    candidates = []
    for idx in indices[0]:
        meta = st.session_state.metadatas[idx]
        if meta.get("department") == "HR":   # Metadata filter
            candidates.append(st.session_state.documents[idx])

    return candidates


# SIMPLE RERANKER (SIMILARITY-BASED)
def rerank(question: str, chunks: List[str], top_k: int = 3):
    if not chunks:
        return []

    q_emb = st.session_state.embedder.encode([question])
    c_embs = st.session_state.embedder.encode(chunks)

    scores = np.dot(c_embs, q_emb.T).squeeze()
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

    return [chunk for chunk, _ in ranked[:top_k]]



# LLM ANSWER GENERATION
def generate_answer(question: str, context_chunks: List[str]) -> str:
    if not context_chunks:
        return "Answer not found in the document."

    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a company policy assistant.

Rules:
- Use ONLY the provided context
- Be concise and specific
- Do NOT explain unrelated policies
- If the answer is missing, say so clearly

Context:
{context}

Question:
{question}

Answer:
"""

    response = ollama.generate(
        model="deepseek-r1:1.5b",
        prompt=prompt,
        options={
            "temperature": 0.2,
            "max_tokens": 250
        }
    )

    return response["response"].strip()


# TYPING EFFECT
def typing_effect(text: str, delay: float = 0.02):
    placeholder = st.empty()
    output = ""

    for char in text:
        output += char
        placeholder.markdown(f"**Answer:** {output}")
        time.sleep(delay)


# STREAMLIT UI
st.set_page_config(
    page_title="Company Policy Q&A Bot",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Company Policy Q&A Bot")
st.sidebar.header("Upload Policy PDF")

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


# PDF PROCESSING
if uploaded_file:
    with st.spinner("Processing PDF..."):
        file_path = "uploaded.pdf"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        pages = load_pdf(file_path)
        chunks = split_documents(pages)

        index, texts, metadatas, embedder = create_vector_store(chunks)

        st.session_state.index = index
        st.session_state.documents = texts
        st.session_state.metadatas = metadatas
        st.session_state.embedder = embedder

        st.success("✅ PDF indexed successfully!")


# QUESTION INPUT
st.subheader("💬 Ask a Question")

query = st.text_input(
    "Enter your question:",
    disabled=st.session_state.processing
)

if st.button("Get Answer"):
    if not query:
        st.warning("Please enter a question.")
    elif st.session_state.index is None:
        st.warning("Please upload a PDF first.")
    else:
        st.session_state.processing = True

        with st.spinner("🤖 Thinking..."):
            candidates = retrieve_candidates(query)
            reranked = rerank(query, candidates)
            answer = generate_answer(query, reranked)

        typing_effect(answer)
        st.session_state.processing = False


st.sidebar.markdown("Built with ❤️ using Streamlit + FAISS + Ollama")
