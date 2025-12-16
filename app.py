import streamlit as st
import faiss
import numpy as np
import time
from typing import List

from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


def init_session_state():
    defaults = {
        "index": None,
        "documents": [],
        "metadatas": [],
        "embedder": None,
        "processing": False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()


def load_pdf(file_path: str):
    return PyPDFLoader(file_path).load()


def split_documents(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(pages)
    for c in chunks:
        c.metadata["department"] = "HR"
    return chunks


def create_vector_store(docs):
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]

    embeddings = embedder.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype("float32"))

    return index, texts, metadatas, embedder


def retrieve_candidates(question: str, k: int = 8) -> List[str]:
    query_vec = st.session_state.embedder.encode([question])
    _, indices = st.session_state.index.search(
        query_vec.astype("float32"), k
    )

    return [
        st.session_state.documents[i]
        for i in indices[0]
        if st.session_state.metadatas[i].get("department") == "HR"
    ]


def rerank(question: str, chunks: List[str], top_k: int = 3):
    if not chunks:
        return []

    q_emb = st.session_state.embedder.encode([question])
    c_embs = st.session_state.embedder.encode(chunks)
    scores = np.dot(c_embs, q_emb.T).squeeze()

    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]


def generate_answer(question: str, context_chunks: List[str]) -> str:
    if not context_chunks:
        return "Answer not found in the document."

    if "GROQ_API_KEY" not in st.secrets:
        return "❌ GROQ_API_KEY missing. Add it in Streamlit → Settings → Secrets."

    context = "\n\n".join(context_chunks)[:3000]

    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a company policy assistant. "
                        "Answer strictly using the provided context. "
                        "If the answer is not in the context, say so."
                    )
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion:\n{question}"
                }
            ],
            temperature=0.2,
            max_tokens=250
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"⚠️ Groq API error: {str(e)}"


def typing_effect(text: str):
    box = st.empty()
    out = ""
    for ch in text:
        out += ch
        box.markdown(f"**Answer:** {out}")
        time.sleep(0.01)


st.set_page_config(
    page_title="Company Policy Q&A Bot",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Company Policy Q&A Bot")
st.sidebar.header("Upload Policy PDF")

uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        with open("uploaded.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        pages = load_pdf("uploaded.pdf")
        chunks = split_documents(pages)
        index, texts, metadatas, embedder = create_vector_store(chunks)

        st.session_state.index = index
        st.session_state.documents = texts
        st.session_state.metadatas = metadatas
        st.session_state.embedder = embedder

        st.success("PDF indexed successfully.")


st.subheader("💬 Ask a Question")

query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if not query:
        st.warning("Please enter a question.")
    elif st.session_state.index is None:
        st.warning("Upload a PDF first.")
    else:
        with st.spinner("Thinking..."):
            candidates = retrieve_candidates(query)
            reranked = rerank(query, candidates)
            answer = generate_answer(query, reranked)

        typing_effect(answer)


st.sidebar.markdown("Built with Streamlit + FAISS + Groq")
