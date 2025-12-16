import streamlit as st
import ollama
import faiss
import numpy as np
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# -----------------------------
# SESSION STATE INITIALIZATION
# -----------------------------
if "index" not in st.session_state:
    st.session_state.index = None

if "documents" not in st.session_state:
    st.session_state.documents = None

if "embedder" not in st.session_state:
    st.session_state.embedder = None

if "processing" not in st.session_state:
    st.session_state.processing = False


# -----------------------------
# PDF LOADING
# -----------------------------
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()


# -----------------------------
# TEXT SPLITTING
# -----------------------------
def split_docs(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_documents(pages)


# -----------------------------
# VECTOR STORE CREATION
# -----------------------------
def create_vector_store(docs):
    embedder = SentenceTransformer(
        "all-MiniLM-L6-v2",
        device="cpu"  # 🔒 avoids meta tensor error
    )

    texts = [doc.page_content for doc in docs]
    embeddings = embedder.encode(texts, show_progress_bar=False)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))

    return index, texts, embedder


# -----------------------------
# RETRIEVAL
# -----------------------------
def retrieve_context(question, k=3):
    query_vec = st.session_state.embedder.encode([question])
    _, indices = st.session_state.index.search(
        np.array(query_vec).astype("float32"), k
    )

    return [st.session_state.documents[i] for i in indices[0]]


# -----------------------------
# OLLAMA ANSWER GENERATION
# -----------------------------
def generate_answer(question, context_chunks):
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a precise assistant.

Answer the question ONLY using the context below.
If the answer is not present, say: "Answer not found in the document."

Context:
{context}

Question:
{question}

Answer (short and direct):
"""

    response = ollama.generate(
        model="deepseek-r1:1.5b",
        prompt=prompt,
        options={
            "temperature": 0.2,
            "max_tokens": 300
        }
    )

    return response["response"].strip()


# -----------------------------
# TYPING EFFECT
# -----------------------------
def typing_effect(text, delay=0.02):
    placeholder = st.empty()
    output = ""

    for char in text:
        output += char
        placeholder.markdown(f"**Answer:** {output}")
        time.sleep(delay)


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="PDF Q&A Bot", page_icon="📄")

st.title("📄 Company Policy Q&A Bot")
st.sidebar.header("Upload PDF")

uploaded_file = st.sidebar.file_uploader(
    "Upload your policy PDF",
    type=["pdf"]
)

# -----------------------------
# PDF PROCESSING
# -----------------------------
if uploaded_file:
    with st.spinner("Processing PDF..."):
        file_path = "uploaded.pdf"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        pages = load_pdf(file_path)
        chunks = split_docs(pages)

        index, texts, embedder = create_vector_store(chunks)

        st.session_state.index = index
        st.session_state.documents = texts
        st.session_state.embedder = embedder

        st.success("✅ PDF processed successfully!")


# -----------------------------
# QUESTION INPUT
# -----------------------------
st.subheader("💬 Ask a Question")

query = st.text_input(
    "Enter your question:",
    disabled=bool(st.session_state.processing)
)

if st.button("Get Answer"):
    if not query:
        st.warning("Please enter a question.")
    elif st.session_state.index is None:
        st.warning("Please upload a PDF first.")
    else:
        st.session_state.processing = True

        with st.spinner("🤖 Thinking..."):
            context = retrieve_context(query)
            answer = generate_answer(query, context)

        typing_effect(answer)
        st.session_state.processing = False


st.sidebar.markdown("Built with ❤️ using Streamlit + Ollama")
