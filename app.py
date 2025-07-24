import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from gemini_llm import ask_gemini

load_dotenv()
VECTOR_DB_PATH = "faiss_index"
PDF_FOLDER = "pdfs"

st.set_page_config(page_title="RAG Chatbot with Gemini", layout="wide")
st.title("📚 Gemini-powered RAG PDF Chatbot")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for document upload
with st.sidebar:
    st.header("📂 Upload PDFs")
    uploaded_files = st.file_uploader("Choose PDFs", type="pdf", accept_multiple_files=True)

    if uploaded_files and st.button("Process Documents"):
        if os.path.exists(VECTOR_DB_PATH):
            shutil.rmtree(VECTOR_DB_PATH)
        os.makedirs(PDF_FOLDER, exist_ok=True)

        docs = []
        for file in uploaded_files:
            path = os.path.join(PDF_FOLDER, file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())

            loader = PyPDFLoader(path)
            docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(VECTOR_DB_PATH)
        st.session_state.vectorstore = vectorstore
        st.success("✅ Documents processed successfully!")

# Chat interface
if st.session_state.vectorstore:
    st.subheader("💬 Chat with your documents")

    # Display previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(prompt)
        context = "\n\n".join([doc.page_content for doc in docs])

        full_prompt = f"""
Use the context below to answer the user's question.

Context:
{context}

Question:
{prompt}

Only answer based on the context provided above.
If the answer cannot be found in the context, respond with:
"I'm sorry, but I don't have enough information from the documents to provide an accurate answer."

Answer:
"""
        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                response = ask_gemini(full_prompt)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("📂 Please upload and process at least one PDF to begin.")
