import streamlit as st
import os
import time

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain_community.chat_models import ChatOllama

st.set_page_config(page_title="RAGify – “RAG-powered search", page_icon="🧠")
st.title("✨ RAGify–\"RAG-powered search\"")

llm = ChatOllama(model="qwen2.5:0.5b")

prompt = ChatPromptTemplate.from_template(
    """
Answer the question using only the provided context.

<context>
{context}
</context>

Question: {input}

Answer:
"""
)

@st.cache_resource
def load_embeddings():
    return OllamaEmbeddings(model="all-minilm:l6-v2")

embeddings = load_embeddings()

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing uploaded PDF..."):
        pdf_bytes = uploaded_file.read()
        pdf_path = "./temp_uploaded_file.pdf"
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        vector_store = FAISS.from_documents(chunks, embeddings)
        st.success(f"✅ Uploaded and indexed: {uploaded_file.name}")
        st.session_state.vector_store = vector_store

if "vector_store" in st.session_state:
    user_question = st.text_input("Ask a question about your uploaded PDF:")
    if user_question:
        with st.spinner("Retrieving answer..."):
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vector_store.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            start = time.process_time()
            response = retrieval_chain.invoke({"input": user_question})
            end = time.process_time()
            st.subheader("💬 Answer")
            st.write(response['answer'])
            st.caption(f"Response time: {end - start:.2f} seconds")
            with st.expander("🔎 Retrieved Context Chunks"):
                for doc in response["context"]:
                    st.markdown(doc.page_content)
                    st.markdown("---")
else:
    st.info("👆 Please upload a PDF to enable Q&A.")
