import streamlit as st
import time
import tempfile
import logging
import logging.config
import os
import re
from datetime import datetime
from copy import deepcopy
import shutil
import uuid
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader,
    TextLoader, CSVLoader, UnstructuredExcelLoader, WebBaseLoader, SitemapLoader,
    UnstructuredHTMLLoader, UnstructuredXMLLoader
)

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"}
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "default", "level": "DEBUG"},
        "file": {"class": "logging.FileHandler", "filename": "ragify_app.log", "formatter": "default", "level": "DEBUG"}
    },
    "root": {"handlers": ["console", "file"], "level": "DEBUG"}
}
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)
logging.getLogger("urllib3").setLevel(logging.WARNING)


FAISS_BASE_DIR = "vector_store_index"
os.makedirs(FAISS_BASE_DIR, exist_ok=True)

try:
    st.set_page_config(page_title="RAGify – RAG-powered search", page_icon="🧠")
    st.title("✨ RAGify – \"RAG-powered search\"")
    logger.info("App started")

    llm = ChatOllama(model="qwen2.5:0.5b")
    prompt = ChatPromptTemplate.from_template("""
        Answer the question using only the provided context.
        If the question is out of the context, then Answer should be "The question is out of the context. Please ask something from the document uploaded."
        Do not create your own answer, only use the context.
        You are a helpful assistant. Answer ONLY based on the provided context.
        If the question is NOT answerable from the context, respond exactly with:
        "The question is out of the context. Please ask something from the document uploaded"

        <context>
        {context}
        </context>
        Question: {input}
        Answer:
    """)
    logger.debug("Prompt designed")

    @st.cache_resource
    def load_embeddings():
        return OllamaEmbeddings(model="all-minilm:l6-v2")

    embeddings = load_embeddings()

    def list_saved_vector_stores():
        return [d for d in os.listdir(FAISS_BASE_DIR) if os.path.isdir(os.path.join(FAISS_BASE_DIR, d))]

    def merge_all_vector_stores():
        logger.debug("merging vector initialised")
        saved_indexes = list_saved_vector_stores()
        if not saved_indexes:
            return None
        try:
            load_path = os.path.join(FAISS_BASE_DIR, saved_indexes[0])
            merged_vector_store = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
            logger.info(f"Loaded base vector store: {saved_indexes[0]}")
            
            for idx in saved_indexes[1:]:
                load_path = os.path.join(FAISS_BASE_DIR, idx)
                additional_store = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
                merged_vector_store.merge_from(additional_store)
                logger.info(f"Merged vector store: {idx}")
                logger.debug("vector merged!")
            return merged_vector_store
        except Exception as e:
            logger.exception("Failed to merge vector stores")
            st.error(f"❌ Failed to merge saved document sets: {e}")
            return None

    if not st.session_state.get("vector_store"):
        merged_vector_store = merge_all_vector_stores()
        if merged_vector_store:
            st.session_state.vector_store = merged_vector_store
            st.session_state.retriever = merged_vector_store.as_retriever(search_kwargs={"k": 3})
            st.session_state.document_chain = create_stuff_documents_chain(llm, prompt)
            st.session_state.file_processed = True
        else:
            st.session_state.file_processed = False

    st.sidebar.header("📚 Manage Saved Documents")
    saved_indexes = list_saved_vector_stores()
    if saved_indexes:
        st.sidebar.markdown("### 🚮 Delete Saved Documents")
        indexes_to_delete = st.sidebar.multiselect("Select documents to delete:", saved_indexes)
        if st.sidebar.button("❌ Delete Selected"):
            for idx in indexes_to_delete:
                try:
                    shutil.rmtree(os.path.join(FAISS_BASE_DIR, idx))
                    st.sidebar.success(f"Deleted: {idx}")
                    logger.info(f"Deleted vector store: {idx}")
                    logger.debug("merging vector store again")
                    st.session_state.vector_store = None
                    merged_vector_store = merge_all_vector_stores()
                    if merged_vector_store:
                        st.session_state.vector_store = merged_vector_store
                        st.session_state.retriever = merged_vector_store.as_retriever(search_kwargs={"k": 3})
                        st.session_state.document_chain = create_stuff_documents_chain(llm, prompt)
                        st.session_state.file_processed = True
                    else:
                        st.session_state.file_processed = False
                        st.session_state.vector_store = None
                        st.session_state.retriever = None
                        st.session_state.document_chain = None
                except Exception:
                    st.sidebar.error(f"❌ Could not delete: {idx}")
                    logger.exception(f"Failed to delete vector store: {idx}")
    else:
        st.sidebar.info("No saved documents yet.")

    uploaded_files = st.file_uploader(
        "Upload one or more documents",
        type=["pdf", "docx", "txt", "pptx", "csv", "xlsx", "xml", "html"],
        accept_multiple_files=True
    )

    url_or_sitemap_input = st.text_input("Or enter a webpage or sitemap URL to parse:")
    parse_links_in_context = st.selectbox("Automatically parse URLs in documents?", ("No", "Yes"))
    pasted_text = st.text_area("Or just paste some text to use as context:")

    user_inputs = []
    if uploaded_files:
        user_inputs.extend([f.name for f in uploaded_files])
    if url_or_sitemap_input:
        user_inputs.append(url_or_sitemap_input)
    if pasted_text:
        user_inputs.append("pasted_text")
    user_inputs.append(f"parse:{parse_links_in_context}")
    logger.debug("upload ing files: %s", user_inputs)

    if user_inputs and st.session_state.get("last_uploaded") != user_inputs:
        st.session_state.last_uploaded = user_inputs
        st.session_state.file_processed = False

    def extract_text_from_file(file_path, file_type):
        logger.debug(f"Extracting text from file: {file_path}")
        try:
            if file_type == "pdf":
                return PyPDFLoader(file_path).load()
            elif file_type == "txt":
                return TextLoader(file_path, encoding="utf-8").load()
            elif file_type == "docx":
                return UnstructuredWordDocumentLoader(file_path).load()
            elif file_type == "pptx":
                return UnstructuredPowerPointLoader(file_path).load()
            elif file_type == "csv":
                return CSVLoader(file_path).load()
            elif file_type == "xlsx":
                return UnstructuredExcelLoader(file_path).load()
            elif file_type == "xml":
                return UnstructuredXMLLoader(file_path).load()
            elif file_type == "html":
                return UnstructuredHTMLLoader(file_path).load()
            else:
                return []
            logger.debug("loading file: %s", file_path)
        except Exception as e:
            logger.warning(f"Failed to extract from {file_type}: {e}")
            return []

    if (uploaded_files or url_or_sitemap_input or pasted_text) and not st.session_state.get("file_processed"):
        vector_store_name = st.text_input("📁 Enter a name to save this document set:")
        if st.button("📥 Process Documents") and vector_store_name:
            with st.spinner("🔄 Processing documents..."):
                logger.debug("processing documents")
                all_docs = []

                if uploaded_files:
                    logger.debug(f"Uploaded files: {[f.name for f in uploaded_files]}")
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            tmp_path = tmp_file.name
                        ext = uploaded_file.name.split(".")[-1].lower()
                        docs = extract_text_from_file(tmp_path, ext)
                        logger.debug(f"Extracted {len(docs)} docs from {uploaded_file.name}")
                        all_docs.extend(docs)

                if url_or_sitemap_input:
                    logger.debug(f"URL input: {url_or_sitemap_input}")
                    if url_or_sitemap_input.strip().endswith(".xml"):
                        all_docs.extend(SitemapLoader(url_or_sitemap_input.strip()).load())
                    else:
                        all_docs.extend(WebBaseLoader(url_or_sitemap_input.strip()).load())

                if pasted_text.strip():
                    logger.debug(f"Pasted text length: {len(pasted_text)}")
                    all_docs.append(Document(page_content=pasted_text.strip()))

                if parse_links_in_context == "Yes":
                    url_pattern = re.compile(r"https?://[^\s]+")
                    detected_urls = []
                    for doc in all_docs:
                        detected_urls.extend(url_pattern.findall(doc.page_content))
                    detected_urls = list(set(detected_urls))
                    logger.debug(f"Detected URLs: {detected_urls}")
                    if detected_urls:
                        all_docs.extend(WebBaseLoader(detected_urls).load())

                # 

                if all_docs:
                    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                    chunks = splitter.split_documents(all_docs)
                    logger.debug(f"Total chunks created: {len(chunks)}")

                    new_vector_store = FAISS.from_documents(chunks, embeddings)

                    save_path = os.path.join(FAISS_BASE_DIR, vector_store_name)
                    new_vector_store.save_local(save_path)
                    logger.info(f"New vector store saved to: {save_path}")

                    st.session_state.vector_store = None
                    st.session_state.retriever = None
                    st.session_state.document_chain = None

                    merged_vector_store = merge_all_vector_stores()
                    if merged_vector_store:
                        st.session_state.vector_store = merged_vector_store
                        st.session_state.retriever = merged_vector_store.as_retriever(search_kwargs={"k": 3})
                        st.session_state.document_chain = create_stuff_documents_chain(llm, prompt)
                        st.session_state.file_processed = True
                        st.success(f"✅ Document set '{vector_store_name}' saved and reloaded!")
                        logger.info(f"✅ Vector store reloaded successfully with new document set: {vector_store_name}")
                    else:
                        st.session_state.file_processed = False
                        st.error("❌ Failed to reload merged vector store after saving new one.")

                logger.debug("documents loaded")

    if st.session_state.get("file_processed") or list_saved_vector_stores():
        user_question = st.text_input("Ask a question about your content:")
        if user_question:
            with st.spinner("🔍 Retrieving answer..."):
                
                logger.debug("Retrieving answer...")
                retrieval_chain = create_retrieval_chain(
                    st.session_state.retriever, st.session_state.document_chain
                )
                start = time.process_time()
                response = retrieval_chain.invoke({"input": user_question})
                end = time.process_time()

                context_docs = response.get("context", [])

                if not context_docs or all(len(doc.page_content.strip()) < 20 for doc in context_docs):
                    st.warning("😕 Sorry, I couldn’t find anything relevant to your question in the uploaded content.")
                else:
                    st.subheader("💬 Answer")
                    st.write(response['answer'])
                    st.caption(f"Response time: {end - start:.2f} seconds")
                    logger.debug("Answer retrieved successfully")

                    with st.expander("🔎 Retrieved Context Chunks"):
                        for doc in context_docs:
                            st.markdown(doc.page_content)
                            st.markdown("---")
    else:
        st.info("👇 Please upload documents, paste text, or enter URLs to begin. No saved documents found.")

except Exception as e:
    logger.exception("Unexpected error occurred")
    st.error(f"An unexpected error occurred: {e}")
