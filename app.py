import streamlit as st
import time
import tempfile
import pandas as pd
import logging
import logging.config

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain_community.chat_models import ChatOllama

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader
)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "DEBUG"
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": "ragify_app.log",
            "formatter": "default",
            "level": "DEBUG"
        }
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "DEBUG"
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)
logging.getLogger("urllib3").setLevel(logging.WARNING)


try:
    st.set_page_config(
        page_title="RAGify – RAG-powered search",
        page_icon="🧠"
    )
    st.title("✨ RAGify – \"RAG-powered search\"")
    logger.info("RAGify app started")

    llm = ChatOllama(model="qwen2.5:0.5b")
    logger.debug("LLM initialized")

    prompt = ChatPromptTemplate.from_template(
        """
Answer the question using only the provided context.
If the question is out of the context, then Answer should be "The question is out of the context. Please ask something from the document uploaded."
Do not create your own answer, only use the context.

<context>
{context}
</context>

Question: {input}

Answer:
"""
    )

    @st.cache_resource
    def load_embeddings():
        logger.info("Loading embeddings")
        return OllamaEmbeddings(model="all-minilm:l6-v2")

    embeddings = load_embeddings()

    # ✅ Multi-file uploader
    uploaded_files = st.file_uploader(
        "Upload one or more documents",
        type=["pdf", "docx", "txt", "pptx", "csv", "xlsx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        logger.info(f"User uploaded files: {[f.name for f in uploaded_files]}")

    if uploaded_files and st.session_state.get("last_uploaded") != [f.name for f in uploaded_files]:
        logger.debug("Clearing session state for new file upload")
        st.session_state.clear()
        st.session_state.last_uploaded = [f.name for f in uploaded_files]

    def extract_text_from_file(file_path, file_type):
        logger.info(f"Extracting text from {file_path} (type: {file_type})")
        try:
            if file_type == "pdf":
                loader = PyPDFLoader(file_path)
                logger.debug("Using PyPDFLoader")
                return loader.load()

            elif file_type == "txt":
                loader = TextLoader(file_path, encoding="utf-8")
                logger.debug("Using TextLoader")
                return loader.load()

            elif file_type == "docx":
                loader = UnstructuredWordDocumentLoader(file_path)
                logger.debug("Using UnstructuredWordDocumentLoader")
                return loader.load()

            elif file_type == "pptx":
                loader = UnstructuredPowerPointLoader(file_path)
                logger.debug("Using UnstructuredPowerPointLoader")
                return loader.load()

            elif file_type == "csv":
                loader = CSVLoader(file_path)
                logger.debug("Using CSVLoader")
                return loader.load()

            elif file_type == "xlsx":
                loader = UnstructuredExcelLoader(file_path)
                logger.debug("Using UnstructuredExcelLoader")
                return loader.load()

            else:
                logger.error(f"Unsupported file type: {file_type}")
                st.error("❌ Unsupported file type.")
                return []

        except Exception as e:
            logger.exception("Failed to extract text from file")
            st.error(f"❌ Failed to extract text from file. See logs for details.")
            raise

    if uploaded_files and "file_processed" not in st.session_state:
        with st.spinner("Processing uploaded documents..."):
            all_docs = []

            for uploaded_file in uploaded_files:
                try:
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name

                    file_ext = uploaded_file.name.split(".")[-1].lower()
                    docs = extract_text_from_file(tmp_path, file_ext)
                    all_docs.extend(docs)

                    logger.info(f"Processed and extracted: {uploaded_file.name}")

                except Exception as e:
                    logger.exception(f"Failed to process {uploaded_file.name}")
                    st.error(f"❌ Failed to process {uploaded_file.name}. See logs for details.")

            if all_docs:
                splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                chunks = splitter.split_documents(all_docs)
                logger.info(f"All documents split into {len(chunks)} chunks")

                vector_store = FAISS.from_documents(chunks, embeddings)
                logger.info("FAISS vector store created successfully")

                st.session_state.vector_store = vector_store
                st.session_state.retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                st.session_state.document_chain = create_stuff_documents_chain(llm, prompt)
                st.session_state.file_processed = True

                st.success(f"✅ Uploaded and indexed: {', '.join([f.name for f in uploaded_files])}")
                logger.info(f"All files processed and indexed")

    if "vector_store" in st.session_state and "document_chain" in st.session_state and "retriever" in st.session_state:
        user_question = st.text_input("Ask a question about your uploaded documents:")

        if user_question:
            logger.info(f"User question: {user_question}")
            with st.spinner("Retrieving answer..."):
                try:
                    retrieval_chain = create_retrieval_chain(
                        st.session_state.retriever,
                        st.session_state.document_chain
                    )

                    start = time.process_time()
                    response = retrieval_chain.invoke({"input": user_question})
                    end = time.process_time()

                    st.subheader("💬 Answer")
                    st.write(response['answer'])
                    st.caption(f"Response time: {end - start:.2f} seconds")

                    logger.debug(f"Response: {response['answer']}")

                    with st.expander("🔎 Retrieved Context Chunks"):
                        for doc in response["context"]:
                            st.markdown(doc.page_content)
                            st.markdown("---")
                            logger.debug(f"Context chunk: {doc.page_content[:100]}...")

                except Exception as e:
                    logger.exception("Failed to retrieve answer")
                    st.error(f"❌ Failed to retrieve answer. See logs for details.")
    else:
        st.info("👆 Please upload one or more documents to enable Q&A.")

except Exception as e:
    logger.exception("Unexpected error in main app")
    st.error("❌ An unexpected error occurred. Please try again.")
