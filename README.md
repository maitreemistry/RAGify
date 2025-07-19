# 🧠 RAGify - "RAG-powered search"

**RAGify** is a Streamlit web app that lets you **upload multiple documents in many formats, paste text, or even crawl webpages/sitemaps** and ask questions about their contents.
It uses **retrieval-augmented generation (RAG)** powered by **Ollama**, **FAISS**, and **LangChain** – all running locally on your machine.
No cloud APIs. No vendor lock-in. Just fast, private document understanding.

---

## ✨ Features

* ✉️ Upload **multiple documents at once**
* 🔗 **Paste text** or **crawl webpages / sitemap URLs**
* 📂 Supports **PDF, DOCX, TXT, PPTX, CSV, XLSX, XML, HTML** formats
* ⚙️ Automatically **extracts URLs from documents** and loads their content
* 📊 Uses **LangChain** built-in loaders for all document types
* 🔄 Splits documents into manageable text chunks with overlap
* 🧩 Embeds chunks locally with **Ollama embeddings**
* 🚪 Stores embeddings in a **FAISS** vector store for local retrieval
* 💡 Retrieves relevant context for any question you ask
* 🤖 Generates context-grounded answers with **Ollama** models (like `qwen2.5:0.5b`)
* ⚡ Transparent: shows **retrieved context** used to answer
* 🚫 Delete saved vector stores easily
* 💻 **Runs 100% locally** – your data never leaves your machine

---

## 🚀 Quickstart

### 1) Install dependencies

Make sure you have **Python 3.9+**.

```bash
pip install streamlit langchain langchain-community
```

**Note:** You must also have [Ollama](https://ollama.com/) installed and running.

Pull the models you want to use:

```bash
ollama pull qwen2.5:0.5b
ollama pull all-minilm:l6-v2
```

---

### 2) Run the app

```bash
streamlit run app.py
```

---

### 3) Use RAGify

1. Open your browser at `http://localhost:8501`
2. Upload one or more documents (PDF, DOCX, TXT, PPTX, CSV, XLSX, XML, HTML)
3. Or paste custom text / enter a webpage or sitemap URL
4. Choose whether to auto-parse URLs found inside content
5. Enter a name to save this document set
6. Click "Process Documents"
7. Ask questions about your content!

---

## 🛩️ How it works

1. **Upload** or **paste/crawl** documents
2. RAGify uses **LangChain loaders**:

   * `PyPDFLoader` for PDFs
   * `UnstructuredWordDocumentLoader` for DOCX
   * `UnstructuredPowerPointLoader` for PPTX
   * `TextLoader` for TXT
   * `CSVLoader` for CSV
   * `UnstructuredExcelLoader` for XLSX
   * `UnstructuredXMLLoader` for XML
   * `UnstructuredHTMLLoader` for HTML
   * `WebBaseLoader` & `SitemapLoader` for URLs
3. **Splits** content into \~2000-character chunks with 200-character overlap
4. **Embeds** using **OllamaEmbeddings**
5. **Stores** in **FAISS** vector DB
6. **Retrieves** top matches using k-NN
7. **Answers** using **ChatOllama**, limited to retrieved context

---

## 🛠️ Configurations

Edit in `app.py`:

* Ollama model names (LLM and Embedding)
* Chunk size and overlap
* Number of retrieved chunks (`k`)
* Prompt template (instruction to the model)

---

## 🛡️ Privacy & Local-first

RAGify runs entirely on your machine:

* No remote API calls ✅
* No cloud storage ✅
* Your documents stay private ✅

---

## ✨ Example Prompt

```text
You are a helpful assistant. Use only the provided context to answer the question.

<context>
{context}
</context>

Question: {input}

If the answer is not in the context, respond exactly:
"The question is out of the context. Please ask something from the document uploaded."

Answer:
```

---

## 📂 Project Structure

```
app.py         # Main Streamlit app
```

---

## 🙏 Acknowledgements

* [Mr. Dheeraj Nair & Mr. Jai Desai](https://bosleo.com/)
* [Streamlit](https://streamlit.io/)
* [LangChain](https://www.langchain.com/)
* [Ollama](https://ollama.com/)
* [FAISS](https://github.com/facebookresearch/faiss)

---

## 📸 RAGify

* Screenshot:

  ![image](image.png)

* Architecture Diagram:

![flowdiagram](RAGify.png)

---

️ Enjoy using **RAGify** to supercharge your understanding of **any** set of documents – all **locally and privately**!
