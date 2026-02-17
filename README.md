# 📚 Wikipedia RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from any Wikipedia article using **LangChain**, **FAISS**, **HuggingFace Embeddings**, and **Groq LLM** — with a clean **Gradio** chat interface.

---

## 🧠 How It Works

```
Wikipedia Article
      ↓
   Chunking  (RecursiveCharacterTextSplitter)
      ↓
  Embeddings  (sentence-transformers/all-MiniLM-L6-v2)
      ↓
 FAISS VectorStore  (saved to disk)
      ↓
  User Question
      ↓
   Router  → too vague? ask to clarify
      ↓
  Retrieval  → top 3 relevant chunks
      ↓
  Groq LLM  (llama-3.1-8b-instant)
      ↓
  JSON Response  (answer + quotes + confidence)
      ↓
  Gradio UI
```

---

## 🚀 Quick Start (Google Colab)

1. Open [Google Colab](https://colab.research.google.com)
2. Open `wikipedia_rag_chatbot.py`
3. Copy each `CELL` block into a **separate Colab cell** in order
4. Add your **Groq API key** in Cell 2
5. Run all cells top to bottom
6. Enter any Wikipedia topic when prompted (e.g. `Diabetes`, `Black hole`)
7. Chat with the bot in the Gradio UI

> ⚠️ Each `%%writefile` cell must be in its **own Colab cell** — do not merge them.

---

## 🔑 Get a Free Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for free
3. Create an API key
4. Paste it in Cell 2

---

## 🗂️ Project Structure

```
wikipedia-rag-agent/
├── src/
│   ├── load_wikipedia.py   # Fetches article from Wikipedia API
│   ├── chunking.py         # Splits text into chunks
│   ├── embeddings.py       # Creates and saves FAISS vectorstore
│   ├── retrieval.py        # Retrieves relevant chunks
│   ├── prompts.py          # Builds the LLM prompt
│   ├── router.py           # Routes vague vs clear questions
│   └── rag_pipeline.py     # Orchestrates the full pipeline
├── data/
│   └── topic.txt           # Saved Wikipedia article
└── vectorstore/            # Saved FAISS index (persists across restarts)
```

---

## ✨ Features

- **Any Wikipedia topic** — not hardcoded, user picks at runtime
- **Smart router** — detects vague questions and asks for clarification
- **Vectorstore saved to disk** — no rebuilding on Colab restart
- **Structured JSON responses** — answer, confidence score, supporting quotes
- **Source chunks displayed** — full transparency on what the LLM used
- **Gradio chat UI** — clean interface with example questions

---

## 🛠️ Tech Stack

| Component | Library |
|-----------|---------|
| LLM | Groq (llama-3.1-8b-instant) |
| Embeddings | HuggingFace sentence-transformers |
| Vector DB | FAISS |
| Orchestration | LangChain |
| UI | Gradio |
| Data Source | Wikipedia API |

---

## 📦 Dependencies

```
langchain
langchain-community
langchain-huggingface
langchain-text-splitters
sentence-transformers
faiss-cpu
groq
requests
gradio
```

---

## 📄 License

MIT License — free to use and modify.
