🤖 RAG-Based PDF Chatbot
A modern Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and interact with them using natural language.
The system retrieves relevant document context using FAISS vector search and generates grounded answers using Google Gemini LLM.

✨ Features
📄 PDF upload support (single document)
✂️ Intelligent text chunking with overlap
🧠 Semantic search using HuggingFace embeddings
🔍 FAISS-powered vector retrieval
💬 Context-aware chatbot (RAG)
🕘 Persistent chat history (session-based)
⚡ Fast inference with Gemini Flash Lite
🔐 Secure API key handling via .env


🧰 Tech Stack
Frontend: Streamlit
LLM: Google Gemini (gemini-2.5-flash-lite)
Embeddings: HuggingFace all-MiniLM-L6-v2
Vector DB: FAISS
Frameworks: LangChain, PyPDF
Language: Python

webapp_link:
https://rag-demo-app2.streamlit.app/

