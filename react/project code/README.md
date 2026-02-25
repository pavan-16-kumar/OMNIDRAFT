# 🧠 OmniDraft

**Industry-Ready Handwriting-to-Text Conversion with RAG Intelligence**

OmniDraft converts any handwritten note format into perfectly typed text using a Multi-Agent verification loop for near-100% accuracy. It features a RAG (Retrieval-Augmented Generation) backend that lets you search and chat with your notes.

---

## ✨ Features

- **Multi-Format Upload:** HEIC, PNG, JPG, and PDF support with drag-and-drop
- **AI-Powered OCR:** Vision LLM transcription with structured Markdown output
- **Verification Agent:** Cross-checks extracted text against the original image for high accuracy
- **RAG Chat:** Chat with your notes using semantic search via ChromaDB
- **Multi-Format Export:** Download as PDF, DOCX, or Markdown
- **Side-by-Side Editor:** Compare original handwriting with transcribed text
- **Modern Dashboard:** Beautiful React UI with real-time processing feedback

## 🏗️ Architecture

```
┌──────────────┐    ┌──────────────────────────────────────┐
│   React UI   │───▶│          FastAPI Backend              │
│  (Vite+TW)   │    │                                      │
└──────────────┘    │  ┌────────┐  ┌──────────┐  ┌──────┐  │
                    │  │ Upload │  │ OCR Agent│  │Export │  │
                    │  │Endpoint│──│ Pipeline │──│Service│  │
                    │  └────────┘  └──────────┘  └──────┘  │
                    │       │           │                   │
                    │  ┌────▼───┐  ┌────▼─────┐            │
                    │  │OpenCV  │  │ Verifier │            │
                    │  │Preproc │  │  Agent   │            │
                    │  └────────┘  └──────────┘            │
                    │                    │                  │
                    │  ┌─────────────────▼──────────────┐  │
                    │  │     ChromaDB (Vector Store)     │  │
                    │  │     LangChain RAG Pipeline      │  │
                    │  └────────────────────────────────┘  │
                    └──────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Google Gemini API Key (or OpenAI API Key)

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
cp .env.example .env       # Add your API keys
uvicorn main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
cp .env.example .env       # Configure API URL
npm run dev
```

## 📁 Project Structure

```
OmniDraft/
├── backend/
│   ├── main.py                  # FastAPI app entry point
│   ├── requirements.txt         # Python dependencies
│   ├── .env.example             # Environment variable template
│   ├── services/
│   │   ├── ocr_agent.py         # Multi-agent OCR pipeline
│   │   ├── verifier_agent.py    # Verification agent for accuracy
│   │   ├── rag_service.py       # RAG pipeline with ChromaDB
│   │   ├── export_service.py    # PDF/DOCX/MD export
│   │   └── image_processor.py   # OpenCV image preprocessing
│   ├── models/
│   │   └── schemas.py           # Pydantic models
│   └── uploads/                 # Temporary upload storage
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main app component
│   │   ├── main.jsx             # Entry point
│   │   ├── index.css            # Global styles
│   │   ├── components/
│   │   │   ├── Dashboard.jsx    # Main dashboard layout
│   │   │   ├── FileUpload.jsx   # Drag-and-drop upload
│   │   │   ├── SideBySideView.jsx # Image vs Text comparison
│   │   │   ├── ChatSidebar.jsx  # RAG chat interface
│   │   │   ├── NotesList.jsx    # Notes library
│   │   │   └── ExportPanel.jsx  # Export options
│   │   └── api/
│   │       └── client.js        # API client
│   └── package.json
└── README.md
```

## 🔑 Environment Variables

### Backend (.env)

```
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_key_optional
LLM_PROVIDER=gemini
CHROMA_PERSIST_DIR=./chroma_db
UPLOAD_DIR=./uploads
MAX_FILE_SIZE_MB=20
```

### Frontend (.env)

```
VITE_API_URL=http://localhost:8000
```

## 📜 License

MIT License — Built for Diploma Project
