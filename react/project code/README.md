# OmniDraft

**Industry-Ready Handwriting-to-Text Conversion with Multi-Agent Verification & RAG Intelligence**

OmniDraft is a powerful, full-stack application designed to transcribe, structure, and query handwritten documents. It transforms messy, multilingual handwritten notes or PDFs into clean, structured Markdown, allowing users to not only digitize their physical documents but also interact with them through a Retrieval-Augmented Generation (RAG) chat interface.

This repository represents a complete, robust implementation originally developed for a Diploma Project, leveraging modern AI agents and local machine learning models for high accuracy.

---

## 🚀 The Vision & Problem Solved

**The Problem:**
Extracting text from handwritten notes—especially those containing mixed languages (like English and Telugu), complex layouts, diagrams, or degraded quality—is notoriously difficult. Standard Optical Character Recognition (OCR) engines often fail to preserve context, misinterpret complex Indian scripts, and output unformatted walls of text.

**The Solution:**
OmniDraft utilizes an **Agentic AI Workflow** that combines the strengths of local traditional OCR tools and modern Vision Large Language Models (LLMs). By orchestrating multiple AI agents, OmniDraft doesn't just "read" the text; it deduces context, verifies accuracy, and formats the output into pristine, modern Markdown.

Furthermore, digitized notes are often hard to search. OmniDraft solves this by embedding transcribed documents into a local Vector Database (ChromaDB), allowing users to "chat" with their notes to instantly recall information or generate summaries.

---

## ✨ Key Features & Capabilities

- **Intelligent, Multi-Agent OCR Pipeline:**
  - **Agent A (Transcription):** Uses Vision LLMs (Google Gemini 2.0 Flash, OpenAI GPT-4o, or OpenRouter) or Local EasyOCR to extract and structure text from images into Markdown.
  - **Agent B (Verification):** Cross-references the initial transcription against the original image using a secondary pass to flag low-confidence words and fix hallucinations, ensuring near-100% accuracy.
- **Multilingual Support (India-Focused):** Specifically engineered to retain the script and formatting of complex languages like Telugu, Hindi, Tamil, Kannada, and Bengali without unwanted translations.
- **RAG-Powered Chat Interface:** Talk to your handwritten notes. OmniDraft chunks uploaded documents and stores them in ChromaDB. When you ask a question, the system retrieves only the relevant chunks and uses an LLM to formulate precise answers based _only_ on your notes.
- **Multi-Format Processing:** Drag-and-drop support for PNG, JPG, JPEG, HEIC, and multi-page PDFs.
- **Dynamic Export Options:** Download the verified transcriptions as neatly formatted PDFs (using `fpdf2` with custom fonts for Indian scripts), DOCX, Markdown, or raw Text.
- **Text-to-Speech (TTS):** Includes a free, unlimited text-to-speech engine powered by Microsoft Neural Voices (`edge-tts`) to read transcribed notes aloud in various regional accents.
- **Side-by-Side Review Editor:** A custom React component allowing users to view their original handwritten image alongside the transcribed text to make manual corrections easily.

---

## 🏗️ Architecture & Implementation Deep Dive

OmniDraft is built on a decoupled Client-Server architecture, ensuring scalability and a snappy user experience.

### 1. The Backend (Python / FastAPI)

The backend acts as the orchestrator for all AI processing, storage, and retrieval.

- **Framework:** Built entirely on **FastAPI**, fully utilizing Python `async`/`await` for non-blocking processing (crucial when handling heavy PDF extractions and API calls).
- **Image Processing:** `OpenCV` and `Pillow` handle image normalization, format conversion (e.g., HEIC to JPEG), and breaking multi-page PDFs into individual frames for processing.
- **The AI Agents:**
  - Implemented in `services/ocr_agent.py` and `services/verifier_agent.py`.
  - Supports dynamic switching of the LLM provider via `.env` (Gemini, OpenAI, OpenRouter).
  - The Verification prompt is carefully tuned with strict _Anti-Hallucination_ rules.
- **Vector Database & RAG Pipeline:**
  - Uses **ChromaDB** as an embedded, purely local vector store (`services/rag_service.py`).
  - Documents are chunked using LangChain's `RecursiveCharacterTextSplitter`.
  - Embeddings are generated completely offline using Chroma's default `all-MiniLM-L6-v2` model, meaning no API costs are incurred for embedding.
- **Data Persistence:** A lightweight JSON flat-file database (`data/notes_db.json`) tracks upload metadata, transcriptions, and confidence scores, making the system highly portable.

### 2. The Frontend (React / Vite.js)

The user interface is a modern Single Page Application focusing on immediate visual feedback and ease of use.

- **Framework:** Built with **React 19** and bootstrapped via **Vite** for incredibly fast hot-module reloading and optimized production builds.
- **Styling:** Uses **Tailwind CSS v4** for utility-first, responsive design, combined with Lucide React for modern iconography.
- **State & Networking:** Standard React Hooks manage component state, while `Axios` drives communication with the FastAPI backend, utilizing robust timeout configurations to handle long-running document processing tasks.
- **Key Components:**
  - `FileUpload.jsx`: Implements `react-dropzone` for an intuitive drag-and-drop area.
  - `SideBySideView.jsx`: A dual-pane editor that syncs the uploaded image with the editable `react-markdown` preview.
  - `ChatSidebar.jsx`: The RAG chat interface integrated gracefully into a sliding side-panel.

---

## 🛠️ Technology Stack Summary

| Domain          | Technology / Library                     | Role                                    |
| :-------------- | :--------------------------------------- | :-------------------------------------- |
| **Frontend**    | React, Vite, Tailwind CSS, Axios         | UI, State Management, API Communication |
| **Backend API** | FastAPI, Uvicorn, Pydantic               | Async HTTP Server, Data Validation      |
| **AI / OCR**    | LangChain, EasyOCR, Gemini / OpenAI SDKs | Agentic Workflows, Text Extraction      |
| **Vector DB**   | ChromaDB (Local), sentence-transformers  | Embedding and Semantic Search for RAG   |
| **Image/PDF**   | OpenCV, Pillow, PyMuPDF, `pillow-heif`   | Image Preprocessing, PDF Splitting      |
| **Export/TTS**  | FPDF2, `python-docx`, `edge-tts`         | Document Generation, Audio Generation   |

---

## 💻 Running the Project Locally

### Prerequisites

- Node.js (v18+)
- Python (3.10+)
- An API Key from Google Gemini Studio (Free) or OpenAI.

### 1. Backend Setup

Navigate to the `backend` directory, set up your Python environment, and start the server:

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# Configure your environment variables
cp .env.example .env
# Add your GEMINI_API_KEY to the .env file

uvicorn main:app --reload --port 8000
```

### 2. Frontend Setup

In a new terminal, navigate to the `frontend` directory, install packages, and start the Vite dev server:

```bash
cd frontend
npm install

# Configure your API URL
cp .env.example .env

npm run dev
```

The app will now be available at `http://localhost:5173`.

---

## 📝 License & Acknowledgements

Created as a comprehensive demonstration of applied Agentic AI and modern web development architectures.

Built under the MIT License.
