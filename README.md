# AI Tutor
An AI-powered tutor leveraging Agentic AI capabilities. The AI users to upload a document (text, PDF, etc.) as the knowledge base. After uploading, users can interact with the AI tutor by asking questions in multiple formats (text, image). 

## 🌟 Features

- **Dual Learning Modes**
  - General Learning: Interactive Q&A with a LLM: Llama 3.2
  - Document-based Learning: Context-aware responses from uploaded materials including images.

- **Smart Document Processing**
  - Supports PDF and TXT files
  - RAG (Retrieval Augmented Generation) for accurate responses
  - Multiple document handling

- **Image Analysis**
  - Upload and analyze images alongside documents
  - OCR capability for text extraction
  - Image description generation

- **Interactive UI**
  - Chat-based interface
  - Note-taking capability
  - Student profile management
  - Real-time mode switching

## 🚀 Getting Started

### Prerequisites

```bash
# Install Ollama
brew install ollama

# Start Ollama service
ollama serve

# Pull the required model
ollama pull llama3.2
```

### Installation

```bash
# Clone the repository
git