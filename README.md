**An intelligent chatbot with OCR integration, persistent storage, and multimodal capabilities**

## 🚀 Overview

This project is an AI-powered conversational chatbot built using Streamlit as the front-end interface and integrated with LLAMA language model for intelligent,
context-aware responses. It includes OCR functionality for extracting text from images/documents and SQLite for persistent data storage.

## 🧠 Key Features

### 🤖 AI Capabilities
- **Conversational AI** using LLAMA for natural, context-aware responses
- **Streaming Responses** - Real-time AI response display
- **System Prompts** - Customizable AI behavior and expertise

### 📄 Document Processing
- **OCR Integration** - Extract text from images (JPG, PNG, BMP, TIFF)
- **PDF Processing** - Text extraction from PDF documents
- **Document Support** - DOCX, TXT, CSV file processing
- **Image Preprocessing** - Enhanced OCR with image optimization

### 💾 Data Management
- **SQLite Database** - Persistent chat history and user sessions
- **Chat Management** - Create, delete, and search conversations
- **File Storage** - Organized upload management with metadata
- **Session Persistence** - Maintain context across app restarts

### 🎨 User Experience
- **Streamlit UI** - Clean, responsive chat interface
- **Sidebar Navigation** - Easy chat history management
- **File Upload** - Drag-and-drop file support
- **Real-time Updates** - Live chat with typing indicators

## 🛠 Tech Stack

### Frontend
- **Streamlit** - Web application framework
- **PIL/Pillow** - Image processing and manipulation

### Backend & AI
- **Ollama** - Local LLM inference engine
- **LLaMA 3/CodeLlama** - Open-source language models
- **Transformers** - NLP model integration

### OCR & Document Processing
- **Tesseract OCR** - Optical character recognition
- **pdfplumber** - PDF text extraction
- **python-docx** - Word document processing
- **pandas** - CSV and data file handling

### Database
- **SQLite** - Lightweight relational database
- **SQLAlchemy** (Optional) - Database ORM

## 📦 Installation

### Prerequisites
- Python 3.8+
- Tesseract OCR
- Ollama (for local AI)

