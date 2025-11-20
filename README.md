**An intelligent chatbot with OCR integration, persistent storage, and multimodal capabilities**

## 🚀 Overview

This project is an AI-powered conversational chatbot built using Streamlit as the front-end interface and integrated with LLAMA language model for intelligent,
context-aware responses. It includes OCR functionality for extracting text from images/documents and SQLite for persistent data storage.

## 🧠 Key Features

- 🤖 AI Capabilities
- 📄 Document Processing
- 💾 Data Management
- 🎨 User Experience


## 🛠 Tech Stack

Frontend
- **Streamlit** - Web application framework
- **Python** - Python manipulation

Backend & AI
- **Ollama** - Local LLM inference engine
- **LLaMA 3/CodeLlama** - Open-source language models

OCR & Document Processing
- **Tesseract OCR** - Optical character recognition

Database
- **SQLite** - Lightweight relational database

## 📦 Installation

### Prerequisites
- Python 3.8+
- Tesseract OCR
- Ollama (for local AI)

- 📚 Use Case This chatbot can serve as a base for: Intelligent document assistants Academic research helpers AI-powered customer support bots Personalized chat companions


All the steps :

Phase 1: Foundation Setup
Install Python 3.8+, Git, and VS Code to establish your development foundation
Create a project folder and Python virtual environment to isolate dependencies
Install core libraries: Streamlit for UI, Pillow for images, Pytesseract for OCR

Phase 2: AI & OCR Setup
4. Install Ollama and run its background service for local AI processing
5. Download AI models like Llama2 and CodeLlama for code generation capabilities
6. Install Tesseract OCR engine and configure it for text extraction from images

Phase 3: Application Development
7. Build the Streamlit interface with sidebar for chat history and main chat area
8. Implement file uploader and OCR processing to extract code from images
9. Create AI communication logic to connect with Ollama models for responses

Phase 4: Core Features
10. Develop the chat system with message display and code formatting
11. Implement persistent chat history using pickle for data storage
12. Add search functionality and date-based organization for past conversations

Phase 5: Advanced Integration
13. Connect OCR output to AI prompts for context-aware code analysis
14. Implement streaming responses for real-time AI interaction
15. Add session management and multi-model fallback capabilities

Phase 6: Final Polish
16. Test complete workflow: image upload → OCR → AI analysis → response
17. Optimize image processing and error handling for robust performance
18. Deploy the final application and prepare for demonstration
