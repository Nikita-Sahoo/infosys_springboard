import streamlit as st
import random
import time
from datetime import datetime, timedelta
import requests
import json
import pickle
import os
import re
from PIL import Image
import pytesseract
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="ChatGPT - Code Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto"
)

# Constants
MAX_CHAT_HISTORY = 15
DATA_FILE = "chat_history.pkl"

def setup_ocr():
    """Setup OCR configuration based on operating system"""
    try:
        # For Windows - you might need to adjust this path
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        return True
    except:
        try:
            # For Linux/Mac - usually in PATH
            pytesseract.pytesseract.tesseract_cmd = 'tesseract'
            return True
        except:
            return False

def extract_text_from_image(image):
    """Extract text from uploaded image using OCR"""
    try:
        # Convert image to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Use pytesseract to extract text
        extracted_text = pytesseract.image_to_string(image)
        return extracted_text.strip()
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def clear_ocr_data():
    """Clear OCR-related session state data"""
    keys_to_remove = [
        'ocr_extracted_text', 
        'ocr_image_uploaded', 
        'ocr_image_preview',
        'ocr_context_ready',
        'ocr_active_image',
        'auto_process_image'
    ]
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]

def enhance_system_prompt_with_ocr():
    """Enhance the system prompt to handle OCR content"""
    ocr_context = ""
    if st.session_state.get('ocr_extracted_text'):
        ocr_context = f"""
        
OCR CONTEXT (Important - user is asking about an uploaded image):
The user has uploaded an image with the following extracted text:
"{st.session_state.ocr_extracted_text}"

When responding to questions about the image content:
1. Reference the extracted text directly when relevant
2. Provide insights, analysis, or explanations based on the OCR content
3. If the question is unclear, ask for clarification about which part of the extracted text they're referring to
4. Maintain context about the OCR content throughout the conversation
"""
    return ocr_context

def save_chats_to_file():
    """Save chat history to pickle file"""
    try:
        data = {
            "chat_history": st.session_state.chat_history,
            "next_chat_id": st.session_state.next_chat_id
        }
        with open(DATA_FILE, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        st.error(f"Error saving to file: {e}")
        return False

def load_chats_from_file():
    """Load chat history from pickle file"""
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'rb') as f:
                data = pickle.load(f)
            return data.get("chat_history", []), data.get("next_chat_id", 1)
        return [], 1
    except Exception as e:
        st.error(f"Error loading from file: {e}")
        return [], 1

def get_code_models_recommendation():
    """Return recommended models for code generation"""
    return [
        "codellama:latest",
        "codeqwen:latest", 
        "deepseek-coder:latest",
        "llama2:latest",
        "mistral:latest",
        "phi:latest",
        "wizardcoder:latest",
        "starcoder:latest"
    ]

def get_code_templates():
    """Return code generation template buttons"""
    templates = {
        
    }
    return templates

def show_code_templates():
    """Display code template buttons"""
    templates = get_code_templates()
    
    cols = st.columns(2)
    for i, (name, prompt) in enumerate(templates.items()):
        with cols[i % 2]:
            if st.button(f"📝 {name}", use_container_width=True, key=f"template_{i}"):
                st.session_state.template_prompt = prompt + " "
                st.rerun()

def format_response_with_code(text):
    """Format response with proper code block styling"""
    parts = text.split('```')
    formatted_text = ""
    
    for i, part in enumerate(parts):
        if i % 2 == 0:
            formatted_text += part
        else:
            lines = part.split('\n', 1)
            if len(lines) > 1 and lines[0].strip():
                language = lines[0].strip()
                code_content = lines[1]
            else:
                language = 'text'
                code_content = part
            
            formatted_text += f"\n```{language}\n{code_content}\n```\n"
    
    return formatted_text

def organize_chats_by_date(chats):
    """Organize chats into date-based sections"""
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    last_week = today - timedelta(days=7)
    
    sections = {
        "Today": [],
        "Yesterday": [],
        "Previous 7 Days": [],
        "Older": []
    }
    
    for chat in chats:
        chat_date_str = chat.get("last_updated", chat["date"])
        try:
            chat_date = datetime.strptime(chat_date_str, "%Y-%m-%d %H:%M").date()
        except:
            chat_date = datetime.strptime(chat_date_str.split()[0], "%Y-%m-%d").date()
        
        if chat_date == today:
            sections["Today"].append(chat)
        elif chat_date == yesterday:
            sections["Yesterday"].append(chat)
        elif chat_date >= last_week:
            sections["Previous 7 Days"].append(chat)
        else:
            sections["Older"].append(chat)
    
    return sections

def initialize_session_state():
    """Initialize all session state variables with persistent data"""
    # Load from file storage
    chat_history, next_chat_id = load_chats_from_file()
    
    default_state = {
        "messages": [],
        "chat_history": chat_history,
        "current_chat_id": None,
        "chat_started": False,
        "next_chat_id": next_chat_id,
        "search_query": "",
        "ollama_url": "http://localhost:11434",
        "model": "codellama:latest",
        "temperature": 0.3,
        "max_tokens": 2000,
        "use_streaming": True,
        "max_chat_history": MAX_CHAT_HISTORY,
        "template_prompt": "",
        "ocr_extracted_text": "",
        "ocr_image_uploaded": False,
        "ocr_image_preview": None,
        "ocr_context_ready": False,
        "ocr_active_image": None,
        "auto_process_image": False,
        "image_chat_mode": False,
        "show_file_uploader": False
    }
    
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialize session state
initialize_session_state()

# Initialize OCR
ocr_available = setup_ocr()

def get_ai_response(user_input, conversation_history=None):
    """Generate AI response using Ollama local LLM with OCR context"""
    try:
        messages = []
        
        system_message = """You are an expert AI assistant specialized in code generation and programming. 
When generating code, follow these guidelines:
1. Provide complete, runnable code examples
2. Include proper syntax highlighting markers (e.g., ```python, ```javascript, etc.)

For non-code questions, provide clear, concise, and accurate responses."""
        
        ocr_context = enhance_system_prompt_with_ocr()
        system_message += ocr_context
        
        messages.append({
            "role": "system",
            "content": system_message
        })
        
        if conversation_history:
            for msg in conversation_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        url = f"{st.session_state.ollama_url}/api/chat"
        
        payload = {
            "model": st.session_state.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": st.session_state.temperature,
                "num_predict": st.session_state.max_tokens
            }
        }
        
        response = requests.post(url, json=payload, timeout=120)
        
        if response.status_code == 200:
            response_data = response.json()
            return response_data['message']['content']
        else:
            return f"Error: Ollama API returned status code {response.status_code}. Make sure Ollama is running and the model is available."
        
    except requests.exceptions.ConnectionError:
        return f"Connection error: Cannot connect to Ollama at {st.session_state.ollama_url}. Please make sure Ollama is running."
    except requests.exceptions.Timeout:
        return "Request timeout: Ollama is taking too long to respond."
    except Exception as e:
        return f"An error occurred: {str(e)}"

def get_ai_response_streamed(user_input, conversation_history=None):
    """Generate AI response using Ollama with streaming for better UX"""
    try:
        messages = []
        
        system_message = """You are an expert AI assistant specialized in code generation and programming. 
When generating code, follow these guidelines:
1. Provide complete, runnable code examples
2. Include proper syntax highlighting markers (e.g., ```python, ```javascript, etc.)

For non-code questions, provide clear, concise, and accurate responses."""
        
        ocr_context = enhance_system_prompt_with_ocr()
        system_message += ocr_context
        
        messages.append({
            "role": "system",
            "content": system_message
        })
        
        if conversation_history:
            for msg in conversation_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        url = f"{st.session_state.ollama_url}/api/chat"
        
        payload = {
            "model": st.session_state.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": st.session_state.temperature,
                "num_predict": st.session_state.max_tokens
            }
        }
        
        response = requests.post(url, json=payload, timeout=180, stream=True)
        
        if response.status_code == 200:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    line_data = json.loads(line)
                    if 'message' in line_data and 'content' in line_data['message']:
                        content = line_data['message']['content']
                        full_response += content
                        yield content
                    elif 'done' in line_data and line_data['done']:
                        break
            return full_response
        else:
            yield f"Error: Ollama API returned status code {response.status_code}"
        
    except requests.exceptions.ConnectionError:
        yield f"Connection error: Cannot connect to Ollama at {st.session_state.ollama_url}"
    except requests.exceptions.Timeout:
        yield "Request timeout: Ollama is taking too long to respond."
    except Exception as e:
        yield f"An error occurred: {str(e)}"

def get_available_models():
    """Get list of available models from Ollama"""
    try:
        url = f"{st.session_state.ollama_url}/api/tags"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
        return []
    except:
        return []

def test_ollama_connection():
    """Test connection to Ollama"""
    try:
        url = f"{st.session_state.ollama_url}/api/tags"
        response = requests.get(url, timeout=10)
        return response.status_code == 200
    except:
        return False

def save_persistent_data():
    """Save chat history to persistent storage"""
    try:
        return save_chats_to_file()
    except Exception as e:
        st.error(f"Error saving persistent data: {e}")
        return False

def manage_chat_history():
    """Manage chat history to ensure it doesn't exceed the maximum limit"""
    if len(st.session_state.chat_history) > st.session_state.max_chat_history:
        chats_to_remove = len(st.session_state.chat_history) - st.session_state.max_chat_history
        st.session_state.chat_history = st.session_state.chat_history[:-chats_to_remove]
        
        save_persistent_data()
        
        if chats_to_remove > 0:
            st.warning(f"⚠️ Memory is full! Removed {chats_to_remove} oldest chat(s) to make space.")

def create_new_chat_id():
    """Generate a new unique chat ID"""
    chat_id = st.session_state.next_chat_id
    st.session_state.next_chat_id += 1
    return chat_id

def get_or_create_chat_session():
    """Get current chat session or create a new one if none exists"""
    if st.session_state.current_chat_id is None:
        st.session_state.current_chat_id = create_new_chat_id()
        st.session_state.chat_started = True
        return st.session_state.current_chat_id
    return st.session_state.current_chat_id

def update_chat_title(first_user_message):
    """Update chat title based on first user message"""
    chat_title = first_user_message[:30] + "..." if len(first_user_message) > 30 else first_user_message
    if not chat_title:
        chat_title = "New Chat"
    return chat_title

def save_current_chat(update_timestamp=True):
    """Save the current chat to history - only saves when there are actual messages"""
    if st.session_state.messages and st.session_state.chat_started:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        chat_id = st.session_state.current_chat_id
        
        existing_chat_index = -1
        for i, chat in enumerate(st.session_state.chat_history):
            if chat["id"] == chat_id:
                existing_chat_index = i
                break
        
        first_user_message = ""
        if existing_chat_index == -1:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    first_user_message = msg["content"]
                    break
            chat_title = update_chat_title(first_user_message)
            
            image_data = st.session_state.get('ocr_active_image')
            extracted_text = st.session_state.get('ocr_extracted_text', '')
            
            chat_data = {
                "id": chat_id,
                "title": chat_title,
                "messages": st.session_state.messages.copy(),
                "date": current_time,
                "last_updated": current_time,
                "image_data": image_data,
                "extracted_text": extracted_text
            }
            
            st.session_state.chat_history.insert(0, chat_data)
            
        else:
            st.session_state.chat_history[existing_chat_index]["messages"] = st.session_state.messages.copy()
            
            if st.session_state.get('ocr_active_image'):
                st.session_state.chat_history[existing_chat_index]["image_data"] = st.session_state.ocr_active_image
                st.session_state.chat_history[existing_chat_index]["extracted_text"] = st.session_state.get('ocr_extracted_text', '')
            
            if update_timestamp:
                st.session_state.chat_history[existing_chat_index]["last_updated"] = current_time
                updated_chat = st.session_state.chat_history.pop(existing_chat_index)
                st.session_state.chat_history.insert(0, updated_chat)
        
        manage_chat_history()
        save_persistent_data()

def start_new_chat():
    """Start a new chat session - creates a completely separate chat"""
    if st.session_state.messages and st.session_state.chat_started and st.session_state.current_chat_id is not None:
        save_current_chat(update_timestamp=False)
    
    new_chat_id = create_new_chat_id()
    
    st.session_state.messages = []
    st.session_state.current_chat_id = new_chat_id
    st.session_state.chat_started = False
    st.session_state.search_query = ""
    st.session_state.template_prompt = ""
    st.session_state.image_chat_mode = False
    clear_ocr_data()
    st.session_state.show_file_uploader = False

def delete_chat(chat_id):
    """Delete a chat from history"""
    st.session_state.chat_history = [chat for chat in st.session_state.chat_history if chat["id"] != chat_id]
    
    if st.session_state.current_chat_id == chat_id:
        st.session_state.messages = []
        st.session_state.current_chat_id = None
        st.session_state.chat_started = False
        st.session_state.image_chat_mode = False
        clear_ocr_data()
        st.session_state.show_file_uploader = False
    
    save_persistent_data()
    
    st.success("Chat deleted successfully!")

def delete_all_chats():
    """Delete all chat history"""
    st.session_state.chat_history = []
    st.session_state.messages = []
    st.session_state.current_chat_id = None
    st.session_state.chat_started = False
    st.session_state.next_chat_id = 1
    st.session_state.template_prompt = ""
    st.session_state.image_chat_mode = False
    clear_ocr_data()
    st.session_state.show_file_uploader = False
    
    save_persistent_data()
    
    try:
        if os.path.exists(DATA_FILE):
            os.remove(DATA_FILE)
    except:
        pass
    
    st.success("All chat history cleared!")

def load_chat(chat_id):
    """Load a specific chat into the current view"""
    for chat in st.session_state.chat_history:
        if chat["id"] == chat_id:
            st.session_state.current_chat_id = chat_id
            st.session_state.messages = chat["messages"].copy()
            st.session_state.chat_started = True
            
            if chat.get("image_data"):
                st.session_state.ocr_active_image = chat["image_data"]
                st.session_state.ocr_extracted_text = chat.get("extracted_text", "")
                st.session_state.ocr_context_ready = True
                st.session_state.image_chat_mode = True
            else:
                st.session_state.image_chat_mode = False
                clear_ocr_data()
            break

def filter_chats(search_query):
    """Filter chats based on search query"""
    if not search_query:
        return st.session_state.chat_history
    
    search_lower = search_query.lower()
    filtered_chats = []
    
    for chat in st.session_state.chat_history:
        if search_lower in chat["title"].lower():
            filtered_chats.append(chat)
        else:
            for message in chat["messages"]:
                if search_lower in message["content"].lower():
                    filtered_chats.append(chat)
                    break
    
    return filtered_chats

def process_uploaded_image(uploaded_image):
    """Process uploaded image and add to current chat (same chat ID)"""
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        
        with st.spinner("Extracting text from image..."):
            extracted_text = extract_text_from_image(image)
            
            st.session_state.ocr_extracted_text = extracted_text
            st.session_state.ocr_image_uploaded = True
            st.session_state.ocr_context_ready = True
            st.session_state.auto_process_image = True
            st.session_state.image_chat_mode = True
            
            current_image_data = uploaded_image.getvalue()
            
            image_message = {
                "role": "user", 
                "content": "📷 Image uploaded for analysis",
                "type": "image",
                "image_data": current_image_data,
                "extracted_text": extracted_text
            }
            st.session_state.messages.append(image_message)
            
            if not st.session_state.chat_started:
                st.session_state.chat_started = True
            
            if extracted_text and len(extracted_text) > 10:
                return True
            else:
                return False

def display_ocr_image_upload():
    """Display image upload button that shows file extract box when clicked"""
    
    st.markdown("""
    <style>
    .file-uploader-hidden {
        display: none !important;
    }
    .custom-upload-btn {
        background-color: #f0f2f6;
        border: 1px solid #d0d0d0;
        border-radius: 4px;
        padding: 8px 12px;
        cursor: pointer;
        text-align: center;
        font-size: 14px;
        margin-bottom: 5px;
    }
    .custom-upload-btn:hover {
        background-color: #e0e2e6;
    }
    .file-uploader-visible {
        display: block !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        if st.session_state.template_prompt:
            prompt = st.chat_input("Complete your code request...", key="chat_input_ocr")
        else:
            prompt = st.chat_input("Message ChatGPT...", key="chat_input_ocr")
    
    with col2:
        if st.button("📷 Upload", key="custom_upload_btn", use_container_width=True):
            st.session_state.show_file_uploader = not st.session_state.show_file_uploader
        
        if st.session_state.get('show_file_uploader', False):
            uploaded_image = st.file_uploader(
                "Choose an image file",
                type=["jpg", "jpeg", "png"],
                key="sidebar_image_upload",
                label_visibility="visible",
                help="Upload image for analysis"
            )
            
            if uploaded_image is not None and not st.session_state.get('auto_process_image', False):
                success = process_uploaded_image(uploaded_image)
                if success:
                    st.success("✅ Image uploaded successfully!")
                    st.session_state.auto_process_image = True
                    st.session_state.show_file_uploader = False
                    st.rerun()
    
    return prompt

# Sidebar
with st.sidebar:
    st.title("🤖 ChatGPT")
    
    if st.button("+ New Chat", use_container_width=True, type="primary"):
        start_new_chat()
        st.rerun()
    
    if st.session_state.chat_history:
        if st.button("🗑️ Clear All Chats", use_container_width=True, type="secondary"):
            delete_all_chats()
            st.rerun()
    
    if not ocr_available:
        st.warning("⚠️ OCR not available. Install Tesseract OCR for text extraction.")
    
    search_query = st.text_input(
        "Search chats",
        value=st.session_state.search_query,
        placeholder="Search chat titles...",
        key="search_input"
    )
    
    if search_query != st.session_state.search_query:
        st.session_state.search_query = search_query
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.subheader("Chat History")
    
    filtered_chats = filter_chats(st.session_state.search_query)
    
    if filtered_chats:
        chat_sections = organize_chats_by_date(filtered_chats)
        
        for section_name, section_chats in chat_sections.items():
            if section_chats:
                st.markdown(f"**{section_name}**")
                
                for chat in section_chats:
                    is_active = st.session_state.current_chat_id == chat["id"]
                    
                    col1, col2 = st.columns([0.8, 0.2])
                    
                    with col1:
                        button_type = "primary" if is_active else "secondary"
                        
                        chat_date_str = chat.get("last_updated", chat["date"])
                        try:
                            chat_date = datetime.strptime(chat_date_str, "%Y-%m-%d %H:%M")
                            formatted_date = chat_date.strftime("%b %d, %Y at %H:%M")
                        except:
                            formatted_date = chat_date_str
                        
                        chat_title = chat['title']
                        if chat.get("image_data"):
                            chat_title = f"🖼️ {chat_title}"
                        
                        if st.button(
                            f"{chat_title}",
                            key=f"load_{chat['id']}",
                            use_container_width=True,
                            type=button_type,
                            help=f"Last updated: {formatted_date}"
                        ):
                            load_chat(chat["id"])
                            st.rerun()
                    
                    with col2:
                        if st.button(
                            "🗑️",
                            key=f"delete_{chat['id']}",
                            help=f"Delete this chat",
                            use_container_width=True
                        ):
                            delete_chat(chat["id"])
                            st.rerun()
                
                st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)
    else:
        if st.session_state.search_query:
            st.warning(f"No chats found for '{st.session_state.search_query}'")
        else:
            st.info("No chat history yet. Start a new chat to begin!")

# Main chat area
st.title("ChatGPT")

if not test_ollama_connection():
    st.error(f"⚠️ Cannot connect to Ollama at {st.session_state.ollama_url}. Please make sure Ollama is running.")
    st.info("**For optimal code generation, install these recommended models:**")
    st.code("""
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull code-specific models (in a new terminal)
ollama pull codellama:latest
ollama pull deepseek-coder:latest
ollama pull codeqwen:latest
ollama pull llama2:latest
    """)

if st.session_state.current_chat_id and st.session_state.messages:
    current_chat = next((chat for chat in st.session_state.chat_history 
                        if chat["id"] == st.session_state.current_chat_id), None)
    if current_chat:
        title_prefix = "🖼️ " if current_chat.get("image_data") else ""
        st.subheader(f"{title_prefix}{current_chat['title']}")

if not st.session_state.image_chat_mode and (not st.session_state.messages or len(st.session_state.messages) == 0):
    show_code_templates()

if st.session_state.template_prompt and not st.session_state.messages:
    st.info(f"💡 Template ready: **{st.session_state.template_prompt}** - Start typing to complete your request.")

if st.session_state.messages:
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.container():
                col1, col2 = st.columns([2, 2])
                with col2:
                    with st.chat_message("user"):
                        if message.get("type") == "image" and message.get("image_data"):
                            try:
                                image = Image.open(io.BytesIO(message["image_data"]))
                                st.image(image, caption="Uploaded Image", use_container_width=True, width=400)
                                st.caption("📷 Image uploaded for analysis")
                                if message.get("extracted_text"):
                                    with st.expander("📝 Extracted Text"):
                                        st.text(message["extracted_text"])
                            except Exception as e:
                                st.write("📷 Image uploaded for analysis")
                        else:
                            st.write(message["content"])
        else:
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    with st.chat_message("assistant"):
                        content = message["content"]
                        
                        if '```' in content:
                            parts = content.split('```')
                            for i, part in enumerate(parts):
                                if i % 2 == 0:
                                    st.write(part)
                                else:
                                    lines = part.split('\n', 1)
                                    if len(lines) > 1 and lines[0].strip() in ['python', 'javascript', 'java', 'cpp', 'html', 'css', 'sql', 'bash', 'json', 'yaml', 'xml']:
                                        language = lines[0].strip()
                                        code_content = lines[1]
                                    else:
                                        language = 'text'
                                        code_content = part
                                    
                                    st.code(code_content, language=language)
                        else:
                            st.write(content)
else:
    if not st.session_state.current_chat_id:
        st.info("👈 Start a new chat or select one from the history to begin!")
    else:
        if st.session_state.image_chat_mode:
            st.info("💡 Upload an image to start analyzing its content!")
        else:
            st.info("💡 Start a conversation with your AI code assistant!")

# Auto-generate response when image is uploaded
if st.session_state.get('auto_process_image') and st.session_state.get('ocr_extracted_text'):
    if not any(msg.get("content", "").startswith("I've analyzed the uploaded image") for msg in st.session_state.messages if msg["role"] == "assistant"):
        analysis_prompt = f"""
        I've uploaded an image with the following extracted text: 
        "{st.session_state.ocr_extracted_text}"
        
        Please analyze this content and provide:
        1. A summary of what the text is about
        
        Keep the response concise but informative.
        """
        
        chat_id = get_or_create_chat_session()
        
        if not st.session_state.chat_started:
            st.session_state.chat_started = True
        
        with st.container():
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Analyzing image content..."):
                        if st.session_state.use_streaming:
                            message_placeholder = st.empty()
                            full_response = ""
                            
                            try:
                                for chunk in get_ai_response_streamed(analysis_prompt, st.session_state.messages[:-1]):
                                    full_response += chunk
                                    if '```' in full_response:
                                        message_placeholder.markdown(format_response_with_code(full_response + "▌"), unsafe_allow_html=True)
                                    else:
                                        message_placeholder.write(full_response + "▌")
                                
                                if '```' in full_response:
                                    message_placeholder.markdown(format_response_with_code(full_response), unsafe_allow_html=True)
                                else:
                                    message_placeholder.write(full_response)
                                ai_response = full_response
                                
                            except Exception as e:
                                error_msg = f"Error during streaming: {str(e)}"
                                st.error(error_msg)
                                ai_response = error_msg
                        else:
                            ai_response = get_ai_response(analysis_prompt, st.session_state.messages[:-1])
                            if '```' in ai_response:
                                st.markdown(format_response_with_code(ai_response), unsafe_allow_html=True)
                            else:
                                st.write(ai_response)

        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        
        save_current_chat()
        
        st.session_state.auto_process_image = False
        st.rerun()

# Chat input with OCR image upload button
prompt = display_ocr_image_upload()

if prompt:
    if not test_ollama_connection():
        st.error(f"Cannot connect to Ollama. Please make sure it's running at {st.session_state.ollama_url}")
        st.stop()
    
    chat_id = get_or_create_chat_session()
    
    if (len(st.session_state.chat_history) >= st.session_state.max_chat_history and 
        not any(chat["id"] == chat_id for chat in st.session_state.chat_history)):
        st.error(f"⚠️ Memory is full! Maximum {st.session_state.max_chat_history} chats allowed. Please delete some chats to create new ones.")
        st.stop()
    
    if not st.session_state.chat_started:
        st.session_state.chat_started = True
    
    if st.session_state.template_prompt:
        full_prompt = st.session_state.template_prompt + prompt
        st.session_state.template_prompt = ""
    else:
        full_prompt = prompt
    
    st.session_state.messages.append({"role": "user", "content": full_prompt})
    
    with st.container():
        col1, col2 = st.columns([0.2, 0.8])
        with col2:
            with st.chat_message("user"):
                st.write(full_prompt)

    with st.container():
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            with st.chat_message("assistant"):
                code_keywords = ['code', 'program', 'function', 'algorithm', 'script', 'class', 'def ', 'import ', 'function ', 'const ', 'let ', 'var ', 'public ', 'private ']
                is_code_request = any(keyword in full_prompt.lower() for keyword in code_keywords)
                
                with st.spinner("💻 Generating code..." if is_code_request else "🤔 Thinking..."):
                    if st.session_state.use_streaming:
                        message_placeholder = st.empty()
                        full_response = ""
                        
                        try:
                            for chunk in get_ai_response_streamed(full_prompt, st.session_state.messages[:-1]):
                                full_response += chunk
                                if '```' in full_response:
                                    message_placeholder.markdown(format_response_with_code(full_response + "▌"), unsafe_allow_html=True)
                                else:
                                    message_placeholder.write(full_response + "▌")
                            
                            if '```' in full_response:
                                message_placeholder.markdown(format_response_with_code(full_response), unsafe_allow_html=True)
                            else:
                                message_placeholder.write(full_response)
                            ai_response = full_response
                            
                        except Exception as e:
                            error_msg = f"Error during streaming: {str(e)}"
                            st.error(error_msg)
                            ai_response = error_msg
                    else:
                        ai_response = get_ai_response(full_prompt, st.session_state.messages[:-1])
                        if '```' in ai_response:
                            st.markdown(format_response_with_code(ai_response), unsafe_allow_html=True)
                        else:
                            st.write(ai_response)

    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    
    save_current_chat(update_timestamp=True)
    
    st.rerun()

if len(st.session_state.chat_history) >= st.session_state.max_chat_history:
    st.warning(f"""
    ⚠️ **Memory Full!** 
    
    You've reached the maximum limit of {st.session_state.max_chat_history} chats. 
    """)
