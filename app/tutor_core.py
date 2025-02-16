import streamlit as st
from typing import Generator
from knowledge_agent import TutorAgent
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os

def initialize_session_state():
    defaults = {
        "messages": [{"role": "assistant", "content": "Hi! I'm your AI Tutor. How can I help?"}],
        "notes": [],
        "toggle_mode": False,
        "student_name": "",
        "uploaded_files": None,
        "image_upload": None
    }

    for key, val in defaults.items():
        st.session_state.setdefault(key, val)

def answer_mode_toggle(switch):
    if switch:
        st.session_state.toggle_mode = not st.session_state.toggle_mode
        mode = "Document-based Learning" if st.session_state.toggle_mode else "General Learning"
        st.toast(f"Switched to {mode} mode")

def setup_sidebar():
    with st.sidebar:

        col1, col2 = st.columns([2, 2])
        with col1:
            with st.popover("Take Notes", icon="📝"):
                notes = st.text_area("Note taking space", label_visibility='hidden')
                st.button("Save Note")
        
        with col2:
            if st.button("New Chat", icon="🗒️"):
                reset_app()
    
        st.header("Student Profile")
        st.session_state.student_name = st.text_input("Enter your name:", label_visibility='hidden', placeholder='How should we call you?')
        
        st.divider()
        st.header("Course Materials")

        uploaded_files = st.file_uploader(
            "Upload study documents (PDF/TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            label_visibility='hidden'
        )
        st.session_state.uploaded_files = uploaded_files

        if uploaded_files:
            st.write("Document-based answering enabled.")
            st.session_state.toggle_mode = True
        else:
            st.session_state.toggle_mode = False
        
        st.divider()
        st.header("Learn from images")
        if not uploaded_files:
            st.warning("Please upload documents first.")
        else:
            if image_upload := st.file_uploader(
                                        "Upload Image",
                                        type=["png", "jpg", "jpeg"],
                                        accept_multiple_files=True,
                                        label_visibility='hidden'
                                    ):
                st.write("Image-based answering enabled.")
                st.session_state.image_upload = image_upload

        st.header("Learning Mode")
        st.session_state.toggle_mode = st.toggle("Document-based Learning", 
                                                 key="doc_learning_toggle", 
                                                 value=st.session_state.toggle_mode)
        
def display_chat():
    for msg in st.session_state.messages:
        avatar = "👾" if msg["role"] == "assistant" else "🤠"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

def reset_app():
    """
    Resets the entire app state by clearing:
    - Session state variables
    - Uploaded files
    - Chat history
    - Cache
    - All stored files in knowledge folder
    """
    # Clear all session state variables
    for key in st.session_state.keys():
        del st.session_state[key]
    
    # Clear cache
    st.cache_data.clear()
    st.cache_resource.clear()

    initialize_session_state()

    # Rerun the app
    st.rerun()

def generate_response(user_input: str) -> Generator[str, None, None]:
    try:
        # Convert messages to LangChain format
        lc_messages = [
            HumanMessage(content=msg["content"]) if msg["role"] == "user" 
            else AIMessage(content=msg["content"])
            for msg in st.session_state.messages
        ]
        
        # Initialize ChatOllama with streaming callback
        llm = ChatOllama(
            model="llama3.2",
            callbacks=[StreamingStdOutCallbackHandler()],
            temperature=0.7,
        )
        
        # Stream the response
        response_chunks = []
        full_response = ""
        for chunk in llm.stream(lc_messages):
            if isinstance(chunk, AIMessage):
                content_chunk = chunk.content
                response_chunks.append(content_chunk)
                full_response += content_chunk
                yield content_chunk
        
        # Add complete response to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
    except Exception as e:
        yield f"⚠️ Error: {str(e)}. Ensure Ollama is running (ollama serve)."

def string_to_stream(text: str) -> Generator[str, None, None]:
    """Convert a string to a stream-like generator for typewriter effect"""
    for chunk in text.split():
        yield chunk + " "

def main():
    st.set_page_config(
        page_title="AI Maestro",
        page_icon="👾",
        layout="wide"
    )
    
    initialize_session_state()
    setup_sidebar()

    st.image("app/static/tutor_image.jpg", width=500)
    st.title("🤖 AI Tutor Assistant")
    display_chat()
    
    if user_input := st.chat_input("Ask your question..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user", avatar="🤠"):
            st.markdown(user_input)
        
        with st.chat_message("assistant", avatar="👾"):
            agent_response = TutorAgent()

            if st.session_state.toggle_mode:
                if not st.session_state.uploaded_files:
                    response = "Document-based learning mode is active. Please upload your study materials first to continue."
                    st.warning(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    uploaded_files = st.session_state.uploaded_files

                    if st.session_state.image_upload:
                        image_upload = st.session_state.image_upload

                    if st.session_state.image_upload:
                        st.image(st.session_state.image_upload, width=200)    
                        st.session_state.messages.append({"role": "user", "content": st.session_state.image_upload})
                        full_response = agent_response.process_documents(uploaded_files, user_input, image_upload)
                    else:
                        full_response = agent_response.process_documents(uploaded_files, user_input)

                    st.write(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                st.write_stream(generate_response(user_input))

if __name__ == "__main__":
    os.system('rm -rf knowledge/*')
    os.system('rm -rf vectorstore_*')
    main()

