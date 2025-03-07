# frontend/app.py
from langchain_ollama.chat_models import ChatOllama
import streamlit as st
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables import chain
from pydantic import BaseModel
import os
import requests
import json

# Update to point to the backend container
llm2 = ChatOllama(model="granite3.2:2b", base_url="http://backend:11434", streaming=True)
llm = ChatOllama(model="command-r7b-arabic", base_url="http://backend:11434", streaming=True)

# Initialize session state for model selection if it doesn't exist
if 'current_model' not in st.session_state:
    st.session_state.current_model = 'الأول'
if 'langchain_messages' not in st.session_state:
    st.session_state['langchain_messages'] = []

# Add model selection dropdown with default value
model_option = st.sidebar.selectbox(
    'اختر النموذج',
    ('الأول', 'الثاني'),
    index=0,
    format_func=lambda x: 'النموذج ' + x,
    key='current_model'
)

# Select the appropriate model based on user choice
selected_llm = llm if model_option == 'الأول' else llm2

st.markdown(
    """
    <style>
    @font-face {
        font-family: 'Cairo Play';
        src: url('app/static/fonts/CairoPlay-SemiBold.woff2') format('truetype');
        font-weight: 300;
        font-style: normal;
        font-display: swap;
    }

    * {
        font-family: 'Cairo Play', sans-serif !important;
        font-size: 18px;
    }

    .stApp {
        direction: rtl;
    }

    .stMarkdown,
    .stChatMessage,
    .stChatInputContainer,
    .stSelectbox,
    .st-emotion-cache-16idsys,
    .st-emotion-cache-1nv5vhh {
        font-family: 'Cairo Play', sans-serif !important;
        direction: rtl;
        text-align: right;
    }

    .stChatMessage > div {
        direction: rtl;
        font-size: 16px;
        text-align: right;
        font-family: 'Cairo Play', sans-serif !important;
    }

    .st-emotion-cache-16idsys p {
        text-align: right;
        font-size: 16px;
    }

    .stChatInput {
        direction: rtl;
        text-align: right;
        font-size: 16px;
        font-family: 'Cairo Play', sans-serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("نظام الذكاء الاصطناعي لكلية القيادة والأركان المشتركة")

UPLOAD_DIR = "./data/uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

uploaded_files = st.sidebar.file_uploader("Upload PDF Files", type="pdf", accept_multiple_files=True)

if uploaded_files:
   if st.sidebar.button("Process Uploaded Files"):
        for uploaded_file in uploaded_files:
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.sidebar.success(f"Saved File: {uploaded_file.name}")

RAG_SERVICE_URL = "http://rag_service:8000/query"
def query_rag_service(question: str, model_name: str) -> str:
    """Queries the RAG service with a question and returns the answer."""
    try:
        response = requests.post(
            RAG_SERVICE_URL,
            json={"question": question, "model_name": model_name},
            timeout=120
         )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error querying RAG service: {e}")
        return "Error: Could not communicate with RAG service."


for message in st.session_state["langchain_messages"]:
    role = "user" if message.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(
            f'<div style="text-align: right; direction: rtl;">{message.content}</div>',
            unsafe_allow_html=True
        )

question = st.chat_input("أدخل سؤالك...")
if question:
    with st.chat_message("user"):
        st.markdown(
            f'<div style="text-align: right; direction: rtl;">{question}</div>',
            unsafe_allow_html=True
        )

    with st.chat_message("assistant"):
        response = query_rag_service(question, selected_llm.model)
        answer = response["answer"]
        answer_from = response["source"]
        if answer_from == "knowledge_base":
             st.markdown(
                 f'<div style="text-align: right; direction: rtl;">{answer} (من قاعدة المعرفة)</div>',
                 unsafe_allow_html=True
            )
        else:
           st.markdown(
               f'<div style="text-align: right; direction: rtl;">{answer} (من المعرفة العامة للنموذج)</div>',
               unsafe_allow_html=True
           )

        st.session_state["langchain_messages"].append(
           BaseModel(content=question, type="human")
        )
        st.session_state["langchain_messages"].append(
             BaseModel(content=answer, type="ai")
        )