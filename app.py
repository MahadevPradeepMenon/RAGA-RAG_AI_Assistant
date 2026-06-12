from pathlib import Path

from PIL import Image
import streamlit as st

from src.loader import load_documents
from src.chunker import chunk_documents
from src.hybrid_retriever import HybridRetriever
from src.local_answerer import generate_local_answer
from src.summarizer import generate_summary_response, is_summary_request
from src.simplifier import (
    is_simple_request,
    is_follow_up_simplification,
    simplify_answer,
)


BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "data" / "uploads"
ICON_PATH = BASE_DIR / "assets" / "wheel.png"

wheel_icon = Image.open(ICON_PATH).convert("RGBA")

st.set_page_config(
    page_title="RAGA",
    page_icon=wheel_icon,
    layout="wide"
)



def get_file_signature():
    """
    Create a simple signature of uploaded files.
    Streamlit uses this to know when documents have changed.
    """
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

    supported_extensions = {".txt", ".pdf",".docx",".pptx"}

    return tuple(
        (file.name, file.stat().st_size, file.stat().st_mtime)
        for file in sorted(UPLOAD_FOLDER.iterdir())
        if file.is_file() and file.suffix.lower() in supported_extensions
    )

@st.cache_resource
def build_pipeline(file_signature):
    """
    Load documents, chunk them, and build the hybrid retriever.

    Cached so the embedding model does not reload on every Streamlit refresh.
    """
    documents = load_documents(str(UPLOAD_FOLDER))
    chunks = chunk_documents(documents)
    retriever = HybridRetriever(chunks)

    return retriever, chunks, documents


def save_uploaded_file(uploaded_file):
    """
    Save uploaded .txt file into data/uploads.
    """
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

    file_path = UPLOAD_FOLDER / uploaded_file.name

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    st.cache_resource.clear()

    return file_path


def apply_custom_styles():
    """
    Apply a professional RAGA interface theme.
    """
    st.markdown(
        """
        <style>
        /* ---------- GLOBAL APP ---------- */

        .stApp {
            background: #f8fafc;
            color: #0f172a;
        }

        .block-container {
            max-width: 1120px;
            padding-top: 3rem;
            padding-bottom: 7rem;
        }

        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }

        header {
            background: transparent !important;
        }

        h1, h2, h3 {
            color: #0f172a;
            letter-spacing: -0.02em;
        }

        p, span, label, div {
            font-family: "Inter", "Segoe UI", sans-serif;
        }

        /* ---------- SIDEBAR ---------- */

        section[data-testid="stSidebar"] {
            background: #0f172a;
            border-right: 1px solid #1e293b;
        }

        section[data-testid="stSidebar"] * {
            color: #f8fafc !important;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #ffffff !important;
            font-weight: 800;
        }

        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span {
            color: #cbd5e1 !important;
        }

        /* Sidebar file names */
        section[data-testid="stSidebar"] .stMarkdown {
            color: #e2e8f0 !important;
        }

        /* File uploader */
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
            background-color: #1e293b !important;
            border: 1px solid #334155 !important;
            border-radius: 14px !important;
            padding: 1rem !important;
        }

        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
            color: #f8fafc !important;
        }

        section[data-testid="stSidebar"] button {
            background-color: #2563eb !important;
            color: #ffffff !important;
            border-radius: 10px !important;
            border: none !important;
            font-weight: 700 !important;
        }

        section[data-testid="stSidebar"] button:hover {
            background-color: #1d4ed8 !important;
            color: #ffffff !important;
        }

        /* ---------- RAGA HEADER ---------- */

        .raga-title {
            font-size: 46px;
            font-weight: 850;
            color: #0f172a;
            margin: 0;
            padding: 0;
            letter-spacing: -1.5px;
            line-height: 1;
        }

        .raga-caption {
            color: #475569;
            font-size: 16px;
            margin-top: 6px;
        }

        /* ---------- STATUS / ALERT BOXES ---------- */

        div[data-testid="stAlert"] {
            border-radius: 14px;
            border: 1px solid #dbeafe;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        }

        /* ---------- CHAT MESSAGES ---------- */

        div[data-testid="stChatMessage"] {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            padding: 16px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
        }

        div[data-testid="stChatMessage"] p {
            color: #0f172a;
            font-size: 16px;
            line-height: 1.6;
        }

        /* ---------- CHAT INPUT ---------- */

        div[data-testid="stChatInput"] {
            background-color: #f8fafc !important;
            border-top: 1px solid #e2e8f0;
        }

        textarea {
            background-color: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 14px !important;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
        }

        textarea::placeholder {
            color: #64748b !important;
        }

        /* ---------- BUTTONS ---------- */

        .stButton > button {
            background-color: #2563eb;
            color: #ffffff !important;
            border-radius: 10px;
            border: none;
            padding: 0.6rem 1rem;
            font-weight: 700;
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25);
        }

        .stButton > button:hover {
            background-color: #1d4ed8;
            color: #ffffff !important;
            border: none;
        }

        /* ---------- EXPANDERS / EVIDENCE ---------- */

        div[data-testid="stExpander"] {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
        }

        div[data-testid="stExpander"] summary {
            color: #0f172a !important;
            font-weight: 650;
        }

        div[data-testid="stExpander"] p {
            color: #334155;
        }

        /* ---------- TABLES / DATAFRAMES ---------- */

        div[data-testid="stDataFrame"] {
            background-color: #ffffff;
            border-radius: 14px;
            border: 1px solid #e2e8f0;
        }

        /* ---------- SMALL POLISH ---------- */

        hr {
            border-color: #e2e8f0;
        }

        a {
            color: #2563eb;
            text-decoration: none;
        }

        a:hover {
            text-decoration: underline;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialise_chat_history():
    """
    Create chat history if it does not already exist.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []


def display_header():
    """
    Display the RAGA logo and title close together.
    """
    col1, col2 = st.columns([0.55, 8], gap="small")

    with col1:
        st.image(wheel_icon, width=65)

    with col2:
        st.markdown("<div class='raga-title'>RAGA</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='raga-caption'>RAG AI Assistant for answering questions from company documents.</div>",
            unsafe_allow_html=True,
        )


def display_chat_history():
    """
    Display previous user questions and assistant answers.
    """
    for message in st.session_state.messages:
        avatar = None

        if message["role"] == "assistant":
            avatar = wheel_icon

        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

            if message["role"] == "assistant" and "evidence" in message:
                with st.expander("View retrieved evidence"):
                    for index, result in enumerate(message["evidence"], start=1):
                        st.markdown(f"### Evidence {index}")

                        st.write(f"**Document:** {result['source']}")

                        if result.get("location"):
                            st.write(f"**Location:** {result['location']}")

                        st.write(result["text"])
                        st.divider()

def get_last_assistant_answer() -> str | None:
    """
    Find the most recent assistant message in chat history.
    """
    for message in reversed(st.session_state.messages):
        if message["role"] == "assistant":
            return message["content"]

    return None

def detect_missing_document_type(user_question: str, documents: list[dict]) -> str | None:
    """
    Detect if the user asks about a document type that has not been uploaded.
    """
    question_lower = user_question.lower()

    available_file_types = {
        document.get("file_type")
        for document in documents
    }

    requested_types = {
        "presentation": ".pptx",
        "powerpoint": ".pptx",
        "pptx": ".pptx",
        "slides": ".pptx",
        "slide": ".pptx",
        "pdf": ".pdf",
        "word document": ".docx",
        "docx": ".docx",
        "document": None,
    }

    for keyword, file_type in requested_types.items():
        if keyword in question_lower:
            if file_type and file_type not in available_file_types:
                return file_type

    return None

def handle_user_question(user_question: str, retriever, documents: list[dict]) -> dict:
    """
    Decide whether the user wants:
    - a simple explanation of the previous answer
    - a document summary
    - a normal answer
    - a simplified normal answer
    """

    missing_file_type = detect_missing_document_type(user_question, documents)

    if missing_file_type:
        return {
            "answer": f"Sorry, I could not find any uploaded {missing_file_type} document for that question.",
            "evidence": [],
        }
    if is_follow_up_simplification(user_question):
        previous_answer = get_last_assistant_answer()

        if previous_answer:
            return {
                "answer": simplify_answer(previous_answer),
                "evidence": [],
            }

        return {
            "answer": "I do not have a previous answer to simplify yet. Ask me a question first.",
            "evidence": [],
        }

    if is_summary_request(user_question):
        summary = generate_summary_response(user_question, documents)

        return {
            "answer": summary,
            "evidence": [],
        }

    results = retriever.search(user_question, top_k=3)
    answer_data = generate_local_answer(user_question, results)

    answer = answer_data["answer"]

    if is_simple_request(user_question):
        answer = simplify_answer(answer)

    return {
        "answer": answer,
        "evidence": results,
    }


def main():
    apply_custom_styles()
    initialise_chat_history()
    display_header()

    with st.sidebar:
        st.header("Documents")

        uploaded_file = st.file_uploader(
            "Upload a company document",
            type=["txt", "pdf", "docx", "pptx"],
        )

        if uploaded_file is not None:
            saved_path = save_uploaded_file(uploaded_file)
            st.success(f"Uploaded: {saved_path.name}")

        st.subheader("Available documents")

        UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

        supported_extensions = {".txt", ".pdf", ".docx", ".pptx"}

        document_files = sorted(
            file for file in UPLOAD_FOLDER.iterdir()
            if file.is_file() and file.suffix.lower() in supported_extensions
        )

        if document_files:
            for file in document_files:
                st.write(file.name)
        else:
            st.warning("No documents found. Add .txt, .pdf, .docx, or .pptx files to data/uploads.")

        if st.button("Refresh documents"):
            st.cache_resource.clear()
            st.rerun()

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    file_signature = get_file_signature()

    if not file_signature:
        st.info("Upload at least one .txt company document to begin.")
        return

    with st.spinner("Building retrieval pipeline..."):
        retriever, chunks, documents = build_pipeline(file_signature)

    st.success(f"RAGA is ready. Loaded {len(chunks)} chunks from company documents.")

    st.divider()

    display_chat_history()

    user_question = st.chat_input("Ask RAGA a question or request a summary...")

    if user_question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question,
            }
        )

        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant", avatar=wheel_icon):
            with st.spinner("Searching company documents..."):
                response = handle_user_question(user_question, retriever, documents)

            st.markdown(response["answer"])

            if response["evidence"]:
                with st.expander("View retrieved evidence"):
                    for index, result in enumerate(response["evidence"], start=1):
                        st.markdown(f"### Evidence {index}")

                        st.write(f"**Document:** {result['source']}")

                        if result.get("location"):
                            st.write(f"**Location:** {result['location']}")

                        st.write(result["text"])
                        st.divider()

        assistant_message = {
            "role": "assistant",
            "content": response["answer"],
        }

        if response["evidence"]:
            assistant_message["evidence"] = response["evidence"]

        st.session_state.messages.append(assistant_message)


if __name__ == "__main__":
    main()