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

st.set_option("client.toolbarMode","minimal")

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
    Apply light blue and white styling.
    """
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #eaf7ff 0%, #ffffff 45%);
            color: #102a43;
        }

        section[data-testid="stSidebar"] {
            background-color: #d8f0ff;
            border-right: 1px solid #b6e0fe;
        }

        h1, h2, h3 {
            color: #0b4f71;
        }

        .raga-title {
            font-size: 42px;
            font-weight: 700;
            color: #003b66;
            margin: 0;
            padding: 0;
        }

        .raga-caption {
            color: #64748b;
            font-size: 16px;
            margin-top: -6px;
        }

        div[data-testid="stChatMessage"] {
            background-color: #ffffff;
            border: 1px solid #ccecff;
            border-radius: 16px;
            padding: 12px;
            box-shadow: 0 2px 8px rgba(11, 79, 113, 0.08);
        }

        .stButton > button {
            background-color: #0ea5e9;
            color: white;
            border-radius: 10px;
            border: none;
            padding: 0.5rem 1rem;
        }

        .stButton > button:hover {
            background-color: #0284c7;
            color: white;
        }

        div[data-testid="stExpander"] {
            background-color: #f8fcff;
            border-radius: 12px;
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


def handle_user_question(user_question: str, retriever, documents: list[dict]) -> dict:
    """
    Decide whether the user wants:
    - a simple explanation of the previous answer
    - a document summary
    - a normal answer
    - a simplified normal answer
    """
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