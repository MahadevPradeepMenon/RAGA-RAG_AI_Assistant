import json
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.loader import load_documents
from src.chunker import chunk_documents
from src.hybrid_retriever import HybridRetriever


EVAL_PATH = Path("data/evaluation/questions.json")
UPLOAD_FOLDER = Path("data/uploads")

BASE_DIR = Path(__file__).parent
DASHBOARD_LOGO_PATH = BASE_DIR / "assets" / "evaluation_logo.png"

dashboard_logo = Image.open(DASHBOARD_LOGO_PATH).convert("RGBA")

def load_evaluation_questions(file_path: Path) -> list[dict]:
    """
    Load evaluation questions from JSON.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def is_relevant_result(result: dict, relevant_sources: list[str]) -> bool:
    """
    Check whether a retrieved result is from an expected source.
    """
    return result["source"] in relevant_sources


def calculate_hit_at_k(results: list[dict], relevant_sources: list[str], k: int) -> int:
    """
    Hit@K = 1 if at least one relevant source appears in the top K results.
    """
    for result in results[:k]:
        if is_relevant_result(result, relevant_sources):
            return 1

    return 0


def calculate_recall_at_k(results: list[dict], relevant_sources: list[str], k: int) -> float:
    """
    Recall@K = relevant sources found in top K / total relevant sources.
    """
    found_sources = set()

    for result in results[:k]:
        if is_relevant_result(result, relevant_sources):
            found_sources.add(result["source"])

    return len(found_sources) / len(relevant_sources)


def calculate_reciprocal_rank(results: list[dict], relevant_sources: list[str]) -> float:
    """
    Reciprocal Rank = 1 / rank of first relevant result.
    """
    for rank, result in enumerate(results, start=1):
        if is_relevant_result(result, relevant_sources):
            return 1 / rank

    return 0.0


@st.cache_resource
def build_retriever():
    """
    Load documents, chunk them, and build the hybrid retriever.
    Cached so the model does not reload on every UI interaction.
    """
    documents = load_documents(str(UPLOAD_FOLDER))
    chunks = chunk_documents(documents)
    retriever = HybridRetriever(chunks)

    return documents, chunks, retriever


def run_evaluation(top_k: int = 3) -> tuple[pd.DataFrame, dict]:
    """
    Run retrieval evaluation and return a results dataframe plus summary metrics.
    """
    questions = load_evaluation_questions(EVAL_PATH)
    documents, chunks, retriever = build_retriever()

    rows = []

    total_hit_at_1 = 0
    total_hit_at_k = 0
    total_recall_at_k = 0
    total_mrr = 0

    for index, item in enumerate(questions, start=1):
        question = item["question"]
        relevant_sources = item["relevant_sources"]

        results = retriever.search(question, top_k=top_k)

        hit_at_1 = calculate_hit_at_k(results, relevant_sources, k=1)
        hit_at_k = calculate_hit_at_k(results, relevant_sources, k=top_k)
        recall_at_k = calculate_recall_at_k(results, relevant_sources, k=top_k)
        reciprocal_rank = calculate_reciprocal_rank(results, relevant_sources)

        total_hit_at_1 += hit_at_1
        total_hit_at_k += hit_at_k
        total_recall_at_k += recall_at_k
        total_mrr += reciprocal_rank

        top_result = results[0]["source"] if results else "No result"

        retrieved_sources = [
            result["source"]
            for result in results
        ]

        rows.append({
            "Question #": index,
            "Question": question,
            "Expected Source(s)": ", ".join(relevant_sources),
            "Top Retrieved Source": top_result,
            "Top 3 Retrieved Sources": ", ".join(retrieved_sources),
            "Hit@1": hit_at_1,
            f"Hit@{top_k}": hit_at_k,
            f"Recall@{top_k}": recall_at_k,
            "MRR": reciprocal_rank,
            "Status": "Pass" if hit_at_1 == 1 else "Needs Review",
        })

    question_count = len(questions)

    summary = {
        "documents_loaded": len(documents),
        "chunks_created": len(chunks),
        "questions_evaluated": question_count,
        "hit_at_1": total_hit_at_1 / question_count,
        f"hit_at_{top_k}": total_hit_at_k / question_count,
        f"recall_at_{top_k}": total_recall_at_k / question_count,
        "mrr": total_mrr / question_count,
    }

    return pd.DataFrame(rows), summary


def get_document_coverage() -> pd.DataFrame:
    """
    Count how many evaluation questions target each document.
    """
    questions = load_evaluation_questions(EVAL_PATH)

    source_counter = Counter()

    for item in questions:
        for source in item["relevant_sources"]:
            source_counter[source] += 1

    rows = [
        {
            "Document": source,
            "Evaluation Question Count": count,
        }
        for source, count in source_counter.items()
    ]

    return pd.DataFrame(rows)

def apply_dashboard_styles():
    """
    Apply professional dashboard styling.
    """
    st.markdown(
        """
        <style>
        .stApp {
            background: #f8fafc;
            color: #0f172a;
        }

        .block-container {
            max-width: 1150px;
            padding-top: 3rem;
            padding-bottom: 5rem;
        }

        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }

        h1, h2, h3 {
            color: #0f172a;
            letter-spacing: -0.02em;
        }

        section[data-testid="stSidebar"] {
            background: #0f172a;
            border-right: 1px solid #1e293b;
        }

        section[data-testid="stSidebar"] * {
            color: #f8fafc !important;
        }

        .stMetric {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 16px;
            box-shadow: 0 3px 12px rgba(15, 23, 42, 0.05);
        }

        div[data-testid="stDataFrame"] {
            background-color: #ffffff;
            border-radius: 14px;
            border: 1px solid #e2e8f0;
        }

        div[data-testid="stExpander"] {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
        }

        .stButton > button {
            background-color: #2563eb;
            color: #ffffff !important;
            border-radius: 10px;
            border: none;
            font-weight: 700;
        }

        .stButton > button:hover {
            background-color: #1d4ed8;
            color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def main():
    st.set_page_config(
        page_title="RAGA Evaluation Dashboard",
        page_icon=dashboard_logo,
        layout="wide",
    )

    apply_dashboard_styles()

    col1, col2 = st.columns([0.7, 8], gap="small")

    with col1:
        st.image(dashboard_logo, width=70)
    with col2:
        st.markdown("## RAGA Evaluation Dashboard")
        st.write(
        "This dashboard helps inspect retrieval performance, failed questions, "
        "and document coverage for the RAGA evaluation set."
        )

    if not EVAL_PATH.exists():
        st.error("Evaluation file not found: data/evaluation/questions.json")
        return

    if not UPLOAD_FOLDER.exists():
        st.error("Upload folder not found: data/uploads")
        return

    with st.sidebar:
        st.header("Evaluation Settings")

        top_k = st.slider(
            "Top K",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
        )

        if st.button("Refresh evaluation"):
            st.cache_resource.clear()
            st.rerun()

        st.markdown("---")
        st.write("Run from terminal:")
        st.code("streamlit run evaluation_dashboard.py")

    with st.spinner("Running retrieval evaluation..."):
        results_df, summary = run_evaluation(top_k=top_k)

    st.subheader("Overall Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Hit@1", f"{summary['hit_at_1']:.2f}")

    with col2:
        st.metric(f"Hit@{top_k}", f"{summary[f'hit_at_{top_k}']:.2f}")

    with col3:
        st.metric(f"Recall@{top_k}", f"{summary[f'recall_at_{top_k}']:.2f}")

    with col4:
        st.metric("MRR", f"{summary['mrr']:.2f}")

    st.markdown("---")

    st.subheader("Evaluation Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.write(f"**Documents loaded:** {summary['documents_loaded']}")

    with summary_col2:
        st.write(f"**Chunks created:** {summary['chunks_created']}")

    with summary_col3:
        st.write(f"**Questions evaluated:** {summary['questions_evaluated']}")

    st.markdown("---")

    st.subheader("Failed / Needs Review Questions")

    failed_df = results_df[results_df["Hit@1"] == 0]

    if failed_df.empty:
        st.success("No Hit@1 failures found.")
    else:
        st.warning(f"{len(failed_df)} question(s) did not retrieve the expected source at rank 1.")
        st.dataframe(
            failed_df[
                [
                    "Question #",
                    "Question",
                    "Expected Source(s)",
                    "Top Retrieved Source",
                    f"Hit@{top_k}",
                    "MRR",
                ]
            ],
            use_container_width=True,
        )

    st.markdown("---")

    st.subheader("Document Coverage")

    coverage_df = get_document_coverage()

    st.write(
        "This shows how many evaluation questions are assigned to each expected source document."
    )

    st.dataframe(
        coverage_df,
        use_container_width=True,
    )

    st.markdown("---")

    st.subheader("Full Evaluation Results")

    st.dataframe(
        results_df,
        use_container_width=True,
    )

    st.markdown("---")

    st.subheader("Question-Level Details")

    for _, row in results_df.iterrows():
        with st.expander(f"Question {row['Question #']}: {row['Question']}"):
            st.write(f"**Expected Source(s):** {row['Expected Source(s)']}")
            st.write(f"**Top Retrieved Source:** {row['Top Retrieved Source']}")
            st.write(f"**Top {top_k} Retrieved Sources:** {row['Top 3 Retrieved Sources']}")
            st.write(f"**Hit@1:** {row['Hit@1']}")
            st.write(f"**Hit@{top_k}:** {row[f'Hit@{top_k}']}")
            st.write(f"**Recall@{top_k}:** {row[f'Recall@{top_k}']:.2f}")
            st.write(f"**MRR:** {row['MRR']:.2f}")
            st.write(f"**Status:** {row['Status']}")


if __name__ == "__main__":
    main()