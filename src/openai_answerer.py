import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def build_context(results: list[dict]) -> str:
    """
    Turn retrieved chunks into a clear context block for the AI model.
    """
    context_parts = []

    for index, result in enumerate(results, start=1):
        source = result["source"]
        text = result["text"]

        context_parts.append(
            f"[Source {index}: {source}]\n{text}"
        )

    return "\n\n".join(context_parts)


def generate_openai_answer(question: str, results: list[dict]) -> dict:
    """
    Generate a natural-language answer using OpenAI,
    grounded only in retrieved company document chunks.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        raise ValueError(
            "Missing OPENAI_API_KEY. Add it to your .env file."
        )

    if not results:
        return {
            "answer": "I could not find this in the available company documents.",
            "sources": []
        }

    context = build_context(results)

    client = OpenAI(api_key=api_key)

    instructions = """
You are a helpful company knowledge assistant.

You answer employee questions using ONLY the provided company document context.

Rules:
1. Do not use outside knowledge.
2. Do not make assumptions.
3. If the answer is not in the context, say:
   "I could not find this in the available company documents."
4. Keep the answer clear and concise.
5. Mention the relevant source document names at the end.
"""

    prompt = f"""
Company document context:

{context}

User question:
{question}

Answer:
"""

    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=prompt
    )

    sources = list({result["source"] for result in results})

    return {
        "answer": response.output_text,
        "sources": sources
    }