import requests


OLLAMA_URL = "http://localhost:11434/api/generate"


def generate_answer(question, retrieved_chunks):

    context = "\n\n".join(
        [
            result["document"]
            for result in retrieved_chunks
        ]
    )

    prompt = f"""
You are a question-answering assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context,
say: "I don't know based on the provided document."

Context:
----------------
{context}
----------------

Question:
{question}

Answer:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )

    response.raise_for_status()

    data = response.json()

    return data["response"]