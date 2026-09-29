from src.extract import extract_text_from_pdf
from src.clean import clean_text
from src.chunk import create_chunks
from src.retrieve import retrieve
from src.generate import generate_answer


PDF_PATH = "data/rag_sample.pdf"


def build_document():

    # ---------------------------
    # 1. Extract
    # ---------------------------

    pages = extract_text_from_pdf(PDF_PATH)

    # ---------------------------
    # 2. Combine pages
    # ---------------------------

    text = "\n".join(
        page["text"]
        for page in pages
    )

    # ---------------------------
    # 3. Clean
    # ---------------------------

    text = clean_text(text)

    # ---------------------------
    # 4. Chunk
    # ---------------------------

    chunks = create_chunks(
        text,
        chunk_size=80,
        overlap=20
    )

    return chunks


def main():

    print("Building RAG system...\n")

    chunks = build_document()

    print(f"Created {len(chunks)} chunks.")

    while True:

        question = input(
            "\nAsk a question "
            "(type 'exit' to stop): "
        )

        if question.lower() == "exit":
            break

        # ---------------------------
        # Retrieval
        # ---------------------------

        results = retrieve(
            question,
            chunks,
            top_k=3
        )

        print("\nRetrieved chunks:")

        for i, result in enumerate(results):

            print(
                f"\n{i + 1}. "
                f"Score: {result['score']:.4f}"
            )

            print(result["document"])

        # ---------------------------
        # Generation
        # ---------------------------

        answer = generate_answer(
            question,
            results
        )

        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main()