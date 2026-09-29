from sentence_transformers import SentenceTransformer
from src.vector_store import VectorStore


MODEL_NAME = "all-MiniLM-L6-v2"


def retrieve(query, chunks, top_k=3):

    model = SentenceTransformer(MODEL_NAME)

    # Create document embeddings
    document_embeddings = model.encode(
        chunks,
        convert_to_numpy=True
    )

    # Create query embedding
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )[0]

    # Create vector store
    vector_store = VectorStore()

    vector_store.add(
        document_embeddings,
        chunks
    )

    # Search
    results = vector_store.search(
        query_embedding,
        top_k=top_k
    )

    return results


if __name__ == "__main__":

    chunks = [
        "Supervised learning uses labeled training data.",
        "Unsupervised learning works with data without labels.",
        "Classification predicts categories.",
        "Regression predicts numerical values.",
        "Overfitting happens when a model learns training data too closely."
    ]

    query = "What type of learning uses labeled data?"

    results = retrieve(
        query,
        chunks,
        top_k=3
    )

    for result in results:

        print("\nScore:", result["score"])
        print("Chunk:", result["document"])