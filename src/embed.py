from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)


def create_embeddings(model, texts):

    embeddings = model.encode(
        texts,
        convert_to_numpy=True
    )

    return embeddings


if __name__ == "__main__":

    model = load_embedding_model()

    texts = [
        "Machine learning uses data to learn patterns.",
        "Supervised learning uses labeled examples.",
        "Pizza is a popular food."
    ]

    embeddings = create_embeddings(model, texts)

    print("Embedding shape:")
    print(embeddings.shape)

    print("\nFirst embedding:")
    print(embeddings[0])