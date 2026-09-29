def create_chunks(text, chunk_size=80, overlap=20):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":

    text = """
    Machine learning is a branch of artificial intelligence that enables
    computers to learn patterns from data and make predictions.

    Supervised learning uses labeled training data. Classification predicts
    categories while regression predicts numerical values.

    Unsupervised learning works with data without target labels.
    Clustering groups similar data points together.
    """

    chunks = create_chunks(
        text,
        chunk_size=30,
        overlap=5
    )

    for i, chunk in enumerate(chunks):

        print(f"\nCHUNK {i}")
        print(chunk)