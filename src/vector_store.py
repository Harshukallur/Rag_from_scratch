import numpy as np


class VectorStore:

    def __init__(self):
        self.embeddings = None
        self.documents = []

    def add(self, embeddings, documents):

        self.embeddings = embeddings
        self.documents = documents

    def search(self, query_embedding, top_k=3):

        # Normalize vectors
        query_norm = np.linalg.norm(query_embedding)

        document_norms = np.linalg.norm(
            self.embeddings,
            axis=1
        )

        # Cosine similarity
        similarities = np.dot(
            self.embeddings,
            query_embedding
        ) / (
            document_norms * query_norm
        )

        # Highest similarity first
        top_indices = np.argsort(
            similarities
        )[::-1][:top_k]

        results = []

        for index in top_indices:

            results.append({
                "document": self.documents[index],
                "score": float(similarities[index])
            })

        return results