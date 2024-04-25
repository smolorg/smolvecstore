from typing import *

import numpy as np
from sentence_transformers import SentenceTransformer


class Vectorstore:
    """
    a lightweight vectorstore built with numpy
    """
    def __init__(self, docs: List[str], embedder: SentenceTransformer = None) -> None:
        self.docs = np.array(docs)
        self.embedder = None or embedder

        # main store
        self._store: np.ndarray = None

    @classmethod
    def from_docs(cls, docs: List[str], embedder: SentenceTransformer = None) -> "Vectorstore":
        store = cls(docs, embedder=embedder)
        return store.build_store()

    def build_store(self):
        """
        use this to embed the documents and build a store
        """
        if self.embedder is not None:
            self._store = self.embedder.encode(self.docs)

        return self
    
    def search(self, query: str, k: int = 5) -> tuple:
        """
        get top K similar documents and their scores, semantically similar to query.
        the lower score, the better.
        """

        assert self.embedder is not None
        assert k >= 1

        q_emb = self.embedder.encode(query)
        assert q_emb.ndim == 1

        return self._get_topk_similar(q_emb, k=k)
    
    def _dist_euclidean__(self, query: np.ndarray):
        """
        calculates the distance between all vectors from the store and query
        """
        assert query.ndim == 1
        assert query.shape[0] == self._store.shape[1], f"Shape mismatch between query and store: {query.shape}, {self._store.shape}"

        # final vector
        # dist_squared = (x1 - x2) ** 2 + (y1 - y2) ** 2 + ...
        # sum along columns for distance

        dist: np.ndarray = np.sqrt((self._store - query) ** 2).sum(axis=1)
        return dist
    
    def _get_topk_similar(self, query: np.ndarray, k: int = 5):
        """
        given a distance matrix, get top k similar indices from docs
        also return distance scores
        """
        dist = self._dist_euclidean__(query)

        sorted_indices = np.argsort(dist)
        top_k_indices = sorted_indices[ :k]

        topk_docs = self.docs[top_k_indices]
        topk_dist = dist[top_k_indices]

        assert topk_docs.ndim == 1 and topk_docs.shape[0] == k
        assert topk_dist.ndim == 1

        return list(topk_docs), topk_dist

    def __repr__(self) -> str:
        return f"Vectorstore(embedder = {self.embedder})"
    

# --------- Usage -----------

# example embedder
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

docs = [
    "Super mario is a nice video game.", 
    "The USA election are on the way!",
    "A video game is fun to play with friends.",
    "What if the earth was covered with plasma instead of water?"
]

vs = Vectorstore.from_docs(docs, embedder=model)

query = "which is a nice game you can think of?"
similar_docs, scores = vs.search(query, k=2)

print(f"Most similar documents: {similar_docs}")
print(f"Scores w.r.t query (lower is better): {list(scores)}")
