import time
from enum import Enum
from functools import wraps
from typing import *

import numpy as np
from sentence_transformers import SentenceTransformer

from documents import document


class SimMetric(Enum):
    EUCLIDEAN = 0
    COSINE = 1


def timeit(func):
    @wraps(func)
    def inner(*args, **kwargs):
        t_start = time.time()
        res = func(*args, **kwargs)
        t_exec = time.time() - t_start
        return res, t_exec * 1000
    return inner


class Vectorstore:
    """
    a lightweight vectorstore built with numpy
    """
    def __init__(
            self, 
            docs: List[str], 
            embedder: SentenceTransformer = None,
            similarity_metric: SimMetric = SimMetric.EUCLIDEAN
        ) -> None:
        self.docs = np.array(docs)
        self.embedder = None or embedder

        # main store
        self._store: np.ndarray = None
        self.similarity_metric = similarity_metric
        self._set_sim_func()

    def set_metric(self, metric: SimMetric):
        assert isinstance(metric, SimMetric)
        self.similarity_metric = metric
        self._sim_func = self._set_sim_func()

    def _set_sim_func(self):
        if self.similarity_metric == SimMetric.EUCLIDEAN:
            self._sim_func = self._dist_euclidean__
        elif self.similarity_metric == SimMetric.COSINE:
            self._sim_func = self._cosine__
        else:
            NotImplementedError(f"Similarity function for {self.similarity_metric} is not implemented.")

    @classmethod
    def from_docs(
        cls, 
        docs: List[str], 
        embedder: SentenceTransformer = None, 
        similarity_metric = SimMetric.EUCLIDEAN
    ) -> "Vectorstore":
        store = cls(docs, embedder=embedder, similarity_metric=similarity_metric)
        print(f"Using similarity metric: {similarity_metric}")
        return store.build_store()

    def build_store(self):
        """
        use this to embed the documents and build a store
        """
        if self.embedder is not None:
            self._store = self.embedder.encode(self.docs)

        return self
    
    @timeit
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
    
    def _cosine__(self, query: np.ndarray):
        """
        calculates the cosine similarity
        """
        assert query.ndim == 1
        assert query.shape[0] == self._store.shape[1], f"Shape mismatch between query and store: {query.shape}, {self._store.shape}"

        norm_a = np.linalg.norm(self._store, axis=1)
        norm_b = np.linalg.norm(query)

        similarity = np.dot(self._store, query) / norm_a * norm_b
        return similarity
    
    def _get_topk_similar(self, query: np.ndarray, k: int = 5):
        """
        given a distance matrix, get top k similar indices from docs
        also return distance scores
        """
        reverse = False
        if self.similarity_metric == SimMetric.COSINE:
            reverse = True
        
        arr = self._sim_func(query)
        sorted_indices = np.argsort(arr)
        top_k_indices = sorted_indices[ :k] if not reverse else sorted_indices[::-1][:k]

        topk_docs = self.docs[top_k_indices]
        topk_dist = arr[top_k_indices]

        assert topk_docs.ndim == 1 and topk_docs.shape[0] == k
        assert topk_dist.ndim == 1

        return list(topk_docs), topk_dist

    def __repr__(self) -> str:
        return f"Vectorstore(embedder = {self.embedder})"
    

# --------- Usage -----------

# example embedder
print(f"Loading the embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

docs = document.split('\n')
print(f"Building vectorstore for {len(docs)} documents...")

vs = Vectorstore.from_docs(docs, embedder=model, similarity_metric=SimMetric.COSINE)

query = "What did emma do in this story?"
result, exectime = vs.search(query, k=2)

print(f"\nMost similar documents: {result[0]}")
print(f"Scores w.r.t query (lower is better): {list(result[1])}")
print(f"\nSearch time: {exectime} ms")
