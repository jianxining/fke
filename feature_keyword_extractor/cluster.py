from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from feature_keyword_extractor.schemas import HeadingNode


@dataclass(frozen=True)
class Cluster:
    cluster_id: str
    items: List[HeadingNode]


class Clusterer:
    def __init__(
        self,
        similarity_threshold: float = 0.45,
        backend: str = "tfidf",
        embedding_model: str = "BAAI/bge-m3",
    ):
        self.similarity_threshold = similarity_threshold
        self.backend = backend
        self.embedding_model = embedding_model

    def cluster(self, headings: List[HeadingNode]) -> List[Cluster]:
        if not headings:
            return []
        if len(headings) == 1:
            return [Cluster(cluster_id=self._cluster_id([headings[0]]), items=headings)]

        corpus = [self._heading_text(node) for node in headings]
        similarity = self._similarity_matrix(corpus)
        if similarity is None:
            return [Cluster(cluster_id=self._cluster_id([node]), items=[node]) for node in headings]

        visited = set()
        clusters: List[Cluster] = []
        for idx, node in enumerate(headings):
            if idx in visited:
                continue
            group_indexes = {idx}
            frontier = [idx]
            visited.add(idx)
            while frontier:
                current = frontier.pop()
                for other, score in enumerate(similarity[current]):
                    if other not in visited and score >= self.similarity_threshold:
                        visited.add(other)
                        group_indexes.add(other)
                        frontier.append(other)
            items = [headings[item_idx] for item_idx in sorted(group_indexes)]
            clusters.append(Cluster(cluster_id=self._cluster_id(items), items=items))
        return clusters

    @staticmethod
    def _heading_text(node: HeadingNode) -> str:
        return " ".join([*node.ancestor_titles, node.title])

    def _similarity_matrix(self, corpus: List[str]):
        if self.backend == "embedding":
            embedding_similarity = self._embedding_similarity(corpus)
            if embedding_similarity is not None:
                return embedding_similarity
        try:
            vectors = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform(corpus)
            return cosine_similarity(vectors)
        except ValueError:
            return None

    def _embedding_similarity(self, corpus: List[str]):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return None
        try:
            model = SentenceTransformer(self.embedding_model)
            vectors = model.encode(corpus, normalize_embeddings=True)
            return cosine_similarity(vectors)
        except Exception:
            return None

    @staticmethod
    def _cluster_id(items: List[HeadingNode]) -> str:
        joined = "|".join(sorted(item.heading_id for item in items))
        digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:10]
        return f"C{digest}"
