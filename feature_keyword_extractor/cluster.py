from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List

import jieba
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
        similarity_threshold: float = 0.3,
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
        groups = self.cluster_texts(corpus)
        clusters: List[Cluster] = []
        for group_indexes in groups:
            items = [headings[i] for i in group_indexes]
            clusters.append(Cluster(cluster_id=self._cluster_id(items), items=items))
        return clusters

    def cluster_texts(self, texts: List[str]) -> List[List[int]]:
        if not texts:
            return []
        if len(texts) == 1:
            return [[0]]

        similarity = self._similarity_matrix(texts)
        if similarity is None:
            return [[i] for i in range(len(texts))]

        visited = set()
        groups: List[List[int]] = []
        for idx in range(len(texts)):
            if idx in visited:
                continue
            group = {idx}
            frontier = [idx]
            visited.add(idx)
            while frontier:
                current = frontier.pop()
                for other, score in enumerate(similarity[current]):
                    if other not in visited and score >= self.similarity_threshold:
                        visited.add(other)
                        group.add(other)
                        frontier.append(other)
            groups.append(sorted(group))
        return groups

    @staticmethod
    def _heading_text(node: HeadingNode) -> str:
        return " ".join([*node.ancestor_titles, node.title])

    def _similarity_matrix(self, corpus: List[str]):
        if self.backend == "embedding":
            embedding_similarity = self._embedding_similarity(corpus)
            if embedding_similarity is not None:
                return embedding_similarity
        try:
            segmented = self._segment(corpus)
            vectors = TfidfVectorizer().fit_transform(segmented)
            return cosine_similarity(vectors)
        except ValueError:
            return None

    def _segment(self, corpus: List[str]) -> List[str]:
        try:
            import jieba
            return [" ".join(jieba.cut(text)) for text in corpus]
        except ImportError:
            return corpus

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
