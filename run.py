from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from feature_keyword_extractor.agents.feature_agent import FeatureExtractionAgent
from feature_keyword_extractor.agents.merge_agent import MergeAgent
from feature_keyword_extractor.cluster import Clusterer
from feature_keyword_extractor.nlp.keyword_extractor import KeywordExtractor
from feature_keyword_extractor.pipeline import ExtractionPipeline


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Extract business capabilities and keywords from DOCX/Markdown docs.")
    parser.add_argument("--input", type=Path, help="Input directory containing .docx/.md files.",default="./input_docs")
    parser.add_argument("--output", type=Path, help="Output directory for CSV/JSON/audit files.",default="./outputs")
    parser.add_argument("--keyword-top-k", type=int, default=10)
    parser.add_argument("--merge-cluster-backend", choices=["none", "tfidf", "embedding"], default="none",
                        help="Merge clustering backend. 'none' uses SequenceMatcher pre-filter, 'tfidf'/'embedding' uses semantic clustering.")
    parser.add_argument("--merge-similarity-threshold", type=float, default=0.6,
                        help="Semantic similarity threshold for merge clustering (used with tfidf/embedding backend).")
    parser.add_argument("--merge-name-similarity", type=float, default=0.6,
                        help="SequenceMatcher pre-filter threshold (used when merge-cluster-backend=none).")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3",
                        help="Embedding model for merge clustering (used with embedding backend).")
    args = parser.parse_args()

    merge_clusterer = None
    if args.merge_cluster_backend != "none":
        merge_clusterer = Clusterer(
            similarity_threshold=args.merge_similarity_threshold,
            backend=args.merge_cluster_backend,
            embedding_model=args.embedding_model,
        )

    feature_agent = FeatureExtractionAgent()
    pipeline = ExtractionPipeline(
        feature_agent=feature_agent,
        keyword_extractor=KeywordExtractor(top_k=args.keyword_top_k),
        merge_agent=MergeAgent(feature_agent.llm),
        merge_name_similarity_threshold=args.merge_name_similarity,
        merge_clusterer=merge_clusterer,
    )
    pipeline.run(args.input, args.output)


if __name__ == "__main__":
    main()
