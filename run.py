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
    parser.add_argument("--input", required=True, type=Path, help="Input directory containing .docx/.md files.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for CSV/JSON/audit files.")
    parser.add_argument("--keyword-top-k", type=int, default=10)
    parser.add_argument("--merge-threshold", type=float, default=0.45)
    parser.add_argument(
        "--cluster-backend",
        choices=["tfidf", "embedding"],
        default="tfidf",
        help="Heading clustering backend. Use embedding only when local model download is acceptable.",
    )
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    args = parser.parse_args()

    feature_agent = FeatureExtractionAgent()
    pipeline = ExtractionPipeline(
        feature_agent=feature_agent,
        clusterer=Clusterer(
            similarity_threshold=args.merge_threshold,
            backend=args.cluster_backend,
            embedding_model=args.embedding_model,
        ),
        keyword_extractor=KeywordExtractor(top_k=args.keyword_top_k),
        merge_agent=MergeAgent(feature_agent.llm),
    )
    pipeline.run(args.input, args.output)


if __name__ == "__main__":
    main()
