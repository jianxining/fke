from __future__ import annotations

from typing import Any, Dict, TypedDict

from feature_keyword_extractor.pipeline import ExtractionPipeline


class PipelineState(TypedDict, total=False):
    input_dir: Any
    output_dir: Any
    result: Dict[str, dict]


def build_langgraph_pipeline(pipeline: ExtractionPipeline):
    """Build a LangGraph wrapper around the deterministic extraction pipeline."""
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise RuntimeError("Missing langgraph. Install requirements.txt before building the agent graph.") from exc

    graph = StateGraph(PipelineState)

    def run_pipeline(state: PipelineState) -> PipelineState:
        result = pipeline.run(state["input_dir"], state["output_dir"])
        return {**state, "result": result}

    graph.add_node("run_extraction_pipeline", run_pipeline)
    graph.set_entry_point("run_extraction_pipeline")
    graph.add_edge("run_extraction_pipeline", END)
    return graph.compile()
