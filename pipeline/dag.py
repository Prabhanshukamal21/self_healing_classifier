"""from langgraph.graph import StateGraph
from pipeline.nodes import inference_node, confidence_check_node, fallback_node

def build_graph():
    graph = StateGraph()
    graph.add_node("inference", inference_node)
    graph.add_node("check", confidence_check_node)
    graph.add_node("fallback", fallback_node)

    graph.set_entry_point("inference")
    graph.add_edge("inference", "check")
    graph.add_conditional_edges("check", {
        "accept": None,  # end
        "fallback": "fallback"
    })
    graph.add_edge("fallback", None)  # end

    return graph.compile()
"""

"""from langgraph.graph import StateGraph
from pipeline.nodes import inference_node, confidence_check_node, fallback_node

def build_graph():
    g = StateGraph(state_schema=GraphState)
    g.add_node("inference", inference_node)
    g.add_node("check", confidence_check_node)
    g.add_node("fallback", fallback_node)
    g.set_entry_point("inference")
    g.add_edge("inference", "check")
    g.add_conditional_edges("check", {"accept": None, "fallback": "fallback"})
    g.add_edge("fallback", None)
    return g.compile()
"""

# pipeline/dag.py
from langgraph.graph import StateGraph
from pydantic import BaseModel
from pipeline.nodes import inference_node, confidence_check_node, fallback_node

# Define the state schema for LangGraph
class GraphState(BaseModel):
    input_text: str
    prediction: str | None = None
    confidence: float | None = None
    fallback_used: bool = False

# Build the LangGraph DAG
def build_graph():
    g = StateGraph(state_schema=GraphState)

    # Define nodes
    g.add_node("inference", inference_node)
    g.add_node("confidence_check", confidence_check_node)
    g.add_node("fallback", fallback_node)
    g.add_node("end", lambda state: state)  # ✅ Add an explicit 'end' node

    # Set up edges and logic
    g.set_entry_point("inference")
    g.add_edge("inference", "confidence_check")

    g.add_conditional_edges(
        "confidence_check",
        lambda state: "fallback" if state.confidence is not None and state.confidence < 0.7 else "end",
        {
            "fallback": "fallback",
            "end": "end"
        }
    )

    g.add_edge("fallback", "end")  # ✅ Make sure fallback leads to end

    return g.compile()


# Optional test
if __name__ == "__main__":
    graph = build_graph()
    result = graph.invoke({"input_text": "The movie was surprisingly good!"})
    print(result)
