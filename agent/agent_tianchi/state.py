# state.py
from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict, total=False):
    question: str

    # parsed narrative
    parsed: Dict[str, Any]

    # plans: list of path (each path is list of step strings)
    plans: List[List[str]]
    plan_index: int

    # per-path results
    answers: List[str]
    evidences: List[Dict[str, Any]]

    # optional: per-path extracted variables
    vars: Dict[str, Any]

    # control / stats
    loop_count: int
    no_new_info: int
    final_answer: str
