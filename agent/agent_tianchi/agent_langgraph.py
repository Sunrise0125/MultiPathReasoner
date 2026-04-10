# agent_langgraph.py

import json
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

from llm import LLMWrapper
from tools import search_serper

llm = LLMWrapper()

MAX_RETRY = 2
CONFIDENCE_THRESHOLD = 0.75


# ================================
# State Definition
# ================================

class AgentState(TypedDict):
    question: str
    
    parsed: Dict[str, Any]
    plans: List[List[str]]
    
    plan_index: int
    step_index: int
    
    current_step: str
    
    variables: Dict[str, Any]
    var_confidence: Dict[str, float]
    
    evidences: List[Dict]
    graded_evidences: List[Dict]
    
    answer: str
    
    retry_count: int
    loop_count: int
    
    status: str


# ================================
# 1️⃣ Parse Node
# ================================

def parse_node(state: AgentState):
    prompt = f"""
解析问题结构：
{state['question']}

输出 JSON:
{{
  "events": [],
  "final_target": ""
}}
"""
    res = llm.safe_invoke(prompt)
    try:
        parsed = json.loads(res)
    except:
        parsed = {"events": [state["question"]], "final_target": ""}
    
    return {"parsed": parsed}


# ================================
# 2️⃣ Plan Node
# ================================

def plan_node(state: AgentState):
    events = state["parsed"]["events"]
    plans = [events]  # 简化版
    
    return {
        "plans": plans,
        "plan_index": 0,
        "step_index": 0,
        "variables": {},
        "var_confidence": {},
        "retry_count": 0
    }


# ================================
# 3️⃣ Select Step
# ================================

def select_step(state: AgentState):
    plan = state["plans"][state["plan_index"]]
    step = plan[state["step_index"]]
    
    return {"current_step": step}


# ================================
# 4️⃣ Adaptive Query Refiner
# ================================

def rewrite_query_node(state: AgentState):
    step = state["current_step"]
    vars_json = json.dumps(state["variables"], ensure_ascii=False)
    
    prompt = f"""
为步骤生成搜索词：
步骤: {step}
变量: {vars_json}

输出 JSON:
{{"queries": ["q1","q2","q3"]}}
"""
    res = llm.safe_invoke(prompt)
    try:
        queries = json.loads(res)["queries"]
    except:
        queries = [step]
    
    return {"queries": queries}


# ================================
# 5️⃣ Search Node
# ================================

def search_node(state: AgentState):
    evidences = []
    
    for q in state["queries"]:
        snippet = search_serper(q)
        if snippet:
            evidences.append({
                "query": q,
                "snippet": snippet
            })
    
    return {"evidences": evidences}


# ================================
# 6️⃣ Evidence Grader ⭐
# ================================

def grade_node(state: AgentState):
    graded = []
    
    for ev in state["evidences"]:
        prompt = f"""
证据：
{ev['snippet']}

问题：
{state['question']}

评分标准：
- 相关性 (0-1)
- 信息完整度 (0-1)
- 可信度 (0-1)

输出 JSON:
{{"relevance":0.8,"completeness":0.7,"credibility":0.9}}
"""
        res = llm.safe_invoke(prompt)
        try:
            score = json.loads(res)
            final_score = (
                score["relevance"] * 0.5 +
                score["completeness"] * 0.3 +
                score["credibility"] * 0.2
            )
        except:
            final_score = 0.3
        
        ev["score"] = final_score
        graded.append(ev)
    
    graded.sort(key=lambda x: x["score"], reverse=True)
    
    return {"graded_evidences": graded}


# ================================
# 7️⃣ Variable Extractor ⭐
# ================================

def extract_node(state: AgentState):
    if not state["graded_evidences"]:
        return {}
    
    best = state["graded_evidences"][0]
    
    prompt = f"""
从证据中提取变量：
{best['snippet']}

输出 JSON:
{{"new_variables": {{"year":"1901"}}}}
"""
    res = llm.safe_invoke(prompt)
    try:
        new_vars = json.loads(res)["new_variables"]
    except:
        new_vars = {}
    
    return {"new_variables": new_vars}


# ================================
# 8️⃣ Variable Confidence Tracker ⭐
# ================================

def update_confidence_node(state: AgentState):
    variables = dict(state["variables"])
    confidence = dict(state["var_confidence"])
    
    new_vars = state.get("new_variables", {})
    
    for k, v in new_vars.items():
        old_conf = confidence.get(k, 0.5)
        
        if k in variables and variables[k] == v:
            confidence[k] = min(1.0, old_conf + 0.2)
        else:
            variables[k] = v
            confidence[k] = 0.6
    
    return {
        "variables": variables,
        "var_confidence": confidence
    }


# ================================
# 9️⃣ Decision Node ⭐⭐
# ================================

def decide_node(state: AgentState):
    confidences = state["var_confidence"].values()
    
    if confidences and min(confidences) > CONFIDENCE_THRESHOLD:
        return {"status": "finish"}
    
    if state["retry_count"] < MAX_RETRY:
        return {"status": "refine"}
    
    # 下一步
    return {"status": "next"}


# ================================
# 🔟 Step Advance
# ================================

def next_step_node(state: AgentState):
    return {
        "step_index": state["step_index"] + 1,
        "retry_count": 0
    }


# ================================
# 1️⃣1️⃣ Finish Node
# ================================

def finish_node(state: AgentState):
    prompt = f"""
问题：
{state['question']}

变量：
{json.dumps(state['variables'],ensure_ascii=False)}

给出最终答案
"""
    ans = llm.safe_invoke(prompt)
    
    return {"answer": ans}


# ================================
# Build Graph
# ================================

def build_graph():
    graph = StateGraph(AgentState)
    
    graph.add_node("parse", parse_node)
    graph.add_node("plan", plan_node)
    graph.add_node("select_step", select_step)
    graph.add_node("rewrite", rewrite_query_node)
    graph.add_node("search", search_node)
    graph.add_node("grade", grade_node)
    graph.add_node("extract", extract_node)
    graph.add_node("update_conf", update_confidence_node)
    graph.add_node("decide", decide_node)
    graph.add_node("next", next_step_node)
    graph.add_node("finish", finish_node)
    
    graph.set_entry_point("parse")
    
    graph.add_edge("parse", "plan")
    graph.add_edge("plan", "select_step")
    graph.add_edge("select_step", "rewrite")
    graph.add_edge("rewrite", "search")
    graph.add_edge("search", "grade")
    graph.add_edge("grade", "extrac+t")
    graph.add_edge("extract", "update_conf")
    graph.add_edge("update_conf", "decide")
    
    graph.add_conditional_edges(
        "decide",
        {
            "refine": "rewrite",
            "next": "next",
            "finish": "finish",
        },
        lambda state: state["status"]
    )
    
    graph.add_edge("next", "select_step")
    graph.add_edge("finish", END)
    
    return graph.compile()