# agent.py
import json
import os
from typing import List, Dict, Any
from llm import LLMWrapper
from tools import search_serper
from utils import normalize_query, safe_text, token_set_similarity
from state import AgentState
import random
import math

# config
MAX_LOOPS = int(os.getenv("MAX_LOOPS", "6"))
MAX_STEPS_PER_PATH = 10
TOP_K_PER_STEP = 2
GLOBAL_EVIDENCE_LIMIT = 10
SIMILARITY_THRESHOLD = 0.7  # token-set similarity threshold for query-level dedup

llm = LLMWrapper()

# ---------------------------
# Var helpers (NEW)
# ---------------------------
# ---------------------------
# Var helpers (MULTI-VALUE VERSION)
# ---------------------------

def priority_of(source: str) -> int:
    return {"question": 3, "search": 2, "inference": 1}.get(source, 0)


def is_packed_var(x: Any) -> bool:
    return isinstance(x, dict) and "candidates" in x


def pack_candidate(value: Any, source: str, confidence: float = 0.5, why: str = "") -> Dict[str, Any]:
    return {
        "value": value,
        "source": source,
        "priority": priority_of(source),
        "confidence": float(confidence),
        "why": why
    }


def pack_var(value: Any, source: str, confidence: float = 0.5, why: str = "") -> Dict[str, Any]:
    return {
        "candidates": [
            pack_candidate(value, source, confidence, why)
        ]
    }


def get_best_candidate(var_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Return highest-confidence candidate."""
    if not is_packed_var(var_obj):
        return {"value": var_obj}

    return max(
        var_obj["candidates"],
        key=lambda c: (c["confidence"], c["priority"])
    )


def get_value(x: Any) -> Any:
    """Return best candidate value for prompting."""
    if is_packed_var(x):
        best = get_best_candidate(x)
        return best.get("value")
    return x


def flatten_vars(vars_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {k: get_value(v) for k, v in (vars_dict or {}).items()}

def merge_var(
    extracted: Dict[str, Any],
    key: str,
    new_value: Any,
    new_source: str,
    confidence: float = 0.5,
    why: str = ""
) -> None:

    new_candidate = pack_candidate(new_value, new_source, confidence, why)

    if key not in extracted:
        extracted[key] = {"candidates": [new_candidate]}
        return

    var_obj = extracted[key]

    if not is_packed_var(var_obj):
        # upgrade legacy format
        extracted[key] = pack_var(var_obj, "search", 0.5)
        var_obj = extracted[key]

    candidates = var_obj["candidates"]

    # Check if same value already exists
    for c in candidates:
        if str(c["value"]) == str(new_value):
            # Increase confidence slightly if seen again
            c["confidence"] = min(1.0, c["confidence"] + 0.1)
            return

    # If new value conflicts with existing ones
    # Do NOT delete old — just append
    candidates.append(new_candidate)

    # Optional: normalize confidence if too many
    total = sum(c["confidence"] for c in candidates)
    if total > 0:
        for c in candidates:
            c["confidence"] = c["confidence"] / total

# ---------------------------
# 1) Narrative Parser
# ---------------------------
def narrative_parse(question: str) -> Dict[str, Any]:
    prompt = f"""
请把下面的题目解析成结构化的叙事信息。

题目：
{question}

输出一个 JSON 对象，包含：

- events: 事件链数组，按题中出现顺序，对于代词和模糊描述，尽量还原成具体的事件描述（比如 "该船长" 就改成 "被处决的船长"）
关于事件时间等事实，从题干出发，不要凭空添加，不要臆测,所有时间表达必须严格保持题干原文粒度,不得将模糊时间具体化
如果题目中事件之间有时间或条件关系，也请在事件描述中体现（比如 "同年" 就改成 "1721年"）。如果题目中没有明确的事件链，也请尽量从题目中提取出2-3个关键事件或条件，作为后续检索的线索。

- final_target: 最终要找的实体类型（如 "出版公司名称"）

只返回JSON，不要多余说明。
"""
    res = llm.safe_invoke(prompt)
    try:
        parsed = json.loads(res)
    except:
        parsed = {"events": [question], "final_target": ""}
    print("narrative_parse:", parsed)
    return parsed

# ---------------------------
# 2) Multi-Plan Generator
# ---------------------------
def generate_plans(parsed: Dict[str, Any], question: str, n_plans: int = 3) -> List[List[str]]:
    prompt = f"""
    
题目原文：
{question}
基于下面的结构化信息，生成 {n_plans} 条不同的线性检索路径（每条路径最多 {MAX_STEPS_PER_PATH} 步）.

后续将记录检索得到的信息，最终目的得到最后的{parsed.get("final_target", "")}。
检索路径的每步写成一句可用于搜索的自然语言短句（简洁）,用于补充信息。输出 JSON: {{ "plans": [ ["step1","step2"], ... ] }}
结构信息：
{json.dumps(parsed, ensure_ascii=False)}

要仔细识别时间，事件，实体的关系，不要搞混了。合理安排检索路径的步骤顺序和内容，每条路径要尽量不同，覆盖不同的检索思路和线索组合。每步都要尽量具体可操作，避免模糊的步骤（比如“了解背景”就太模糊了）。
"""
    res = llm.safe_invoke(prompt)
    try:
        out = json.loads(res)
        plans = out.get("plans", [])
        clean_plans = []
        for p in plans:
            if isinstance(p, list):
                clean = [str(s).strip() for s in p if str(s).strip()]
                if clean:
                    clean_plans.append(clean[:MAX_STEPS_PER_PATH])
        if not clean_plans:
            raise ValueError("empty")
    except Exception:
        events = parsed.get("events", [question])
        linear = [[f"查找：{e}" for e in events][:MAX_STEPS_PER_PATH]]
        clean_plans = [linear[0]]
        while len(clean_plans) < n_plans:
            clean_plans.append(linear[0])

    if len(clean_plans) < n_plans:
        while len(clean_plans) < n_plans:
            clean_plans.append(clean_plans[-1])

    print("generate_plans:", clean_plans[:n_plans])
    return clean_plans[:n_plans]

# ---------------------------
# 3) Query Rewriter (improve hit-rate)
# ---------------------------
def rewrite_query(question: str, step: str, state_vars: Dict[str, Any], history_context: List[str] = None) -> List[str]:
    """
    生成 3 个不同维度的搜索词：
    1. 强约束：包含所有已知变量（最精准）。
    2. 松弛约束：故意去掉年份/地点限制（防止上一步变量提取错误导致死胡同）。
    3. 关联探索：直接搜索步骤中两个实体的关系。
    """
    # IMPORTANT: flatten packed vars to avoid dict noise in prompt
    flat_vars = flatten_vars(state_vars)
    vars_json = json.dumps(flat_vars, ensure_ascii=False)

    prompt = f"""
你是一个搜索策略大师。请根据一直变量将思考步骤转化为 **2个策略不同** 的 Google 搜索queries。
【question】"{question}"
【步骤】"搜索{step}"
【已知变量】{vars_json}

请生成 JSON 格式的 "queries" 列表，包含以下两种策略：
1. **精准利用 (Strict)**: 使用所有相关变量（特别是年份、人名）。例如："pirate captain executed 1721"
2. **关系突破 (Relation)**: 直接搜两个实体的关系。例如："privateer captain executed same year Charles Messier born"

输出示例：
{{
    "queries": [
        "Charles Vane execution 1721",
        "pirate executed same year astronomer Messier born"
    ]
}}
"""
    res = llm.safe_invoke(prompt)

    queries = []
    try:
        clean_json = res.replace("```json", "").replace("```", "").strip()
        if clean_json.startswith("["):
            queries = json.loads(clean_json)
        else:
            data = json.loads(clean_json)
            queries = data.get("queries", [])
    except:
        queries = [step]

    unique_queries = []
    seen = set()
    for q in queries:
        q_str = str(q).strip()
        if q_str and q_str not in seen:
            unique_queries.append(q_str)
            seen.add(q_str)

    # print(f"🔀 Rewrite Strategies: {unique_queries}")
    return unique_queries[:3]

# ---------------------------
# 4) Path Executor (Sequential with Feedback Loop)
# ---------------------------
def execute_plan(plan: List[str], question: str, global_evidences: List[Dict], state_vars: Dict[str, Any]) -> Dict[str, Any]:
    path_evidences = []
    extracted = dict(state_vars) if state_vars else {}
    history_context = []
    context_for_answer = ""

    for step_idx, step in enumerate(plan[:MAX_STEPS_PER_PATH]):
        print(f"\n--- Step {step_idx + 1}: {step} ---")

        candidate_queries = rewrite_query(question,step, extracted, history_context)

        step_snippets = []

        for q in candidate_queries:
            q_norm = normalize_and_protect(q)
            if any(token_set_similarity(q_norm, ev.get("query", "")) >= SIMILARITY_THRESHOLD for ev in global_evidences):
                continue

            print(f"🔍 Searching: {q_norm}")
            snippet = search_serper(q_norm)

            if snippet :
                step_snippets.append(snippet)
                global_evidences.append({"query": q_norm, "snippet": snippet})
                path_evidences.append({"query": q_norm, "snippet": snippet, "facts": {}})

        if not step_snippets:
            print("❌ No results found.")
            continue

        combined_text = "\n\n".join(step_snippets)[:3500]

        flat_current = flatten_vars(extracted)
        current_vars_json = json.dumps(flat_current, ensure_ascii=False)
       

        # NOTE: prompt changed slightly to avoid "search overrides question"

        extract_prompt = f"""
你是一个多假设变量管理系统。

目标：从搜索结果中提取变量，并允许同一变量存在多个候选值（不同假设）。

--------------------------------------
【问题】
{question}

【当前任务】
{step}

【已知变量（可能包含多个候选值）】
{current_vars_json}

【搜索结果聚合】
{combined_text}
--------------------------------------

请执行以下逻辑：

==========================
1️⃣ 变量抽取（支持多值）
==========================

- 每个变量可以包含多个候选 value
- 每个 value 代表一个可能假设
- 不要因为冲突就删除旧值
- 只有在与“题干明确事实”冲突时才舍弃

==========================
2️⃣ 冲突策略（关键）
==========================

如果搜索结果与“之前提取的变量”冲突：

- 且不与题干冲突：
  ✅ 保留新变量
  ✅ 保留旧变量
  ✅ 根据“与题干条件的匹配程度”重新分配 confidence

confidence 分配原则：

- 更符合题干时间线 / 条件 / 逻辑链 → 提高 confidence
- 仅来自单一来源、缺少支持 → 降低 confidence
- 与当前任务强相关 → 提高 confidence

必须在 correction_note 中说明：
- 为什么保留多个假设
- 为什么调整置信度

==========================
3️⃣ source 规则
==========================

- "question" → 明确来自题干
- "search" → 明确来自搜索结果
- "inference" → 推断得出

==========================
4️⃣ 输出格式
==========================

输出 JSON：

{{
  "new_variables": {{
    "变量名": [
      {{
        "value": "候选值1",
        "source": "search | question | inference",
        "confidence": 0.0~1.0,
        "reasoning": "为何提取该值"}}
        ,
     {{
        "value": "候选值2",
        "source": "...",
        "confidence": ...,
        "reasoning": "..."
      }}
    ]
  }},
  "correction_note": "如果出现冲突或置信度调整，解释原因，否则为空字符串",
  "summary": "本步骤新增或更新的关键信息总结"
}}

注意：

- 如果没有新变量，new_variables返回空对象
- confidence 代表你当前的相对判断强度
- 不要删除旧变量（除非与题干明确冲突）
- 只返回 JSON
""" 
        extract_res = llm.safe_invoke(extract_prompt)

        try:
            clean_json = extract_res.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean_json)

            new_vars = parsed.get("new_variables", {}) or {}
            correction = parsed.get("correction_note", "") or ""
            summary = parsed.get("summary", "") or ""

            # ---- UPDATE VARS (CHANGED) ----
            if new_vars:
                print(f"✅ Extracted: {new_vars}")
                # Heuristic: treat step 1 extracted vars as "question anchors"
                # to avoid later web snippets overriding the main worldline.
                inferred_source = "question" if step_idx == 0 else "search"

                for k, v in new_vars.items():
                    why = summary or correction
                    merge_var(extracted, k, v, inferred_source, why=why)

            if correction:
                print(f"⚠️ Correction: {correction}")
                history_context.append(f"**修正**: {correction}")

            if summary:
                history_context.append(f"第{step_idx+1}步: {summary}")

            context_for_answer += combined_text + "\n"

        except Exception as e:
            print(f"⚠️ Extraction Error: {e}")

    # End Loop

    flat_extracted = flatten_vars(extracted)

    answer_prompt = f"""
基于检索路径回答问题。
问题：{question}
推理链：
{chr(10).join(history_context)}

提取变量：
{json.dumps(flat_extracted, ensure_ascii=False)}

请给出最终答案。按问题给出答案（像填空题一样不要包含其他东西）
返回 JSON: {{"answer": string}}
"""
    answer = llm.safe_invoke(answer_prompt).strip()
    print(f"🏁 Path Answer: {answer}")
    return {"answer": answer, "evidences": path_evidences, "extracted": extracted}

# small helper
def normalize_and_protect(q: str) -> str:
    q = (q or "").strip()
    if len(q) > 300:
        q = q[:300]
    return q

# ---------------------------
# 5) Evidence Scoring & Ranking
# ---------------------------
def score_evidences(evidences: List[Dict[str, Any]], question: str) -> List[Dict[str, Any]]:
    scored = []
    for ev in evidences:
        snippet = ev.get("snippet", "")
        sim = token_set_similarity(snippet, question)
        score = sim * 7 + min(len(snippet) / 400, 3)
        score = max(0.0, min(score, 10.0))
        ev_copy = dict(ev)
        ev_copy["score"] = round(score, 3)
        scored.append(ev_copy)
    scored = sorted(scored, key=lambda x: x["score"], reverse=True)
    return scored[:GLOBAL_EVIDENCE_LIMIT]

def extract_final_answer(answer: str) -> str:
    for prefix in ["最终答案是:", "因此，答案是:", "所以答案是:"]:
        if prefix in answer:
            parts = answer.split(prefix, 1)
            if len(parts) > 1:
                return parts[1].strip()
    return answer.strip()

# ---------------------------
# 6) Self-Consistency Voting + Final Verifier
# ---------------------------
def aggregate_answers(answers: List[str], evidences_by_path: List[List[Dict[str, Any]]], question: str) -> str:
    if not answers:
        return ""

    path_descriptions = []
    for idx, (ans, evs) in enumerate(zip(answers, evidences_by_path)):
        context = "\n".join([ev.get("snippet", "") for ev in evs[:5]])
        path_descriptions.append(f"""
路径编号：{idx}
候选答案：{ans}

证据片段：
{context}
""")

    judge_prompt = f"""
下面是针对同一个问题通过不同推理路径得到的候选答案及其证据。

问题：
{question}

{''.join(path_descriptions)}

请判断哪一个路径的证据最充分、逻辑最合理。
只输出最合理路径的编号（例如 0 或 1 或 2），不要输出其它内容。
"""
    judge_result = llm.safe_invoke(judge_prompt).strip()

    try:
        best_idx = int(judge_result)
    except:
        best_idx = 0

    if best_idx >= len(answers):
        best_idx = 0

    final = answers[best_idx].strip()

    best_evidences = evidences_by_path[best_idx]
    top_context = "\n".join([ev.get("snippet", "") for ev in best_evidences[:3]])

    verify_prompt = f"""
候选答案：
{final}

问题：
{question}

支持该答案的证据（片段）：
{top_context}

请简短回答：该答案是否被证据支持？回答 YES 或 NO，并给出一句简短理由。
"""
    vres = llm.safe_invoke(verify_prompt)

    extract_prompt = f"""根据下面的问题要求和答案，提取最终答案的核心部分，翻译转换语言，如果没有语言要求，答案和question严格用同语言
question为中文则答案用中文!!，question为英文则答案用英文!!，只需要提取关键名词（如果有多余的解释或不确定性，请去掉）
即便是人名，也根据question语言要求翻译成对应语言的人名（如果有对应的翻译）。如果答案里有不确定性描述（比如 "可能是"、"大概是"、"我觉得是"、"不太确定但可能是"等），请去掉这些描述，直接提取核心答案。
question: {question}
answer:{final}
请按如下 JSON 输出：
{{
    "final_answer": str
}}
如answer为"因此，该期刊的卷号为：**3**。"则final_answer为"3"
"""
   
    extracted = llm.safe_invoke(extract_prompt)

    if "YES" in vres.upper():
        return extracted

    return extracted + "（未经强证据确认）"

def normalize_final_answer(final):
    if isinstance(final, dict):
        return str(final.get("final_answer", "")).strip()

    if isinstance(final, str):
        text = final.strip()
        if text.startswith("{"):
            try:
                data = json.loads(text)
                return str(data.get("final_answer", "")).strip()
            except:
                pass
        text = text.split("（")[0].strip()
        return text

    return str(final).strip()

# ---------------------------
# 7) Top-level run function
# ---------------------------
def run_agent(question: str, n_paths: int = 3) -> Dict[str, Any]:
    parsed = narrative_parse(question)
    plans = generate_plans(parsed, question, n_plans=n_paths)

    all_evidences_by_path = []
    answers = []
    global_evidences = []

    for idx, plan in enumerate(plans):
        if idx >= n_paths:
            break 
        result = execute_plan(plan, question, global_evidences, state_vars= {})
        answers.append(result.get("answer", ""))
        all_evidences_by_path.append(result.get("evidences", []))

    final = aggregate_answers(answers, all_evidences_by_path, question)
    final_answer = normalize_final_answer(final)
    return final_answer


if __name__ == "__main__":
    test_question_0 = "An essay titled \"Letters to the Deaf,\" published in 1834, addresses societal challenges of deafness and recommends something that is discussed in an article that is part of a children's biography series. This series includes an illustrated biography about historical figures and has an associated application containing over 5,500 questions and 100 levels. The article was published in a journal that has volume number \"?\". What is the volume number of that journal?"

    test_question_1 = "在某一年，一位法国天文学家对一颗彗星的光谱进行了开创性观测，同年的一张太阳黑子照片后来在东亚某大都市的天文展览中展出。也正是在这一年，一位尚不满二十岁的南欧创业者，在家乡小镇创办了他的出版事业。十余年后，他将公司总部迁往了该国北部的商业中心。他所创立的这家出版公司的名字是什么？"

    test_question_2 = "在那位曾参与某次重大军事行动的军人中，他是其军事专业领域（Military Occupational Specialty）中第一位获得“炮兵军士长”（Master Gunnery Sergeant）军衔的西班牙裔人士。他隶属于这样一个美国军种：该军种在美国独立战争后被解散，并于1798年重新建立；曾在巴哈马群岛实施首次两栖突袭行动；向不到100人授予“荣誉海军陆战队员”（Honorary Marine）称号；并且有一种传统——为激励士兵，将下一军衔的徽章提前佩戴（pinning the next rank）。该重大军事行动的名称是?"

    test_question_3 = "一位物理学领域的学者为一种经典棋盘游戏设计的评分系统，后来被一家北美游戏公司广泛应用于其一款多人在线战术竞技游戏中。这家公司的母公司是一家亚洲科技巨头，该巨头在21世纪10年代完成了对前者的全资收购，并涉足量子计算等前沿科技领域。在这家北美公司开发的另一款第一人称射击游戏中，有一件适合近距离作战的武器，其名称与上述亚洲巨头代理发行的一款格斗手游中的一名在登场角色中年龄偏大的武术教官角色相同。这款格斗手游的名字是什么？"

    test_question_4 = "Which protein was identified as an interactor of PAD4 yet shows no evidence of interacting with ADF3 or contributing to powdery mildew defense or EHM targeting?"

    test_question_5 = "A European architect who played a key role in introducing a new architectural style to the United States in the early 20th century authored a book in the 1920s analyzing the development and potential of architecture and urban planning in America. In this book, he used a well-known hotel in a major Midwestern city as a primary example of tall building construction. What is the title of this book?"

    test_question_6 = "在21世纪10年代中期，某国从其西北内陆地区发射了一颗科学实验卫星，为一项前沿的远距离保密通信技术奠定了基础。八年后，该国在与此技术相关的下一代计算领域取得了多项突破，其中一个成果涉及超过五百个处理单元的系统。有一个专注于遗传性疾病研究的医学中心，其成立时间恰好是卫星发射的同一年，而它的一项大规模基因组数据计划的目标节点则定在八年后的这个年份。请问这个医学中心的名字是什么？"

    test_question_7 = "一位15世纪末为某欧洲王室服务的航海家发现了一座美洲岛屿。这座岛屿后来成为一名私掠船长的据点，他在18世纪20年代俘获了一艘满载贵金属的船只。该船长被处决的年份，也诞生了一位以其星云星团表闻名的法国天文学家。请问，是哪位作家，在上述船长被处决的几十年前以契约劳工的身份抵达该岛，后来作为外科医生记录了这些海上冒险者的事迹？"

    test_question_8 = "一家位于华中地区大型农业示范区的种子公司，曾卷入一场知识产权纠纷，该案件后来被某省级法院作为典型案例公布。这场纠纷的核心是一种在21世纪10年代末推出的作物新品种，其亲本是两个曾分别在不同区域广泛种植的品种。该新品种也曾被用于一项专利研究，以验证其对某种土壤胁迫的耐受性标记。此作物是一种主要的粮食作物，其不同播种季节的类型分化与某个基因的拷贝数变异有关，并且它的副产品可用于生产一种常见的增味剂。请问这是什么作物？"

    test_question_9 = "There is a well-known photography collective that was established in the early 1990s in a European country that had recently experienced major political changes. This group was inspired by an internationally renowned photographers' cooperative founded in the late 1940s by a French photographer famous for his candid and street photography. Who is the photographer who co-founded the younger collective in the early 1990s?"

    test_question_10 = "在一场持续了约十年的政治运动期间，一个东亚国家发生了一起以事发日期命名的突发事件，该事件被视为这场运动的转折点。事件发生的那一年，一款被某家未来软件巨头的创始人所使用的大型计算机在多所大学的实验室中得到应用。五年后，这场政治运动正式结束；同年，在另一个国家，一艘可重复使用的航天器因一部科幻作品而得名，其航天机构也为此首次划分出新的宇航员类别。请问这起以日期命名的突发事件是什么？"

    # result = run_agent(test_question_0, n_paths=1) #volume 3   错
    # print(result)
    # result = run_agent(test_question_1, n_paths=1) #阿诺尔多·蒙达多利出版社
    # print(result)
    result = run_agent(test_question_2, n_paths=1) #Operation Desert Shield and Desert Storm 沙漠之盾行动和沙漠风暴行动
    print(result)
    # result = run_agent(test_question_3, n_paths=1)#魂武者
    # print(result)
    # result = run_agent(test_question_4, n_paths=1)#HR4
    # print(result)
    # result = run_agent(test_question_5, n_paths=1)#Wie Baut Amerika?
    # print(result)
    # result = run_agent(test_question_6, n_paths=1)#华西罕见病医学中心
    # print(result)
    # result = run_agent(test_question_7, n_paths=1)#亚历山大·埃克斯梅林  √
    # print(result)
    # result = run_agent(test_question_8, n_paths=1)#小麦    ×
    # print(result)
    # result = run_agent(test_question_9, n_paths=1) #Harald Hauswald
    # print(result)
    # # result = run_agent(test_question_10, n_paths=1) #九一三   有问题
    # # print(result)

