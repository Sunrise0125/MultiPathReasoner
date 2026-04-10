# agent.py
import json
import os
import re
import threading
import concurrent.futures
from collections import Counter
from typing import List, Dict, Any
from llm import LLMWrapper
from tools import search_serper
from utils import normalize_query, safe_text, token_set_similarity

# config
N_PATHS = 3  # 执行路径数
MAX_STEPS_PER_PATH = 6
# GLOBAL_EVIDENCE_LIMIT = 15
# SIMILARITY_THRESHOLD = 0.7
MAX_REACT_TURNS = 6 # ReAct 框架最大反馈轮数

llm = LLMWrapper()

# ---------------------------
# 1) Graph Builder (生成 条件+目标 分离的 DAG)
# ---------------------------
def generate_dag(question: str) -> Dict[str, Any]:
    prompt = f"""
你是一个高级逻辑解析器和图计算架构师。你的任务是将复杂的多跳推理问题拆解为一个“约束满足图 (Constraint Satisfaction DAG)”，供搜索 Agent 按拓扑顺序执行。

【核心架构理念：条件与目标分离】
你必须将问题拆分为两个明确的部分：
1. **前置条件节点 (Condition Nodes)**：问题中提供的已知背景、中间实体、时间锚点等。这些节点负责找出确切的固定答案（如某家公司、某一年份、某个科学名词），作为后续推理的“硬约束”。
2. **最终目标节点 (Target Node)**：题目最终要求解答的那个实体。并且需要输出他所依赖的前置实体。

【通用拆解法则（必须严格遵守）】
1. **全局背景继承**：如果原题中包含明确的国家、地区（如“中国”、“某国”、“西北地区”）或时代背景，如果该信息在问题的后续文本中一直涉及，那么需要在 task_description 中明确保留，防止搜索范围漂移。
2. **静态计算前置**：如果题目涉及时间推演（如“成立的同一年”、“8年后”），必须在生成 DAG 时直接算出具体的年份（例如 2016+8=2024），并将明确的年份（2024）写进任务描述中。
3. **强制提取中间实体**：遇到“该岛屿”、“这家公司”等代词，必须设立独立的 Condition Node 查出具体专有名词，禁止跳步。
4. **单向约束注入**：遇到同名比对时，先提取列表，再在下一节点注入列表作为条件过滤。绝对禁止生成两个庞大的列表然后取交集！
【输入题目】
{question}

【输出格式】
请严格输出 JSON 格式，必须包含 `condition_nodes` 列表和 `target_node` 对象。
- "condition_nodes": 包含所有前置条件节点。每个 node 包含：
  - "node_id": 节点标识（如 "cond_1", "cond_2"）。
  - "dependencies": 依赖的前置节点 ID 列表。
  - "task_description": 具体的搜索任务（带有 `<前置节点_目标变量>` 的占位符）。
  - "target_variable": 需要提取的关键实体名称变量。
- "target_node": 最终要推导的目标节点。包含：
  - "node_id": "target_final"
  - "dependencies": 所有与其相关的 Condition Node ID 列表。
  - "task_description": 综合前置条件，描述最终需要寻找的实体。必须包含 `<前置变量>` 占位符。
  - "target_variable": 最终实体变量名。

请直接输出合法 JSON：
"""
    res = llm.safe_invoke(prompt)
    try:
        match = re.search(r'\{.*\}', res, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            print("[Graph Builder] DAG 成功生成!")
            return parsed
    except Exception as e:
        print(f"⚠️ DAG 生成失败: {e}")
    
    return {
        "condition_nodes": [],
        "target_node": {
            "node_id": "target_final",
            "dependencies": [],
            "task_description": f"直接搜索：{question}",
            "target_variable": "answer"
        }
    }

# ---------------------------
# 2) Query Rewriter (主要服务于 Condition Nodes)
# ---------------------------
def rewrite_query(task: str, global_vars: Dict[str, Any], path_idx: int, original_question: str) -> List[str]:
    vars_json = json.dumps(global_vars, ensure_ascii=False)

    prompt = f"""
You are an expert SEO search query generator for a Multi-Hop Reasoning AI. Translate the current task into **3 highly optimized search queries**.
This is execution path {path_idx}/{N_PATHS}. Try to use slightly different angles or synonyms to ensure diverse search coverage.

【Original Question】: "{original_question}"
【Current Task】: "{task}"
【Known Variables Context】: {vars_json}

CRITICAL RULES FOR MULTI-HOP REASONING:
1. **Language Strategy**: Autonomously determine the query language based on the target entity's cultural context. Use English for global science, medicine, and world history to access international databases. STRICTLY USE CHINESE for Chinese literature, domestic entertainment, Chinese legal cases, or local events to maximize local search recall.
2. **Context Resolution (代词消解与聚焦)**: Focus heavily on the 【Current Task】. If the task references variables from the 【Known Variables Context】, ensure their values are the core of your queries.
3. **Tactical Diversification (海陆空战术分层，必须严格遵守)**: You MUST generate 3 queries with completely different tactical search scopes:
   - **query 1 (Broad & Exact)**: Core keywords only. Directly translate the 【Current Task】 without domain restrictions.
   - **query 2 (Domain-Specific Sniper)**: Append a professional database footprint. (e.g., if biology/medicine, append "PubMed" or "NCBI"; if law, append "court case" or "典型案例"; if business, append "annual report").
   - **query 3 (Lateral Synonym / 同义词与侧面迂回)**: DO NOT mention Wikipedia. Rephrase core concepts using specific synonyms (e.g., unpack "power unit/功率单位" to "horsepower watt/马力 瓦特").
4. **Keyword Economy**: Use as few words as possible. Strip out all verbs, stop words, and conversational English. Keep boolean "OR" if present.

Output JSON ONLY: {{ "queries": ["query 1", "query 2", "query 3"] }}
""" 
    res = llm.safe_invoke(prompt)
    queries = []
    try:
        match = re.search(r'\{.*\}', res, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            queries = data.get("queries", [])
    except:
        queries = [task]

    # 1. 先提取大模型生成的 3 个基础 Query
    unique_base_queries = []
    seen = set()
    for q in queries:
        q_str = str(q).strip()
        if q_str and q_str not in seen:
            unique_base_queries.append(q_str)
            seen.add(q_str)
            
    # 2. 💡 核心恢复：交替穿插组合！
    # 顺序变为：[普通1, 百科1, 普通2, 百科2, 普通3, 百科3]
    expanded_queries = []
    for q in unique_base_queries[:3]: 
        # 加入原始的普通查询（打前阵）
        expanded_queries.append(q)
        
        # 衍生并加入维基百科专属查询（紧随其后作为事实锚点）
        clean_q_for_wiki = re.sub(r'site:\S+', '', q).strip()
        wiki_q = clean_q_for_wiki + " site:wikipedia.org"
        expanded_queries.append(wiki_q)
            
    print(f"🔀 衍生后 6 条精简查询 (普通与百科交替穿插): {expanded_queries}")
    return expanded_queries

def normalize_and_protect(q: str) -> str:
    return (q or "").strip()[:300]

# ---------------------------
# 3) Path Executor (Condition收集 + ReAct决策)
# ---------------------------
# 💡 修改点 2：取消传入 global_evidences，强制在函数内部生成局部的 path_evidences 保证绝对隔离
def execute_single_path(dag: Dict[str, Any], question: str, path_idx: int) -> Dict[str, Any]:
    global_vars = {}
    known_facts_board = [] 
    
    # 局部专有证据列表，防止多线程脏写
    path_evidences = [] 
    
    # ==========================================
    # 阶段一：执行 Condition Nodes (收集约束并生成极简记忆)
    # ==========================================
    condition_nodes = dag.get("condition_nodes", [])
    
    for node in condition_nodes:
        node_id = node.get("node_id")
        raw_task = node.get("task_description", "")
        target_var_name = node.get("target_variable", "result")
        full_var_key = f"{node_id}_{target_var_name}"
        
        injected_task = raw_task
        for k, v in global_vars.items():
            placeholder = f"<{k}>"
            if placeholder in injected_task:
                val_str = str(v) if v != "NOT_FOUND" else "UNKNOWN_ENTITY"
                if isinstance(v, list):
                    val_str = " OR ".join([str(item) for item in v])
                injected_task = injected_task.replace(placeholder, val_str)
                
        candidate_queries = rewrite_query(injected_task, global_vars, path_idx, question)
        step_snippets = []
        for q in candidate_queries:
            q_norm = normalize_and_protect(q)
            snippet = search_serper(q_norm)
            if snippet and len(snippet) > 40:
                step_snippets.append(snippet)
                path_evidences.append({"query": q_norm, "snippet": snippet})
        
        combined_text = "\n\n".join(step_snippets)[:8000] if step_snippets else "No search results obtained."
        
        extract_prompt = f"""
You are an intelligent reasoning analyst. 
【Original Question】: {question}
【Condition Task】: {injected_task}
【Search Results】:
{combined_text}

1. Extract the target variable requested. 
2. If multiple matches exist, output a JSON array (e.g. ["A", "B"]).
3. If not found, output "NOT_FOUND".

Output JSON STRICTLY:
{{
    "extracted_value": "A single string, OR a [list, of, strings], OR NOT_FOUND"
}}
"""
        extract_res = llm.safe_invoke(extract_prompt)
        extracted_value = "NOT_FOUND"
        try:
            match = re.search(r'\{.*\}', extract_res, re.DOTALL)
            if match:
                parsed_extract = json.loads(match.group(0))
                extracted_value = parsed_extract.get("extracted_value", "NOT_FOUND")
        except:
            pass
            
        global_vars[full_var_key] = extracted_value
        known_facts_board.append(f"任务：{injected_task} => 结果：{extracted_value}")

    # ==========================================
    # 阶段二：执行 Target Node (使用 ReAct 动态推理)
    # ==========================================
    target_node = dag.get("target_node", {})
    raw_target_task = target_node.get("task_description", f"查找最终答案：{question}")
    
    injected_target_task = raw_target_task
    for k, v in global_vars.items():
        placeholder = f"<{k}>"
        if placeholder in injected_target_task:
            val_str = str(v) if v != "NOT_FOUND" else "UNKNOWN_ENTITY"
            if isinstance(v, list):
                val_str = " OR ".join([str(item) for item in v])
            injected_target_task = injected_target_task.replace(placeholder, val_str)

    print(f"\n🎯 [Path {path_idx}] 启动 ReAct 引擎解决终极目标: {injected_target_task}")
    
    formatted_facts = "\n".join(known_facts_board) if known_facts_board else "无前置已知条件"
    
    react_history = ""
    final_ans = "无法确定"

    for turn in range(MAX_REACT_TURNS):
        print(f"   🔄 [Path {path_idx}] ReAct 思考轮次 {turn + 1}/{MAX_REACT_TURNS} ...")
        
        if turn < 3:
            quitting_rule = """
🚨【禁止提前放弃法则（前2轮必须死磕）】🚨：
1. 当前处于前二轮探索，绝对禁止提前交白卷！在执行 Finish 行动时，Action_Input 必须是一个【确实存在的实体名称】。
2. 【绝对禁止】输出 `Action: Finish` 并在 Action_Input 中填写 "NOT_FOUND"、"无法确定" 等消极词汇！
3. 如果查不到信息，你【必须】强迫自己换一个完全不同的角度继续 Search。
"""
        else:
            quitting_rule = """
🚨【智能放弃法则（允许及时止损）】🚨：
你已经进行了 2 轮以上的深度搜索尝试。如果你认为所有可能的线索确实都已穷尽，且毫无希望找到确切答案，允许你提前结束以节省算力。
你可以输出 `Action: Finish` 并在 Action_Input 中严格填写 "无法确定"。
"""

        react_prompt = f"""
你是一个具备强大推理能力和搜索决策能力的侦探 Agent。你需要回答以下问题：

【原问题】：
{question}
【当前核心任务】：
{injected_target_task}

【我们已经掌握的绝对事实（全局记忆）】：
{formatted_facts}

【之前的思考与动作历史】：
{react_history if react_history else "无"}

🚨【证据冲突解决与信源分级法则 (Evidence Hierarchy)】🚨：
当全局记忆中的线索发生逻辑冲突（即不同的线索指向了截然不同的实体）时，你必须遵循以下“信源权重优先级”进行裁决，【绝对禁止强行缝合】相互矛盾的信息！
1. 顶级权重（排他性硬指标 / 学术客观事实）：包含独特的科学机制（如特定的分子/基因/物理/化学名词）、精确的历史定论、唯一的专属标识（如专利号、法案名）、高度生僻的专业术语。这类信息在全网具有极强的排他性，一旦查证，准确率极高！
2. 次级权重（通用或共有属性）：泛泛的分类分类（如“某种作物”、“某个欧洲国家”）、常规的用途或宽泛的时代背景。这类信息往往对应大量候选实体，只能作为辅助过滤条件。
3. 最低权重（易受污染的社会/商业/时效性线索）：涉及具体的公司企业、近期社会新闻、常见的地方法院纠纷、人物八卦等。这类信息极易受“搜索引擎近期新闻污染 (Recency Bias)”和“媒体发稿偏好”影响，产生“假阳性 (False Positive)”错误抓取的概率极高！
裁决铁律：当【顶级权重】的硬核线索明确指向实体 A，而【最低权重】的时效性/商业/新闻线索却指向实体 B 时，这必然意味着底层的普通搜索抓取了错误的噪音数据！此时，你必须【果断抛弃】低权重的错误干扰项，坚决且唯一地采信顶级硬指标推导出的答案，不要陷入无休止的自我怀疑！

🚨【搜索降维与重试策略 (Search Backoff Strategy)】🚨：
当你的搜索返回空值（NOT_FOUND）或毫无用处的线索时，说明你的搜索词“过度约束（Over-constrained）”了！在下一次 Search 时你必须强制执行以下降维操作：
1. 核心提纯（做减法）：果断删掉非核心的修饰词（如：具体的期刊名、常规的实验条件、宽泛的地点与年份、介词连词等），只保留 2-3 个最具排他性的核心实体词（例如：基因名 + 细胞系名 + 罕见疾病）。
2. 解除硬锁定：尽量避免使用双引号 `""` 强行包裹一长串词组。允许搜索引擎使用“词汇共现（Co-occurrence）”和模糊匹配来大范围召回包含冷门知识的学术文献或档案。

{quitting_rule}

【你的任务】：
请仔细分析当前的任务和掌握的事实，运用上述法则进行思考。
- 如果依据顶级权重事实已经足够让你锁定唯一答案，请坚决执行 Finish，无视且抛弃那些不匹配的低权重干扰项。
- 如果顶级核心事实仍未查清，你需要继续执行 Search，并优先使用英文搜索最冷门的专业词汇。如果上轮搜索失败，必须应用《搜索降维与重试策略》放宽搜索条件！

你必须严格按照以下格式输出：
Thought: [最多4句话：评估全局记忆中的线索权重，指出哪些是不可推翻的硬核铁证，哪些是抓错的噪音干扰，坚决做出决断。如果上轮搜索失败，请简述你将如何删减多余条件进行降维搜索]
Action: [只能是 Search 或 Finish]
Action_Input: [如果是 Search，填入精简降维后的搜索词；如果是 Finish，直接填入最终实体答案（不要带任何标点或废话）]
"""
        resp = llm.safe_invoke(react_prompt)
        
        thought_match = re.search(r'Thought:\s*(.*?)\nAction:', resp, re.IGNORECASE | re.DOTALL)
        action_match = re.search(r'Action:\s*(.*?)\nAction_Input:', resp, re.IGNORECASE | re.DOTALL)
        input_match = re.search(r'Action_Input:\s*(.*)', resp, re.IGNORECASE | re.DOTALL)
        
        thought = thought_match.group(1).strip() if thought_match else "思考解析失败"
        action = action_match.group(1).strip() if action_match else "Search"
        action_input = input_match.group(1).strip() if input_match else injected_target_task

        print(f"      🤔 [Path {path_idx}] Thought: {thought}")
        print(f"      🛠️ [Path {path_idx}] Action: {action} => {action_input}")

        if "Finish" in action:
            final_ans = action_input
            break
            
        elif "Search" in action:
            q_norm = normalize_and_protect(action_input)
            wiki_q_norm = normalize_and_protect(action_input + " site:wikipedia.org")
            
            snippet_general = search_serper(q_norm)
            snippet_wiki = search_serper(wiki_q_norm)
            
            combined_snippets = []
            
            # 百科优先
            if snippet_wiki and len(snippet_wiki) > 30:
                path_evidences.append({"query": wiki_q_norm, "snippet": snippet_wiki})
                combined_snippets.append(f"【维基百科结果 (高优可信)】:\n{snippet_wiki}")
                
            # 常规搜索随后
            if snippet_general and len(snippet_general) > 30:
                path_evidences.append({"query": q_norm, "snippet": snippet_general})
                combined_snippets.append(f"【常规搜索结果 (需甄别)】:\n{snippet_general}")
            
            if combined_snippets:
                full_snippet = "\n\n".join(combined_snippets)
                observation = safe_text(full_snippet, max_len=3000)
            else:
                observation = "No relevant search results found."
            
            print(f"      👀 [Path {path_idx}] Observation: 获得长达 {len(observation)} 字符的线索 (百科优先排版)。")
            
            react_history += f"Thought: {thought}\nAction: {action}\nAction_Input: {action_input}\nObservation: {observation}\n\n"
            
        else:
            break

    return {
        "candidate_answer": final_ans,
        "known_facts": known_facts_board,
        "reasoning_history": react_history,
        "path_evidences": path_evidences # 将本路径独立收集的证据打包返回
    }

# ---------------------------
# 4) Voting Aggregator (超级裁判：结合证据与格式清洗)
# ---------------------------
def aggregate_answers(path_details: List[Dict[str, Any]], question: str) -> str:
    valid_details = []
    valid_answers = []
    
    for detail in path_details:
        ans = detail.get("candidate_answer", "").strip()
        if ans and "无法确定" not in ans and "NOT_FOUND" not in ans.upper():
            valid_details.append({
                "candidate": ans,
                "facts": detail.get("known_facts", []),
                "reasoning": detail.get("reasoning_history", "")[-1000:] 
            })
            valid_answers.append(ans)
            
    if not valid_answers:
        return "无法确定"

    print(f"\n🗳️ 提交给法官的候选答案池: {valid_answers}")
    print("⚖️ 启动超级法官进行底层证据核查与裁决 (Evidence-Based Cross-Examination)...")

    # ==========================================
    # 💡 裁判大模型 Prompt 重构：取消硬拦截，引入“证据优于共识”法则，格式规则依然前置
    # ==========================================
    prompt = f"""You are the strict Chief Judge for an AI reasoning system. 
You must evaluate the candidates provided by independent agents, select the correct answer strictly based on their evidence, and format it perfectly.

【CRITICAL JUDGING & FORMATTING RULES (必须绝对遵守)】

1. **No Hallucination (绝对禁止联想)**: You MUST ONLY select your answer from the [Candidate Pool] provided below. DO NOT invent, hallucinate, or guess any answers outside of this pool.
2. **Evidence over Blind Consensus (证据优于盲目共识 - 核心评判法则)**: 
   - First, check if there is a Semantic Consensus (e.g., 2 or more agents outputting the same entity or semantically identical entities like "Apple" and "苹果公司").
   - HOWEVER, DO NOT blindly trust a consensus! You MUST evaluate the `facts` and `reasoning` for these matching candidates. Are they based on solid, objective search evidence, or are they just speculative, unsupported guesses?
   - If the matching candidates lack solid evidence (just guessing), and a minority candidate provides a different answer with undeniable, high-confidence factual proof, you MUST reject the false consensus and choose the minority candidate with the strongest evidence.
   - If the matching candidates do have solid factual evidence, they win.
3. **Target Format Alignment (终极格式对齐 - 极其重要)**: Once you select the winning entity, you MUST format it as follows:
   - **Language Rule**: Determine the language of the 【Original Question】. If it is Chinese, you MUST output the Chinese name. IF AND ONLY IF the Chinese question explicitly asks for an English name (e.g., "英文名称", "英文全称是什么"), you MUST output the English name.
   - **Lowercase Rule**: ALL English letters in your final output MUST be converted to lowercase (e.g., output "apple" instead of "Apple").
   - **Numeric Rule**: If the answer is a numerical value or year, output ONLY the integer (e.g., convert "2024.0" or "2024年" strictly to "2024").
   - **Multi-Entity Punctuation**: If the answer contains multiple entities, separate them using a comma or semicolon followed by a space (e.g., "entity1, entity2" or "张三, 李四").
   - **Full Name**: Use the official full name of the entity.

【Original Question】: 
{question}

【Candidate Pool (候选集)】:
{valid_answers}

【Agents' Submissions Details (候选依据)】:
{json.dumps(valid_details, ensure_ascii=False, indent=2)}

You MUST output your response strictly in the following format:
Thought: [1. Identify any semantic consensus. 2. Strictly evaluate if the consensus is backed by facts or if it's a guess. Compare with the evidence of other candidates. 3. Explicitly select the candidate with the highest confidence evidence. 4. State how you will apply the exact formatting rules to this candidate.]
Final_Answer: [Output ONLY the final cleaned entity. No punctuation at the end, no explanations.]
"""
    
    # 获取大模型回复
    res = llm.safe_invoke(prompt)
    
    # 使用正则安全提取 Thought 和 Final_Answer
    thought_match = re.search(r'Thought:\s*(.*?)\nFinal_Answer:', res, re.IGNORECASE | re.DOTALL)
    answer_match = re.search(r'Final_Answer:\s*(.*)', res, re.IGNORECASE | re.DOTALL)
    
    if thought_match:
        print(f"   🧐 法官思考过程: {thought_match.group(1).strip()}")
        
    if answer_match:
        preliminary_winner = answer_match.group(1).strip().strip('"').strip("'").strip("。").strip(".")
    else:
        # 兜底：如果大模型没有按照格式输出，直接清洗全文
        preliminary_winner = res.strip().strip('"').strip("'").strip("。").strip(".")
        
    print(f"⚖️ 超级法官(初审)裁决: {preliminary_winner}")
    return preliminary_winner

# ---------------------------
# 4.5) Entity Canonicalization (终极百科洗稿)
# ---------------------------
def refine_final_answer(preliminary_answer: str, question: str) -> str:
    if preliminary_answer == "无法确定" or "NOT_FOUND" in preliminary_answer.upper():
        return preliminary_answer
        
    print(f"🔍 启动终极百科洗稿，尝试获取官方全称...")
    
    wiki_query = normalize_and_protect(f"{preliminary_answer} site:wikipedia.org")
    snippet = search_serper(wiki_query)
    
    if not snippet or len(snippet) < 20:
        print(f"   ⚠️ 未查到有效维基百科信息，保留初审结果。")
        return preliminary_answer
        
    # 💡 核心优化：格式规则绝对前置，强制思考对齐逻辑
    refine_prompt = f"""You are a meticulous Entity Standardizer. Your absolute TOP PRIORITY is to ensure FORMAT ACCURACY, followed by finding the most official name based on the Wikipedia context.

【CRITICAL FORMATTING & STANDARDIZATION RULES (必须绝对遵守)】
1. **Target Format Alignment (语言与格式对齐 - 核心铁律)**: 
   - **Language Rule**: You MUST match the language of the 【Original Question】. If the question is in Chinese, you MUST output the Chinese official name. IF AND ONLY IF the question explicitly asks for an English name (e.g., "英文名称", "英文全称"), you MUST output the English name.
   - **Lowercase Rule**: ALL English letters MUST be strictly converted to lowercase.
   - **Numeric Rule**: Numeric answers MUST be integers only.
   - **Multi-Entity Punctuation**: Separate multiple entities with a comma or semicolon followed by a space (e.g., "a, b").
2. **Official Full Name**: Use the 【Wikipedia Context】 to expand the 【Preliminary Answer】 to its formal/official registered name, but ONLY if it perfectly obeys the Language Rule above.
3. **Fallback**: If the Wikipedia context is irrelevant, confusing, or contradicts the Language Rule, you must simply apply Rule 1 to the exact 【Preliminary Answer】 and output it.

【Original Question】: 
{question}

【Preliminary Answer】: 
{preliminary_answer}

【Wikipedia Context】:
{snippet[:2000]}

You MUST output your response strictly in the following format:
Thought: [1. Analyze the language of the Original Question. 2. Find the official name in the Wiki context. 3. Explicitly state how you apply the lowercase/numeric/punctuation rules to the final entity.]
Final_Answer: [Output ONLY the final cleaned entity. No punctuation at the end, no explanations.]
"""
    
    res = llm.safe_invoke(refine_prompt)
    
    # 💡 使用正则安全提取，分离思考过程与最终结果
    thought_match = re.search(r'Thought:\s*(.*?)\nFinal_Answer:', res, re.IGNORECASE | re.DOTALL)
    answer_match = re.search(r'Final_Answer:\s*(.*)', res, re.IGNORECASE | re.DOTALL)
    
    if thought_match:
        print(f"   🧐 洗稿思考过程: {thought_match.group(1).strip()}")
        
    if answer_match:
        refined_winner = answer_match.group(1).strip().strip('"').strip("'").strip("。").strip(".")
    else:
        # 兜底截断
        refined_winner = res.strip().strip('"').strip("'").strip("。").strip(".")
        
    print(f"🌟 终极洗稿定稿: {refined_winner}")
    return refined_winner
# ---------------------------
# 5) Top-level run function (统一 DAG + 线程隔离控制版)
# ---------------------------
def run_agent(question: str) -> Dict[str, Any]:
    print(f"\n[Agent Started] 准备并发执行 {N_PATHS} 条探索路径...")
    
    # 💡 修改点 1：将 DAG 提取到最外层，统一生成唯一的标准蓝图！
    print(f"==========================================")
    print("🗺️ 正在生成全局唯一知识图谱 (Unified DAG)...")
    unified_dag = generate_dag(question)
    
    global_evidences = []
    path_details = []
    path_answers_str = [] 
    
    MAX_RETRIES = 3
    retries_used = 0
    valid_paths_completed = 0
    total_threads_launched = N_PATHS 
    
    # 💡 修改点 3：引入线程锁，保证汇总数据写入时的绝对线程安全
    data_lock = threading.Lock()
    
    def path_worker(p_idx: int) -> tuple:
        print(f"\n==========================================")
        print(f"🚀 [线程启动] 正在基于 Unified DAG 执行路径 [Path {p_idx}] ...")
        print(f"==========================================")
        
        # 将统一的图谱传入
        detail_dict = execute_single_path(unified_dag, question, path_idx=p_idx)
        return detail_dict, p_idx

    print(f"\n==========================================")
    print(f"🚀 正在 [并发开启] {N_PATHS} 条底层推理线程 ...")
    print(f"==========================================")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_PATHS + MAX_RETRIES) as executor:
        pending_futures = {executor.submit(path_worker, i + 1): i + 1 for i in range(N_PATHS)}
        
        while pending_futures and valid_paths_completed < N_PATHS:
            done, not_done = concurrent.futures.wait(
                pending_futures.keys(), 
                return_when=concurrent.futures.FIRST_COMPLETED
            )
            
            for future in done:
                p_idx = pending_futures.pop(future)
                try:
                    detail_dict, _ = future.result()
                    ans = detail_dict["candidate_answer"]
                    print(f"✅ [并发返回] Path {p_idx} 最终得出: {ans}")
                    
                    is_invalid = (ans == "无法确定" or "NOT_FOUND" in ans.upper())
                    
                    if is_invalid and retries_used < MAX_RETRIES:
                        retries_used += 1
                        print(f"⚠️ [Path {p_idx}] 判定为无效路径。触发第 {retries_used}/{MAX_RETRIES} 次额外重试！(热启动新线程...)")
                        total_threads_launched += 1
                        new_p_idx = total_threads_launched
                        
                        new_future = executor.submit(path_worker, new_p_idx)
                        pending_futures[new_future] = new_p_idx
                        continue  
                        
                    # 💡 修改点 4：使用线程锁（Lock）安全、有序地将各独立路径收集到的隔离数据写入主列表
                    with data_lock:
                        path_details.append(detail_dict)
                        path_answers_str.append(ans)
                        
                        # 提取该路径特有的 evidences 追加到全局列表中，彻底消除乱序和覆盖
                        local_evidences = detail_dict.get("path_evidences", [])
                        if local_evidences:
                            global_evidences.extend(local_evidences)
                            
                        valid_paths_completed += 1
                    
                except Exception as exc:
                    print(f"❌ [Path {p_idx}] 线程执行引发底层异常: {exc}")

    print(f"\n🛑 所有并发路径执行完毕，准备进入最终裁判环节...")
    
    preliminary_answer = aggregate_answers(path_details, question)
    final_answer = refine_final_answer(preliminary_answer, question)
    
    # 💡 终极兜底清洗：利用 Python 原生能力将所有英文字母强制小写！
    final_answer = final_answer.lower()
    
    return {
        "answer": final_answer, 
        "path_answers": path_answers_str,
        "evidences": global_evidences
    }