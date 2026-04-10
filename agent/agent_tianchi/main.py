import os

import json
from agent.agent_tianchi.agent_dag_react import run_agent

def load_questions(file_path: str) -> dict:
    """加载 questions.jsonl，返回 {id: question} 字典"""
    questions = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                questions[item["id"]] = item["question"]
    return questions

def load_answers(file_path: str) -> dict:
    """加载 answers.jsonl，返回 {id: answer} 字典"""
    answers = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                answers[item["id"]] = item["answer"]
    return answers

def normalize_answer(ans: str) -> str:
    """标准化答案：去空格、转小写，便于比较"""
    if not isinstance(ans, str):
        ans = str(ans)
    return ans.strip().lower()

def main(
    start_id: int = 0,
    end_id: int = None,          # None 表示测到最大 id
    output_results_file: str = "results.jsonl",
    output_errors_file: str = "errors.jsonl",
    n_paths: int = 1
):
    # 加载数据
    questions = load_questions("question_example.jsonl")

    ground_truth = load_answers("answer_example.jsonl")  
    

    # 确定要测试的 id 范围
    all_ids = sorted(questions.keys())
    if end_id is None:
        test_ids = [i for i in all_ids if i >= start_id]
    else:
        test_ids = [i for i in all_ids if start_id <= i <= end_id]

    print(f"🔍 将测试 {len(test_ids)} 个问题 (ID: {start_id} ~ {end_id or 'max'})")

    correct = 0
    total = len(test_ids)

    # 清空输出文件
    with open(output_results_file, 'w', encoding='utf-8') as rf, \
         open(output_errors_file, 'w', encoding='utf-8') as ef:

        for idx, q_id in enumerate(test_ids):
            question = questions[q_id]
            true_answer = ground_truth.get(q_id, "").strip()
            
            print(f"\n[{idx+1}/{total}] 处理问题 ID {q_id}...")

            try:
                # 调用你的智能代理
                result = run_agent(question)
                pred_answer = result.get("answer", "").strip()
            except Exception as e:
                print(f"⚠️ 问题 {q_id} 执行出错: {e}")
                pred_answer = ""

            # 判断是否正确（标准化后比较）
            is_correct = normalize_answer(pred_answer) == normalize_answer(true_answer)

            if is_correct:
                correct += 1

            # 构造结果记录
            record = {
                "id": q_id,
                # "question": question,
                "predicted_answer": pred_answer,
                "ground_truth": true_answer,
                "correct": is_correct
            }

            # 写入 results.jsonl
            rf.write(json.dumps(record, ensure_ascii=False) + "\n")
            rf.flush()

            # 如果错误，也写入 errors.jsonl
            if not is_correct:
                ef.write(json.dumps(record, ensure_ascii=False) + "\n")
                ef.flush()

            # print(f"✅ 正确: {is_correct} | 预测: '{pred_answer}' | 真实: '{true_answer}'")

    # 最终统计
    accuracy = correct / total if total > 0 else 0
    print("\n" + "="*60)
    print(f"📊 总体准确率: {correct}/{total} = {accuracy:.2%}")
    print(f"📄 详细结果已保存至: {output_results_file}")
    print(f"❌ 错误案例已保存至: {output_errors_file}")
    print("="*60)

if __name__ == "__main__":
    # ========== 配置区 ==========
    START_ID = 0      # 起始问题 ID
    END_ID = 1  # 结束 ID（设为 None 表示全部）
    N_PATHS = 3       # 每个问题生成的检索路径数
    # ==========================

    main(start_id=START_ID, end_id=END_ID, n_paths=N_PATHS)