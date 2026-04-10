# Search Agent 项目说明

本项目包含多种基于大语言模型（LLM）的自动解题 Agent 实现，支持复杂问题的多跳推理、搜索与变量融合。适用于学术搜索、复杂问答、信息抽取等场景。

## 目录结构

- `agent/agent_tianchi/`
  - `agent_0303.py`：多候选变量融合 Agent，适合变量不确定性高的场景。
  - `agent_langgraph.py`：基于状态机/有向图的 Agent，流程自动化、可视化。
  - `agent.py`：条件-目标分离 DAG Agent，适合复杂多跳推理。
  - `agent0305.py`：DAG+ReAct 推理 Agent，支持动态多轮推理与证据分级。
  - 其他：`llm.py`（大模型接口）、`tools.py`（搜索工具）、`state.py`（状态管理）、`utils.py`（工具函数）等。

- `agent/agent_tianchi/search_/`：搜索相关模块与工具。

- `*.jsonl`：示例问题、答案与实验结果。

## 主要 Agent 方案简介

- **agent_0303.py**：多候选变量融合，变量置信度管理，适合变量不确定性高的场景。
- **agent_langgraph.py**：状态机/有向图自动流程，适合需要灵活流程编排和可视化的场景。
- **agent.py / agent0305.py**：DAG条件-目标分离，变量注入与动态推理，适合复杂多跳、需要显式条件管理的场景。

## 依赖环境

- Python 3.8+
- 推荐使用虚拟环境（如 venv）
- 主要依赖见 `requirements.txt`（如有）

## 快速开始

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 运行示例：
   ```bash
   cd agent/agent_tianchi
   python sample_run.py
   ```

## 贡献与维护

如需扩展 Agent 能力或适配新场景，请参考各 Agent 文件的注释与实现风格。

---

如有问题请联系项目维护者。
