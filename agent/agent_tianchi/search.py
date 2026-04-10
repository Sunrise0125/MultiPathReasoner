# tools.py

import os
import json
import requests
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from utils import safe_text
from llm import LLMWrapper

load_dotenv()

SERPER_KEY = os.getenv("SERPER_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")
BING_KEY = os.getenv("BING_API_KEY")

# ============================================================
# 单个搜索引擎实现
# ============================================================

class BaseSearchEngine:
    def search(self, query: str, max_snippets: int = 3, timeout: int = 10) -> str:
        raise NotImplementedError


class SerperEngine(BaseSearchEngine):
    def search(self, query: str, max_snippets: int = 3, timeout: int = 10) -> str:
        if not SERPER_KEY:
            return ""

        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"}
        payload = {"q": query}

        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            j = r.json()
            organics = j.get("organic", [])

            texts = []
            for item in organics[:max_snippets]:
                snippet = item.get("snippet") or item.get("title") or ""
                texts.append(safe_text(snippet, max_len=800))

            return "\n".join(texts)
        except Exception as e:
            print("Serper error:", e)
            return ""


class TavilyEngine(BaseSearchEngine):
    def search(self, query: str, max_snippets: int = 3, timeout: int = 10) -> str:
        if not TAVILY_KEY:
            return ""

        url = "https://api.tavily.com/search"
        payload = {
            "api_key": TAVILY_KEY,
            "query": query,
            "max_results": max_snippets,
        }

        try:
            r = requests.post(url, json=payload, timeout=timeout)
            j = r.json()
            results = j.get("results", [])

            texts = []
            for item in results[:max_snippets]:
                snippet = item.get("content", "")
                texts.append(safe_text(snippet, max_len=800))

            return "\n".join(texts)
        except Exception as e:
            print("Tavily error:", e)
            return ""


# ============================================================
# 多引擎智能搜索类
# ============================================================

class MultiSearch:

    def __init__(self):
        self.llm = LLMWrapper()
        self.engines = {
            "serper": SerperEngine(),
            "tavily": TavilyEngine(),
        }

    # --------------------------------------------------------
    # Step 1: 让 LLM 选择搜索引擎
    # --------------------------------------------------------
    def _choose_engines(self, query: str, motivation: str) -> List[str]:

        prompt = f"""
You are a search engine selector.

Query:
{query}

Motivation:
{motivation}

Available engines:
- serper (Google-based, good for general factual queries)
- tavily (good for long-form analytical content)

Return a JSON list of engine names to use.
Example:
["serper"]
or
["serper", "tavily"]

Only output JSON.
"""

        res = self.llm.safe_invoke(prompt)

        try:
            engines = json.loads(res)
            engines = [e for e in engines if e in self.engines]
            return engines if engines else ["serper"]
        except:
            return ["serper"]

    # --------------------------------------------------------
    # Step 2: 并行搜索
    # --------------------------------------------------------
    def _parallel_search(self, engines: List[str], query: str, max_snippets: int) -> Dict[str, str]:

        results = {}

        with ThreadPoolExecutor(max_workers=len(engines)) as executor:
            futures = {
                executor.submit(self.engines[name].search, query, max_snippets): name
                for name in engines
            }

            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception:
                    results[name] = ""

        return results

    # --------------------------------------------------------
    # Step 3: LLM 抽取 relevant evidence
    # --------------------------------------------------------
    def _extract_relevant(self, query: str, motivation: str, raw_results: Dict[str, str]) -> str:

        combined_text = "\n\n".join(
            f"[{engine}]\n{text}" for engine, text in raw_results.items() if text
        )

        if not combined_text.strip():
            return ""

        prompt = f"""
You are an evidence extractor.

Query:
{query}

Motivation:
{motivation}

Search Results:
{combined_text}

Extract ONLY information relevant to answering the query.
Remove duplicated or irrelevant content.
Return concise evidence paragraphs.
"""

        return self.llm.safe_invoke(prompt)

    # --------------------------------------------------------
    # 主接口（兼容原 search_serper）
    # --------------------------------------------------------
    def search(self, query_motivation: str, query: str,
               max_snippets: int = 3, timeout: int = 10) -> str:

        # 1. 选择引擎
        engines = self._choose_engines(query, query_motivation)

        # 2. 并行搜索
        raw_results = self._parallel_search(engines, query, max_snippets)

        # 3. LLM 抽取 relevant
        final_text = self._extract_relevant(query, query_motivation, raw_results)

        return final_text or ""