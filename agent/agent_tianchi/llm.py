# llm.py
import os
import time
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
# Wrapper with robust invoke + content-safe handling
class LLMWrapper:
    def __init__(self, model=None, temperature=0.2):
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        # use DashScope if available (as user used before), else fallback to OpenAI
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            model=model or os.getenv("LLM_MODEL", "qwen3-max"),
            temperature=float(os.getenv("LLM_TEMPERATURE", temperature))
        )

    def safe_invoke(self, prompt_or_messages, max_retries:int=2, timeout:float=30.0):
        """
        Accepts either a string prompt or a list of messages (langchain style).
        Returns content string (possibly empty on failure).
        Wraps exceptions such as content filtering and returns fallback value.
        """
        last_exc = None
        for attempt in range(max_retries+1):
            try:
                if isinstance(prompt_or_messages, str):
                    res = self.llm.invoke(prompt_or_messages)
                else:
                    # assume list/dict messages
                    res = self.llm.invoke(prompt_or_messages)
                content = getattr(res, "content", None)
                if content is None:
                    # sometimes the return shape differs
                    content = str(res)
                return content
            except Exception as e:
                last_exc = e
                # If content filter error, shorten or sanitize and retry once
                msg = str(e).lower()
                print(f"[LLM] invoke error (attempt {attempt}):", e)
                if "inappropriate" in msg or "data_inspection_failed" in msg:
                    # signal caller to sanitize input; return empty to be safe
                    return ""
                time.sleep(0.5 + attempt)
                continue
        print("[LLM] failed after retries:", last_exc)
        return ""
