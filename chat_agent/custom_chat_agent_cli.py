# encoding: utf-8
"""
@version: python3.13
@author: tommy
@contact: tingzai.yang@gmail.com
@software: PyCharm
@file: custom_chat_agent_cli.py
@time: 2025/5/4 22:41
"""
from langchain_ollama.llms import OllamaLLM

class CustomChatAgentCLi(object):
    @property
    def _llm_type(self) -> str:
        # 返回我们自定义的模型标记
        return "qwen3-1.7b"

    def __init__(self):
        self.llm = OllamaLLM(base_url="http://127.0.0.1:11434", model="deepseek-r1:8b")

    def llama_completion(self, user_input) -> str:
        # 调用llama的接口，返回响应
        # return "Hello from llama!"
        try:
            response = self.llm.invoke(user_input)
            return response
        except Exception as e:
            print(f"error:{e}")
        return "error"

llm = CustomChatAgentCLi()
print(llm.llama_completion("你是谁？"))