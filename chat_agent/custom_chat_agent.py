# encoding: utf-8
"""
@version: python3.13
@author: tommy
@contact: tingzai.yang@gmail.com
@software: PyCharm
@file: custom_chat_agent.py
@time: 2025/5/4 17:07
"""
from typing import List

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from ollama import Client

class CustomChatAgent(object):
    @property
    def _llm_type(self) -> str:
        # 返回我们自定义的模型标记
        return "qwen3-1.7b"

    def __init__(self):
        self.llm = Client(
            host='http://localhost:11434',
            headers={'x-some-header': 'some-value'},
        )

    def __call__(self, prompt) -> str:
        # 这里是调用自定义模型或API接口的逻辑
        messages = [
            {
                "role": "user",
                "content": prompt.to_string()
            },
            # 如果需要，可以在这里添加更多的消息历史
        ]

        response = self.llama_completion(messages)
        return response

    def llama_completion(self, messages: List[dict]) -> str:
        # 调用llama的接口，返回响应
        # return "Hello from llama!"
        print(f"messages:{messages}")
        try:
            response = self.llm.chat(model="qwen3:1.7b", messages=messages)
            print(f"response:{response}")
            return response['message']['content']
        except Exception as e:
            print(f"error:{e}")
        return "error"

# 1. Create prompt template
# prompt_template = PromptTemplate(input_variables=[],
#                                  template="""
#                                  您是世界级的前沿技术(大模型)分享博主。
#                                  你可以基于用户输入的主题："{topic}"进行文档创作，
#                                  每篇文档创作字数为500-1000字，文档基于markdown格式输出。
#                                  """)
prompt_template = ChatPromptTemplate.from_messages([
 ("system", """
             您是专注于前沿技术(大模型)知识分享的博主。
             您可以基于用户输入的主题进行文档创作，
             每篇文档创作字数为500-1000字，文档基于markdown格式输出。
             """),
 ("user", "{topic}")
])

# 2. Create model
model = CustomChatAgent()

# 3. Create parser
parser = StrOutputParser()

# 4. Create chain
chain = prompt_template | model | parser


# 4. App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 5. Adding chain route
add_routes(
    app,
    chain,
    path="/writer",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)