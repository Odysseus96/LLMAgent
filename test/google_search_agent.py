import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, List
from dotenv import load_dotenv
import json
import requests

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

load_dotenv()

os.environ['http_proxy'] = '127.0.0.1:8181'
os.environ['https_proxy'] = '127.0.0.1:8181'

llm = ChatOpenAI(
    model="qwen2.5:7b-instruct",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key="ollama",
    base_url="http://127.0.0.1:8181/v1",
)

# 自定义工具类
class GoogleSearchTool(BaseTool):
    name: str = "google_search"  # 工具的名称
    description: str = (
        "使用 Google API 搜索相关内容。输入一个字符串，"
        "根据num_results返回的内容包含标题、链接和简要描述。"
    )

    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)

    def _run(self, query: str) -> str:
        """
        同步运行工具逻辑。
        """
        num_results = 3

        url = 'https://google.serper.dev/search'
        payload = json.dumps({"q": query})

        headers = {
            'X-API-KEY': os.getenv('X-API-KEY'),
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)

        if response.status_code != 200:
            raise Exception(f"Google Search API 请求失败: {response.status_code}, {response.text}")

        data = response.json()['organic']

        results = []
        for idx, item in enumerate(data):
            if idx >= num_results: break
            results.append(f"标题: {item['title']}\n链接: {item['link']}\n简介: {item['snippet']}\n")

        return "\n".join(results)

if __name__ == '__main__':
    search_tool = GoogleSearchTool()
    tool_input = "今天北京天气怎么样"
    result = search_tool.run(tool_input)
    # print(result)
    #
    tools = [search_tool]
    #
    # model_with_tools = llm.bind_tools(tools)
    #
    # response = model_with_tools.invoke([HumanMessage(content="今天北京天气怎么样")])
    #
    # print(f"ContentString: {response.content}")
    # print(f"ToolCalls: {response.tool_calls}")



    agent_executor = create_react_agent(llm, tools, debug=True)

    query_messages = {"messages":[HumanMessage(content="今天北京天气怎么样")]}

    for chunk in agent_executor.stream(query_messages):
        if not chunk.get('agent'): continue
        if chunk['agent']['messages'][0].content != '':
            print(chunk['agent']['messages'][0].content)