import arxiv
import os

from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

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

client = arxiv.Client()

def get_least_ai_papers(n=10):
    search_query = "cat:cs.AI"
    papers = arxiv.Search(
        query="React agent",
        max_results=n,
        sort_by=arxiv.SortCriterion.Relevance
        # sort_by=arxiv.SortCriterion.SubmittedDate
    )
    results = []
    for paper in client.results(papers):
        # print(f"papers title: {paper.title}\npapers abstract: {paper.summary}\n papers link: {paper.links}\tdownload link: {paper.pdf_url}")
        results.append(f"papers title: {paper.title}\npapers abstract: {paper.summary}\n download link: {paper.pdf_url}")

    return ''.join(results)


class ArxivPaperSearch(BaseTool):
    name: str = "arxiv_paper_search"  # 工具的名称
    description: str = (
        "使用 Arxiv API 搜索相关内容。输入一个字符串，"
        "根据num_results返回每片论文的题目，摘要和pdf链接"
    )

    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)

    def _run(self, query: str) -> str:
        print(f"query: {query}")
        papers = arxiv.Search(
            query=query,
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance
            # sort_by=arxiv.SortCriterion.SubmittedDate
        )
        results = []
        for paper in client.results(papers):
            results.append(
                f"papers title: {paper.title}\npapers abstract: {paper.summary}\n download link: {paper.pdf_url}")
        return '\n'.join(results)


if __name__ == '__main__':
    arxiv_tools = ArxivPaperSearch()
    results = arxiv_tools.run("AI Agent")

    # print(results)

    tools = [arxiv_tools]

    # model_with_tools = llm.bind_tools(tools)
    #
    # response = model_with_tools.invoke([HumanMessage(content="帮我搜集一些AI Agent相关的论文")])
    #
    # print(f"ContentString: {response.content}")
    # print(f"ToolCalls: {response.tool_calls}")

    agent_executor = create_react_agent(llm, tools, debug=True)

    query_messages = {"messages": [HumanMessage(content="帮我搜集一些AI Agent相关的论文")]}

    for chunk in agent_executor.stream(query_messages):
        if not chunk.get('agent'): continue
        if chunk['agent']['messages'][0].content != '':
            print(chunk['agent']['messages'][0].content)
