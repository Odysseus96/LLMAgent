import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

os.environ['http_proxy'] = '127.0.0.1:11434'
os.environ['https_proxy'] = '127.0.0.1:11434'

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个强大的翻译助手，可以把{input_language}翻译成{output_language}"),
        ("human", "{input}"),
    ]
)

os.environ['OPENAI_API_KEY'] = 'ollama'
os.environ['OPENAI_BASE_URL'] = 'http://127.0.0.1:11434/v1'

llm = ChatOpenAI(
    model="qwen2.5:7b-instruct",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
    # base_url="...",
    # organization="...",
    # other params...
)

chain = prompt | llm

res = chain.invoke(
    {
        "input_language": "汉语",
        "output_language": "俄语",
        "input": "如何变得更强"
    }
)

print(res.content)