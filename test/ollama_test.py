import os
from openai import OpenAI
from langchain_openai import ChatOpenAI

os.environ['http_proxy'] = '127.0.0.1:8181'
os.environ['https_proxy'] = '127.0.0.1:8181'

# client = OpenAI(
#     base_url="http://127.0.0.1:11434/v1",
#     api_key="ollama",
# )
#
# chat_response = client.chat.completions.create(
#     model="qwen2.5:7b-instruct",
#     messages=[
#         {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#         {"role": "user", "content": "如何变得更强"},
#     ],
#     temperature=0.7,
#     top_p=0.8,
#     max_tokens=512,
#     extra_body={
#         "repetition_penalty": 1.05,
#     },
#     stream=True
# )
#
# for chunk in chat_response:
#     print(chunk.choices[0].delta.content, end='', flush=True)
# print('\n')

os.environ['OPENAI_API_KEY'] = 'ollama'
os.environ['OPENAI_BASE_URL'] = 'http://127.0.0.1:8181/v1'

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

messages = [
    # (
    #     "system",
    #     "You are a helpful assistant that translates English to French. Translate the user sentence.",
    # ),
    ("human", "如何变得更强"),
]
ai_msg = llm.invoke(messages)
print(ai_msg.content)