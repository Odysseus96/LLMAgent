import os
from openai import OpenAI

os.environ['http_proxy'] = '127.0.0.1:11434'
os.environ['https_proxy'] = '127.0.0.1:11434'

client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",
)

chat_response = client.chat.completions.create(
    model="qwen2.5:7b-instruct",
    messages=[
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": "如何变得更强"},
    ],
    temperature=0.7,
    top_p=0.8,
    max_tokens=512,
    extra_body={
        "repetition_penalty": 1.05,
    },
    stream=True
)

for chunk in chat_response:
    print(chunk.choices[0].delta.content, end='', flush=True)
print('\n')