from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from abc import ABC, abstractmethod


class BaseModel:
    def __init__(self, base_url, model_name):
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = 'EMPTY'

        self.history = []
        self.max_history = 5

    @abstractmethod
    def add_to_history(self, user_message, assistant_response=None):
        pass

    @abstractmethod
    def generate_response(self, user_message):
        pass


class QwenOllamaChat(BaseModel):
    def __init__(self, base_url, model_name):
        super().__init__(base_url, model_name)
        # self.base_url = base_url
        # self.model_name = model_name
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        print(f'model name: {self.model_name}')

    def add_to_history(self, user_message, assistant_response=None):
        """
        将用户消息和助手的回应保存到历史记录中

        参数:
        user_message (str): 用户的消息。
        assistant_response (str): 助手的回应（可选）。
        """
        self.history.append({"role": "user", "content": user_message})
        if assistant_response:
            self.history.append({"role": "assistant", "content": assistant_response})

        # 控制历史记录长度
        if len(self.history) > self.max_history * 2:  # user + assistant pairs
            self.history = self.history[-self.max_history * 2:]

    def generate_response(self, user_message):
        """
        调用 OpenAI API，并结合历史信息生成回复

        参数:
        user_message (str): 用户的最新消息。

        返回:
        response_text (str): OpenAI 模型生成的回复。
        """
        # 将用户的消息添加到历史中
        self.add_to_history(user_message)

        # 调用 OpenAI 的 Chat API
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.history,
                max_tokens=512,
                top_p=0.8,
                temperature=0.7,
                extra_body={
                    "repetition_penalty": 1.05,
                },
            )

            # 提取助手的回应
            generated_text = response.choices[0].message.content.strip()

            # 保存助手的回应到历史记录
            self.add_to_history(user_message, generated_text)

            return generated_text
        except Exception as e:
            return f"Error: {str(e)}"



class QwenVLLMChat(BaseModel):
    def __init__(self, base_url, model_name="Qwen2.5-7B-Instruct"):
        super().__init__(base_url, model_name)
        self.model_name = model_name
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url
        )
        self.history = []
        self.max_history = 5

    def add_to_history(self, user_message, assistant_response=None):
        """
        将用户消息和助手的回应保存到历史记录中

        参数:
        user_message (str): 用户的消息。
        assistant_response (str): 助手的回应（可选）。
        """
        self.history.append({"role": "user", "content": user_message})
        if assistant_response:
            self.history.append({"role": "assistant", "content": assistant_response})

        # 控制历史记录长度
        if len(self.history) > self.max_history * 2:  # user + assistant pairs
            self.history = self.history[-self.max_history * 2:]

    def generate_response(self, user_message):
        """
        调用 OpenAI API，并结合历史信息生成回复

        参数:
        user_message (str): 用户的最新消息。

        返回:
        response_text (str): OpenAI 模型生成的回复。
        """
        # 将用户的消息添加到历史中
        self.add_to_history(user_message)

        # 调用 OpenAI 的 Chat API
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.history,
                max_tokens=512,
                top_p=0.8,
                temperature=0.7,
                extra_body = {
                    "repetition_penalty": 1.05,
                },
            )

            # 提取助手的回应
            generated_text = response.choices[0].message.content.strip()

            # 保存助手的回应到历史记录
            self.add_to_history(user_message, generated_text)

            return generated_text
        except Exception as e:
            return f"Error: {str(e)}"



if __name__ == '__main__':
    base_url = "http://127.0.0.1:8181/v1"
    # model_name = "qwen2.5:7b-instruct"
    model_name = "internlm2:7b-chat-v2.5-q4_K_M"
    llm = QwenOllamaChat(base_url, model_name=model_name)
    user_message = "天为什么是蓝色的？"
    response = llm.generate_response(user_message)
    print(f"resp: {response}")
