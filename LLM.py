from typing import Dict, List, Optional, Tuple, Union, ClassVar, Any, Iterator
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from abc import ABC, abstractmethod
import openai

from langchain_core.language_models import BaseChatModel
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    BaseMessage
)
from openai import OpenAI
from langchain_core.outputs import ChatResult, ChatGenerationChunk

os.environ['http_proxy'] = '127.0.0.1:11434'
os.environ['https_proxy'] = '127.0.0.1:11434'


class OllamaChatModel(BaseChatModel):
    model_name: str = 'qwen2.5:7b-instruct'
    client: ClassVar[openai.OpenAI] = OpenAI(
        base_url="http://127.0.0.1:11434/v1",
        api_key="ollama",
    )
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    def _generate(
            self,
            messages: list[BaseMessage],
            stop: Optional[list[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ):
        print(f'model name: {self.model_name}')
        openai_messages = [
            {"role": "system", "content": msg.content} if isinstance(msg, AIMessage) else
            {"role": "user", "content": msg.content} if isinstance(msg, HumanMessage) else
            {"role": "assistant", "content": msg.content}
            for msg in messages
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=openai_messages,
            temperature=self.temperature,
            top_p=1,
            max_tokens=self.max_tokens,
            extra_body={
                "is_sensitive_enable": True
            },
            stream=False
        )
        message = response.choices[0].message.content
        ai_message = AIMessage(content=message)
        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])

    def _stream(
            self,
            messages: list[BaseMessage],
            stop: Optional[list[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        openai_messages = [
            {"role": "system", "content": msg.content} if isinstance(msg, AIMessage) else
            {"role": "user", "content": msg.content} if isinstance(msg, HumanMessage) else
            {"role": "assistant", "content": msg.content}
            for msg in messages
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=openai_messages,
            temperature=self.temperature,
            top_p=1,
            max_tokens=self.max_tokens,
            extra_body={
                "repetition_penalty": 1.05,
            },
            stream=True
        )

        for chunk in response:
            ai_message = AIMessageChunk(content=chunk.choices[0].delta.content)
            yield ChatGenerationChunk(message=ai_message)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": "OllamaChatModel",
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "custom_ollama"


if __name__ == '__main__':
    llm = OllamaChatModel()
    user_message = "天为什么是蓝色的？"
    response = llm.stream(user_message)
    for resp in response:
        print(resp.content, end='', flush=True)
