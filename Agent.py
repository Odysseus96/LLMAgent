from typing import Dict, List, Optional, Tuple, Union
import json5
from LLM import QwenLMChat
from tools import Tools


TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters} Format the arguments as a JSON object."""
REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tool_descs}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
"""

class Agent:
    def __init__(self, model_name: str = '', base_url="http://127.0.0.1:8181/v1") -> None:
        self.model_name = model_name
        self.tool = Tools()
        self.system_prompt = self.build_system_input()
        self.model = QwenLMChat(base_url=base_url, model_name=model_name)

    def build_system_input(self):
        # 构造上文中所说的系统提示词
        tool_descs, tool_names = [], []
        for tool in self.tool.toolConfig:
            tool_descs.append(TOOL_DESC.format(**tool))
            tool_names.append(tool['name_for_model'])
        tool_descs = '\n\n'.join(tool_descs)
        tool_names = ','.join(tool_names)
        sys_prompt = REACT_PROMPT.format(tool_descs=tool_descs, tool_names=tool_names)
        return sys_prompt

    def parse_latest_plugin_call(self, text):
        # 解析第一次大模型返回选择的工具和工具参数
        plugin_name, plugin_args = '', ''
        i = text.rfind('\nAction:')
        j = text.rfind('\nAction Input:')
        k = text.rfind('\nObservation:')
        if 0 <= i < j:  # If the text has `Action` and `Action input`,
            if k < j:  # but does not contain `Observation`,
                text = text.rstrip() + '\nObservation:'  # Add it back.
            k = text.rfind('\nObservation:')
            plugin_name = text[i + len('\nAction:'): j].strip()
            plugin_args = text[j + len('\nAction Input:'): k].strip()
            text = text[:k]
        return plugin_name, plugin_args, text

    def call_plugin(self, plugin_name, plugin_args):
        # 调用选择的工具
        plugin_args = json5.loads(plugin_args)
        if plugin_name == 'google_search':
            return '\nObservation:' + self.tool.google_search(**plugin_args)

    def text_completion(self, text, history=[]):
        # 整合两次调用
        text = "\nQuestion:" + text
        response, his = self.model.generate_response(text, history, self.system_prompt)
        print(response)
        plugin_name, plugin_args, response = self.parse_latest_plugin_call(response)
        if plugin_name:
            response += self.call_plugin(plugin_name, plugin_args)
        response, his = self.model.chat(response, history, self.system_prompt)
        return response, his

if __name__ == '__main__':
    agent = Agent('/Users/wangyaozhong/hf_models/internlm2-chat-20b')
    prompt = agent.build_system_input()
    print(prompt)