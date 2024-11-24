import os, json
import requests
from requests import Response
from PIL import Image
import imagesize
from transformers import BlipProcessor, BlipForConditionalGeneration, MarianMTModel, MarianTokenizer
from utils import *


class Tools:
    def __init__(self):
        self.toolConfig = self._tools()
        self.object_detection_url = "http://127.0.0.1:8000/predict"

        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        self.translate_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
        self.translate_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-zh")

    def _tools(self):
        tools = [
            {
                'name_for_human':'谷歌搜索',
                'name_for_model':'google_search',
                'description_for_model':'谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。',
                'parameters': [
                    {
                        'name': 'search_query',
                        'description': '搜索关键词或短语',
                        'required': True,
                        'schema': {'type': 'string'},
                    }
                ],
            },
            {
                'name_for_human':'PersonCount',
                'name_for_model':'person_count',
                'description_for_model': "用于查询图片大小以及图中人数。",
                'parameters': [
                    {
                        'name': 'image_caption',
                        'description': '返回图像描述，包括图像大小和图中人数',
                        'required': True,
                        'schema': {'type': 'string'},
                    }
                ]
            }
        ]
        return tools

    def google_search(self, search_query: str):
        url = 'https://google.serper.dev/search'
        payload = json.dumps({"q": search_query})

        headers = {
            'X-API-KEY': '0d15d366db01ca14061edb164e0704c8fa3243d3',
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload).json()
        return response['organic'][0]['snippet']

    def image_caption(self, image_path):
        width, height = imagesize.get(image_path)
        image = Image.open(image_path)
        # text = "照片里"
        inputs = self.processor(image, return_tensors="pt")
        out = self.model.generate(**inputs)
        describe = self.processor.decode(out[0], skip_special_tokens=True)
        # describe = self.translate_en_to_zh(describe)
        caption = f"这张图大小为({width}x{height}), 图片描述: {describe}"
        return caption

    def translate_en_to_zh(self, text: str) -> str:
        """将英文翻译成中文"""
        # 对输入文本进行编码
        inputs = self.translate_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        # 模型生成翻译
        outputs = self.translate_model.generate(**inputs)
        # 解码生成的翻译
        translated_text = self.translate_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text


    def person_count(self, image_path):
        # image = Image.open(image_path)
        width, height = imagesize.get(image_path)
        results = send_prediction_request(image_path, self.object_detection_url)
        numbers = len(results['predictions'])
        caption = f"这张图大小为({width}x{height}), 图中有{numbers}个人"
        return caption

if __name__ == '__main__':
    tool = Tools()
    caption = tool.image_caption('assert/bus.jpg')
    print(caption)