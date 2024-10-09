import os, json
import requests
from requests import Response


class Tools:
    def __init__(self):
        self.toolConfig = self._tools()

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

    def send_prediction_result(self, image_path : str, api_url : str):
        try:
            with open(image_path, 'rb') as img_file:
                response = requests.post(api_url, files={'file': img_file})

                if response.status_code == 200:
                    result = response.json()
                    return {len(result['predictions'])}
                else:
                    print(f"请求失败，状态码: {response.status_code}")
                    print(f"响应内容: {response.text}")
                    return {"error": "请求失败", "status_code": response.status_code, "response": response.text}
        except Exception as e:
            print(f"发生错误: {e}")
            return {"error": str(e)}

if __name__ == '__main__':
    url = 'https://google.serper.dev/search'
    search_query = "天为什么是蓝色的"
    payload = json.dumps({"q": search_query})
    headers = {
        'X-API-KEY': '0d15d366db01ca14061edb164e0704c8fa3243d3',
        'Content-Type': 'application/json'
    }
    resp = requests.request("POST", url=url, data=payload, headers=headers).json()
    print(resp['organic'][0]['snippet'])