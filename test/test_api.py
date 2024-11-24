import requests
from requests import Response
import json


def send_prediction_request(image_path : str, api_url : str):
    try:
        with open(image_path, 'rb') as img_file:
            response = requests.post(api_url, files={'file':img_file})

            if response.status_code == 200:
                result = response.json()
                return result
            else:
                print(f"请求失败，状态码: {response.status_code}")
                print(f"响应内容: {response.text}")
                return {"error": "请求失败", "status_code": response.status_code, "response": response.text}
    except Exception as e:
        print(f"发生错误: {e}")
        return {"error": str(e)}

if __name__ == '__main__':
    # API 的 URL
    api_url = "http://127.0.0.1:8000/predict"

    # 要上传的图像文件路径
    image_path = "assert/bus.jpg"

    # 发送请求并获取结果
    result = send_prediction_request(image_path, api_url)
    print(result)
    # result = json.dumps(result)

    # 打印结果
    print(f"结果个数: {len(result['predictions'])}")
    # print(result['predictions'])