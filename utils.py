import requests


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