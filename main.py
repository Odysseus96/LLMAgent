from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
from ultralytics import YOLO

app = FastAPI()

# 加载 YOLOv8 模型
model = YOLO('yolo11s.pt')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 读取上传的文件
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # 进行预测
    results = model(image)

    # 处理结果
    predictions = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            confidence = box.conf[0]
            class_id = box.cls[0]
            if class_id != 0:
                continue
            predictions.append({
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "confidence": float(confidence),
                "class_id": int(class_id)
            })

    return JSONResponse(content={"predictions": predictions})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)