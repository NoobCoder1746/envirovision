from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import numpy as np
import torch
import cv2

# import các hàm, model, class_names, device... từ app chính
from app import detect_and_classify, yolo_model, classification_model, preprocess_image, class_names, device

app = FastAPI(title="EnviroVision API")

@app.post("/predict/")
async def predict(file: UploadFile = File(...), conf_threshold: float = 0.5):
    try:
        # Đọc ảnh
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Chạy hàm detect_and_classify gốc của bạn
        result_img, results = detect_and_classify(image, conf_threshold)

        # Chuyển kết quả sang JSON-friendly format
        predictions = [
            {
                "label": label,
                "confidence": float(conf),
                "box": [int(x1), int(y1), int(x2), int(y2)]
            }
            for (label, conf, (x1, y1, x2, y2)) in results
        ]

        return JSONResponse(content={"predictions": predictions})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
