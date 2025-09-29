import streamlit as st
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

yolo_model_path = hf_hub_download(
    repo_id="Noob1746/EnviroVision",
    filename="best.pt"
)
yolo_model = YOLO(yolo_model_path)

num_classes = 10
classification_model = models.resnet18()
num_ftrs = classification_model.fc.in_features
classification_model.fc = nn.Linear(num_ftrs, num_classes)

resnet_model_path = hf_hub_download(
    repo_id="Noob1746/EnviroVision",
    filename="class.pt"
)
classification_model.load_state_dict(torch.load(resnet_model_path, map_location=torch.device("cpu")))
classification_model.eval()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class_names = ['battery', 'biological', 'brown-glass', 'cardboard',
               'green-glass', 'metal', 'paper', 'plastic', 'trash', 'white-glass']


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((180, 180)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def detect_and_classify(image):
    img = np.array(image)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    results = yolo_model(img, conf=0.2)

    final_results = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            cropped_obj = img[y1:y2, x1:x2]

            if cropped_obj.size == 0:
                continue

            pre_img = preprocess_image(cropped_obj).to(device)

            with torch.no_grad():
                outputs = classification_model(pre_img)
                _, predicted = torch.max(outputs, 1)
                conf_score = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()
                label = class_names[predicted.item()]

            final_results.append((label, conf_score, (x1, y1, x2, y2)))

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_bgr, f"{label} {conf_score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb, final_results


st.title("♻️ EnviroVision - Smart Waste Classification")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Detection"):
        with st.spinner("Processing..."):
            result_img, results = detect_and_classify(image)

        st.image(result_img, caption="Detection Result", use_column_width=True)

        st.subheader("Classification Results:")
        for label, conf, _ in results:
            st.write(f"**{label}** - Confidence: {conf:.2f}")
