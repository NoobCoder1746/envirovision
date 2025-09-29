import streamlit as st
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import base64


#load models
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

class_names = [
    'battery', 'biological', 'brown-glass', 'cardboard',
    'green-glass', 'metal', 'paper', 'plastic', 'trash', 'white-glass'
]


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((180, 180)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def detect_and_classify(image, conf_threshold):
    img = np.array(image)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    results = yolo_model(img, conf=conf_threshold)
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


#streamlit ui
st.set_page_config(page_title="EnviroVision", page_icon="♻️", layout="centered")


def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


background_path = "realfish-Kh8aGCgWZLg-unsplash.jpg"  
base64_bg = get_base64_of_bin_file(background_path)

st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)),
                    url("data:image/png;base64,{base64_bg}");
        background-size: cover;
        background-attachment: fixed;
        color: white;
    }}

    h1 {{
        color: #00e676;
        text-align: center;
        font-weight: 900;
        text-shadow: 2px 2px 5px black;
    }}

    /* Slider */
    .stSlider label, .stSlider span {{
        color: white !important;
        font-weight: bold;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

#layout
st.title("♻️ EnviroVision - AI phân loại rác") 
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"]) 
conf_threshold = st.slider("🔧 Ngưỡng độ tin cậy (Càng thấp thì mô hình sẽ nhận diện được nhiều hơn nhưng độ chính xác giảm dần)", 0.1, 0.9, 0.3, 0.05) 
if uploaded_file is not None: 
    image = Image.open(uploaded_file).convert("RGB") 
    st.image(image, caption="Ảnh gốc", use_column_width=True)

st.markdown(
    f"""
    <style>
    /* Ẩn menu, footer, GitHub icon */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    .stApp {{
        background: linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)),
                    url("data:image/png;base64,{base64_bg}");
        background-size: cover;
        background-attachment: fixed;
        font-family: 'Montserrat', sans-serif;
    }}

    /* Card container */
    .block-container {{
        background: rgba(0,0,0,0.55);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 30px rgba(0,0,0,0.4);
        max-width: 750px;
        margin: auto;
    }}

    h1 {{
        color: #00e676;
        text-align: center;
        font-weight: 900;
        text-shadow: 2px 2px 5px black;
    }}

    /* File uploader */
    .stFileUploader label {{
        color: white !important;
        font-weight: bold;
        text-align: center;
    }}
    .stFileUploader div div {{
        background-color: rgba(0,0,0,0.6) !important;
        border: 2px dashed #00e676 !important;
        border-radius: 12px;
        text-align: center;
        color: white !important;
    }}

    /* Slider */
    .stSlider label, .stSlider span {{
        color: white !important;
        font-weight: bold;
    }}

    /* Nút xanh */
    div.stButton > button:first-child {{
        background-color: #00c853;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.3);
    }}
    div.stButton > button:first-child:hover {{
        background-color: #00e676;
        color: black;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

if st.button("Chạy nhận diện"): 
    with st.spinner("⚙️ Đang xử lý..."): 
        result_img, results = detect_and_classify(image, conf_threshold) 
        st.image(result_img, caption="Kết quả nhận diện", use_column_width=True) 
        
        st.subheader("Kết quả phân loại:") 
        for label, conf, _ in results: 
            st.write(f"**{label}** - Độ tin cậy: {conf:.2f}")
