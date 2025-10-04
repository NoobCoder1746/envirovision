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

# ===============================
# Thiết bị
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# Hàm load mô hình EfficientNet-V2-S Phase3b
# ===============================
def load_efficientnet(checkpoint_path, num_classes=8, device="cpu"):
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"✅ Loaded EfficientNet-V2-S model from {checkpoint_path}")
    return model

# ===============================
# Load YOLOv8 mô hình phát hiện vật thể
# ===============================
yolo_model_path = hf_hub_download(
    repo_id="Noob1746/EnviroVision",
    filename="best.pt"
)
yolo_model = YOLO(yolo_model_path)

# ===============================
# Load EfficientNet-V2-S Phase3b model
# ===============================
efficientnetv2s_model_path = hf_hub_download(
    repo_id="Noob1746/EnviroVision",
    filename="class.pth"  # Đảm bảo tên file trùng với file bạn upload
)
classification_model = load_efficientnet(
    efficientnetv2s_model_path, num_classes=8, device=device
)

# ===============================
# Danh sách nhãn
# ===============================
class_names = [
    "biodegradable",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes"
]

# ===============================
# Hàm xử lý ảnh đầu vào
# ===============================
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# ===============================
# Hàm detect + classify
# ===============================
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
            cv2.putText(img_bgr, f"{label} {conf_score:.2f}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb, final_results

# ===============================
# Giao diện Streamlit
# ===============================
st.set_page_config(page_title="EnviroVision", page_icon="♻️", layout="centered")

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

background_path = "realfish-Kh8aGCgWZLg-unsplash.jpg"
base64_bg = get_base64_of_bin_file(background_path)

# Giao diện CSS
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
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# Nội dung chính
# ===============================
st.title("♻️ EnviroVision - AI phân loại rác")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
conf_threshold = st.slider("🔧 Ngưỡng độ tin cậy", 0.1, 0.9, 0.3, 0.05)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh gốc", use_container_width=True)

if st.button("Chạy nhận diện"):
    with st.spinner("⚙️ Đang xử lý..."):
        result_img, results = detect_and_classify(image, conf_threshold)
        st.image(result_img, caption="Kết quả nhận diện", use_container_width=True)

        st.subheader("Kết quả phân loại:")
        for label, conf, _ in results:
            st.write(f"**{label}** - Độ tin cậy: {conf:.2f}")
