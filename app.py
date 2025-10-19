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

import torch
import torch.nn as nn
from torchvision import transforms
import timm
import numpy as np
from PIL import Image
import os
import cv2 # For reading/writing/displaying images
from huggingface_hub import hf_hub_download

# --- Configuration ---
NUM_CLASSES = 22 # The number of classes from your previous evaluation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Hugging Face Configuration ---
HF_REPO_ID = "Noob1746/EnviroVision"
HF_MODEL_FILENAME = "efficientnetv2-b0 .pth" # Note the space in the filename

# Class names must match the order used during training for the 22-class model
CLASS_NAMES_22 = [
    'HDPE', 'LDPE', 'Other plastic', 'PET', 'PP', 'PS', 'PVC', 
    'aerosol', 'battery', 'cardboard', 'charger', 'clothes', 
    'computer', 'glass', 'keyboard', 'mouse', 'organic', 
    'paper', 'phone', 'recyclable metal', 'remote control', 'shoes'
]

# --- Model Loading Function ---

def load_efficientnet_b0(repo_id, filename, num_classes, device):
    try:
        checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename)
        print(f"✅ Download complete. Checkpoint saved locally at: {checkpoint_path}")
    except Exception as e:
        print(f"❌ Error downloading model from Hugging Face: {e}")
        raise
    
    # Use timm to create the model structure, matching your training script
    model = timm.create_model(
        'tf_efficientnetv2_b0',
        pretrained=False, # We load our custom checkpoint
        num_classes=num_classes
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load the state dictionary. Prioritize EMA state which typically performed better.
    if "ema_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["ema_state_dict"]) 
        print(f"Loaded EMA state. Val Acc: {checkpoint.get('val_acc', 'N/A'):.4f}")
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded Raw state. Val Acc: {checkpoint.get('val_acc', 'N/A'):.4f}")
    else:
        # Fallback for old/simple checkpoints that only save the model state
        try:
            model.load_state_dict(checkpoint)
            print("Loaded model state directly (assuming simple state dict structure).")
        except:
             raise ValueError("Checkpoint file does not contain a recognizable model state ('ema_state_dict', 'model_state_dict', or simple state dict).")

    model.to(device)
    model.eval()

    print(f"✅ Model loaded and set to evaluation mode on {device}.")
    return model

# --- Preprocessing Function ---

def preprocess_image(image: np.ndarray):
    """
    Transforms the input NumPy image (H, W, C) into the required PyTorch tensor (1, 3, 224, 224).
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    # The output shape is (1, 3, 224, 224)
    return transform(image).unsqueeze(0)

def classify_single_image(image_input, classification_model, class_names, device):
    """
    Takes an image (NumPy array or PIL Image), preprocesses it, 
    and classifies it using the EfficientNet model.
    """
    # Convert input to NumPy array
    if isinstance(image_input, Image.Image):
        img_np = np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        img_np = image_input
    else:
        raise TypeError("Input must be a PIL Image or NumPy array.")

    # 1. Preprocess
    pre_img_tensor = preprocess_image(img_np)
    pre_img_tensor = pre_img_tensor.to(device)

    # 2. Classify
    with torch.no_grad():
        outputs = classification_model(pre_img_tensor)
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get the class with the highest probability
        conf_score, predicted = torch.max(probs, 1)
        
        conf_score = conf_score.item()
        label = class_names[predicted.item()]

    # 3. Return result and the original image (for display)
    return label, conf_score, img_np


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

st.title("♻️ EnviroVision - AI phân loại rác")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
conf_threshold = st.slider(
    "🔧 Ngưỡng độ tin cậy (Càng thấp thì mô hình sẽ nhận diện được nhiều hơn nhưng độ chính xác giảm dần)", 0.1, 0.9,
    0.3, 0.05)
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh gốc", use_container_width=True)

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
        border: 2px solid #555555 !important;
        border-radius: 3px;
        text-align: center;
        color: white !important;
        transition: all 0.3s ease-in-out;
    }}

    .stFileUploader div div:hover {{
        background-color: rgba(255,255,255,0.1) !important;
        border-color: #cccccc !important;
        box-shadow: 0 0 10px rgba(255,255,255,0.2);
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
        st.image(result_img, caption="Kết quả nhận diện", use_container_width=True)

        # Màu chữ khớp với màu khung trên ảnh
        color_map = {
            "biodegradable": "rgb(0, 200, 0)",      # Xanh lá
            "cardboard": "rgb(42, 157, 244)",       # Xanh dương nhạt
            "clothes": "rgb(255, 105, 180)",        # Hồng
            "glass": "rgb(255, 255, 0)",            # Vàng
            "metal": "rgb(192, 192, 192)",          # Xám bạc
            "paper": "rgb(255, 128, 0)",            # Cam đậm
            "plastic": "rgb(0, 165, 255)",          # Xanh biển
            "shoes": "rgb(219, 112, 147)",          # Tím hồng
        }

        vietnamese_labels = {
            "biodegradable": "Rác hữu cơ",
            "cardboard": "Bìa cứng",
            "clothes": "Quần áo",
            "glass": "Thủy tinh",
            "metal": "Kim loại",
            "paper": "Giấy",
            "plastic": "Nhựa",
            "shoes": "Giày dép",
        }
        st.subheader("Kết quả phân loại:")
        for label, conf, _ in results:
            color = color_map.get(label, "rgb(0, 255, 0)")
            vietnamese_name = vietnamese_labels.get(label, label)
            st.markdown(
                f"""
                <span style="
                    color:{color};
                    font-weight:bold;
                    font-size:16px;
                ">
                    {vietnamese_name}
                </span>
                <span style="color:white;"> — Độ tin cậy: {conf:.2f}</span>
                """,
                unsafe_allow_html=True
            )
