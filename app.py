import streamlit as st
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import base64
import timm
import os

NUM_CLASSES = 22
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HF_REPO_ID = "Noob1746/EnviroVision"
HF_MODEL_FILENAME = "efficientnetv2-b0 .pth"

CLASS_NAMES_22 = [
    'HDPE', 'LDPE', 'Other plastic', 'PET', 'PP', 'PS', 'PVC', 
    'aerosol', 'battery', 'cardboard', 'charger', 'clothes', 
    'computer', 'glass', 'keyboard', 'mouse', 'organic', 
    'paper', 'phone', 'recyclable metal', 'remote control', 'shoes'
]


TRASH_BIN_MAP = {
    'HDPE': {"bin": "Tái chế", "emoji": "♻️", "color": "#0099cc"}, 
    'LDPE': {"bin": "Tái chế", "emoji": "♻️", "color": "#0099cc"},
    'PET': {"bin": "Tái chế", "emoji": "♻️", "color": "#0099cc"},
    'PP': {"bin": "Tái chế", "emoji": "♻️", "color": "#0099cc"},
    'PS': {"bin": "Tái chế", "emoji": "♻️", "color": "#0099cc"},
    'PVC': {"bin": "Tái chế", "emoji": "♻️", "color": "#0099cc"},
    'Other plastic': {"bin": "Vô cơ", "emoji": "🗑️", "color": "#999999"}, 
    'battery': {"bin": "Rác nguy hại (nếu có). Tuyệt đối KHÔNG bỏ vào những thùng kia", "emoji": "🔋⚠️", "color": "#ff3333"},
    'charger': {"bin": "Rác nguy hại (nếu có). Tuyệt đối KHÔNG bỏ vào những thùng kia", "emoji": "💻🔌", "color": "#cc66ff"},
    'computer': {"bin": "Rác nguy hại (nếu có). Tuyệt đối KHÔNG bỏ vào những thùng kia", "emoji": "💻", "color": "#cc66ff"},
    'keyboard': {"bin": "Rác nguy hại (nếu có). Tuyệt đối KHÔNG bỏ vào những thùng kia", "emoji": "⌨️", "color": "#cc66ff"},
    'mouse': {"bin": "Rác nguy hại (nếu có). Tuyệt đối KHÔNG bỏ vào những thùng kia", "emoji": "🖱️", "color": "#cc66ff"},
    'phone': {"bin": "Rác nguy hại (nếu có). Tuyệt đối KHÔNG bỏ vào những thùng kia", "emoji": "📱", "color": "#cc66ff"},
    'remote control': {"bin": "Rác nguy hại (nếu có). Tuyệt đối KHÔNG bỏ vào những thùng kia", "emoji": "📺", "color": "#cc66ff"},
    'aerosol': {"bin": "Rác nguy hại (Bình xịt) nên bỏ vào thùng rác nguy hại (nếu có) nếu không có thì tuyệt đối không bỏ vào những thùng kia", "emoji": "💥", "color": "#ff3333"},
    'cardboard': {"bin": "Tái chế", "emoji": "📰", "color": "#00cc66"},
    'paper': {"bin": "Tái chế", "emoji": "📰", "color": "#00cc66"},
    'glass': {"bin": "Tái chế", "emoji": "🥂", "color": "#ffcc00"},
    'recyclable metal': {"bin": "Tái chế", "emoji": "⚙️", "color": "#9999ff"},
    'clothes': {"bin": "Vô cơ. Nhưng nên từ thiện hoặc cho đi", "emoji": "🗑️", "color": "#999999"}, 
    'shoes': {"bin": "Vô cơ. Nhưng nên từ thiện hoặc cho đi", "emoji": "🗑️", "color": "#999999"},
    'organic': {"bin": "Hữu cơ", "emoji": "🥬", "color": "#ff6600"},
}


@st.cache_resource
def load_efficientnet_b0(repo_id, filename, num_classes, device):
    try:
        with st.spinner(f"Downloading model '{filename}' from Hugging Face..."):
            checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception as e:
        st.error(f"❌ Error downloading model: {e}")
        st.stop()
    
    model = timm.create_model(
        'tf_efficientnetv2_b0',
        pretrained=False,
        num_classes=num_classes
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    state_dict_keys = ["ema_state_dict", "model_state_dict"]
    loaded = False
    for key in state_dict_keys:
        if key in checkpoint:
            model.load_state_dict(checkpoint[key])
            loaded = True
            break
    
    if not loaded:
        try:
            model.load_state_dict(checkpoint)
        except Exception:
            raise ValueError("Checkpoint file does not contain a recognizable model state.")

    model.to(device)
    model.eval()
    return model


def preprocess_image(image: np.ndarray):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def classify(image_input, classification_model, class_names, device):
    if isinstance(image_input, Image.Image):
        img_np = np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        img_np = image_input
    else:
        raise TypeError("Input must be a PIL Image or NumPy array.")
        
    pre_img_tensor = preprocess_image(img_np)
    pre_img_tensor = pre_img_tensor.to(device)
    
    with torch.no_grad():
        outputs = classification_model(pre_img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf_score, predicted = torch.max(probs, 1)
        
        conf_score = conf_score.item()
        label = class_names[predicted.item()]
        
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    text = f"{label}: {conf_score:.2f}"
    color = (0, 255, 0)
    cv2.putText(img_bgr, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    
    result_img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    results_list = [(label, conf_score, None)]
    
    return result_img_rgb, results_list

try:
    classification_model = load_efficientnet_b0(HF_REPO_ID, HF_MODEL_FILENAME, NUM_CLASSES, DEVICE)
except Exception as e:
    st.error(f"Failed to initialize model. Please check the Hugging Face path. Error: {e}")
    st.stop()


def get_base64_of_bin_file(bin_file):
    if not os.path.exists(bin_file):
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
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
    
    .stSlider {{ display: none; }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    .block-container {{
        background: rgba(0,0,0,0.55);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 30px rgba(0,0,0,0.4);
        max-width: 750px;
        margin: auto;
    }}

    .stFileUploader label, .stFileUploader div div {{
        color: white !important;
        font-weight: bold;
        text-align: center;
    }}

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

st.title("♻️ EnviroVision - AI phân loại rác")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

conf_threshold = 0.0

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh gốc", use_container_width=True)

if st.button("Chạy nhận diện") and uploaded_file is not None:
    with st.spinner("⚙️ Đang xử lý..."):
        try:
            result_img, results = classify(image, classification_model, CLASS_NAMES_22, DEVICE)
            st.image(result_img, caption="Kết quả phân loại", use_container_width=True)

            VIETNAMESE_LABELS = {
                "HDPE": "Nhựa HDPE", "LDPE": "Nhựa LDPE", "Other plastic": "Nhựa khác", "PET": "Nhựa PET",
                "PP": "Nhựa PP", "PS": "Nhựa PS", "PVC": "Nhựa PVC", "aerosol": "Bình xịt",
                "battery": "Pin/ắc quy", "cardboard": "Bìa cứng", "charger": "Sạc điện thoại/máy tính",
                "clothes": "Quần áo", "computer": "Thiết bị điện tử", "glass": "Thủy tinh",
                "keyboard": "Bàn phím", "mouse": "Chuột máy tính", "organic": "Rác hữu cơ",
                "paper": "Giấy", "phone": "Điện thoại", "recyclable metal": "Kim loại tái chế",
                "remote control": "Điều khiển", "shoes": "Giày dép",
            }

            st.subheader("Kết quả phân loại:")
            for label, conf, _ in results:
                bin_info = TRASH_BIN_MAP.get(label, {"bin": "Không xác định", "emoji": "❓", "color": "#ff0000"})
                
                color = "rgb(0, 255, 0)"
                vietnamese_name = VIETNAMESE_LABELS.get(label, label)
                
                st.markdown(
                    f"""
                    <span style="
                        color:{color};
                        font-weight:bold;
                        font-size:20px; 
                    ">
                        {vietnamese_name}
                    </span>
                    <span style="color:white; font-size: 20px;"> — Độ tin cậy: {conf:.4f}</span>
                    """,
                    unsafe_allow_html=True
                )
                
                st.markdown(
                    f"""
                    <div style="
                        background-color: {bin_info['color']}20; 
                        border-left: 5px solid {bin_info['color']}; 
                        padding: 10px; 
                        margin-top: 10px; 
                        border-radius: 5px;
                    ">
                        <p style="
                            font-weight: bold; 
                            font-size: 18px; 
                            color: {bin_info['color']}; 
                            margin: 0;
                        ">
                            {bin_info['emoji']} Bỏ vào thùng: {bin_info['bin']}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
        except Exception as e:
            st.error(f"❌ Lỗi trong quá trình phân loại: {e}")
