# import streamlit as st
# from ultralytics import YOLO
# from PIL import Image
# import tempfile
# import cv2
# import av
# from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image
# import os
# from huggingface_hub import hf_hub_download

# cache_dir = "models_cache"
# os.makedirs(cache_dir, exist_ok=True)

# VGG16_best = hf_hub_download(
#     repo_id="jgvghf/smartvision",
#     filename="VGG16_best.pth",
#     token=st.secrets["HuggingFace_token"],
#     cache_dir=cache_dir
# )

# yolo_best = hf_hub_download(
#     repo_id="jgvghf/smartvision",
#     filename="best.pt",
#     token=st.secrets["HuggingFace_token"],
#     cache_dir=cache_dir
# )


# st.set_page_config(page_title="SmartVision AI", layout="centered")
# st.title("🔍 SmartVision AI – Object Detection")

# # ---------------- LOAD MODEL ----------------
# model = YOLO(yolo_best)

# # ---------------- MODE SELECTION ----------------
# mode = st.radio("Select Mode", ["📁 Image Upload", "📷 Webcam","🧠 Image Classification"])

# # ==================================================
# # 📁 IMAGE UPLOAD MODE
# # ==================================================
# if mode == "📁 Image Upload":

#     uploaded_img = st.file_uploader(
#         "Upload Image",
#         type=["jpg", "jpeg", "png"]
#     )

#     if uploaded_img is not None:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
#             tmp.write(uploaded_img.read())
#             image_path = tmp.name

#         results = model(image_path, conf=0.35)
#         res = results[0]

#         annotated_img = res.plot()
#         annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

#         st.image(annotated_img, caption="Detected Objects", use_container_width=True)

#         if res.boxes is not None:
#             st.success(f"Detected Objects: {len(res.boxes)}")

# # ==================================================
# # 📷 WEBCAM MODE
# # ==================================================
# elif mode == "📷 Webcam":

#     class YOLOVideoProcessor(VideoProcessorBase):
#         def recv(self, frame):
#             img = frame.to_ndarray(format="bgr24")

#             results = model(img, conf=0.35)
#             res = results[0]

#             annotated_frame = res.plot()
#             return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

#     webrtc_streamer(
#         key="yolo-webcam",
#         video_processor_factory=YOLOVideoProcessor,
#         media_stream_constraints={
#             "video": True,
#             "audio": False
#         },
#         async_processing=True
#     )
# else:
    


#     # =====================================================
#     # PAGE CONFIG
#     # =====================================================
#     # st.set_page_config(page_title="SmartVision AI - Classification", layout="centered")
#     st.title("🧠 SmartVision AI – Image Classification (VGG16)")
    
#     # =====================================================
#     # LOAD CLASS NAMES
#     # =====================================================
#     # CLASS_DIR = "smartvision_dataset/classification/train"
#     class_names= ['airplane', 'bed', 'bench', 'bicycle', 'bird', 'bottle', 'bowl', 'bus', 'cake', 
#                   'car', 'cat', 'chair', 'couch', 'cow', 'cup', 'dog', 'elephant', 'horse', 'motorcycle', 
#                   'person', 'pizza', 'potted plant', 'stop sign', 'traffic light', 'truck']
#     # class_names = sorted(os.listdir(CLASS_DIR))
#     NUM_CLASSES = len(class_names)
    
#     # =====================================================
#     # LOAD VGG16 MODEL
#     # =====================================================
#     @st.cache_resource
#     def load_vgg16():
#        vggmodel = models.vgg16(pretrained=False)
#        vggmodel.classifier[6] = nn.Linear(4096, NUM_CLASSES)

#        vggmodel.load_state_dict(
#         torch.load(VGG16_best, map_location="cpu")
#         )

#        vggmodel.eval()
#        return vggmodel

#     vggmodel = load_vgg16()

#     # =====================================================
#     # IMAGE TRANSFORMS
#     # =====================================================
#     transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
#     ])
    
#     # =====================================================
#     # IMAGE UPLOAD
#     # =====================================================
#     uploaded_img = st.file_uploader(
#     "📁 Upload an image for classification",
#     type=["jpg", "jpeg", "png"]
#     )

#     if uploaded_img:

#       image = Image.open(uploaded_img).convert("RGB")
#       st.image(image, caption="Uploaded Image", use_container_width=True)

#       input_tensor = transform(image).unsqueeze(0)

#       with torch.no_grad():
#         outputs = vggmodel(input_tensor)
#         probs = torch.softmax(outputs, dim=1)
#         confidence, predicted = torch.max(probs, 1)

#       st.success(
#         f"🧠 Prediction: **{class_names[predicted.item()]}**\n\n"
#         f"🎯 Confidence: **{confidence.item()*100:.2f}%**"
#        )


import streamlit as st
import os
import tempfile
import cv2
import av
import torch
import torch.nn as nn
from PIL import Image
from ultralytics import YOLO
from torchvision import models, transforms
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from huggingface_hub import hf_hub_download

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="SmartVision AI", layout="centered")
st.title("🚀 SmartVision AI")

# =====================================================
# DOWNLOAD MODELS (HF HUB)
# =====================================================
@st.cache_resource
def download_models():
    cache_dir = "models_cache"
    os.makedirs(cache_dir, exist_ok=True)

    vgg_path = hf_hub_download(
        repo_id="jgvghf/smartvision",
        filename="VGG16_best.pth",
        token=st.secrets["HuggingFace_token"],
        cache_dir=cache_dir
    )

    yolo_path = hf_hub_download(
        repo_id="jgvghf/smartvision",
        filename="best.pt",
        token=st.secrets["HuggingFace_token"],
        cache_dir=cache_dir
    )

    return vgg_path, yolo_path

VGG16_PATH, YOLO_PATH = download_models()

# =====================================================
# LOAD MODELS
# =====================================================
@st.cache_resource
def load_yolo():
    model = YOLO(YOLO_PATH)
    return model

@st.cache_resource
def load_vgg16():
    class_names = [
        'airplane','bed','bench','bicycle','bird','bottle','bowl','bus','cake',
        'car','cat','chair','couch','cow','cup','dog','elephant','horse',
        'motorcycle','person','pizza','potted plant','stop sign',
        'traffic light','truck'
    ]

    model = models.vgg16(pretrained=False)
    model.classifier[6] = nn.Linear(4096, len(class_names))
    model.load_state_dict(torch.load(VGG16_PATH, map_location="cpu"))
    model.eval()

    return model, class_names

yolo_model = load_yolo()
vgg_model, CLASS_NAMES = load_vgg16()

# =====================================================
# TABS (2 PAGE APP)
# =====================================================
tab1, tab2 = st.tabs(["🔍 Object Detection", "🧠 Image Classification"])

# =====================================================
# 🔍 OBJECT DETECTION PAGE
# =====================================================
with tab1:
    st.header("🔍 Object Detection (YOLO)")

    mode = st.radio("Select Mode", ["📁 Image Upload", "📷 Webcam"])

    if mode == "📁 Image Upload":
        img_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"],key="detector_uploader")

        if img_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(img_file.read())
                img_path = tmp.name

            results = yolo_model(img_path, conf=0.4)
            annotated = results[0].plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            st.image(annotated, caption="Detected Objects", use_container_width=True)
            st.success(f"Objects Detected: {len(results[0].boxes)}")

    else:
        class YOLOProcessor(VideoProcessorBase):
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                results = yolo_model(img, conf=0.4)
                return av.VideoFrame.from_ndarray(results[0].plot(), format="bgr24")

        webrtc_streamer(
            key="yolo-webcam",
            video_processor_factory=YOLOProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )

# =====================================================
# 🧠 IMAGE CLASSIFICATION PAGE
# =====================================================
with tab2:
    st.header("🧠 Image Classification (VGG16)")

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    img_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"],key="classifier_uploader")

    if img_file:
        image = Image.open(img_file).convert("RGB")
        st.image(image, use_container_width=True)

        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = vgg_model(tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)

        st.success(
            f"### 🧠 Prediction: **{CLASS_NAMES[pred.item()]}**\n"
            f"### 🎯 Confidence: **{conf.item()*100:.2f}%**"
        )
