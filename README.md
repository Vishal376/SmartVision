SmartVision AI – Intelligent Multi-Class Object Recognition System

SmartVision AI is a computer vision project that performs image classification and multi-object detection using deep learning and transfer learning techniques. The system is designed to work on real-world images containing multiple objects and provides fast and accurate predictions through a web-based interface.

The project uses pre-trained CNN models such as VGG16, ResNet50, MobileNetV2, and EfficientNetB0 for image classification and YOLOv8 for real-time object detection. A curated subset of 25 classes from the COCO 2017 dataset is used to ensure balanced training and efficient learning.

Tech Stack:
Python, PyTorch, OpenCV, CNN, Transfer Learning, YOLOv8, Streamlit, Hugging Face Spaces

Dataset:
COCO 2017 – 25 class subset
Total images: 2,500 (100 images per class)
Train/Validation/Test split: 70% / 15% / 15%

Models:
VGG16 – Transfer learning based classification model
ResNet50 – Fine-tuned deep residual network
MobileNetV2 – Lightweight and fast inference model
EfficientNetB0 – High accuracy optimized model
YOLOv8 – Multi-object detection with bounding boxes

Performance (Expected):
Classification Accuracy: 80% – 93%
YOLOv8 mAP@0.5: 85% – 90%
Inference Speed: 30–50 FPS (GPU)

Application:
A Streamlit-based web application with image classification, object detection, model comparison, and performance visualization features. The application is deployed on Hugging Face Spaces for public access.

Use Cases:
Smart cities, traffic monitoring, retail analytics, security and surveillance, wildlife conservation, healthcare, smart homes, agriculture, and logistics.

Deployment:
Streamlit application deployed on Hugging Face Spaces with cloud-ready and scalable architecture.

Created By: Vishal Singla

