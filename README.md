SmartVision AI – Intelligent Multi-Class Object Recognition System

SmartVision AI is an end-to-end computer vision application designed for accurate image classification and real-time multi-object detection. The system leverages deep learning and transfer learning techniques to recognize and localize multiple objects within a single image under real-world conditions.

The project integrates state-of-the-art CNN architectures including VGG16, ResNet50, MobileNetV2, and EfficientNetB0 for image classification, along with YOLOv8 for high-speed object detection. A balanced 25-class subset of the COCO 2017 dataset is used to ensure robust training, fair evaluation, and efficient performance.

Tech Stack:
Python, PyTorch, OpenCV, Deep Learning, CNN, Transfer Learning, YOLOv8, Streamlit, Hugging Face Spaces

Dataset:
COCO 2017 – Curated 25-Class Subset  
Total Images: 2,500 (100 images per class)  
Train / Validation / Test Split: 70% / 15% / 15%

Models Implemented:
VGG16 – Baseline transfer learning model for classification  
ResNet50 – Fine-tuned deep residual network for improved accuracy  
MobileNetV2 – Lightweight model optimized for fast inference  
EfficientNetB0 – High-accuracy model with optimized scaling  
YOLOv8 – Real-time multi-object detection with bounding boxes  

Expected Performance:
Classification Accuracy: 80% – 93%  
YOLOv8 mAP@0.5: 85% – 90%  
Inference Speed: 30–50 FPS on GPU  

Application Overview:
The project is deployed as a Streamlit-based web application that allows users to upload images and perform image classification, multi-object detection, model comparison, and performance visualization through an interactive interface.

Use Cases:
Smart cities and traffic monitoring, retail and e-commerce analytics, security and surveillance systems, wildlife conservation, healthcare monitoring, smart home automation, agriculture, and logistics management.

Deployment:
The application is cloud-deployed on Hugging Face Spaces with a scalable and production-ready architecture.

Created By: Vishal Singla
