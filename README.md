# 🩺 LuminaDerm

## AI-Powered Skin Condition Classification

LuminaDerm is an AI-powered web application designed to analyze skin images and predict the most likely skin condition using deep learning and computer vision.

The project uses a **ResNet18 deep learning model** trained on a 7-class dermatology image dataset. The trained model is integrated with a **FastAPI backend**, allowing users to upload an image and receive a predicted condition along with a confidence score.

> ⚠️ **Medical Disclaimer:** LuminaDerm is an experimental educational and research project. It is not intended to provide medical diagnosis, treatment, or professional medical advice. Users should consult a qualified healthcare professional for medical concerns.

---

## 🚀 Features

- 🖼️ Upload a skin image for analysis
- 🤖 AI-powered skin-condition classification
- 📊 Prediction confidence score
- 🧠 ResNet18-based deep learning model
- ⚡ FastAPI backend
- 🌐 Web-based application
- 🏋️ GPU-accelerated model training
- 📈 Model evaluation using accuracy, precision, recall, and F1-score
- 💾 Model checkpoint storage and backup

---

## 🩻 Supported Skin Conditions

The current LuminaDerm model is trained to classify the following **7 skin conditions**:

1. Acne
2. Eczema
3. Impetigo
4. Psoriasis
5. Rosacea
6. Tinea
7. Vitiligo

---

## 🧠 Deep Learning Model

LuminaDerm currently uses **ResNet18**, a convolutional neural network architecture designed for image classification.

The model was adapted and trained for multi-class skin-condition classification.

### Model Pipeline

```text
Skin Image
     │
     ▼
Image Preprocessing
     │
     ▼
ResNet18 Model
     │
     ▼
Feature Extraction
     │
     ▼
7-Class Classification
     │
     ▼
Predicted Condition
     │
     ▼
Confidence Score