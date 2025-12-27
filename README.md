# 🖼️ Intelligent Image Captioning System

## 📌 Introduction
This project implements an **Intelligent Image Captioning System** that generates a natural language description for an input image.  
It integrates **Computer Vision** and **Natural Language Processing (NLP)** using deep learning techniques.

The system allows users to:
- Upload an image
- Provide an image URL
- Receive an automatically generated caption




## 🗂️ Project Folder Structure

Root/
├── app.py # Main Streamlit app
├── requirements.txt # All required Python packages
├── vocab.pkl # Vocabulary file
├── caption_model.pth # Trained model weights
├── last_ver.py # Model classes (Encoder/Decoder/Vocabulary)
├── dataset/
└── subset_coco/ # Dataset folder
├── images/ # All training/test images
│ ├── train2017/
│ │ ├── 000000000001.jpg
│ │ ├── 000000000002.jpg
│ │ └── ...
│ └── val2017/
│ ├── 000000000001.jpg
│ └── ...
└── annotations/
├── captions_train2017.json
└── captions_val2017.json
├── README.md # Project documentation     
│  
│
└── README.md                  # Project documentation

---

## 🧠 System Overview
The model follows an **Encoder–Decoder architecture**:

### 🔹 Encoder (CNN)
- Extracts visual features from the input image
- Converts the image into a feature vector

### 🔹 Decoder (LSTM)
- Takes image features as input
- Generates a sentence word-by-word
- Uses special tokens:
  - `<SOS>`: Start of sentence
  - `<EOS>`: End of sentence

---

## ⚙️ Caption Generation Process
1. Image is resized and normalized
2. CNN extracts image features
3. Decoder initializes hidden and cell states from features
4. LSTM predicts the next word iteratively
5. Generation stops when `<EOS>` is predicted or max length is reached

---

## 🖥️ User Interface (Streamlit)
- Built using **Streamlit**
- Supports:
  - Image upload
  - Image URL input
- Sidebar settings allow caption length adjustment
- Results include:
  - Generated caption
  - Word statistics

---

## 🛠️ Technologies Used
- Python
- PyTorch
- Torchvision
- Streamlit
- PIL
- Pickle

---

## 📂 Project Structure
