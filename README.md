# 😄 Human Emotion Detection (Multimodal)

This project is a **multimodal emotion classification system** that detects human emotions using **both facial expressions (images)** and **textual input (sentences or phrases)**. Built using **TensorFlow, NLTK, and Streamlit**, the system integrates **computer vision (CNN)** and **natural language processing (LSTM)** models to provide accurate emotion predictions in **real time**.

---

## 📌 Features

- 🧠 **Multimodal Emotion Detection**  
  Combines facial and textual inputs to predict emotions across **five categories**.

- 🖼️ **Computer Vision Stream (CV)**  
  CNN-based image classifier trained on facial datasets with ~70% accuracy.

- 📝 **Natural Language Processing Stream (NLP)**  
  LSTM-based text classifier trained on annotated emotional sentences with ~92% accuracy.

- ⚡ **Real-Time Inference via Streamlit**  
  User-friendly UI deployed using Streamlit for live image and text input.

---

## 🎯 Emotion Classes

The model classifies both text and image inputs into the following **five emotions**:

- 😄 Happy  
- 😠 Angry  
- 😢 Sad  
- 😨 Fear  
- 😐 Neutral  

---

## 🛠️ Tech Stack

| Category        | Tools / Frameworks     |
|----------------|-------------------------|
| Language        | Python                  |
| Deep Learning   | TensorFlow, Keras       |
| NLP             | NLTK, LSTM              |
| Computer Vision | CNN, OpenCV             |
| UI              | Streamlit               |
| Data Handling   | NumPy, Pandas           |


---

## 🚀 Running the Project

### 🔧 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### ▶️ 2. Run the Streamlit App
```bash
streamlit run streamlit_app/app.py
```

---

## 🎥 How It Works

1. Upload a facial image or type an emotional sentence.
2. The backend model processes each input separately:
   - **Image → CNN → Emotion**
   - **Text → LSTM → Emotion**
3. The predicted emotions are displayed in real time via Streamlit.

---

## ✅ Results

| Model           | Accuracy |
|----------------|----------|
| CNN (facial)   | ~70%     |
| LSTM (textual) | ~92%     |

---
