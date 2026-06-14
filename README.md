# 🌿 KisanAI - Crop Disease Detection for Indian Farmers

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red)](https://streamlit.io)

## 🎯 Problem Statement

Indian farmers lose 20-30% of their crops every year due to plant diseases. Early detection is critical but most farmers lack access to agricultural experts. KisanAI solves this by letting a farmer photograph a diseased leaf and instantly receive a diagnosis with treatment advice in both Hindi and English.

## 🌾 Supported Crops & Diseases

| Crop | Diseases Detected |
|------|------------------|
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus |
| Potato | Early Blight, Late Blight |
| Bell Pepper | Bacterial Spot |
| Rice | Bacterial Leaf Blight, Brown Spot, Leaf Blast, Sheath Blight, Leaf Scald |
| Wheat | Leaf Rust, Loose Smut, Crown and Root Rot |
| Cotton | Diseased Leaf, Diseased Plant |

**Total: 29 classes across 6 crops**

## 🧠 Model Architecture

- **Base Model:** EfficientNetB0 (pre-trained on ImageNet)
- **Approach:** Transfer Learning with Fine Tuning
- **Training Data:** 23,994 images across 29 classes
- **Validation Accuracy:** 89%
- **Input Size:** 224×224 RGB

## ✨ Features

- 🔍 **Disease Detection** — identifies 29 plant diseases across 6 crops
- 🗺️ **Grad-CAM Visualization** — highlights which part of the leaf the model focused on
- 💊 **Treatment Advice** — chemical and organic treatment recommendations
- 🇮🇳 **Bilingual Output** — advice in both Hindi and English
- 🌐 **Web Interface** — simple Streamlit app for easy use

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/soveshika/kisanai.git
cd kisanai

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📁 Project Structure
## 🔬 Tech Stack

- **Deep Learning:** TensorFlow, Keras, EfficientNetB0
- **Explainability:** Grad-CAM
- **Web App:** Streamlit
- **Image Processing:** OpenCV, PIL
- **Language:** Python 3.11

## 🔮 Future Work

- WhatsApp bot integration for farmers without internet access
- Support for wheat, maize, and sugarcane diseases
- Voice output in Hindi for farmers with low literacy
- Mobile app deployment
- Multi-language support (Punjabi, Gujarati, Marathi)

## 👨‍💻 Author

**Sonu Ravish** — AI/ML Engineer
- GitHub: [@soveshika](https://github.com/soveshika)

## 📄 License

MIT License

kisanai/

├── src/

│   ├── predict.py          # Disease prediction

│   ├── treatment_advice.py # Hindi + English treatment advice

│   ├── gradcam.py          # Grad-CAM visualization

│   └── whatsapp_bot.py     # WhatsApp integration (coming soon)

├── models/

│   ├── kisanai_model.h5    # Trained model

│   └── class_indices.json  # Class mappings

├── app.py                  # Streamlit web app

├── requirements.txt

└── README.md