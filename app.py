import gradio as gr
import numpy as np
from PIL import Image
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from predict import predict
from treatment_advice import get_advice
from gradcam import save_gradcam

def analyse_leaf(image):
    temp_path = "temp_leaf.jpg"
    image.save(temp_path)
    
    disease, confidence = predict(temp_path)
    advice = get_advice(disease)
    gradcam_path = save_gradcam(temp_path, "gradcam_output.jpg")
    
    result = f"""
## 🌿 Diagnosis Result

**Disease Detected:** {advice['english_name']}
**रोग:** {advice['hindi_name']}
**Confidence:** {confidence:.1f}%

---

### 💊 Chemical Treatment
{advice['chemical']}

### 🌱 Organic Treatment
{advice['organic']}

### 🛡️ Prevention
{advice['prevention']}

### 🇮🇳 Hindi Advice | हिंदी सलाह
**{advice['hindi_advice']}**
"""
    
    gradcam_img = Image.open(gradcam_path)
    os.remove(temp_path)
    
    return result, gradcam_img

demo = gr.Interface(
    fn=analyse_leaf,
    inputs=gr.Image(type="pil", label="Upload Leaf Image"),
    outputs=[
        gr.Markdown(label="Diagnosis"),
        gr.Image(label="Grad-CAM: Where the model looked")
    ],
    title="🌿 KisanAI - Crop Disease Detection",
    description="फसल रोग पहचान प्रणाली | Upload a crop leaf photo to get instant disease diagnosis in Hindi and English.",
    examples=None
)

demo.launch()
