from fastapi import FastAPI, Request
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
import sys
import os
import requests
from requests.auth import HTTPBasicAuth

sys.path.append(os.path.join(os.path.dirname(__file__)))

from predict import predict
from treatment_advice import get_advice

app = FastAPI()

TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")

@app.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    form_data = await request.form()
    media_url = form_data.get("MediaUrl0")
    
    print(f"DEBUG: Media URL = {media_url}")
    print(f"DEBUG: SID set = {bool(TWILIO_ACCOUNT_SID)}")
    
    resp = MessagingResponse()
    
    if not media_url:
        resp.message("Please send a photo of the diseased crop leaf.")
        return Response(content=str(resp), media_type="application/xml")
    
    image_response = requests.get(media_url, auth=HTTPBasicAuth(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
    
    print(f"DEBUG: Status code = {image_response.status_code}")
    print(f"DEBUG: Content type = {image_response.headers.get('content-type')}")
    print(f"DEBUG: Content length = {len(image_response.content)}")
    
    temp_path = "temp_whatsapp_leaf.jpg"
    with open(temp_path, "wb") as f:
        f.write(image_response.content)
    
    disease, confidence = predict(temp_path)
    advice = get_advice(disease)
    
    os.remove(temp_path)
    
    reply = f"""🌿 *KisanAI Diagnosis*

*Disease:* {advice['english_name']}
*रोग:* {advice['hindi_name']}
*Confidence:* {confidence:.1f}%

💊 *Chemical Treatment:*
{advice['chemical']}

🌱 *Organic Treatment:*
{advice['organic']}

🛡️ *Prevention:*
{advice['prevention']}

🇮🇳 *Hindi Advice:*
{advice['hindi_advice']}"""
    
    resp.message(reply)
    return Response(content=str(resp), media_type="application/xml")

@app.get("/")
def home():
    return {"status": "KisanAI WhatsApp bot is running"}
