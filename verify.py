from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()
api_key = ("AIzaSyCO6fRFlmVfC7mlJuz63DwxZntVPGM8luw")
print("Loaded key:", api_key[:12])

genai.configure(api_key=api_key)

try:
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content("Say hi in one word.")
    print("✅ Gemini reply:", response.text)
except Exception as e:
    print("❌ Error:", e)
