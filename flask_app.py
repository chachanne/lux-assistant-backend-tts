from flask import Flask, request, jsonify, Response
import requests
import os # Importez os pour accéder aux variables d'environnement
import io
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
CORS(app) # Active CORS pour toutes les routes

# --- Configuration des APIs ---
LUXASR_API_URL = "https://luxasr.uni.lu/v2/asr"

# Configuration de Gemini API Key:
# LA CLÉ EST MAINTENANT RÉCUPÉRÉE DEPUIS LES VARIABLES D'ENVIRONNEMENT.
# NE JAMAIS CODER LA CLÉ EN DUR DANS LE FICHIER POUR LA PRODUCTION OU UN DÉPÔT PUBLIC !
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") # Récupère la variable d'environnement nommée GEMINI_API_KEY

# Vérification si la clé est présente (très important pour le débogage)
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY n'est pas définie dans les variables d'environnement. Veuillez la configurer.")

genai.configure(api_key=GEMINI_API_KEY) # Utilise la clé récupérée de l'environnement

# Le code pour lister les modèles n'est plus nécessaire après vérification, mais je le laisse commenté
# @app.route('/list_gemini_models')
# def list_gemini_models():
#     try:
#         models = genai.list_models()
#         model_list = []
#         for m in models:
#             model_info = {
#                 "name": m.name,
#                 "display_name": m.display_name,
#                 "supported_generation_methods": m.supported_generation_methods
#             }
#             model_list.append(model_info)
#         return jsonify(model_list)
#     except Exception as e:
#         return jsonify({"error": f"Failed to list models: {e}"}), 500

# Fonction pour appeler l'API LuxASR
def call_luxasr_api(audio_data, filename="audio.wav", content_type="audio/wav"):
    headers = {
        "accept": "application/json"
    }
    files = {
        "audio_file": (filename, audio_data, content_type)
    }
    params = {
        "diarization": "Disabled",
        "outfmt": "text"
    }
    try:
        response = requests.post(LUXASR_API_URL, headers=headers, files=files, params=params)
        response.raise_for_status()
        return response.text.strip()
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de l'appel à l'API LuxASR : {e}")
        return f"Erreur de transcription : {e}"

# Fonction pour appeler l'API Gemini
def call_gemini_api(prompt_text):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        chat = model.start_chat(history=[])
        response = chat.send_message(
            f"La question est : '{prompt_text}'. "
            f"Réponds à cette question en allemand, en t'adressant à un jeune adolescent, et sois concis. "
            f"Ensuite, traduis cette réponse en luxembourgeois. "
            f"Présente la réponse comme suit : [Votre réponse en luxembourgeois]"
        )
        return response.text
    except Exception as e:
        print(f"Erreur lors de l'appel à l'API Gemini : {e}")
        return f"Erreur de génération de réponse Gemini : {e}"


# Point d'accès de test simple
@app.route('/')
def home():
    return "Bienvenue sur l'API de l'Assistant Luxembourgeois PoC !"

# Point d'accès pour recevoir l'audio et renvoyer une réponse
@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    audio_data = audio_file.read()
    uploaded_content_type = audio_file.content_type

    # Appel à l'API LuxASR
    transcribed_text = call_luxasr_api(audio_data, audio_file.filename, uploaded_content_type)

    # Vérifiez si LuxASR a rencontré une erreur avant d'appeler Gemini
    if "Erreur de transcription :" in transcribed_text:
        return jsonify({
            "message": "LuxASR transcription failed.",
            "transcribed_text": transcribed_text,
            "gemini_response": "Impossible de traiter la demande sans transcription."
        }), 500

    # Appel à l'API Gemini
    gemini_response_text = call_gemini_api(transcribed_text)

    # Vérifiez si Gemini a rencontré une erreur
    if "Erreur de génération de réponse Gemini :" in gemini_response_text:
        return jsonify({
            "message": "Gemini response generation failed.",
            "transcribed_text": transcribed_text,
            "gemini_response": gemini_response_text
        }), 500

    # Retourne la transcription et la réponse de Gemini
    return jsonify({
        "message": "Audio processed successfully.",
        "transcribed_text": transcribed_text,
        "gemini_response": gemini_response_text
    }), 200

# Bloc pour lancer l'application localement
if __name__ == '__main__':
    app.run(debug=True, port=5000) # Lance le serveur de développement Flask sur le port 5000
