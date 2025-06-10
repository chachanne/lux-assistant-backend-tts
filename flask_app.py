from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
import requests
import base64
import io

app = Flask(__name__)
CORS(app) # Activez CORS pour toutes les requêtes

# --- Configuration des APIs ---

# Configuration de l'API Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in environment variables.")
genai.configure(api_key=GEMINI_API_KEY)

# URL de l'API LuxASR
LUX_ASR_API_URL = "https://luxasr.uni.lu/api/recognize_file/"

# URL CONFIRMÉE de l'API Hugging Face Piper LU TTS
PIPER_LU_TTS_API_URL = "https://mbarnig-rhasspy-piper-lu-streaming.hf.space/run/predict"

# Contrôle d'activation/désactivation de Gemini (si vous l'avez implémenté)
GEMINI_ENABLED = os.environ.get("GEMINI_ENABLED", "true").lower() == "true"

# --- Fonctions d'API ---

def call_gemini_api_lux(prompt_text):
    """
    Appelle l'API Gemini. Demande une réponse et une question de relance
    principalement en luxembourgeois.
    """
    if not GEMINI_ENABLED:
        print("Gemini API est désactivé par la variable d'environnement.")
        return "L'assistant est actuellement en mode maintenance. Veuillez réessayer plus tard."

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        chat = model.start_chat(history=[])

        # Prompt simplifié pour obtenir une réponse en luxembourgeois
        # On inclut une instruction de fallback si le luxembourgeois est difficile.
        prompt_for_gemini = (
            f"La question est : '{prompt_text}'. "
            f"Réponds à cette question en t'adressant à un jeune adolescent, sois concis et utile. "
            f"Réponds en luxembourgeois. Si tu ne peux pas répondre en luxembourgeois, réponds en allemand. "
            f"Après ta réponse, pose une question courte et pertinente pour relancer la conversation. "
            f"Cette question de relance doit être en luxembourgeois. Si pas possible, en allemand. "
            f"Présente la réponse et la question de relance de la manière suivante : "
            f"LU : [Votre réponse en luxembourgeois] "
            f"Question suivante LU : [Question en luxembourgeois] "
            f"Si tu dois utiliser l'allemand, utilise le format 'DE : ...' au lieu de 'LU : ...' "
            f"pour la réponse et 'Question suivante DE : ...' pour la relance."
        )

        response = chat.send_message(prompt_for_gemini)
        return response.text
    except Exception as e:
        print(f"Erreur lors de l'appel à l'API Gemini : {e}")
        return f"Erreur de génération de réponse Gemini : {e}"

def get_luxembourgish_tts(text_luxembourgish):
    """
    Appelle l'API Hugging Face Piper LU pour générer de l'audio en base64.
    """
    try:
        payload = {"fn_index": 0, "data": [text_luxembourgish]}
        headers = {"Content-Type": "application/json"}

        response = requests.post(PIPER_LU_TTS_API_URL, json=payload, headers=headers)
        response.raise_for_status()

        result_data = response.json()
        if result_data and result_data.get('data') and len(result_data['data']) > 0:
            audio_data_base64_with_prefix = result_data['data'][0]
            if "," in audio_data_base64_with_prefix:
                base64_string = audio_data_base64_with_prefix.split(",", 1)[1]
            else:
                base64_string = audio_data_base64_with_prefix
            return base64_string
        else:
            print("Aucune donnée audio trouvée dans la réponse de Piper LU.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Erreur d'appel à l'API Piper LU TTS : {e}")
        return None
    except Exception as e:
        print(f"Erreur inattendue lors de la récupération audio de Piper LU : {e}")
        return None

# --- Route principale de l'API Flask ---

@app.route('/process_audio', methods=['POST'])
def process_audio():
    transcribed_text = ""
    gemini_response = ""
    
    # On assume que la langue d'entrée est toujours le luxembourgeois pour cette version simplifiée
    input_language_from_frontend = 'lb' 

    # Traitement si l'entrée est du texte directement (e.g., via la zone de texte du frontend)
    if request.is_json:
        data = request.get_json()
        transcribed_text = data.get('text', '')
        print(f"Backend - Reçu texte du navigateur ('{input_language_from_frontend}'): {transcribed_text}")
        gemini_response = call_gemini_api_lux(transcribed_text) # Utilise la fonction simplifiée

    # Traitement si l'entrée est un fichier audio (e.g., via l'enregistrement vocal)
    elif 'audio' in request.files:
        audio_file = request.files['audio']
        use_luxasr_flag = request.form.get('use_luxasr', 'true') # On force 'true' pour cette version

        if use_luxasr_flag == 'true': # On utilise toujours LuxASR ici
            print(f"Backend - Reçu audio pour LuxASR (langue assumée: {input_language_from_frontend})...")
            try:
                files = {'audio': (audio_file.filename, audio_file.read(), audio_file.mimetype)}
                response_luxasr = requests.post(LUX_ASR_API_URL, files=files)
                response_luxasr.raise_for_status()

                luxasr_result = response_luxasr.json()
                transcribed_text = luxasr_result.get('recognized_text', 'Erreur LuxASR: Pas de texte reconnu.')
                print(f"LuxASR Transcription: {transcribed_text}")
                gemini_response = call_gemini_api_lux(transcribed_text) # Utilise la fonction simplifiée

            except requests.exceptions.RequestException as e:
                print(f"Erreur d'appel à LuxASR : {e}")
                transcribed_text = "Erreur de transcription LuxASR."
                gemini_response = "Désolé, je n'ai pas pu transcrire votre demande."
            except Exception as e:
                print(f"Erreur inattendue LuxASR : {e}")
                transcribed_text = "Erreur inattendue de transcription LuxASR."
                gemini_response = "Désolé, une erreur est survenue lors du traitement."
        else:
            return jsonify({"error": "Audio reçu mais LuxASR n'est pas activé ou la langue n'est pas le luxembourgeois."}), 400
    else:
        return jsonify({"error": "Type de contenu non supporté ou données manquantes"}), 400

    audio_base64 = None
    if gemini_response:
        # --- Extraction du texte luxembourgeois pour la synthèse vocale ---
        lux_text_to_speak = ""
        # Logique pour extraire la partie LU et la question de relance LU de la réponse de Gemini
        
        # Prioriser l'extraction du texte après "LU :"
        if "LU :" in gemini_response:
            start_idx = gemini_response.find("LU :") + len("LU :")
            # Cherche la fin de la réponse LU (avant "Question suivante LU :" ou "Question suivante DE :")
            end_idx_q_lu = gemini_response.find("Question suivante LU :", start_idx)
            end_idx_q_de = gemini_response.find("Question suivante DE :", start_idx) # Au cas où Gemini fallback sur DE

            if end_idx_q_lu != -1 and (end_idx_q_de == -1 or end_idx_q_lu < end_idx_q_de):
                # Si "Question suivante LU" est présente et vient avant "Question suivante DE"
                lux_text_to_speak = gemini_response[start_idx:end_idx_q_lu].strip()
            elif end_idx_q_de != -1:
                # Si "Question suivante DE" est présente (Gemini a fallback sur DE)
                lux_text_to_speak = gemini_response[start_idx:end_idx_q_de].strip()
            else:
                # Si aucune question de relance n'est trouvée, prends jusqu'à la fin de la section LU
                lux_text_to_speak = gemini_response[start_idx:].strip()
        else:
            # Si "LU :" n'est pas présent, cela signifie que Gemini a répondu sans le préfixe
            # (par exemple, si la réponse était 100% LU sans ambiguïté ou formatage).
            # On cherche alors directement la question de relance "Question suivante LU :"
            if "Question suivante LU :" in gemini_response:
                split_text = gemini_response.split("Question suivante LU :", 1)
                lux_text_to_speak = split_text[0].strip()
            elif "Question suivante DE :" in gemini_response: # Fallback si Gemini a écrit la relance en DE
                split_text = gemini_response.split("Question suivante DE :", 1)
                lux_text_to_speak = split_text[0].strip()
            else:
                lux_text_to_speak = gemini_response.strip()


        if lux_text_to_speak:
            print(f"Demande TTS pour le luxembourgeois : '{lux_text_to_speak}'")
            audio_base64 = get_luxembourgish_tts(lux_text_to_speak)
        else:
            print("Aucun texte luxembourgeois identifié pour le TTS.")
            audio_base64 = None
    else:
        audio_base64 = None

    return jsonify({
        "transcribed_text": transcribed_text,
        "gemini_response": gemini_response,
        "audio_response_base64": audio_base64
    })

# --- Démarrage de l'application Flask ---

if __name__ == '__main__':
    # Pour le déploiement sur Render, Gunicorn sera utilisé.
    # Pour le développement local, utilisez app.run
    app.run(debug=True, host='127.0.0.1', port=5000)
