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
    # Pour un déploiement sur Render, Render doit fournir cette variable.
    # Pour un test local, décommentez et mettez votre clé ici (NE PAS PUSHER SUR GITHUB)
    # GEMINI_API_KEY = "VOTRE_CLE_API_GEMINI_ICI"
    raise ValueError("GEMINI_API_KEY is not set in environment variables.")
genai.configure(api_key=GEMINI_API_KEY)

# URL de l'API LuxASR (MISE À JOUR VERS LA V2 !)
LUX_ASR_API_URL = "https://luxasr.uni.lu/v2/asr"

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
        response.raise_for_status() # Lève une exception pour les codes d'état HTTP d'erreur

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
                # Lire le contenu binaire du fichier audio
                audio_content = audio_file.read()

                # Déterminer le MIME type basé sur le nom du fichier
                filename = audio_file.filename if audio_file.filename else "audio.wav"
                ext = filename.lower()
                if ext.endswith(".wav"):
                    mime_type = "audio/wav"
                elif ext.endswith(".mp3"):
                    mime_type = "audio/mpeg"
                elif ext.endswith(".m4a"):
                    mime_type = "audio/mp4"
                else:
                    mime_type = "application/octet-stream"

                # Créer le dictionnaire 'files' pour requests.post
                files = {
                    'audio_file': (filename, audio_content, mime_type)
                }

                # Préparer les paramètres de requête pour l'API LuxASR v2
                params = {
                    "diarization": "Disabled",
                    "outfmt": "json" # Demander la sortie JSON pour une extraction facile
                }
                headers = {
                    "accept": "application/json"
                }

                # Envoyer la requête à la nouvelle URL LuxASR avec les paramètres
                response_luxasr = requests.post(LUX_ASR_API_URL, params=params, headers=headers, files=files)
                response_luxasr.raise_for_status() # Lève une exception pour les codes d'état HTTP d'erreur
            # --- NOUVELLES LIGNES DE DÉBOGAGE POUR LUXASR ---
            print(f"LuxASR Raw Response Status Code: {response_luxasr.status_code}")
            print(f"LuxASR Raw Response Headers: {response_luxasr.headers}")
            luxasr_raw_text = response_luxasr.text
            print(f"LuxASR Raw Response Body (first 500 chars): {luxasr_raw_text[:500]}")
            # --- FIN NOUVELLES LIGNES DE DÉBOGAGE -
            
                luxasr_result = response_luxasr.json()
                
                # Extrait le texte transcrit du JSON de LuxASR.
                # La structure exacte peut varier, nous supposons 'text' ou 'recognized_text'
                transcribed_text = ""
                if 'text' in luxasr_result:
                    transcribed_text = luxasr_result['text']
                elif 'recognized_text' in luxasr_result: # Au cas où une autre clé serait utilisée
                    transcribed_text = luxasr_result['recognized_text']
                elif 'segments' in luxasr_result and isinstance(luxasr_result['segments'], list):
                    # Si la réponse est une liste de segments avec du texte à l'intérieur
                    transcribed_text = " ".join([s['text'] for s in luxasr_result['segments'] if 'text' in s])
                
                if not transcribed_text:
                    transcribed_text = "Erreur LuxASR: Pas de texte reconnu ou format inattendu."


                print(f"LuxASR Transcription: {transcribed_text}")
                
                # Si LuxASR n'a rien transcrit, ne pas appeler Gemini avec une chaîne vide
                if transcribed_text == "Erreur LuxASR: Pas de texte reconnu ou format inattendu." or not transcribed_text.strip():
                    gemini_response = "Désolé, je n'ai pas pu transcrire votre demande."
                else:
                    gemini_response = call_gemini_api_lux(transcribed_text)

            except requests.exceptions.RequestException as e:
                print(f"Erreur d'appel à LuxASR : {e}")
                transcribed_text = "Erreur de transcription LuxASR."
                gemini_response = f"Désolé, je n'ai pas pu transcrire votre demande en raison d'une erreur LuxASR : {e}"
            except Exception as e:
                print(f"Erreur inattendue LuxASR : {e}")
                transcribed_text = "Erreur inattendue de transcription LuxASR."
                gemini_response = f"Désolé, une erreur inattendue est survenue lors du traitement : {e}"
        else:
            return jsonify({"error": "Audio reçu sans instructions claires ou langue non gérée par LuxASR."}), 400
    else:
        return jsonify({"error": "Type de contenu non supporté ou données manquantes"}), 400

    audio_base64 = None
    if gemini_response:
        # --- Extraction du texte luxembourgeois pour la synthèse vocale ---
        lux_text_to_speak = ""
        
        if "LU :" in gemini_response:
            start_idx = gemini_response.find("LU :") + len("LU :")
            end_idx_q_lu = gemini_response.find("Question suivante LU :", start_idx)
            end_idx_q_de = gemini_response.find("Question suivante DE :", start_idx) 

            if end_idx_q_lu != -1 and (end_idx_q_de == -1 or end_idx_q_lu < end_idx_q_de):
                lux_text_to_speak = gemini_response[start_idx:end_idx_q_lu].strip()
            elif end_idx_q_de != -1:
                lux_text_to_speak = gemini_response[start_idx:end_idx_q_de].strip()
            else:
                lux_text_to_speak = gemini_response[start_idx:].strip()
        else:
            # Si "LU :" n'est pas présent, cela signifie que Gemini a répondu sans le préfixe
            if "Question suivante LU :" in gemini_response:
                split_text = gemini_response.split("Question suivante LU :", 1)
                lux_text_to_speak = split_text[0].strip()
            elif "Question suivante DE :" in gemini_response:
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
    app.run(debug=True, host='127.0.0.1', port=5000)
