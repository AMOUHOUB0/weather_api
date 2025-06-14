from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Initialisation de Flask
app = Flask(__name__)
CORS(app) # Pour autoriser les requêtes depuis Flutter

# Chargement des modèles et encodeurs
clf = joblib.load("risk_type_model.pkl")
reg = joblib.load("risk_level_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
recommendation_map = joblib.load("recommendation_map.pkl")

@app.route('/')
def hello():
    return "API météo ML en ligne ✔️"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Requête reçue")  # Debug
        data = request.get_json()
        print("Données reçues:", data)  # Debug
        

        # Récupération des données météo
        temperature = data['temperature']
        humidity = data['humidity']
        wind_speed = data['wind_speed']
        rain_intensity = data['rain_intensity']

        # Préparation des features pour le modèle
        features = np.array([[temperature, humidity, wind_speed, rain_intensity]])

        # Prédictions
        predicted_type_encoded = clf.predict(features)[0]
        predicted_type = label_encoder.inverse_transform([predicted_type_encoded])[0]

        predicted_level = float(reg.predict(features)[0])
        predicted_level = round(predicted_level, 2)

        recommendation = recommendation_map.get(predicted_type, "Pas de recommandation disponible")

        # Construction de la réponse
        response = {
            "risk_type": predicted_type,
            "risk_level": predicted_level,
            "recommendation": recommendation
        }
        print("Prédiction terminée:", response)  # Debug
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Lancer le serveur
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
