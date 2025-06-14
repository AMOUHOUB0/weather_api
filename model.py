import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import joblib

# Charger les données
df = pd.read_csv("weather_data.csv")

# Encode risk_type (catégorique)
label_encoder = LabelEncoder()
df["risk_type_encoded"] = label_encoder.fit_transform(df["risk_type"])

# Features d'entrée
X = df[["temperature", "humidity", "wind_speed", "rain_intensity"]]

# Cible 1 : Classification (risk_type)
y_class = df["risk_type_encoded"]

# Cible 2 : Régression (risk_level)
y_reg = df["risk_level"]

# Modèle 1 : classification
clf = RandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(X, y_class)

# Modèle 2 : régression
reg = LinearRegression()
reg.fit(X, y_reg)

# Dictionnaire de recommandations basé sur le type de risque
recommendation_map = df.set_index("risk_type")["recommendation"].to_dict()

# Sauvegarder les modèles et les encodeurs
joblib.dump(clf, "risk_type_model.pkl")
joblib.dump(reg, "risk_level_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(recommendation_map, "recommendation_map.pkl")

print("✅ Modèles entraînés et enregistrés.")



