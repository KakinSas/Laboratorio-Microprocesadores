# ================
#   LIBRERÍAS
# ================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns
import joblib

# ===============================
#   Cargar dataset temperatura
# ===============================
df = pd.read_csv('data.csv')

# Eliminamos columna innecesaria si existe
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# -----------------------------------------------------------------------------
# IMPUTACIÓN DE NULOS
# Numéricas → Interpolación lineal
# Categóricas → Moda (valor más frecuente)
# -----------------------------------------------------------------------------

# Detectar columnas numéricas y categóricas
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Interpolación numérica
df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')

# Imputación categórica con moda
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ===============================
#   Definir X (features) y y (target)
# ===============================
y = df['maxtemp']  # Target = temperatura máxima
X = df.drop(columns=['maxtemp'])  # Features

# ===============================
#   ONE HOT ENCODER para variables categóricas
# ===============================
cat_cols = [col for col in X.columns if X[col].dtype == "object"]
num_cols = [col for col in X.columns if col not in cat_cols]

# Preprocesador
preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', num_cols)
    ]
)

# ===============================
#   MODELO RANDOM FOREST + PIPELINE
# ===============================
model = Pipeline(steps=[
    ('preprocess', preprocess),
    ('rf', RandomForestRegressor(n_estimators=250, random_state=42))
])

# ===============================
#   Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# ===============================
#   ENTRENAR MODELO
# ===============================
model.fit(X_train, y_train)

# ===============================
#   Predicciones y Métricas
# ===============================
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}")
print(f"R2 Score: {r2:.3f}")

# =============================
#   📈 GRAFICOS
# =============================

# === 1️⃣ Histograma de temperatura ===
plt.figure(figsize=(6,4))
sns.histplot(df['temperatura'], kde=True, color='orange')
plt.title("Distribución de Temperatura")
plt.xlabel("Temperatura")
plt.ylabel("Frecuencia")
plt.savefig("grafico_histograma_temp.png")
plt.close()

# === 2️⃣ Línea de evolución de temperatura ===
plt.figure(figsize=(7,4))
plt.plot(df['temperatura'].values, label="Temperatura", color='red')
plt.title("Temperatura a lo largo del tiempo")
plt.xlabel("Tiempo (index)")
plt.ylabel("Temperatura")
plt.legend()
plt.savefig("grafico_linea_tiempo.png")
plt.close()

# === 3️⃣ Importancia de variables ===
plt.figure(figsize=(5,4))
sns.barplot(x=model.feature_importances_, y=['Humedad'], palette="viridis")
plt.title("Importancia de Variables en Random Forest")
plt.savefig("grafico_importancia_variables.png")
plt.close()

# === 4️⃣ Real vs Predicho COLOR + LEYENDA ===
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred, alpha=0.7, c='blue', label="Predicciones")
plt.scatter(y_test, y_test, alpha=0.4, c='red', label="Valores Reales")
plt.xlabel("Real")
plt.ylabel("Predicho")
plt.title("Comparación Temperatura Real vs Predicha")
plt.legend()
plt.savefig("grafico_real_vs_predicho.png")
plt.close()

# ===============================
#   Guardar Modelo Entrenado
# ===============================
joblib.dump(model, "random_forest_model.pkl")

print("Modelo guardado como random_forest_model.pkl")
print("Gráfico guardado como prediction_plot.png")
