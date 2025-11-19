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

# ===============================
#   GRÁFICO DE RESULTADOS
# ===============================
plt.figure(figsize=(7,5))

# Puntos reales vs predichos
plt.scatter(y_test, y_pred, alpha=0.7, edgecolor='k', label='Predicciones')

# Línea ideal (si el modelo fuera perfecto)
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Real = Predicho')

plt.xlabel("Temperatura Real")
plt.ylabel("Temperatura Predicha")
plt.title(f"Predicción RF Temp - MSE: {mse:.2f} | R2: {r2:.3f}")
plt.legend()
plt.grid(True)
plt.savefig("prediction_plot.png")

# ===============================
#   Guardar Modelo Entrenado
# ===============================
joblib.dump(model, "random_forest_model.pkl")

print("Modelo guardado como random_forest_model.pkl")
print("Gráfico guardado como prediction_plot.png")
