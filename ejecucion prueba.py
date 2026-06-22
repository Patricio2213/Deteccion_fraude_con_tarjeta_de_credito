#Carga de funciones
import gc
from sklearn.model_selection import train_test_split
from Subida_data import *
from procesamiento_bases import *
from EDA import *
from models import *
import statsmodels.api as sm
import matplotlib.pyplot as plt
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': False          # False para quitar las cuadriculas
})

data_train=buscar_y_cargar("fraudTrain.csv")
data_test=buscar_y_cargar("fraudTest.csv")

data=pd.concat([data_train, data_test], ignore_index=True)
data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
#Se calcula nuevo comercio sobre la data original para que no existan 3 periodos distintos de enfriamiento, las bases solo tienen información anterior y no futura
data["es_nuevo"] = nuevo_comercio(data)

#--------------------------
#Codigo para ver fechas de inicio y fin de las transacciones de cada mes para dividir de forma exacta la base
#--------------------------
data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])

df_filtrado = data[
    (data['trans_date_trans_time'].dt.year == 2020) &
    (data['trans_date_trans_time'].dt.month.isin([7, 8, 9, 10, 11, 12]))
].copy()

reporte_fechas = df_filtrado.groupby(df_filtrado['trans_date_trans_time'].dt.month)['trans_date_trans_time'].agg(['min', 'max'])

print(reporte_fechas)

print("fechas data entrenamiento")
print(data_train["trans_date_trans_time"].min())
print(data_train["trans_date_trans_time"].max())

print("fechas data testeo")
print(data_test["trans_date_trans_time"].min())
print(data_test["trans_date_trans_time"].max())

fecha_entreno_inicio = '2019-01-01 00:00:18'
fecha_entreno_fin    = '2020-06-21 12:13:37'
fecha_val_inicio="2020-06-21 12:14:25"
fecha_val_fin="2020-08-31 23:59:31"
fecha_prueba_inicio  = '2020-09-01 00:01:26'
fecha_prueba_fin     = '2020-11-30 23:59:45'

# Extraemos los conjuntos completos respetando los cortes temporales estrictos
data_train = data[(data['trans_date_trans_time'] >= fecha_entreno_inicio) &
                        (data['trans_date_trans_time'] <= fecha_entreno_fin)].copy()
data_valid=data[(data["trans_date_trans_time"]>= fecha_val_inicio)&(data["trans_date_trans_time"]<= fecha_val_fin)]
data_test = data[(data['trans_date_trans_time'] >= fecha_prueba_inicio) &
                      (data['trans_date_trans_time'] <= fecha_prueba_fin)].copy()



# ==============================================================================
# creacion variables train
# ==============================================================================
print("\n" + "="*60)
print("🚀 INICIANDO BLOQUE 2: INGENIERÍA DE VARIABLES CON AISLAMIENTO TEMPORAL")
print("="*60)

# Aseguramos el formato datetime de manera estrictamente independiente
data_train['trans_date_trans_time'] = pd.to_datetime(data_train['trans_date_trans_time'])
data_test['trans_date_trans_time'] = pd.to_datetime(data_test['trans_date_trans_time'])

# --- variables train ---
print("⏳ Procesando métricas geográficas y de comportamiento en DATA_TRAIN...")
data_train = distancia_entre_comercios(data_train)
data_train = calcular_velocidad(data_train)
data_train = distancia_cliente_comercio(data_train)

data_train["edad"] = calcular_edad(data_train)
#data_train["tasa_categoria"] = calcular_anomaliaencategoria(data_train)
#data_train["es_nuevo"] = nuevo_comercio(data_train)

# Aplicación de transformaciones logarítmicas en Train
data_train["monto_log"] = np.log1p(data_train["amt"])
data_train["poblacion_ciudad_log"] = np.log1p(data_train["city_pop"])
data_train["velocidad_log_local"] = np.log1p(data_train["velocidad_local"])
data_train["velocidad_log_internet"] = np.log1p(data_train["velocidad_internet"])
data_train["diferencia_tiempo_log_local"] = np.log1p(data_train["delta_tiempo_local"])
data_train["diferencia_tiempo_log_internet"] = np.log1p(data_train["delta_tiempo_internet"])
data_train["distancia_log_local"] = np.log1p(data_train["distancia_local"])
data_train["distancia_log_internet"] = np.log1p(data_train["distancia_internet"])
data_train["d_cliente_comercio_log_local"] = np.log1p(data_train["d_cliente_comercio_loc"])
data_train["d_cliente_comercio_log_int"] = np.log1p(data_train["d_cliente_comercio_int"])

categorias_net = ["grocery_net", "misc_net", "shopping_net"]
data_train["es_online"] = data_train["category"].isin(categorias_net).astype(int)
"""
# Calculamos la proporción de fraude por categoría usando SOLO los datos de entrenamiento
# Categoría -> Probabilidad de fraude histórica
tasa_categoria_map = (
    data_train.groupby("category")["is_fraud"].mean().to_dict()
)

# Mapeamos ese diccionario en el set de Entrenamiento
data_train["tasa_fraude_categoria"] = data_train["category"].map(tasa_categoria_map)

# Mapeamos el MISMO diccionario en el resto de datas (Heredamos el pasado al futuro)
data_test["tasa_fraude_categoria"] = data_test["category"].map(tasa_categoria_map)
data_valid["tasa_fraude_categoria"] = data_valid["category"].map(tasa_categoria_map)


# Control de seguridad: Si en Test aparece una categoría que no existía en Train,
# el mapa arrojará NaN. Lo llenamos con la tasa de fraude global de Train.
tasa_global_train = data_train["is_fraud"].mean()
data_test["tasa_fraude_categoria"] = data_test["tasa_fraude_categoria"].fillna(
    tasa_global_train)
data_valid["tasa_fraude_categoria"] = data_valid["tasa_fraude_categoria"].fillna(
    tasa_global_train)
"""
# --- Variables para test---
print("⏳ Procesando métricas geográficas y de comportamiento en DATA_TEST...")
data_test = distancia_entre_comercios(data_test)
data_test = calcular_velocidad(data_test)
data_test = distancia_cliente_comercio(data_test)

data_test["edad"] = calcular_edad(data_test)
#data_test["tasa_categoria"] = calcular_anomaliaencategoria(data_test)
#data_test["es_nuevo"] = nuevo_comercio(data_test)

# Aplicación de transformaciones logarítmicas en Test
data_test["monto_log"] = np.log1p(data_test["amt"])
data_test["poblacion_ciudad_log"] = np.log1p(data_test["city_pop"])
data_test["velocidad_log_local"] = np.log1p(data_test["velocidad_local"])
data_test["velocidad_log_internet"] = np.log1p(data_test["velocidad_internet"])
data_test["diferencia_tiempo_log_local"] = np.log1p(data_test["delta_tiempo_local"])
data_test["diferencia_tiempo_log_internet"] = np.log1p(data_test["delta_tiempo_internet"])
data_test["distancia_log_local"] = np.log1p(data_test["distancia_local"])
data_test["distancia_log_internet"] = np.log1p(data_test["distancia_internet"])
data_test["d_cliente_comercio_log_local"] = np.log1p(data_test["d_cliente_comercio_loc"])
data_test["d_cliente_comercio_log_int"] = np.log1p(data_test["d_cliente_comercio_int"])

data_test["es_online"] = data_test["category"].isin(categorias_net).astype(int)

# --- Variables para validacion---
print("⏳ Procesando métricas geográficas y de comportamiento en DATA_VALID...")
data_valid = distancia_entre_comercios(data_valid)
data_valid = calcular_velocidad(data_valid)
data_valid = distancia_cliente_comercio(data_valid)

data_valid["edad"] = calcular_edad(data_valid)
#data_valid["tasa_categoria"] = calcular_anomaliaencategoria(data_valid)
#data_valid["es_nuevo"] = nuevo_comercio(data_valid)

# Aplicación de transformaciones logarítmicas para validacion
data_valid["monto_log"] = np.log1p(data_valid["amt"])
data_valid["poblacion_ciudad_log"] = np.log1p(data_valid["city_pop"])
data_valid["velocidad_log_local"] = np.log1p(data_valid["velocidad_local"])
data_valid["velocidad_log_internet"] = np.log1p(data_valid["velocidad_internet"])
data_valid["diferencia_tiempo_log_local"] = np.log1p(data_valid["delta_tiempo_local"])
data_valid["diferencia_tiempo_log_internet"] = np.log1p(data_valid["delta_tiempo_internet"])
data_valid["distancia_log_local"] = np.log1p(data_valid["distancia_local"])
data_valid["distancia_log_internet"] = np.log1p(data_valid["distancia_internet"])
data_valid["d_cliente_comercio_log_local"] = np.log1p(data_valid["d_cliente_comercio_loc"])
data_valid["d_cliente_comercio_log_int"] = np.log1p(data_valid["d_cliente_comercio_int"])

data_valid["es_online"] = data_valid["category"].isin(categorias_net).astype(int)



# Liberamos las bases de que ya no usaremos
del data
gc.collect()

#Con este codigo dividimos la muestra de forma estratificada y usé _, pq divide la base en 2, así que la parte que no usamos se va a "_"
_,train_df = train_test_split(data_train, test_size=160000, stratify=data_train["is_fraud"], random_state=42)
_,test_df = train_test_split(data_test, test_size=40000, stratify=data_test["is_fraud"], random_state=42)
_,val_df = train_test_split(data_valid, test_size=40000, stratify=data_valid["is_fraud"], random_state=42)


# --- Ingeniería de Variables Temporales en los DataFrames ---
for df_temp in [train_df, test_df,val_df]:
    df_temp['hora'] = df_temp['trans_date_trans_time'].dt.hour
    df_temp['mes'] = df_temp['trans_date_trans_time'].dt.month


target = 'is_fraud'
columnas_drop = [target, "trans_date_trans_time", "street", "first", "last",
                 "merchant", "city", "dob", "lat", "long", "merch_lat", "merch_long", "job", "amt", "city_pop",
                 "velocidad", "delta_tiempo_local", "delta_tiempo_internet", "velocidad_local", "velocidad_internet",
                 "distancia_local", "distancia_internet", "d_cliente_comercio_loc", "d_cliente_comercio_int", "is_first_buy",'Unnamed: 0', 'cc_num', 'unix_time',"is_fraud","zip","trans_num","trans_date_trans_time"]

X_train_raw = train_df.drop(columns=[col for col in columnas_drop if col in train_df.columns])
X_test_raw  = test_df.drop(columns=[col for col in columnas_drop if col in test_df.columns])
X_val_raw  = val_df.drop(columns=[col for col in columnas_drop if col in val_df.columns])

y_train = train_df[target]
y_test  = test_df[target]
y_val  = val_df[target]


cat_columns = X_train_raw.select_dtypes(include=['object', 'string']).columns.tolist()
num_columns = X_train_raw.select_dtypes(include=['number']).columns.tolist()

# Pipeline de preprocesamiento estándar
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_columns),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_columns)
    ],verbose_feature_names_out=False,)

X_train_scaled = preprocessor.fit_transform(X_train_raw)
X_test_scaled = preprocessor.transform(X_test_raw)

# --- 4. MODELOS SUPERVISADOS TRADICIONALES ---
print("\n--- Entrenando Modelos Supervisados ---")

# A. REGRESIÓN LOGÍSTICA (Statsmodels)
print("⏳ Ajustando Logit...")

# 1. Recuperar nombres de columnas
nombres_columnas = preprocessor.get_feature_names_out()

# 2. Crear DataFrames y REINICIAR ÍNDICES de las etiquetas
X_train_final = pd.DataFrame(X_train_scaled, columns=nombres_columnas)
X_test_final = pd.DataFrame(X_test_scaled, columns=nombres_columnas)

# Reseteamos y_train e y_test
y_train_reset = y_train.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

# 3. Agregar constante
X_train_stat = sm.add_constant(X_train_final)
X_test_stat = sm.add_constant(X_test_final)

#Guardar data 

X_train_final.to_csv("X_train_procesado.csv", index=False)
X_test_final.to_csv("X_test_procesado.csv", index=False)
y_train_reset.to_frame(name='is_fraud').to_csv("y_train_procesado.csv", index=False)
y_test_reset.to_frame(name='is_fraud').to_csv("y_test_procesado.csv", index=False)


# 4. Asegurar que las columnas de Test sean IDENTICAS a las de Train
# Si falta una columna en Test que estaba en Train, la crea con ceros
X_test_stat = X_test_stat.reindex(columns=X_train_stat.columns, fill_value=0)

print(f"✅ Columnas Train: {X_train_stat.shape[1]} | Columnas Test: {X_test_stat.shape[1]}")


#Calcular los pesos inversamente proporcionales al desbalance
conteo_clases = np.bincount(y_train_reset)
peso_fraudes = conteo_clases[0] / conteo_clases[1]  # Ej: Si hay 300 veces más legítimas, el fraude pesa 300

from sklearn.linear_model import LogisticRegression

# 5. Ajustar modelo
try:
    logit_sk = LogisticRegression(C=20.0, solver='saga', max_iter=300, random_state=42)
    logit_sk.fit(
        X_train_scaled,
        y_train_reset,
        sample_weight=np.where(y_train_reset == 1, peso_fraudes, 1.0)
    )

    print("\n=== MODELO LOGÍSTICA (SAGA) AJUSTADO CON ÉXITO ===")

    y_prob_logit = logit_sk.predict_proba(X_test_scaled)[:, 1]

# Evaluamos con umbral estadístico fijo
    evaluar_modelo("Regresión Logística", y_test_reset, y_prob_logit, umbral=0.1882)

except Exception as e:
    print(f"⚠️ Nota de Tesis: La Regresión Logística con solver SAGA falló o no convergió.")
    print(f"Detalle: {e}")

print("ESCANER")
for col in X_train_final.columns:
    # Agrupamos los datos reales por el valor de la variable escalada
    # y miramos el promedio del target para ver si da 0 o 1 de forma perfecta
    agrupado = pd.DataFrame({"valor": X_train_final[col], "is_fraud": y_train_reset})

    # Redondeamos para agrupar variables continuas que actúen como flags
    summary_col = agrupado.groupby("valor")["is_fraud"].agg(["count", "mean"])

    # Buscamos grupos significativos (más de 50 filas) donde el riesgo sea 100% o 0% perfecto
    sospechosos = summary_col[(summary_col["mean"].isin([0.0, 1.0])) & (summary_col["count"] > 50)]

    if not sospechosos.empty:
        print(f"⚠️ ¡ALERTA! Variable problemática detectada: '{col}'")
        print(sospechosos.head(2))
        print("-" * 60)


# B. XGBOOST con OPTIMIZACIÓN BAYESIANA
print(f"\n🚀 Iniciando Optimización Bayesiana para XGBoost (Rangos: Tayebi & El Kafhali)...")

# Crear el estudio de Optuna
sampler_aleatorio = optuna.samplers.TPESampler(seed=None)
study = optuna.create_study(direction="maximize", sampler=sampler_aleatorio)
# Ejecutamos la optimización (n_trials=20 )
study.optimize(
    lambda trial: objective_xgboost(trial, X_train_raw, y_train, X_val_raw, y_val, preprocessor),
    n_trials=20
)
datos_a_guardar = study.best_params.copy()
datos_a_guardar['mejor_auc_val_mediana'] = study.best_value
df_mejores_parametros = pd.DataFrame([datos_a_guardar])
df_mejores_parametros.to_excel("mejores_hiperparametros_xgboost.xlsx", index=False)
print(f"✅ Mejores Hiperparámetros encontrados: {study.best_params}")

# Entrenamos el modelo final con los mejores parámetros encontrados
print(f"📊 Entrenando modelo final con parámetros óptimos...")
xgb_model_final = get_xgboost(study.best_params)
xgb_model_final.fit(X_train_scaled, y_train)

# Predicción y Evaluación
y_prob_xgb = xgb_model_final.predict_proba(X_test_scaled)[:, 1]
evaluar_modelo("XGBoost Optimizado", y_test, y_prob_xgb, umbral=0.2843)



# --- 5. MODELOS NO SUPERVISADOS (Isolation Forest & LOF) ---
print("\n--- Entrenando Modelos de Anomalías ---")

# A. ISOLATION FOREST
iso_forest = get_isolation_forest()
iso_forest.fit(X_train_scaled)
y_prob_iso = -iso_forest.decision_function(X_test_scaled)
y_prob_iso = (y_prob_iso - y_prob_iso.min()) / (y_prob_iso.max() - y_prob_iso.min() + 1e-9)
evaluar_modelo("Isolation Forest", y_test, y_prob_iso, umbral=0.8109)

# B. LOCAL OUTLIER FACTOR (LOF) - METODOLOGÍA DE RANGOS (20-50)
print("⏳ Ejecutando LOF con metodología de rangos (Breunig et al.)...")
#soo usaremos variables continuas para no tener problemas con a distancia euclidiana
variables_numericas = preprocessor.named_transformers_['num'].get_feature_names_out()
num_only_idx = [list(nombres_columnas).index(col) for col in variables_numericas]
# Definimos el rango de vecinos a probar
rango_vecinos = [20, 30, 40, 50]
scores_acumulados = []

for k in rango_vecinos:
    print(f"   Procesando k={k}...")
    lof = get_lof(neighbors=k)
    lof.fit(X_train_scaled[:,num_only_idx])

    # Obtenemos el score para este k, usamos signo negativo porque score_samples devuelve valores negativos para anomalías
    current_scores = -lof.score_samples(X_test_scaled[:,num_only_idx])
    scores_acumulados.append(current_scores)

# Tomar el máximo score LOF entre todos los k
# np.maximum.reduce compara los arrays y se queda con el valor más alto en cada posición

y_prob_lof_max = np.maximum.reduce(scores_acumulados)

# Normalización para el Benchmark
y_prob_lof_final = (y_prob_lof_max - y_prob_lof_max.min()) / (y_prob_lof_max.max() - y_prob_lof_max.min() + 1e-9)

evaluar_modelo("LOF (Rango 20-50)", y_test, y_prob_lof_final, umbral=0.0143)

# --- 6. PREPARACIÓN PARA REDES NEURONALES (PyTorch) ---
X_train_tensor = torch.from_numpy(X_train_scaled.copy()).float()
X_test_tensor = torch.from_numpy(X_test_scaled.copy()).float()
y_train_tensor = torch.from_numpy(y_train.values.copy()).float().view(-1, 1)

input_dim = X_train_tensor.shape[1]
#esto es para que no cambien los resultados
g_mlp = torch.Generator()
g_mlp.manual_seed(42)
train_loader = DataLoader(
    TensorDataset(X_train_tensor, y_train_tensor),
    batch_size=32,
    shuffle=True,
    generator=g_mlp
)
# --- 7. ENTRENAMIENTO MLP BALANCEADO ---
conteo_clases = y_train.value_counts()
peso_fraude = conteo_clases[0] / conteo_clases[1]
pos_weight = torch.tensor([peso_fraude]).float()

mlp_model = MLP(input_dim,seed=42)
optimizer_mlp = torch.optim.Adam(mlp_model.parameters(), lr=0.001)
criterion_mlp = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

historial_loss_mlp = []  # Guardamos los datos para graficar después
num_epochs = 20

print(f"🚀 Entrenando MLP Balanceado...")

for epoch in range(num_epochs):
    mlp_model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer_mlp.zero_grad()
        loss = criterion_mlp(mlp_model(batch_X), batch_y)
        loss.backward()
        optimizer_mlp.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    historial_loss_mlp.append(avg_loss)

    if (epoch + 1) % 5 == 0:
        print(f"   Época [{epoch + 1}/{num_epochs}] completada.")

mlp_model.eval()
with torch.no_grad():
    y_prob_mlp = torch.sigmoid(mlp_model(X_test_tensor)).numpy().flatten()
    evaluar_modelo("MLP Balanceado", y_test, y_prob_mlp, umbral=0.9949)
"""
# --- GRÁFICA DE SALIDA (Separada del flujo de entrenamiento) ---
plt.figure(figsize=(8, 4))
plt.plot(historial_loss_mlp, label='Pérdida Entrenamiento', color='royalblue', linewidth=2)
plt.title('Evolución del Error - MLP (Tesis)')
plt.xlabel('Épocas')
plt.ylabel('BCE Loss')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
"""

# --- 8. ENTRENAMIENTO AUTOENCODER ---
# --- CONFIGURACIÓN DE DATOS ---
X_train_normal = X_train_tensor[y_train_tensor.flatten() == 0]
ae_train_loader = DataLoader(
    TensorDataset(X_train_normal),
    batch_size=256,  # Batch size más grande sugerido para estabilidad
    shuffle=True
)

# --- INICIALIZACIÓN ---
ae_model = AutoencoderProgresivoVariable(input_dim, c1=20, c2=10, bottleneck=5)

# Optimizador con Learning Rate reducido y Regularización L2 (weight_decay)
optimizer_ae = torch.optim.Adam(
    ae_model.parameters(),
    lr=0.001,
    weight_decay=1e-3  # Penalización L2 para mejorar generalización
)

criterion_ae = torch.nn.MSELoss()
num_epochs_ae = 50

# --- BUCLE DE ENTRENAMIENTO ---

print(f"🚀 Entrenando Autoencoder (Solo datos normales)...")

ae_model.train()

for epoch in range(num_epochs_ae):

    running_loss = 0.0

    for batch in ae_train_loader:
        # Extraer el Tensor desde el batch que entrega TensorDataset
        batch_X = batch[0]

        optimizer_ae.zero_grad()

        outputs = ae_model(batch_X)

        loss = criterion_ae(outputs, batch_X)

        loss.backward()

        optimizer_ae.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(ae_train_loader)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"   Época [{epoch + 1}/{num_epochs_ae}] - Loss: {avg_loss:.6f}")
# --- EVALUACIÓN ---
ae_model.eval()
with torch.no_grad():
    # Obtener errores de reconstrucción (MSE)
    reconst_train = ae_model(X_train_tensor)
    mse_train_torch = torch.mean((X_train_tensor - reconst_train) ** 2, dim=1).numpy()
    mse_train_normal=mse_train_torch[y_train_tensor==0]
    # Cálculo del umbral dinámico (Media + 3 * Desv. Estándar)
    umbral_3sigma = mse_train_normal.mean() + 3 * mse_train_normal.std()

    #evaluar sobre test
    reconst_test = ae_model(X_test_tensor)
    mse_test = torch.mean((X_test_tensor - reconst_test) ** 2, dim=1).numpy()
    evaluar_modelo(
        "Autoencoder Optimizado",
        y_test,
        mse_test,
        umbral=umbral_3sigma
    )

# ------------------------------------------
# OBTENER MEJORES UMBRALES
# ------------------------------------------
# El Autoencoder entrega MSE (errores), necesitamos pasarlo a escala 0-1
y_prob_ae_norm = (mse_test - mse_test.min()) / (mse_test.max() - mse_test.min() + 1e-9)

# --- DICCIONARIO DE PROBABILIDADES CORREGIDO ---
modelos_probs = {
    "Regresión Logística": y_prob_logit,  # Corregido de y_prob_reg
    "XGBoost": y_prob_xgb,
    "MLP": y_prob_mlp,
    "Autoencoder": y_prob_ae_norm,  # Usamos la versión normalizada arriba
    "Isolation Forest": y_prob_iso,  # Corregido de y_prob_if_norm
    "LOF": y_prob_lof_final
}

resultados_umbrales = []

for nombre, probas in modelos_probs.items():
    res = encontrar_mejor_umbral(nombre, y_test, probas)
    resultados_umbrales.append(res)

# Creamos tabla comparativa
df_umbrales = pd.DataFrame(resultados_umbrales)
print("\n=== RESULTADOS DE OPTIMIZACIÓN DE UMBRALES ===")
print(df_umbrales[['Modelo', 'Umbral_Optimo', 'F1_Score', 'FP', 'FN']])

# ==============================================================================
# 🎯 EVALUACIÓN DE APRENDIZAJE EN TRAIN (LOS 6 MODELOS DE LA TESIS)
# ==============================================================================
print("\n" + "=" * 60)
print("🔥 INICIANDO AUDITORÍA DE APRENDIZAJE: MUESTRA TRAIN")
print("=" * 60)

# Reseteamos el target de entrenamiento a un array limpio de NumPy
y_train_arr = y_train.values

if 'logit_sk' in locals():
    y_prob_logit_train = logit_sk.predict(X_train_stat)
    evaluar_aprendizaje_train("Regresión Logística", y_train_arr, y_prob_logit_train)

if 'xgb_model_final' in locals():
    y_prob_xgb_train = xgb_model_final.predict_proba(X_train_scaled)[:, 1]
    evaluar_aprendizaje_train("XGBoost Optimizado", y_train_arr, y_prob_xgb_train)

if 'mlp_model' in locals():
    mlp_model.eval()
    with torch.no_grad():
        y_prob_mlp_train = torch.sigmoid(mlp_model(X_train_tensor)).cpu().numpy().flatten()
    evaluar_aprendizaje_train("MLP Balanceado", y_train_arr, y_prob_mlp_train)

if 'iso_forest' in locals():
    y_prob_iso_train_raw = -iso_forest.decision_function(X_train_scaled)
    # Normalización Min-Max local para consistencia en el reporte 0-1
    y_prob_iso_train = (y_prob_iso_train_raw - y_prob_iso_train_raw.min()) / (
                y_prob_iso_train_raw.max() - y_prob_iso_train_raw.min() + 1e-9)
    evaluar_aprendizaje_train("Isolation Forest", y_train_arr, y_prob_iso_train)
"""
if 'rango_vecinos' in locals():
    scores_acum_train = []
    for k in rango_vecinos:
        lof_t = get_lof(neighbors=k)
        lof_t.fit(X_train_scaled)  # Se ajusta y extrae sobre el mismo set
        scores_acum_train.append(-lof_t.score_samples(X_train_scaled))

    y_prob_lof_max_train = np.maximum.reduce(scores_acum_train)
    y_prob_lof_train = (y_prob_lof_max_train - y_prob_lof_max_train.min()) / (
                y_prob_lof_max_train.max() - y_prob_lof_max_train.min() + 1e-9)
    evaluar_aprendizaje_train("LOF (Rango 20-50)", y_train_arr, y_prob_lof_train)
"""
if 'ae_model' in locals():
    ae_model.eval()
    with torch.no_grad():
        reconst_train = ae_model(X_train_tensor)
        mse_train_raw = torch.mean((X_train_tensor - reconst_train) ** 2, dim=1).cpu().numpy()

    # Normalización de los errores de reconstrucción a escala 0-1
    y_prob_ae_train = (mse_train_raw - mse_train_raw.min()) / (mse_train_raw.max() - mse_train_raw.min() + 1e-9)
    evaluar_aprendizaje_train("Autoencoder", y_train_arr, y_prob_ae_train)

print("=" * 60)
print("✅ Auditoría de entrenamiento completada de forma exitosa.")
print("=" * 60)
# --- EVALUACIÓN ECONÓMICA FINAL (TESIS) ---

# 1. Cálculo de Frecuencia Dinámica con Suavizado (Smoothing) con toda la muestra de entrenamiento disponible
dias_data = (data_train['trans_date_trans_time'].max() - data_train['trans_date_trans_time'].min()).days
if dias_data == 0: dias_data = 1

# Calculamos frecuencia anual y aplicamos un piso de 6 transacciones/año para compensar el muestreo
conteo_muestral = data_train.groupby('cc_num').size()
frecuencia_extrapolada = (conteo_muestral * 365 / dias_data)
freq_map = frecuencia_extrapolada.clip(lower=6).to_dict()

# 2. Monto promedio por cliente (solo legítimas)
clv_map = data_train[data_train['is_fraud'] == 0].groupby('cc_num')['amt'].mean().to_dict()


# 3. Construcción del Vector de CLV para el set de Test
def calc_clv_dinamico(cc):
    amt_avg = clv_map.get(cc, train_df['amt'].mean())
    freq = freq_map.get(cc, 6)  # Si el cliente no está en train, usamos el piso
    return (amt_avg * freq) * 0.018 * 5  # Margen 1.8% y 5 años vida media


clv_test_vector = data_train['cc_num'].apply(calc_clv_dinamico).values
amt_test_vector = data_train['amt'].values
y_test_values = y_test.values

# 4. Ejecución del Benchmark para todos los modelos del script
resultados_economicos = []

# Supervisados
if 'y_prob_logit' in locals():
    resultados_economicos.append(
        calcular_matriz_costos_economica("Regresión Logística", y_test_values, y_prob_logit, amt_test_vector,
                                         clv_test_vector))

if 'y_prob_xgb' in locals():
    resultados_economicos.append(
        calcular_matriz_costos_economica("XGBoost Optimizado", y_test_values, y_prob_xgb, amt_test_vector,
                                         clv_test_vector))

if 'y_prob_mlp' in locals():
    resultados_economicos.append(
        calcular_matriz_costos_economica("MLP Balanceado", y_test_values, y_prob_mlp, amt_test_vector, clv_test_vector))

# Anomalías (Normalizando scores para que actúen como probabilidad 0-1)
if 'y_prob_iso' in locals():
    y_prob_iso_norm = (y_prob_iso - y_prob_iso.min()) / (y_prob_iso.max() - y_prob_iso.min() + 1e-9)
    resultados_economicos.append(
        calcular_matriz_costos_economica("Isolation Forest", y_test_values, y_prob_iso_norm, amt_test_vector,
                                         clv_test_vector))

if 'y_prob_lof_final' in locals():
    y_prob_lof_norm = (y_prob_lof_final - y_prob_lof_final.min()) / (
                y_prob_lof_final.max() - y_prob_lof_final.min() + 1e-9)
    resultados_economicos.append(
        calcular_matriz_costos_economica("LOF (20-50)", y_test_values, y_prob_lof_norm, amt_test_vector,
                                         clv_test_vector))

if 'mse_test' in locals():
    mse_ae_norm = (mse_test - mse_test.min()) / (mse_test.max() - mse_test.min() + 1e-9)
    resultados_economicos.append(
        calcular_matriz_costos_economica("Autoencoder", y_test_values, mse_ae_norm, amt_test_vector, clv_test_vector))

# 5. Visualización del Ranking Económico
df_resumen = pd.DataFrame(resultados_economicos).sort_values(by='costo_total')
print("\n" + "=" * 50)
print("RANKING DE MODELOS POR IMPACTO FINANCIERO")
print("=" * 50)
print(df_resumen[['modelo', 'costo_total', 'ahorro', 'ahorro_pct', 'fp', 'fn']])

info_modelos = {
    'Regresión Logística': {'probabilidades': y_prob_logit, 'umbral_f1': 0.1882},
    'XGBoost': {'probabilidades': y_prob_xgb, 'umbral_f1': 0.2843},
    'MLP': {'probabilidades': y_prob_mlp, 'umbral_f1': 0.9949},  # Umbral actualizado según tu entrenamiento
    'Autoencoder': {'probabilidades': y_prob_ae_norm, 'umbral_f1': 0.0553},  # Usamos versión normalizada
    'Isolation Forest': {'probabilidades': y_prob_iso, 'umbral_f1': 0.8109},
    'LOF': {'probabilidades': y_prob_lof_final, 'umbral_f1': 0.0143}
}

# Ejecución de la matriz comparativa total (F1 vs Bayes)
df_final = generar_matriz_comparativa_total(
    info_modelos,
    y_test_values,
    amt_test_vector,
    clv_test_vector
)

# Visualización de la tabla definitiva para la tesis
print("\n=== AUDITORÍA ESTRATÉGICA: IMPACTO ECONÓMICA (F1 VS BAYES) ===")
print(df_final.to_string(index=False))

#GRAFICO RESULTADOS ECONOMICOS
print("\n📊 Generando Gráfico de Ahorro Económico...")
graficar_ahorro_modelos_propios(resultados_economicos)
#GRAFICO RESULTADOS ESTADISTICOS
print("\n📊 Generando Gráfico de AUC-ROC Estructural...")
graficar_auc_roc_modelos_propios(info_modelos, y_test_values)

# ---------------------------------------------
# PRUEBA DE ROBUSTEZ
# -------------------------------------------

print("\n" + "=" * 60)
print(" INICIANDO PRUEBA DE ROBUSTEZ INDEPENDIENTE: DICIEMBRE 2020")
print("=" * 60)

# 1. Extracción y filtrado del universo exclusivo de diciembre de 2020
fecha_rob_inicio = '2020-12-01 00:00:00'
fecha_rob_fin = '2020-12-31 23:59:34'

df_robustez_raw = data_test[(data_test['trans_date_trans_time'] >= fecha_rob_inicio) &
                       (data_test['trans_date_trans_time'] <= fecha_rob_fin)].copy()

print(f"🔹 Registros encontrados para la prueba de estrés de diciembre: {len(df_robustez_raw)}")

# 2. Ingeniería de variables temporales exclusiva para el set de robustez
df_robustez_raw['hora'] = df_robustez_raw['trans_date_trans_time'].dt.hour
df_robustez_raw['mes'] = df_robustez_raw['trans_date_trans_time'].dt.month

# 3. Separación de matriz de características y etiquetas reales
X_robust_raw = df_robustez_raw.drop(columns=[col for col in columnas_drop if col in df_robustez_raw.columns])
y_robust_real = df_robustez_raw[target].values

# 4. Transformación de los datos usando el preprocessor original (YA ENTRENADO)
X_robust_scaled = preprocessor.transform(X_robust_raw)

# 5. Vectores financieros específicos y exclusivos para el mes de diciembre
amt_robust_vector = df_robustez_raw['amt'].values
clv_robust_vector = df_robustez_raw['cc_num'].apply(calc_clv_dinamico).values

# 6. Preparación de tensores en PyTorch para los modelos de redes neuronales
X_robust_tensor = torch.from_numpy(X_robust_scaled.copy()).float()

# ------------------------------------------------------------------------------
# PARTE A: EVALUACIÓN DE ROBUSTEZ TÉCNICA (Umbrales Fijos Obtenidos en Entrenamiento)
# ------------------------------------------------------------------------------
print("\n [ROBUSTEZ TÉCNICA] Evaluación en Diciembre con Umbrales Previos:")

resultados_robustez_tecnica = []

if 'logit_sk' in locals():
    y_prob_logit_rob = logit_sk.predict_proba(X_robust_scaled)[:, 1]
    res = evaluar_modelo("Regresión Logística (Rob)", y_robust_real, y_prob_logit_rob,
                         umbral=info_modelos['Regresión Logística']['umbral_f1'])
    resultados_robustez_tecnica.append(res)

if 'xgb_model_final' in locals():
    y_prob_xgb_rob = xgb_model_final.predict_proba(X_robust_scaled)[:, 1]
    res = evaluar_modelo("XGBoost (Rob)", y_robust_real, y_prob_xgb_rob, umbral=info_modelos['XGBoost']['umbral_f1'])
    resultados_robustez_tecnica.append(res)

if 'mlp_model' in locals():
    mlp_model.eval()
    with torch.no_grad():
        y_prob_mlp_rob = torch.sigmoid(mlp_model(X_robust_tensor)).numpy().flatten()
    res = evaluar_modelo("MLP (Rob)", y_robust_real, y_prob_mlp_rob, umbral=info_modelos['MLP']['umbral_f1'])
    resultados_robustez_tecnica.append(res)

if 'iso_forest' in locals():
    y_prob_iso_rob = -iso_forest.decision_function(X_robust_scaled)
    y_prob_iso_rob_norm = (y_prob_iso_rob - y_prob_iso_rob.min()) / (y_prob_iso_rob.max() - y_prob_iso_rob.min() + 1e-9)
    res = evaluar_modelo("Isolation Forest (Rob)", y_robust_real, y_prob_iso_rob_norm,
                         umbral=info_modelos['Isolation Forest']['umbral_f1'])
    resultados_robustez_tecnica.append(res)

if 'rango_vecinos' in locals():
    scores_acum_rob = []
    for k in rango_vecinos:
        lof = get_lof(neighbors=k)
        lof.fit(X_train_scaled[:, num_only_idx])
        scores_acum_rob.append(-lof.score_samples(X_robust_scaled[:, num_only_idx]))
    y_prob_lof_rob_max = np.maximum.reduce(scores_acum_rob)
    y_prob_lof_rob_norm = (y_prob_lof_rob_max - y_prob_lof_rob_max.min()) / (
                y_prob_lof_rob_max.max() - y_prob_lof_rob_max.min() + 1e-9)
    res = evaluar_modelo("LOF (Rob)", y_robust_real, y_prob_lof_rob_norm, umbral=info_modelos['LOF']['umbral_f1'])
    resultados_robustez_tecnica.append(res)

if 'ae_model' in locals():
    ae_model.eval()
    with torch.no_grad():
        reconst_rob = ae_model(X_robust_tensor)
        mse_rob_torch = torch.mean((X_robust_tensor - reconst_rob) ** 2, dim=1)
        mse_robust_test = mse_rob_torch.numpy()
    y_prob_ae_rob_norm = (mse_robust_test - mse_robust_test.min()) / (
                mse_robust_test.max() - mse_robust_test.min() + 1e-9)
    res = evaluar_modelo("Autoencoder (Rob)", y_robust_real, y_prob_ae_rob_norm,
                         umbral=info_modelos['Autoencoder']['umbral_f1'])
    resultados_robustez_tecnica.append(res)

# Matriz resumen técnica del estrés temporal
df_rob_tec = pd.DataFrame(resultados_robustez_tecnica)
print("\n [RESUMEN MATRICES - ROBUSTEZ DICIEMBRE 2020]:")
print(df_rob_tec[['nombre', 'auc_roc', 'tp', 'fp', 'tn', 'fn']])

# ------------------------------------------------------------------------------
# PARTE B: EVALUACIÓN DE ROBUSTEZ FINANCIERA (Auditoría de Impacto Económico)
# ------------------------------------------------------------------------------
print("\n [ROBUSTEZ FINANCIERA] Auditoría de Decisiones Estratégicas (F1 VS BAYES):")

info_modelos_robustez = {}
if 'y_prob_logit_rob' in locals(): info_modelos_robustez['Regresión Logística'] = {'probabilidades': y_prob_logit_rob,
                                                                                   'umbral_f1': info_modelos[
                                                                                       'Regresión Logística'][
                                                                                       'umbral_f1']}
if 'y_prob_xgb_rob' in locals():   info_modelos_robustez['XGBoost'] = {'probabilidades': y_prob_xgb_rob,
                                                                       'umbral_f1': info_modelos['XGBoost'][
                                                                           'umbral_f1']}
if 'y_prob_mlp_rob' in locals():   info_modelos_robustez['MLP'] = {'probabilidades': y_prob_mlp_rob,
                                                                   'umbral_f1': info_modelos['MLP']['umbral_f1']}
if 'y_prob_iso_rob_norm' in locals(): info_modelos_robustez['Isolation Forest'] = {
    'probabilidades': y_prob_iso_rob_norm, 'umbral_f1': info_modelos['Isolation Forest']['umbral_f1']}
if 'y_prob_lof_rob_norm' in locals(): info_modelos_robustez['LOF'] = {'probabilidades': y_prob_lof_rob_norm,
                                                                      'umbral_f1': info_modelos['LOF']['umbral_f1']}
if 'y_prob_ae_rob_norm' in locals():  info_modelos_robustez['Autoencoder'] = {'probabilidades': y_prob_ae_rob_norm,
                                                                              'umbral_f1': info_modelos['Autoencoder'][
                                                                                  'umbral_f1']}

# Ejecutamos tu función de matriz comparativa total pasándole los datos de diciembre
df_final_robustez = generar_matriz_comparativa_total(
    info_modelos_robustez,
    y_robust_real,
    amt_robust_vector,
    clv_robust_vector
)

print(df_final_robustez.to_string(index=False))
print("\n" + "=" * 60 + "\n")

# -----------------------------------------------------
# TEST DE ROBUSTEZ IMAN-DAVENPORT-->HOLM
# -----------------------------------------------------
print("\n" + "=" * 60)
print("📊 GENERANDO MATRICES MENSUALES PARA IMAN-DAVENPORT")
print("=" * 60)

meses_test = test_df.loc[y_test.index, 'trans_date_trans_time'].dt.month.values
meses_unicos = np.unique(meses_test)

df_iman_auc = pd.DataFrame(index=[f"Mes_{m}" for m in meses_unicos])
df_iman_ahorro = pd.DataFrame(index=[f"Mes_{m}" for m in meses_unicos])

mult_lexis = 5.75
costo_ca = 2.50

for nombre_modelo, info in info_modelos.items():
    lista_auc_mes = []
    lista_ahorro_mes = []

    for mes in meses_unicos:
        mask = (meses_test == mes)

        y_real_mes = y_test_values[mask]
        amt_mes = amt_test_vector[mask]
        clv_mes = clv_test_vector[mask]
        y_prob_mes = info['probabilidades'][mask]

        try:
            auc_m = roc_auc_score(y_real_mes, y_prob_mes)
        except:
            auc_m = 0.5
        lista_auc_mes.append(auc_m)

        costo_fn_vector = amt_mes * mult_lexis
        costo_fp_vector = costo_ca + clv_mes
        costo_tp = costo_ca

        thresholds_bayes = costo_fp_vector / (costo_fn_vector - costo_tp + costo_fp_vector)
        y_pred_riesgo = (y_prob_mes > thresholds_bayes).astype(int)

        tp = (y_real_mes == 1) & (y_pred_riesgo == 1)
        fp = (y_real_mes == 0) & (y_pred_riesgo == 1)
        fn = (y_real_mes == 1) & (y_pred_riesgo == 0)

        costo_total = (tp.sum() * costo_tp) + (fp * costo_fp_vector).sum() + (fn * costo_fn_vector).sum()
        costo_base = (y_real_mes * costo_fn_vector).sum()

        ahorro_mes = costo_base - costo_total
        lista_ahorro_mes.append(ahorro_mes)

    df_iman_auc[nombre_modelo] = lista_auc_mes
    df_iman_ahorro[nombre_modelo] = lista_ahorro_mes

# LLAMADAS FINALES (Llamando a la función que ahora vive en estadisticas.py)
ranks_auc, holm_auc = pipeline_iman_davenport_holm_puro(df_iman_auc, "AUC-ROC (Estructural)", buscar_maximo=True)
ranks_ahorro, holm_ahorro = pipeline_iman_davenport_holm_puro(df_iman_ahorro, "Ahorro Financiero (Bayes)",                                                            buscar_maximo=True)

print("\n[ANÁLISIS METODOLÓGICO DE CONTROL]")
ranks_auc_ctrl, holm_auc_ctrl = pipeline_iman_davenport_holm_control(
    df_iman_auc, "AUC-ROC (Estructural)", modelo_control="XGBoost", buscar_maximo=True
)

ranks_ahorro_ctrl, holm_ahorro_ctrl = pipeline_iman_davenport_holm_control(
    df_iman_ahorro, "Ahorro Financiero (Bayes)", modelo_control="XGBoost", buscar_maximo=True
)



# --- FASE FINAL: INTERPRETABILIDAD COMPARATIVA ---
nombres_columnas = preprocessor.get_feature_names_out()

# Modelos Supervisados
aplicar_shap_tesis_final(xgb_model_final, X_test_scaled, nombres_columnas, "XGBoost")
aplicar_shap_tesis_final(mlp_model, X_test_scaled, nombres_columnas, "MLP", es_pytorch=True)

# CORREGIDO: Pasamos logit_sk y activamos el flag de statsmodels
aplicar_shap_tesis_final(logit_sk, X_test_scaled, nombres_columnas, "Regresión Logística", es_statsmodels=False)

# Modelos No Supervisados / Anomalías
aplicar_shap_tesis_final(iso_forest, X_test_scaled, nombres_columnas, "Isolation Forest")
aplicar_shap_tesis_final(ae_model, X_test_scaled, nombres_columnas, "Autoencoder", es_pytorch=True)
# Ejecución de la interpretabilidad para LOF
df_res_lofo = aplicar_lofo_tesis_final(
    X_train_scaled[:, num_only_idx],
    X_test_scaled[:, num_only_idx],
    nombres_columnas[num_only_idx],
    nombre_modelo="LOF_Rango_20_50"
)
