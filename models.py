import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim

# --- FAMILIA 1: ENFOQUE SUPERVISADO ---

def get_logistic_regression():
    # Eliminamos 'penalty' explícito para evitar avisos de deprecación.
    # Por defecto usa L2 (Ridge). C=1.0 es el estándar.
    return LogisticRegression(C=1.0, solver='liblinear')  # @ajuste: C menor aumenta regularización


def objective_xgboost(trial, X, y):
    # Definimos rangos exactos basados en Tayebi y El Kafhali
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 400),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.7),
        'gamma': trial.suggest_float('gamma', 0.01, 1.0),
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': -1
    }

    # Implementamos Stratified 5-Fold Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # El objetivo es maximizar el AUC
    # validacion cruzada con kfold 5
    score = cross_val_score(
        XGBClassifier(**param), X, y,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1
    ).mean()

    return score


def get_xgboost(params=None):
    # Si no pasamos parámetros, usa unos por defecto
    if params is None:
        return XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, eval_metric='logloss', random_state=42)
    # Si pasamos los optimizados, los usa
    return XGBClassifier(**params)


# MLP: Red Neuronal para Clasificación
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Eliminamos el Sigmoid final
        )
    def forward(self, x):
        return self.network(x)


# --- FAMILIA 2: ENFOQUE NO SUPERVISADO ---

def get_isolation_forest():
    return IsolationForest(
        n_estimators=500,
        contamination=0.0058,  #ajuste: % de fraude esperado en la base
        random_state=42
    )


def get_lof(neighbors):

    return LocalOutlierFactor(
        n_neighbors=neighbors,
        novelty=True,         # Necesario para usarlo en el set de testeo (X_test)
        contamination=0.01,   # Proporción estimada de fraude
        n_jobs=-1             # Uso de todos los núcleos del procesador
    )


# Autoencoder en PyTorch
import torch.nn as nn

class AutoencoderProgresivoVariable(nn.Module):
    def __init__(self, input_dim, c1=20, c2=10, bottleneck=5):
        super(AutoencoderProgresivoVariable, self).__init__()

        # Encoder: Reducción progresiva con ReLU y Dropout
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, c1),
            nn.ReLU(),             # ReLU es preferida sobre Tanh en estas capas
            nn.Dropout(0.2),       # Dropout para evitar el sobreajuste
            nn.Linear(c1, c2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(c2, bottleneck),
            nn.ReLU()
        )

        # Decoder: Reconstrucción simétrica
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, c2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(c2, c1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(c1, input_dim) # Salida lineal para datos estandarizado [3]
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def evaluar_modelo(nombre, y_real, y_prob, umbral=None):
    # 1. Cálculos de Ranking (AUC) - Esto no cambia, es independiente del umbral
    roc_auc = roc_auc_score(y_real, y_prob)
    precision, recall, _ = precision_recall_curve(y_real, y_prob)
    pr_auc = auc(recall, precision)

    # 2. Matriz de Confusión (Umbral Ajustado)
    if umbral is None:
        # Si no pasas un umbral, usamos el percentil 95 de los scores
        # Esto marcará el 5% de los datos con más error como fraude
        umbral = np.percentile(y_prob, 95)

    y_pred = (y_prob > umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_real, y_pred).ravel()

    # 3. Impresión en consola
    print(f"\n📊 RESULTADOS: {nombre}")
    print(f"  - Umbral utilizado: {umbral:.6f}")  # Importante para tu reporte
    print(f"  - AUC-ROC: {roc_auc:.4f} | AUC-PR: {pr_auc:.4f}")
    print(f"  - Matriz:  TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
    print(f"  - Tasa Falsos Positivos (FPR): {fp / (fp + tn):.4%}")

    return {
        "nombre": nombre,
        "auc_roc": roc_auc,
        "auc_pr": pr_auc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "umbral": umbral
    }


def aplicar_shap_tesis_final(model, X_data_numpy, feature_names, nombre_modelo="Modelo", es_pytorch=False,
                             es_lof=False):
    print(f"\n" + "=" * 40)
    print(f"🧐 ANALIZANDO INTERPRETABILIDAD: {nombre_modelo}")
    print("=" * 40)

    sample_size = min(2000, len(X_data_numpy))
    indices = np.random.choice(X_data_numpy.shape[0], sample_size, replace=False)
    X_sample_np = X_data_numpy[indices]

    try:
        # --- Lógica de Explicadores ---
        if es_pytorch:
            def predict_fn_torch(x_np):
                model.eval()
                x_tensor = torch.from_numpy(x_np).float()
                with torch.no_grad():
                    if "autoencoder" in nombre_modelo.lower():
                        reconst = model(x_tensor)
                        return torch.mean((x_tensor - reconst) ** 2, dim=1).numpy()
                    else:
                        return torch.sigmoid(model(x_tensor)).numpy().flatten()

            masker = shap.maskers.Independent(X_data_numpy, max_samples=100)
            explainer = shap.Explainer(predict_fn_torch, masker)
            shap_values = explainer(X_sample_np)
        elif es_lof:
            def predict_fn_lof(x_np):
                return -model.score_samples(x_np)

            masker = shap.maskers.Independent(X_data_numpy, max_samples=100)
            explainer = shap.Explainer(predict_fn_lof, masker)
            shap_values = explainer(X_sample_np)
        else:
            explainer = shap.Explainer(model)
            shap_values = explainer(X_sample_np)

        # --- EXTRACCIÓN E IMPRESIÓN DE TODAS LAS VARIABLES ---
        if hasattr(shap_values, "values"):
            if len(shap_values.values.shape) == 3:  # Multiclase (XGBoost)
                vals = np.abs(shap_values.values[:, :, 1]).mean(0)
            else:
                vals = np.abs(shap_values.values).mean(0)
        else:
            vals = np.abs(shap_values).mean(0)

        # Crear DataFrame con el 100% de las variables
        df_total = pd.DataFrame(list(zip(feature_names, vals)), columns=['Variable', 'Impacto_Medio_SHAP'])
        df_total = df_total.sort_values(by='Impacto_Medio_SHAP', ascending=False)

        print(f"\n📋 IMPACTO TOTAL DE VARIABLES (Orden descendente) - {nombre_modelo}:")
        pd.set_option('display.max_rows', None)  # Para que PyCharm no esconda filas
        print(df_total.to_string(index=False))
        pd.reset_option('display.max_rows')

        # --- GENERACIÓN DEL GRÁFICO (SOLO TOP 15) ---
        plt.figure(figsize=(12, 8))
        # Asignamos los nombres para el gráfico
        if hasattr(shap_values, "values"):
            shap_values.feature_names = list(feature_names)

        # max_display=15 hace la magia de limpiar el gráfico automáticamente
        if hasattr(shap_values, "values") and len(shap_values.values.shape) == 3:
            shap.plots.bar(shap_values[:, :, 1], show=False, max_display=15)
        else:
            shap.plots.bar(shap_values, show=False, max_display=15)

        plt.title(f"Top 15 Variables de Mayor Impacto (SHAP) - {nombre_modelo}")
        plt.tight_layout()
        plt.savefig(f"shap_top15_{nombre_modelo.lower().replace(' ', '_')}.png", dpi=300)
        plt.show()

    except Exception as e:
        print(f"❌ Error en {nombre_modelo}: {e}")




def calcular_matriz_costos_economica(nombre_modelo, y_real, y_prob, amt_test, clv_vector,
                                     multiplicador_lexis=5.75, ca=2.50):
    """
    Calcula el impacto económico real basado en la matriz de Correa Bahnsen & LexisNexis.
    Aplica Bayes Minimum Risk para optimizar el umbral de decisión.
    """
    # 1. Definición de vectores de costo por transacción
    costo_fn_vector = amt_test * multiplicador_lexis
    costo_fp_vector = ca + clv_vector
    costo_tp = ca  # Costo fijo administrativo por gestión de alerta

    # 2. Umbral de Bayes Óptimo (BMR)
    # Se calcula un umbral específico para cada transacción i
    # Threshold_i = C_FP_i / (C_FN_i - C_TP + C_FP_i)
    thresholds_bayes = costo_fp_vector / (costo_fn_vector - costo_tp + costo_fp_vector)

    # 3. Clasificación basada en Riesgo Mínimo
    y_pred_riesgo = (y_prob > thresholds_bayes).astype(int)

    # 4. Cálculo de métricas económicas
    tp = (y_real == 1) & (y_pred_riesgo == 1)
    fp = (y_real == 0) & (y_pred_riesgo == 1)
    fn = (y_real == 1) & (y_pred_riesgo == 0)

    costo_tp_total = tp.sum() * costo_tp
    costo_fp_total = (fp * costo_fp_vector).sum()
    costo_fn_total = (fn * costo_fn_vector).sum()

    costo_total = costo_tp_total + costo_fp_total + costo_fn_total

    # Referencia: Dinero perdido si no hubiera ningún modelo (solo Falsos Negativos)
    costo_base = (y_real * costo_fn_vector).sum()
    ahorro = costo_base - costo_total

    return {
        'modelo': nombre_modelo,
        'costo_total': costo_total,
        'ahorro': ahorro,
        'tp': tp.sum(),
        'fp': fp.sum(),
        'fn': fn.sum(),
        'ahorro_pct': (ahorro / costo_base) * 100 if costo_base > 0 else 0
    }


def aplicar_lofo_tesis_final(X_train_scaled, X_test_scaled, feature_names, nombre_modelo="LOF",
                             rango_vecinos=[20, 30, 40, 50]):
    print(f"\n" + "=" * 40)
    print(f"🧐 ANALIZANDO INTERPRETABILIDAD (LOFO): {nombre_modelo}")
    print("=" * 40)

    # 1. Reducimos muestra para que sea comparable en tiempos y representatividad
    sample_size = min(2000, len(X_test_scaled))
    indices = np.random.choice(X_test_scaled.shape[0], sample_size, replace=False)
    X_test_sample = X_test_scaled[indices]

    # 2. Score Base (con todas las variables) usando tu lógica de rangos
    def get_score_rangos(X_tr, X_te):
        scores_acum = []
        for k in rango_vecinos:
            lof = LocalOutlierFactor(n_neighbors=k, novelty=True)
            lof.fit(X_tr)
            scores_acum.append(-lof.score_samples(X_te))
        return np.maximum.reduce(scores_acum)

    score_base = get_score_rangos(X_train_scaled, X_test_sample)

    importancias = []

    try:
        print(f"🚀 Calculando impacto por omisión para {len(feature_names)} variables...")
        for i, nombre in enumerate(feature_names):
            # Omitimos la variable i
            X_tr_reduced = np.delete(X_train_scaled, i, axis=1)
            X_te_reduced = np.delete(X_test_sample, i, axis=1)

            # Score sin la variable
            score_sin_col = get_score_rangos(X_tr_reduced, X_te_reduced)

            # El impacto es el cambio absoluto medio en el score de anomalía
            impacto = np.mean(np.abs(score_base - score_sin_col))
            importancias.append({'Variable': nombre, 'Impacto_Medio_LOFO': impacto})

            if (i + 1) % 20 == 0:
                print(f"   ✅ {i + 1}/{len(feature_names)} procesadas...")

        # --- EXTRACCIÓN E IMPRESIÓN DE TODAS LAS VARIABLES ---
        df_total = pd.DataFrame(importancias)
        df_total = df_total.sort_values(by='Impacto_Medio_LOFO', ascending=False)

        print(f"\n📋 IMPACTO TOTAL DE VARIABLES (LOFO) - {nombre_modelo}:")
        pd.set_option('display.max_rows', None)
        print(df_total.to_string(index=False))
        pd.reset_option('display.max_rows')

        # --- GENERACIÓN DEL GRÁFICO (TOP 15) ---
        df_top15 = df_total.head(15).iloc[::-1]  # Invertir para que la más alta esté arriba en barh

        plt.figure(figsize=(12, 8))
        plt.barh(df_top15['Variable'], df_top15['Impacto_Medio_LOFO'], color='skyblue')
        plt.xlabel('Impacto Medio en el Score de Anomalía')
        plt.title(f"Top 15 Variables de Mayor Impacto (LOFO) - {nombre_modelo}")
        plt.tight_layout()

        plt.savefig(f"lofo_top15_{nombre_modelo.lower().replace(' ', '_')}.png", dpi=300)
        plt.show()

        return df_total

    except Exception as e:
        print(f"❌ Error en LOFO {nombre_modelo}: {e}")
        return None


def encontrar_mejor_umbral(nombre_modelo, y_real, y_prob):
    # Calculamos precision, recall y los umbrales posibles
    precision, recall, thresholds = precision_recall_curve(y_real, y_prob)

    # Calculamos F1-Score para encontrar el equilibrio técnico
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    ix = np.argmax(f1_scores)

    best_threshold = thresholds[ix]

    # Calculamos la matriz de confusión con el mejor umbral
    y_pred = (y_prob >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_real, y_pred).ravel()

    return {
        'Modelo': nombre_modelo,
        'Umbral_Optimo': round(best_threshold, 4),
        'F1_Score': round(f1_scores[ix], 4),
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn
    }


def generar_matriz_comparativa_total(modelos_data, y_real, amt_test, clv_vector,
                                     multiplicador_lexis=5.75, ca=2.50):
    """
    Cuantifica el costo total y el ahorro de los 6 modelos bajo dos paradigmas:
    1. Optimización Técnica (F1-Score)
    2. Optimización de Riesgo (Bayes)
    """
    comparativa = []
    costo_base = (y_real * amt_test * multiplicador_lexis).sum()

    for nombre, info in modelos_data.items():
        y_prob = info['probabilidades']
        threshold_f1 = info['umbral_f1']

        # --- ESTRATEGIA ESTADÍSTICA (F1) ---
        y_pred_f1 = (y_prob > threshold_f1).astype(int)
        tp_f1 = (y_real == 1) & (y_pred_f1 == 1)
        fp_f1 = (y_real == 0) & (y_pred_f1 == 1)
        fn_f1 = (y_real == 1) & (y_pred_f1 == 0)

        costo_f1 = (tp_f1.sum() * ca) + (ca + clv_vector[fp_f1]).sum() + (amt_test[fn_f1] * multiplicador_lexis).sum()
        ahorro_f1_pct = ((costo_base - costo_f1) / costo_base) * 100

        # --- ESTRATEGIA FINANCIERA (BAYES) ---
        c_fn_v = amt_test * multiplicador_lexis
        c_fp_v = ca + clv_vector
        t_bayes = c_fp_v / (c_fn_v - ca + c_fp_v)

        y_pred_b = (y_prob > t_bayes).astype(int)
        tp_b = (y_real == 1) & (y_pred_b == 1)
        fp_b = (y_real == 0) & (y_pred_b == 1)
        fn_b = (y_real == 1) & (y_pred_b == 0)

        costo_b = (tp_b.sum() * ca) + (c_fp_v[fp_b]).sum() + (c_fn_v[fn_b]).sum()
        ahorro_b_pct = ((costo_base - costo_b) / costo_base) * 100

        comparativa.append({
            'Modelo': nombre,
            'Ahorro F1 (%)': f"{ahorro_f1_pct:.2f}%",
            'Ahorro Bayes (%)': f"{ahorro_b_pct:.2f}%",
            'Diferencia (Valor)': f"{ahorro_b_pct - ahorro_f1_pct:.2f}%",
            'Mejora USD': f"${(costo_f1 - costo_b):,.0f}"
        })

    return pd.DataFrame(comparativa)

#------------------------------
#ROBUSTEZ
#-----------------------------
def evaluar_robustez_oot(nombre_modelo, y_real, y_prob, amt_vector, clv_vector, umbral_f1):
    from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

    # 1. Escenario Base (Inacción)
    costo_base = (y_real * (amt_vector * 5.75)).sum()
    if costo_base == 0: return None

    # --- MUNDO F1: MÉTRICAS ESTADÍSTICAS Y ECONÓMICAS ---
    y_pred_f1 = (y_prob >= umbral_f1).astype(int)
    f1_oot = f1_score(y_real, y_pred_f1)

    # Estas métricas son independientes del umbral, miden la "calidad" del score
    auc_roc = roc_auc_score(y_real, y_prob)
    auc_pr = average_precision_score(y_real, y_prob)

    # Cálculo económico con umbral F1
    tp_f1 = (y_real == 1) & (y_pred_f1 == 1)
    fp_f1 = (y_real == 0) & (y_pred_f1 == 1)
    fn_f1 = (y_real == 1) & (y_pred_f1 == 0)

    costo_f1 = (tp_f1.sum() * 2.50) + (fp_f1 * (2.50 + clv_vector)).sum() + (fn_f1 * (amt_vector * 5.75)).sum()
    ahorro_f1_pct = ((costo_base - costo_f1) / costo_base) * 100

    # --- MUNDO BAYES: MÉTRICAS ECONÓMICAS ---
    res_bayes = calcular_matriz_costos_economica(nombre_modelo, y_real, y_prob, amt_vector, clv_vector)

    # F1 resultante de usar el umbral de Bayes
    y_pred_bayes = (y_prob >= res_bayes['umbral_optimo']).astype(int)
    f1_bajo_bayes = f1_score(y_real, y_pred_bayes)

    return {
        'Modelo': nombre_modelo,
        'AUC_ROC': auc_roc,  # Métrica de calidad técnica
        'AUC_PR': auc_pr,  # Métrica de calidad técnica (clave en desbalance)
        'F1_Max_F1': f1_oot,
        'Ahorro_F1_%': ahorro_f1_pct,
        'F1_Bayes': f1_bajo_bayes,
        'Ahorro_Bayes_%': res_bayes['ahorro_pct'],
        'Mejora_Económica': res_bayes['ahorro_pct'] - ahorro_f1_pct
    }