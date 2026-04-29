import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
import torch.optim as optim

# --- FAMILIA 1: ENFOQUE SUPERVISADO ---

def get_logistic_regression():
    # Eliminamos 'penalty' explícito para evitar avisos de deprecación.
    # Por defecto usa L2 (Ridge). C=1.0 es el estándar.
    return LogisticRegression(C=1.0, solver='liblinear')  # @ajuste: C menor aumenta regularización


def get_xgboost():
    return XGBClassifier(
        n_estimators=100,  # @ajuste: n_estimadores (cantidad de árboles)
        max_depth=6,  # @ajuste: profundidad (evita sobreajuste si es bajo)
        learning_rate=0.1,  # @ajuste: tasa de aprendizaje (eta)
        eval_metric='logloss'  # Métrica interna para clasificación binaria
    )


# MLP: Red Neuronal para Clasificación
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),  # @ajuste: neuronas capa 1
            nn.ReLU(),
            nn.Linear(32, 16),  # @ajuste: neuronas capa 2
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Obligatorio para probabilidad 0 a 1
        )

    def forward(self, x):
        return self.network(x)


# --- FAMILIA 2: ENFOQUE NO SUPERVISADO ---

def get_isolation_forest():
    return IsolationForest(
        n_estimators=100,
        contamination=0.01,  # @ajuste: % de fraude esperado en la base
        random_state=42
    )


def get_lof():
    return LocalOutlierFactor(
        n_neighbors=20,
        novelty=True,
        n_jobs=-1  # Esto acelera el proceso usando todos tus núcleos
    )


# Autoencoder en PyTorch
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def evaluar_modelo(nombre, y_real, y_prob):
    # 1. Cálculos de Ranking (AUC)
    roc_auc = roc_auc_score(y_real, y_prob)
    precision, recall, _ = precision_recall_curve(y_real, y_prob)
    pr_auc = auc(recall, precision)

    # 2. Matriz de Confusión (Umbral estándar 0.5)
    y_pred = (y_prob > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_real, y_pred).ravel()

    # 3. Impresión en consola (Lo que querías ver)
    print(f"\n📊 RESULTADOS: {nombre}")
    print(f"  - AUC-ROC: {roc_auc:.4f} | AUC-PR: {pr_auc:.4f}")
    print(f"  - Matriz:  TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
    print(f"  - Tasa Falsos Positivos (FPR): {fp / (fp + tn):.4%}")

    # 4. RETURNO (Para poder guardar los resultados)
    return {
        "nombre": nombre,
        "auc_roc": roc_auc,
        "auc_pr": pr_auc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn
    }