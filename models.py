import torch
import numpy as np
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
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
class AutoencoderProgresivoVariable(nn.Module):
    def __init__(self, input_dim, c1=16, c2=8, bottleneck=4):
        super(AutoencoderProgresivoVariable, self).__init__()

        # Encoder Progresivo: Entrada -> C1 -> C2 -> Cuello
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, c1),
            nn.Tanh(),
            nn.Linear(c1, c2),
            nn.Tanh(),
            nn.Linear(c2, bottleneck),  # @ajuste: El cuello de botella final
            nn.Tanh()
        )

        # Decoder Simétrico: Cuello -> C2 -> C1 -> Entrada
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, c2),
            nn.Tanh(),
            nn.Linear(c2, c1),
            nn.Tanh(),
            nn.Linear(c1, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


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