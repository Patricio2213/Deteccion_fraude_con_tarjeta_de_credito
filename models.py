from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import torch.nn as nn

# --- 1. Regresión Logística (Baseline) ---
def get_logistic_pipeline(preprocessor):
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            penalty='l2',
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ))
    ])

# --- 2. XGBoost (Estado del Arte) ---
def get_xgboost_pipeline(preprocessor, scale_pos_weight):
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, # Manejo de desbalanceo
            eval_metric='logloss',
            random_state=42
        ))
    ])

# --- 3. PyTorch MLP (Deep Learning) ---
class FraudMLP(nn.Module):
    def __init__(self, input_dim):
        super(FraudMLP, self).__init__()
        self.red = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.red(x)