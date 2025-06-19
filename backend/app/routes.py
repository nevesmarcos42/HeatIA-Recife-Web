from fastapi import APIRouter
from pydantic import BaseModel

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

router = APIRouter()


# Arquitetura do modelo -  mesma usada no treinamento
class ModeloAcidentes(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)


# Reconstrói o modelo e carrega os pesos
modelo = ModeloAcidentes()
pesos = torch.load("app/modelo_preditivo.pth", map_location="cpu")
modelo.load_state_dict(pesos)
modelo.eval()

# Carrega o scaler salvo do Colab
scaler = joblib.load("app/scaler.pkl")


# Entrada esperada
class EntradaAno(BaseModel):
    ano: int


@router.post("/previsao")
def prever_vitimas(entrada: EntradaAno):
    ano = entrada.ano

    # Lê os dados dos bairros e força atualização do ano
    df = pd.read_csv("app/dados_2025.csv")
    df["ano"] = ano

    # Define as fetures usadas no treinamento
    features_scaler = [
        "local_cod",
        "tipo_acidente_cod",
        "ano",
        "mes",
        "dia",
        "latitude",
        "longitude"
    ]

    X_base = df[features_scaler].values.astype(np.float32)
    X_norm = scaler.transform(X_base)

    # Junta com densidade (sem normalizar)
    densidade = df["densidade_acidentes"].to_numpy().reshape(
        -1, 1).astype(np.float32)
    X_final = np.hstack((X_norm, densidade))

    # Gera previsões com o modelo pytorch
    with torch.no_grad():
        entrada_tensor = torch.tensor(X_final)
        saida = modelo(entrada_tensor).numpy().flatten()

    df["vitimas"] = saida

    # Agrupa por bairro (média por ocorrência)
    df_grouped = df.groupby("bairro").agg({
        "latitude": "mean",
        "longitude": "mean",
        "vitimas": "mean"  # ou "sum" se quiser o total por bairro
    }).reset_index()

    # Prepara resposta agrupada
    resposta = [
        {
            "bairro": row["bairro"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "vitimas": float(row["vitimas"])
        }
        for _, row in df_grouped.iterrows()
    ]
    return {
        "ano": ano,
        "acidentes": resposta
    }
