import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from .data_prep import load_data

# Caminhos de arquivos
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

def train_and_evaluate(df: pd.DataFrame):
    """
    Treina um modelo de Regressão Logística e avalia seu desempenho.
    """
    print("Iniciando treinamento e avaliação do modelo...")
    
    # Separação de features e target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Divisão em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Treinamento do modelo
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Previsão e avaliação
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Acurácia do Modelo: {accuracy:.4f}")
    print(f"F1-Score do Modelo: {f1:.4f}")
    
    # Salva o modelo e os metadados
    save_model_and_metadata(model, accuracy, f1)

def save_model_and_metadata(model, accuracy: float, f1: float):
    """
    Salva o modelo treinado e registra os metadados (simulando o registro em MLflow).
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Salva o modelo
    joblib.dump(model, MODEL_PATH)
    print(f"Modelo salvo em: {MODEL_PATH}")
    
    # 2. Salva os metadados
    metadata = {
        "model_name": "LogisticRegressionClassifier",
        "version": "1.0.0",
        "metrics": {
            "accuracy": accuracy,
            "f1_score": f1
        },
        "parameters": model.get_params(),
        "model_path": MODEL_PATH
    }
    
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadados do modelo salvos em: {METADATA_PATH}")

def main():
    """
    Função principal para executar o pipeline de treinamento.
    """
    try:
        df = load_data()
        train_and_evaluate(df)
    except FileNotFoundError:
        print("Erro: Dados não encontrados. Execute 'data_prep.py' primeiro.")

if __name__ == "__main__":
    main()
