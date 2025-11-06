import pandas as pd
from sklearn.datasets import make_classification
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw_data.csv")

def generate_synthetic_data(n_samples=1000, n_features=10, random_state=42):
    """
    Gera um dataset sintético para um problema de classificação binária.
    """
    print("Gerando dados sintéticos...")
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=5,
        n_redundant=0,
        n_classes=2,
        random_state=random_state
    )
    
    # Cria um DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df

def save_data(df: pd.DataFrame):
    """
    Salva o DataFrame gerado no diretório de dados.
    """
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Dados salvos em: {DATA_PATH}")

def load_data() -> pd.DataFrame:
    """
    Carrega os dados do diretório de dados.
    """
    print(f"Carregando dados de: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)

if __name__ == "__main__":
    # Exemplo de uso: gerar e salvar os dados
    df = generate_synthetic_data()
    save_data(df)
