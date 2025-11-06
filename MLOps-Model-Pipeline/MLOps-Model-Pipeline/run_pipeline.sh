#!/bin/bash

# Script para executar o pipeline de MLOps

echo "--- 1. Preparação do Ambiente ---"
# Instala as dependências (assumindo que o ambiente virtual já está ativo)
pip install -r requirements.txt

echo "--- 2. Geração e Preparação dos Dados ---"
python3 src/data_prep.py

echo "--- 3. Treinamento e Avaliação do Modelo ---"
python3 src/train_model.py

echo "--- Pipeline de MLOps Concluído com Sucesso! ---"
echo "Modelo salvo em: models/model.pkl"
echo "Metadados salvos em: models/model_metadata.json"
