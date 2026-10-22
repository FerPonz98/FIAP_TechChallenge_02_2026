import pandas as pd
import requests


dados = pd.read_csv(
    "data/data.csv"
)

dados = dados.drop(
    columns=[
        "id",
        "Unnamed: 32"
    ],
    errors="ignore"
)

linha = dados.iloc[0].copy()

diagnostico_real = linha.pop(
    "diagnosis"
)

payload = {
    "dados": {
        coluna: float(valor)
        for coluna, valor
        in linha.items()
    }
}


resposta = requests.post(
    "http://localhost:8000/predict",
    json=payload,
    timeout=30
)


print(
    "Status:",
    resposta.status_code
)

print()

print(
    "Diagnóstico real:",
    diagnostico_real
)

print()

print(
    "Resposta da API:"
)

print(
    resposta.json()
)