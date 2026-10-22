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
    "http://localhost:8000/interpret",
    json=payload,
    timeout=120
)


print(
    "Status:",
    resposta.status_code
)

print(
    "Diagnóstico real:",
    diagnostico_real
)

print()


if resposta.status_code == 200:

    resultado = resposta.json()

    print(
        "Classificação:",
        resultado["classificacao"]
    )

    print(
        "Probabilidade:",
        resultado[
            "probabilidade_malignidade"
        ]
    )

    print(
        "Threshold:",
        resultado["threshold"]
    )

    print()

    print(
        "Principais fatores SHAP:"
    )

    for fator in resultado[
        "principais_fatores"
    ]:

        print(
            "-",
            fator["variavel"],
            "|",
            fator["efeito"]
        )

    print()

    print(
        "Explicação do LLM:"
    )

    print(
        resultado[
            "explicacao_llm"
        ]
    )

else:

    print(
        "Erro:"
    )

    print(
        resposta.text
    )