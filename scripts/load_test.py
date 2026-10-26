import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests


# Por padrão, o teste de carga é feito contra a API do Kubernetes (porta 8010).
# Para usar outra URL: python scripts/load_test.py http://localhost:8000
BASE_URL = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.getenv("API_URL", "http://localhost:8010")
)

URL = BASE_URL.rstrip("/") + "/predict"

DURACAO_SEGUNDOS = 120

TRABALHADORES = 50


dados = pd.read_csv(
    "data/data.csv"
)

dados = dados.drop(
    columns=[
        "id",
        "Unnamed: 32",
    ],
    errors="ignore",
)

linha = dados.iloc[0].copy()

linha.pop(
    "diagnosis"
)

payload = {
    "dados": {
        coluna: float(valor)
        for coluna, valor
        in linha.items()
    }
}


def enviar_requisicao():
    try:
        resposta = requests.post(
            URL,
            json=payload,
            timeout=10,
        )

        return resposta.status_code

    except Exception:
        return 0


def executar_carga():
    inicio = time.time()

    total = 0

    sucessos = 0

    print(
        "Iniciando teste de carga..."
    )

    print(
        f"URL: {URL}"
    )

    print(
        f"Duração: {DURACAO_SEGUNDOS} segundos"
    )

    print(
        f"Trabalhadores simultâneos: {TRABALHADORES}"
    )

    with ThreadPoolExecutor(
        max_workers=TRABALHADORES
    ) as executor:

        while (
            time.time() - inicio
            < DURACAO_SEGUNDOS
        ):

            futuros = [
                executor.submit(
                    enviar_requisicao
                )
                for _ in range(
                    TRABALHADORES
                )
            ]

            for futuro in futuros:
                total += 1

                if futuro.result() == 200:
                    sucessos += 1

    print()
    print(
        "Teste concluído."
    )

    print(
        f"Total de requisições: {total}"
    )

    print(
        f"Sucessos: {sucessos}"
    )


if __name__ == "__main__":
    executar_carga()