import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests


URL = "http://localhost:8000/predict"

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