from unittest.mock import patch

import pandas as pd

from fastapi.testclient import TestClient

from api.main import app


cliente = TestClient(
    app
)


def carregar_payload():
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

    return {
        "dados": {
            coluna: float(valor)
            for coluna, valor
            in linha.items()
        }
    }


def test_rota_raiz():
    resposta = cliente.get(
        "/"
    )

    assert resposta.status_code == 200

    dados = resposta.json()

    assert dados["status"] == "online"


def test_health():
    resposta = cliente.get(
        "/health"
    )

    assert resposta.status_code == 200

    dados = resposta.json()

    assert dados["status"] == "ok"


def test_features():
    resposta = cliente.get(
        "/features"
    )

    assert resposta.status_code == 200

    dados = resposta.json()

    assert dados["quantidade"] == 30

    assert len(
        dados["variaveis"]
    ) == 30


def test_metrics():
    resposta = cliente.get(
        "/metrics"
    )

    assert resposta.status_code == 200

    assert (
        "api_requisicoes_total"
        in resposta.text
    )

    assert (
        "api_tempo_requisicao_segundos"
        in resposta.text
    )


def test_predict():
    payload = carregar_payload()

    resposta = cliente.post(
        "/predict",
        json=payload,
    )

    assert resposta.status_code == 200

    dados = resposta.json()

    assert (
        "probabilidade_malignidade"
        in dados
    )

    assert "threshold" in dados

    assert "classificacao" in dados

    assert (
        0.0
        <= dados[
            "probabilidade_malignidade"
        ]
        <= 1.0
    )

    assert dados[
        "classificacao"
    ] in [
        "Benigno",
        "Maligno",
    ]


@patch(
    "api.main.gerar_explicacao",
    return_value=(
        "Explicação simulada para teste."
    ),
)
def test_interpret(
    mock_gerar_explicacao,
):
    payload = carregar_payload()

    resposta = cliente.post(
        "/interpret",
        json=payload,
    )

    assert resposta.status_code == 200

    dados = resposta.json()

    assert (
        "probabilidade_malignidade"
        in dados
    )

    assert "threshold" in dados

    assert "classificacao" in dados

    assert (
        "principais_fatores"
        in dados
    )

    assert (
        len(
            dados["principais_fatores"]
        )
        == 5
    )

    assert (
        dados["explicacao_llm"]
        ==
        "Explicação simulada para teste."
    )

    mock_gerar_explicacao.assert_called_once()