import pandas as pd

from src.prediction_service import (
    obter_variaveis_modelo,
    prever_caso,
)


def carregar_primeiro_caso():
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

    linha.pop(
        "diagnosis"
    )

    return {
        coluna: float(valor)
        for coluna, valor
        in linha.items()
    }


def test_modelo_possui_30_variaveis():
    variaveis = (
        obter_variaveis_modelo()
    )

    assert len(variaveis) == 30


def test_previsao_retorna_campos_esperados():
    paciente = (
        carregar_primeiro_caso()
    )

    resultado = prever_caso(
        paciente
    )

    assert (
        "probabilidade_malignidade"
        in resultado
    )

    assert (
        "threshold"
        in resultado
    )

    assert (
        "classificacao"
        in resultado
    )


def test_probabilidade_esta_entre_zero_e_um():
    paciente = (
        carregar_primeiro_caso()
    )

    resultado = prever_caso(
        paciente
    )

    probabilidade = resultado[
        "probabilidade_malignidade"
    ]

    assert 0.0 <= probabilidade <= 1.0


def test_classificacao_e_valida():
    paciente = (
        carregar_primeiro_caso()
    )

    resultado = prever_caso(
        paciente
    )

    assert resultado[
        "classificacao"
    ] in [
        "Benigno",
        "Maligno"
    ]