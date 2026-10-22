import pandas as pd

from src.explainability_service import (
    explicar_caso,
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


def test_shap_retorna_cinco_fatores():
    paciente = (
        carregar_primeiro_caso()
    )

    fatores = explicar_caso(
        paciente
    )

    assert len(fatores) == 5


def test_fatores_shap_possuem_campos_esperados():
    paciente = (
        carregar_primeiro_caso()
    )

    fatores = explicar_caso(
        paciente
    )

    for fator in fatores:

        assert "variavel" in fator
        assert "valor" in fator
        assert "valor_shap" in fator
        assert "efeito" in fator


def test_efeito_shap_e_valido():
    paciente = (
        carregar_primeiro_caso()
    )

    fatores = explicar_caso(
        paciente
    )

    efeitos_validos = [
        (
            "aumenta a tendência "
            "para malignidade"
        ),
        (
            "reduz a tendência "
            "para malignidade"
        )
    ]

    for fator in fatores:
        assert (
            fator["efeito"]
            in efeitos_validos
        )