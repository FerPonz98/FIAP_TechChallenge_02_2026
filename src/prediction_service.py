from pathlib import Path

import json
import joblib
import pandas as pd


RAIZ_PROJETO = Path(__file__).resolve().parent.parent

CAMINHO_MODELO = (
    RAIZ_PROJETO
    / "models"
    / "modelo_ga_fase2.joblib"
)

CAMINHO_THRESHOLD = (
    RAIZ_PROJETO
    / "results"
    / "threshold_otimizado.json"
)


def carregar_modelo():
    if not CAMINHO_MODELO.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {CAMINHO_MODELO}"
        )

    return joblib.load(
        CAMINHO_MODELO
    )


def carregar_threshold():
    if not CAMINHO_THRESHOLD.exists():
        raise FileNotFoundError(
            f"Threshold não encontrado: {CAMINHO_THRESHOLD}"
        )

    with open(
        CAMINHO_THRESHOLD,
        "r",
        encoding="utf-8"
    ) as arquivo:
        configuracao = json.load(
            arquivo
        )

    return float(
        configuracao[
            "threshold_otimizado"
        ]
    )


modelo = carregar_modelo()
threshold = carregar_threshold()


def obter_variaveis_modelo():
    if hasattr(
        modelo,
        "feature_names_in_"
    ):
        return list(
            modelo.feature_names_in_
        )

    raise ValueError(
        "O modelo não contém os nomes "
        "das variáveis de entrada."
    )


def prever_caso(
    dados_paciente: dict
):
    variaveis = obter_variaveis_modelo()

    variaveis_ausentes = [
        variavel
        for variavel in variaveis
        if variavel not in dados_paciente
    ]

    if variaveis_ausentes:
        raise ValueError(
            "Variáveis ausentes: "
            + ", ".join(
                variaveis_ausentes
            )
        )

    paciente = pd.DataFrame(
        [
            {
                variavel:
                    dados_paciente[
                        variavel
                    ]
                for variavel
                in variaveis
            }
        ]
    )

    probabilidade = float(
        modelo.predict_proba(
            paciente
        )[0, 1]
    )

    classificacao = (
        "Maligno"
        if probabilidade >= threshold
        else "Benigno"
    )

    return {
        "probabilidade_malignidade":
            probabilidade,

        "threshold":
            threshold,

        "classificacao":
            classificacao
    }