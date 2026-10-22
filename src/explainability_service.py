from pathlib import Path

import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import train_test_split

from src.prediction_service import (
    modelo,
    obter_variaveis_modelo,
)


RAIZ_PROJETO = Path(__file__).resolve().parent.parent

CAMINHO_DADOS = (
    RAIZ_PROJETO
    / "data"
    / "data.csv"
)

SEMENTE = 42


def preparar_explicador():
    dados = pd.read_csv(
        CAMINHO_DADOS
    )

    dados = dados.drop(
        columns=[
            "id",
            "Unnamed: 32"
        ],
        errors="ignore"
    )

    dados["diagnosis"] = (
        dados["diagnosis"]
        .map({
            "B": 0,
            "M": 1
        })
        .astype(int)
    )

    X = dados.drop(
        columns=["diagnosis"]
    )

    y = dados["diagnosis"]

    X_train, _, _, _ = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=SEMENTE
        )
    )

    scaler = (
        modelo.named_steps[
            "padronizacao"
        ]
    )

    regressao = (
        modelo.named_steps[
            "modelo"
        ]
    )

    X_train_padronizado = pd.DataFrame(
        scaler.transform(
            X_train
        ),
        columns=X_train.columns
    )

    explicador = shap.LinearExplainer(
        regressao,
        X_train_padronizado
    )

    return (
        scaler,
        explicador
    )


scaler, explicador_shap = (
    preparar_explicador()
)


def explicar_caso(
    dados_paciente: dict,
    quantidade_fatores: int = 5
):
    variaveis = obter_variaveis_modelo()

    paciente = pd.DataFrame([
        {
            variavel:
                dados_paciente[
                    variavel
                ]
            for variavel
            in variaveis
        }
    ])

    paciente_padronizado = pd.DataFrame(
        scaler.transform(
            paciente
        ),
        columns=variaveis
    )

    valores_shap = (
        explicador_shap(
            paciente_padronizado
        )
    )

    valores = (
        valores_shap.values[0]
    )

    indices = np.argsort(
        np.abs(valores)
    )[::-1][
        :quantidade_fatores
    ]

    fatores = []

    for indice in indices:

        valor_shap = float(
            valores[indice]
        )

        nome_variavel = (
            variaveis[indice]
        )

        valor_original = float(
            paciente.iloc[
                0,
                indice
            ]
        )

        efeito = (
            "aumenta a tendência para malignidade"
            if valor_shap > 0
            else
            "reduz a tendência para malignidade"
        )

        fatores.append({
            "variavel":
                nome_variavel,

            "valor":
                valor_original,

            "valor_shap":
                valor_shap,

            "efeito":
                efeito
        })

    return fatores