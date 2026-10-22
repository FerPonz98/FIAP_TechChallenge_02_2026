import os

from openai import OpenAI


API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

MODELO_LLM = os.getenv(
    "LLM_MODEL"
)


if not API_KEY:
    raise ValueError(
        "OPENAI_API_KEY não encontrada."
    )

if not MODELO_LLM:
    raise ValueError(
        "LLM_MODEL não encontrado."
    )


cliente = OpenAI(
    api_key=API_KEY
)


def criar_prompt(
    resultado_predicao: dict,
    fatores_shap: list
):
    fatores_texto = "\n".join([
        (
            f"- {fator['variavel']}: "
            f"valor={fator['valor']:.4f}; "
            f"SHAP={fator['valor_shap']:.4f}; "
            f"{fator['efeito']}."
        )
        for fator
        in fatores_shap
    ])

    return f"""
Você é um assistente de apoio à interpretação
de resultados de Machine Learning para
classificação de câncer de mama.

REGRAS:
- Não realize um novo diagnóstico.
- Não altere a classificação do modelo.
- Não invente dados clínicos.
- Utilize somente as informações fornecidas.
- O resultado deve ser tratado como apoio à decisão.
- Recomende revisão por profissional de saúde.

RESULTADO DO MODELO:

Probabilidade estimada de malignidade:
{resultado_predicao['probabilidade_malignidade']:.2%}

Threshold utilizado:
{resultado_predicao['threshold']:.2f}

Classificação:
{resultado_predicao['classificacao']}

PRINCIPAIS FATORES SHAP:

{fatores_texto}

Produza a resposta com:

1. Resumo do resultado
2. Principais fatores
3. Interpretação
4. Limitações e recomendação de revisão clínica
"""


def gerar_explicacao(
    resultado_predicao: dict,
    fatores_shap: list
):
    prompt = criar_prompt(
        resultado_predicao,
        fatores_shap
    )

    resposta = (
        cliente.responses.create(
            model=MODELO_LLM,
            input=prompt
        )
    )

    return resposta.output_text