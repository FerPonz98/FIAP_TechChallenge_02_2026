"""
Serviço de interpretação dos resultados do modelo com LLM.

A LLM NÃO classifica: recebe a probabilidade, o threshold, a classificação
e os fatores SHAP calculados pelo modelo de Machine Learning e os
transforma em uma explicação em linguagem natural.
"""

import json
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from openai import OpenAI
from sklearn.model_selection import train_test_split


RAIZ_PROJETO = Path(__file__).resolve().parent.parent

CAMINHO_DADOS = RAIZ_PROJETO / "data" / "data.csv"
CAMINHO_THRESHOLD = RAIZ_PROJETO / "results" / "threshold_otimizado.json"

SEMENTE = 42

LIMITE_ZONA_INCERTEZA = 0.10

TIMEOUT_SEGUNDOS = 60
MAX_TENTATIVAS = 2


class LLMIndisponivelError(Exception):
    """Erro levantado quando a LLM não pode ser utilizada."""


# =========================================================
# Cliente OpenAI (criado somente na primeira chamada)
# =========================================================

@lru_cache(maxsize=1)
def obter_cliente():
    """
    Cria o cliente da OpenAI apenas quando necessário.

    Assim a API inicia normalmente (/health, /predict) mesmo
    sem OPENAI_API_KEY configurada.
    """

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise LLMIndisponivelError(
            "OPENAI_API_KEY não configurada."
        )

    return OpenAI(
        api_key=api_key,
        timeout=TIMEOUT_SEGUNDOS,
        max_retries=MAX_TENTATIVAS,
    )


def obter_modelo_llm():

    modelo_llm = os.getenv("LLM_MODEL")

    if not modelo_llm:
        raise LLMIndisponivelError(
            "LLM_MODEL não configurado."
        )

    return modelo_llm


# =========================================================
# Dados de referência para o contexto do prompt
# =========================================================

@lru_cache(maxsize=1)
def carregar_x_train():
    """
    Recria o conjunto de treino com o mesmo split dos notebooks.

    É usado apenas para calcular o percentil de cada variável,
    sem utilizar o conjunto de teste.
    """

    dados = pd.read_csv(CAMINHO_DADOS)

    dados = dados.drop(
        columns=["id", "Unnamed: 32"],
        errors="ignore",
    )

    dados["diagnosis"] = (
        dados["diagnosis"]
        .map({"B": 0, "M": 1})
        .astype(int)
    )

    X = dados.drop(columns=["diagnosis"])
    y = dados["diagnosis"]

    X_train, _, _, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=SEMENTE,
    )

    return X_train


@lru_cache(maxsize=1)
def carregar_desempenho_modelo():
    """
    Lê o desempenho do modelo no teste com o threshold adotado
    (salvo pelo notebook 03).
    """

    try:
        with open(CAMINHO_THRESHOLD, "r", encoding="utf-8") as arquivo:
            configuracao = json.load(arquivo)

        resultado = configuracao["resultado_otimizado_teste"]

        return {
            "recall": float(resultado["recall"]),
            "precision": float(resultado["precision"]),
        }

    except (FileNotFoundError, KeyError):
        return None


# =========================================================
# Formatação
# =========================================================

def formatar_percentual(valor):

    if valor > 0.9999:
        return ">99,99%"

    if valor < 0.0001:
        return "<0,01%"

    return f"{valor * 100:.2f}%".replace(".", ",")


def classificar_percentil(percentil):

    if percentil >= 75:
        return "alto"

    if percentil <= 25:
        return "baixo"

    return "intermediário"


def descrever_zona_decisao(probabilidade, threshold):

    distancia = probabilidade - threshold

    if abs(distancia) < LIMITE_ZONA_INCERTEZA:
        zona = (
            "ZONA DE INCERTEZA: a probabilidade está a menos de "
            f"{LIMITE_ZONA_INCERTEZA * 100:.0f} pontos percentuais do threshold"
        )
    elif distancia >= 0:
        zona = "acima do threshold, com margem"
    else:
        zona = "abaixo do threshold, com margem"

    distancia_texto = f"{distancia * 100:+.2f}".replace(".", ",")

    return f"{zona} (distância: {distancia_texto} p.p.)"


# =========================================================
# Prompt
# =========================================================

def criar_prompt(
    resultado_predicao: dict,
    fatores_shap: list,
    observacoes_clinicas: str | None = None,
):
    X_train = carregar_x_train()

    linhas_fatores = []

    for fator in fatores_shap:

        percentil = float(
            (X_train[fator["variavel"]] <= fator["valor"]).mean() * 100
        )

        linhas_fatores.append(
            f"- {fator['variavel']}: valor={fator['valor']:.4f} "
            f"(percentil {percentil:.0f} do treino, "
            f"nível {classificar_percentil(percentil)}); "
            f"SHAP={fator['valor_shap']:+.4f}; {fator['efeito']}."
        )

    fatores_texto = "\n".join(linhas_fatores)

    probabilidade = resultado_predicao["probabilidade_malignidade"]
    threshold = resultado_predicao["threshold"]

    desempenho = carregar_desempenho_modelo()

    if desempenho:
        desempenho_texto = (
            f"- Recall (sensibilidade): {formatar_percentual(desempenho['recall'])}\n"
            f"- Precision: {formatar_percentual(desempenho['precision'])}"
        )
    else:
        desempenho_texto = "- Não disponível."

    observacoes = (
        observacoes_clinicas
        or "Não disponíveis nesta fase (dados textuais serão integrados no Módulo 3)."
    )

    return f"""
Você é um assistente de apoio à interpretação de resultados de um modelo
de Machine Learning para classificação de câncer de mama. Seu público são
profissionais de saúde.

REGRAS:
- Não realize um novo diagnóstico e não altere a classificação do modelo.
- Não invente informações clínicas; utilize somente os dados fornecidos.
- Não apresente a classificação como certeza ou diagnóstico definitivo.
- Não recomende tratamentos, medicamentos ou exames específicos.
- Se o caso estiver na ZONA DE INCERTEZA, deixe isso explícito no resumo.
- Use os percentis para dizer se cada valor é baixo, intermediário ou alto
  em relação aos pacientes do conjunto de treino.
- Valores SHAP estão em escala de log-odds: positivos aumentam e negativos
  reduzem a tendência para malignidade.

RESULTADO DO MODELO:
- Probabilidade estimada de malignidade: {formatar_percentual(probabilidade)}
- Threshold utilizado: {f"{threshold:.2f}".replace(".", ",")}
- Classificação do modelo: {resultado_predicao['classificacao']}
- Zona de decisão: {descrever_zona_decisao(probabilidade, threshold)}

DESEMPENHO DO MODELO NO CONJUNTO DE TESTE:
{desempenho_texto}

PRINCIPAIS FATORES (SHAP):
{fatores_texto}

OBSERVAÇÕES CLÍNICAS EM TEXTO:
{observacoes}

Produza a resposta em português, com no máximo 350 palavras, nas seções:

1. Resumo do resultado
2. Principais fatores que influenciaram a previsão
3. Interpretação do resultado
4. Pontos de atenção para o profissional de saúde
5. Limitações e recomendação de revisão clínica
"""


# =========================================================
# Geração da explicação
# =========================================================

def gerar_explicacao(
    resultado_predicao: dict,
    fatores_shap: list,
    observacoes_clinicas: str | None = None,
):
    """
    Gera a explicação em linguagem natural.

    Levanta LLMIndisponivelError em qualquer falha, para que a API
    possa responder com a predição e o SHAP mesmo sem a LLM.
    """

    prompt = criar_prompt(
        resultado_predicao,
        fatores_shap,
        observacoes_clinicas,
    )

    try:
        cliente = obter_cliente()

        resposta = cliente.responses.create(
            model=obter_modelo_llm(),
            input=prompt,
        )

        return resposta.output_text

    except LLMIndisponivelError:
        raise

    except Exception as erro:
        # A mensagem original não é repassada ao usuário,
        # pois pode conter detalhes internos da requisição.
        raise LLMIndisponivelError(
            f"Falha na chamada à LLM ({type(erro).__name__})."
        ) from erro