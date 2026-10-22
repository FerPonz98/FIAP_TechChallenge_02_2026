import time

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
)

from fastapi.responses import Response

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

from pydantic import BaseModel

from src.explainability_service import (
    explicar_caso,
)

from src.llm_service import (
    gerar_explicacao,
)

from src.logging_config import (
    logger,
)

from src.prediction_service import (
    obter_variaveis_modelo,
    prever_caso,
)


# =========================================================
# Aplicação FastAPI
# =========================================================

app = FastAPI(
    title="FIAP Tech Challenge - Fase 2",
    description=(
        "API para classificação de câncer de mama "
        "utilizando modelo otimizado por Algoritmo Genético, "
        "explicabilidade com SHAP e interpretação com LLM."
    ),
    version="1.0.0",
)


# =========================================================
# Métricas Prometheus
# =========================================================

TOTAL_REQUISICOES = Counter(
    "api_requisicoes_total",
    "Quantidade total de requisições da API",
    [
        "metodo",
        "rota",
        "status",
    ],
)


TEMPO_REQUISICAO = Histogram(
    "api_tempo_requisicao_segundos",
    "Tempo de processamento das requisições da API",
    [
        "metodo",
        "rota",
    ],
)


# =========================================================
# Middleware de logging e monitoramento
# =========================================================

@app.middleware("http")
async def registrar_requisicao(
    request: Request,
    call_next,
):
    inicio = time.perf_counter()

    resposta = await call_next(
        request
    )

    duracao = (
        time.perf_counter()
        - inicio
    )

    rota = request.url.path

    logger.info(
        "Método=%s | Rota=%s | Status=%s | Tempo=%.4fs",
        request.method,
        rota,
        resposta.status_code,
        duracao,
    )

    TOTAL_REQUISICOES.labels(
        metodo=request.method,
        rota=rota,
        status=str(
            resposta.status_code
        ),
    ).inc()

    TEMPO_REQUISICAO.labels(
        metodo=request.method,
        rota=rota,
    ).observe(
        duracao
    )

    return resposta


# =========================================================
# Modelos de entrada e saída
# =========================================================

class PacienteRequest(BaseModel):
    dados: dict[str, float]


class PredicaoResponse(BaseModel):
    probabilidade_malignidade: float
    threshold: float
    classificacao: str


class FatorShap(BaseModel):
    variavel: str
    valor: float
    valor_shap: float
    efeito: str


class InterpretacaoResponse(BaseModel):
    probabilidade_malignidade: float
    threshold: float
    classificacao: str
    principais_fatores: list[FatorShap]
    explicacao_llm: str


# =========================================================
# Endpoint raiz
# =========================================================

@app.get("/")
def raiz():
    return {
        "projeto": "FIAP Tech Challenge - Fase 2",
        "versao": "1.0.0",
        "status": "online",
    }


# =========================================================
# Health Check
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "modelo": "carregado",
    }


# =========================================================
# Métricas Prometheus
# =========================================================

@app.get("/metrics")
def metricas():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# =========================================================
# Variáveis utilizadas pelo modelo
# =========================================================

@app.get("/features")
def listar_variaveis():
    variaveis = (
        obter_variaveis_modelo()
    )

    return {
        "quantidade":
            len(variaveis),

        "variaveis":
            variaveis,
    }


# =========================================================
# Predição
# =========================================================

@app.post(
    "/predict",
    response_model=PredicaoResponse,
)
def prever(
    requisicao: PacienteRequest,
):
    try:
        resultado = prever_caso(
            requisicao.dados
        )

        logger.info(
            "Predição realizada | "
            "Classificação=%s | "
            "Probabilidade=%.6f | "
            "Threshold=%.4f",
            resultado[
                "classificacao"
            ],
            resultado[
                "probabilidade_malignidade"
            ],
            resultado[
                "threshold"
            ],
        )

        return resultado

    except ValueError as erro:
        logger.warning(
            "Erro de validação na predição: %s",
            erro,
        )

        raise HTTPException(
            status_code=400,
            detail=str(erro),
        )

    except Exception:
        logger.exception(
            "Erro interno durante a predição."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Erro interno durante a previsão."
            ),
        )


# =========================================================
# Predição + SHAP + LLM
# =========================================================

@app.post(
    "/interpret",
    response_model=InterpretacaoResponse,
)
def interpretar(
    requisicao: PacienteRequest,
):
    try:
        resultado = prever_caso(
            requisicao.dados
        )

        fatores = explicar_caso(
            requisicao.dados
        )

        explicacao = gerar_explicacao(
            resultado,
            fatores,
        )

        logger.info(
            "Interpretação realizada | "
            "Classificação=%s | "
            "Probabilidade=%.6f | "
            "Fatores_SHAP=%d",
            resultado[
                "classificacao"
            ],
            resultado[
                "probabilidade_malignidade"
            ],
            len(fatores),
        )

        return {
            **resultado,
            "principais_fatores":
                fatores,
            "explicacao_llm":
                explicacao,
        }

    except ValueError as erro:
        logger.warning(
            "Erro de validação na interpretação: %s",
            erro,
        )

        raise HTTPException(
            status_code=400,
            detail=str(erro),
        )

    except Exception:
        logger.exception(
            "Erro interno durante a interpretação."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Erro interno durante a interpretação."
            ),
        )