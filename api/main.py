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
    LLMIndisponivelError,
    gerar_explicacao,
)

from src.logging_config import (
    logger,
)

from src.prediction_service import (
    obter_variaveis_modelo,
    prever_caso,
)



app = FastAPI(
    title="FIAP Tech Challenge - Fase 2",
    description=(
        "API para classificação de câncer de mama "
        "utilizando modelo otimizado por Algoritmo Genético, "
        "explicabilidade com SHAP e interpretação com LLM."
    ),
    version="1.0.0",
)


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


class PacienteRequest(BaseModel):
    dados: dict[str, float]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "dados": {
                    "radius_mean": 17.99,
                    "texture_mean": 10.38,
                    "perimeter_mean": 122.8,
                    "area_mean": 1001.0,
                    "smoothness_mean": 0.1184,
                    "compactness_mean": 0.2776,
                    "concavity_mean": 0.3001,
                    "concave points_mean": 0.1471,
                    "symmetry_mean": 0.2419,
                    "fractal_dimension_mean": 0.07871,
                    "radius_se": 1.095,
                    "texture_se": 0.9053,
                    "perimeter_se": 8.589,
                    "area_se": 153.4,
                    "smoothness_se": 0.006399,
                    "compactness_se": 0.04904,
                    "concavity_se": 0.05373,
                    "concave points_se": 0.01587,
                    "symmetry_se": 0.03003,
                    "fractal_dimension_se": 0.006193,
                    "radius_worst": 25.38,
                    "texture_worst": 17.33,
                    "perimeter_worst": 184.6,
                    "area_worst": 2019.0,
                    "smoothness_worst": 0.1622,
                    "compactness_worst": 0.6656,
                    "concavity_worst": 0.7119,
                    "concave points_worst": 0.2654,
                    "symmetry_worst": 0.4601,
                    "fractal_dimension_worst": 0.1189
                    }
                }
            ]
        }
    }


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
    explicacao_llm: str | None = None
    aviso: str | None = None



@app.get("/")
def raiz():
    return {
        "projeto": "FIAP Tech Challenge - Fase 2",
        "versao": "1.0.0",
        "status": "online",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "modelo": "carregado",
    }


@app.get("/metrics")
def metricas():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )

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

    # A falha da LLM não invalida a predição nem o SHAP:
    # a API responde com o resultado do modelo e um aviso.
    explicacao = None
    aviso = None

    try:
        explicacao = gerar_explicacao(
            resultado,
            fatores,
        )

    except LLMIndisponivelError as erro:
        logger.error(
            "LLM indisponível: %s",
            erro,
        )

        aviso = (
            "LLM indisponível. A classificação e os fatores SHAP "
            "foram gerados normalmente pelo modelo."
        )

    logger.info(
        "Interpretação realizada | "
        "Classificação=%s | "
        "Probabilidade=%.6f | "
        "Fatores_SHAP=%d | "
        "LLM=%s",
        resultado[
            "classificacao"
        ],
        resultado[
            "probabilidade_malignidade"
        ],
        len(fatores),
        "ok" if explicacao else "indisponível",
    )

    return {
        **resultado,
        "principais_fatores":
            fatores,
        "explicacao_llm":
            explicacao,
        "aviso":
            aviso,
    }