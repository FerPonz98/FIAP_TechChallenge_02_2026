from fastapi import FastAPI

app = FastAPI(
    title="FIAP Tech Challenge - Fase 2",
    description="Otimização de modelos de diagnóstico com Algoritmos Genéticos e LLMs.",
    version="1.0.0"
)


@app.get("/")
def raiz():
    return {
        "projeto": "FIAP Tech Challenge - Fase 2",
        "status": "online"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }