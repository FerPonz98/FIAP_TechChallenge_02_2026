import logging
from pathlib import Path


RAIZ_PROJETO = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

PASTA_LOGS = (
    RAIZ_PROJETO
    / "logs"
)

PASTA_LOGS.mkdir(
    parents=True,
    exist_ok=True
)

CAMINHO_LOG = (
    PASTA_LOGS
    / "api.log"
)


def configurar_logging():
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
        handlers=[
            logging.FileHandler(
                CAMINHO_LOG,
                encoding="utf-8"
            ),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(
        "fiap_fase2"
    )


logger = configurar_logging()