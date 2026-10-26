# FIAP Tech Challenge — Fase 2

**Projeto 1 — Otimização de Modelos de Diagnóstico**

Evolução do modelo de classificação de câncer de mama da Fase 1 com **Algoritmo Genético** para otimização de hiperparâmetros, **SHAP** para explicabilidade, **LLM** para interpretação em linguagem natural, API REST, observabilidade e **escalabilidade automática com Kubernetes (HPA)**.

**Integrantes:** Fernando Ponz Vargas

**Vídeo de demonstração:** <!-- PREENCHER: link do YouTube/Vimeo -->

**Documentação técnica completa:** [`docs/arquitetura.md`](docs/arquitetura.md)

> Projeto acadêmico. Nenhum resultado deve ser usado como diagnóstico médico.

---

## 1. Resultados principais

Conjunto de teste (114 casos, 42 malignos), threshold 0,50:

| Modelo                          |   Accuracy |  Precision |     Recall |         F1 |    ROC-AUC |    FN |    FP |
| ------------------------------- | ---------: | ---------: | ---------: | ---------: | ---------: | ----: | ----: |
| Fase 1 — GridSearchCV           |     0,9737 |     0,9756 |     0,9524 |     0,9639 |     0,9954 |     2 |     1 |
| **Fase 2 — Algoritmo Genético** | **0,9912** | **1,0000** | **0,9762** | **0,9880** | **0,9977** | **1** | **0** |

- **Algoritmo Genético:** 3 experimentos convergiram para `C = 0,1`, `class_weight = balanced`, `liblinear/l2`.
- **Threshold:** 0,34 selecionado por F2-score com probabilidades _out-of-fold_ (sem usar o teste). A redução para 0,30, recomendada na avaliação da Fase 1, **não eliminou** os falsos negativos; o ganho veio do Algoritmo Genético.
- **SHAP:** 8 das 10 variáveis mais importantes coincidem entre o modelo da Fase 1 e o otimizado.
- **LLM:** explica o resultado sem classificar; prompt v2 com zona de incerteza, percentis e pontos de atenção; avaliação manual, automática e por LLM.
- **Kubernetes:** HPA escalou de **1 → 2 → 4 pods** sob carga.
- **Testes:** **15 testes automatizados aprovados**.

---

## 2. Arquitetura

```text
Usuário ──HTTP──▶ FastAPI ──▶ Pipeline (StandardScaler → LogisticRegression otimizada pelo AG)
                     │                 │
                     │                 ▼
                     │        Probabilidade ≥ 0,34 → Classificação
                     │                 │
                     │                 ▼
                     │        SHAP (5 principais fatores)
                     │                 │
                     │                 ▼
                     │        Prompt estruturado → LLM → Explicação
                     │
                     ├── logs/api.log  (logging)
                     └── /metrics      (formato Prometheus)

Pods ──▶ Metrics Server ──▶ HPA ──▶ 1 a 5 réplicas   (independente do /metrics)
```

A LLM **não** realiza a classificação: ela apenas interpreta o resultado do modelo. Se a LLM falhar, `/interpret` continua respondendo com a predição e o SHAP.

---

## 3. Estrutura

```text
api/main.py                  # FastAPI: endpoints, logging e métricas
src/
├── prediction_service.py    # carrega modelo e threshold, gera a predição
├── explainability_service.py# SHAP LinearExplainer
├── llm_service.py           # prompt v2, timeout, retries e fallback
└── logging_config.py
notebooks/
├── 01_baseline_fase1.ipynb              # reproduz o modelo da Fase 1
├── 02_algoritmo_genetico.ipynb          # AG: genes, operadores, 3 experimentos
├── 03_threshold_e_explicabilidade.ipynb # threshold, recomendação 0,30, SHAP
└── 04_llm.ipynb                         # LLM, prompt v1 × v2, avaliação
tests/                       # testes automatizados (pytest)
scripts/                     # testes manuais e teste de carga (API em execução)
k8s/                         # Deployment, Service, HPA, Secret (exemplo), patch do Metrics Server
docs/                        # arquitetura.md e evidências do autoscaling
models/modelo_ga_fase2.joblib
results/                     # métricas, gráficos e avaliações geradas pelos notebooks
```

---

## 4. Como executar

### 4.1 Configuração

```powershell
Copy-Item .env.example .env      # Windows
```

```bash
cp .env.example .env             # Linux/macOS
```

```env
OPENAI_API_KEY=sua_chave
LLM_MODEL=seu_modelo
TZ=America/La_Paz
```

Sem `OPENAI_API_KEY` a API inicia normalmente; apenas `/interpret` responde sem a explicação da LLM.

### 4.2 Docker

```bash
docker compose up -d --build
```

| Serviço       | Endereço                   |
| ------------- | -------------------------- |
| API (Swagger) | http://localhost:8000/docs |
| Jupyter       | http://localhost:8888      |

Execute os notebooks no Jupyter, nesta ordem: `01 → 02 → 03 → 04`.

### 4.3 Sem Docker (ambiente virtual)

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
uvicorn api.main:app --reload
```

### 4.4 Testes

```bash
docker compose exec api python -m pytest tests/ -v
```

Resultado: **15 testes aprovados** (API, predição, SHAP, `/interpret` com LLM simulada e indisponível, validação de entrada e `/metrics`). A LLM é simulada com `mock` para não consumir créditos.

Testes manuais com a API em execução:

```bash
python scripts/test_predict_manual.py
python scripts/test_interpret_manual.py
```

---

## 5. API

| Método | Rota         | Descrição                                       |
| ------ | ------------ | ----------------------------------------------- |
| GET    | `/health`    | Health check (usado pelas probes do Kubernetes) |
| GET    | `/features`  | As 30 variáveis esperadas                       |
| POST   | `/predict`   | Probabilidade, threshold e classificação        |
| POST   | `/interpret` | Predição + 5 fatores SHAP + explicação da LLM   |
| GET    | `/metrics`   | Métricas em formato compatível com Prometheus   |

No Swagger, `/predict` e `/interpret` já trazem um paciente de exemplo.

---

## 6. Kubernetes e escalabilidade automática

```bash
docker build -t fiap-fase2-api:latest .
kubectl create secret generic fiap-fase2-secrets --from-env-file=.env
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/hpa.yaml
```

- **HPA:** 1 a 5 réplicas, meta de 60% de CPU (sobre `requests` de 100m).
- **Metrics Server:** no Docker Desktop exige `--kubelet-insecure-tls` (`k8s/metrics-server-patch.yaml`, somente ambiente local).
- **Teste de carga** (`scripts/load_test.py`, 50 workers, 120 s, `POST /predict`): 26.150 requisições, 16.017 com sucesso (61,3%). Antes do ajuste do `timeoutSeconds` das probes, a taxa era de 38%. É um teste de estresse para validar o HPA, não um benchmark.

### Evidências

**HPA aumentando réplicas (1 → 2 → 4):**

![Escalabilidade automática do HPA](docs/evidencias/01_hpa_escalado_replicas.png)

**Criação automática de pods:**

![Criação automática de Pods](docs/evidencias/02_kubernetes_criacao_pods.png)

**Resultado do teste de carga:**

![Resultado do teste de carga](docs/evidencias/03_resultado_teste_carga.png)

---

## 7. Avaliação da LLM

Casos avaliados: alta probabilidade, próximo ao threshold (falso positivo), baixa probabilidade e falso negativo.

| Método         | O que avalia                                                                                                                                       |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Manual         | Notas de 1 a 5 (fidelidade, clareza, segurança, completude) atribuídas pelo grupo                                                                  |
| Automática     | Probabilidade/classificação corretas, fatores citados, seções, recomendação de revisão, ausência de linguagem de certeza, sinalização de incerteza |
| LLM-as-a-judge | Rubrica de 1 a 5, incluindo acionabilidade                                                                                                         |

<!-- PREENCHER: tabela de results/comparacao_prompts_v1_v2.csv -->

Detalhes em [`docs/arquitetura.md`](docs/arquitetura.md#10-avaliação-das-interpretações-da-llm).

---

## 8. Segurança

- `.env` e `k8s/secret.yaml` não são versionados; apenas `.env.example` e `k8s/secret.example.yaml`.
- O log não registra a chave da API nem as variáveis clínicas do paciente.

---

## 9. Limitações

- Dataset público, sem validação clínica; o conjunto de teste é pequeno (114 casos).
- Kubernetes executado localmente (Docker Desktop).
- A avaliação da LLM utiliza poucos casos.

Detalhes, decisões de implementação, desafios e evoluções futuras: [`docs/arquitetura.md`](docs/arquitetura.md).
