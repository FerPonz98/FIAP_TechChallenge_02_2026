# Arquitetura da Solução

## 1. Visão Geral

Este documento descreve a arquitetura implementada para o **Tech Challenge – Fase 2**, Projeto 1: **Otimização de Modelos de Diagnóstico**.

A solução evolui o modelo de classificação de câncer de mama desenvolvido na Fase 1, incorporando:

- otimização de hiperparâmetros por Algoritmo Genético;
- definição de threshold de decisão;
- explicabilidade com SHAP;
- integração com LLM para interpretação dos resultados;
- disponibilização do modelo através de API REST;
- logging e monitoramento;
- containerização com Docker;
- escalabilidade automática com Kubernetes e HPA;
- testes automatizados.

A arquitetura foi organizada para separar treinamento, inferência, explicabilidade, interpretação e infraestrutura.

---

# 2. Arquitetura Geral

```text
                         ┌─────────────────────────┐
                         │       Usuário /         │
                         │   Sistema Consumidor    │
                         └────────────┬────────────┘
                                      │
                                      │ HTTP / REST
                                      ▼
                         ┌─────────────────────────┐
                         │        FastAPI          │
                         │                         │
                         │ /health                 │
                         │ /features               │
                         │ /predict                │
                         │ /interpret              │
                         │ /metrics                │
                         └────────────┬────────────┘
                                      │
                    ┌─────────────────┼──────────────────┐
                    │                 │                  │
                    ▼                 ▼                  ▼
          ┌─────────────────┐ ┌──────────────┐ ┌─────────────────┐
          │ Modelo de ML    │ │     SHAP     │ │       LLM       │
          │ otimizado por   │ │ Explicação   │ │ Interpretação   │
          │ Algoritmo       │ │ do modelo    │ │ em linguagem    │
          │ Genético        │ │              │ │ natural         │
          └────────┬────────┘ └──────┬───────┘ └─────────────────┘
                   │                 │
                   ▼                 ▼
             Probabilidade      Principais fatores
                   │                 │
                   ▼                 │
             Threshold 0,34          │
                   │                 │
                   └────────┬────────┘
                            ▼
                  Resultado interpretável
```

A API atua como camada de integração entre o modelo de Machine Learning, o mecanismo de explicabilidade e a LLM.

---

# 3. Estrutura do Projeto

```text
FIAP_TechChallenge_02_2026/
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── src/
│   ├── __init__.py
│   ├── prediction_service.py
│   ├── explainability_service.py
│   ├── llm_service.py
│   ├── logging_config.py
│   └── outros módulos de suporte
│
├── notebooks/
│   ├── 01_baseline_fase1.ipynb
│   ├── 02_algoritmo_genetico.ipynb
│   ├── 03_threshold_e_explicabilidade.ipynb
│   └── 04_llm.ipynb
│
├── data/
│   └── data.csv
│
├── models/
│   └── modelo_ga_fase2.joblib
│
├── results/
│   ├── baseline_fase1.json
│   ├── resultado_final_ga.json
│   ├── threshold_otimizado.json
│   ├── importancia_variaveis_shap.csv
│   └── resultados das avaliações
│
├── tests/
│   ├── test_api.py
│   ├── test_prediction_service.py
│   ├── test_explainability_service.py
│   ├── test_predict_manual.py
│   ├── test_interpret_manual.py
│   └── load_test.py
│
├── docs/
│   ├── arquitetura.md
│   └── evidencias/
│
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── hpa.yaml
│   └── secret.example.yaml
│
├── logs/
│   └── api.log
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env
├── .env.example
└── README.md
```

---

# 4. Camada de Machine Learning

## 4.1 Modelo Base

O projeto utiliza o dataset **Breast Cancer Wisconsin Diagnostic**, contendo 30 variáveis numéricas relacionadas às características dos tumores.

A variável alvo é:

```text
B = Benigno
M = Maligno
```

Na Fase 1 foram avaliados diferentes modelos de classificação.

A Regressão Logística apresentou bom equilíbrio entre desempenho, interpretabilidade e estabilidade, sendo utilizada como base para a otimização realizada na Fase 2.

---

# 5. Otimização com Algoritmo Genético

O Algoritmo Genético foi implementado manualmente seguindo os princípios estudados no módulo.

Cada indivíduo representa uma configuração de hiperparâmetros do modelo.

## 5.1 Representação dos genes

Os genes representam:

- `C`;
- `class_weight`;
- combinação de `solver`;
- `penalty`.

Exemplo conceitual:

```text
Indivíduo
[5, 0, 0]

↓ decodificação

C = 2.0
class_weight = None
solver = liblinear
penalty = l1
```

## 5.2 Operadores genéticos

Foram implementados:

- geração inicial da população;
- seleção;
- crossover uniforme;
- mutação;
- elitismo;
- substituição de gerações.

## 5.3 Função fitness

A avaliação utiliza validação cruzada estratificada sobre o conjunto de treinamento.

A função fitness combina:

```text
50% Recall
30% F1-score
20% ROC-AUC
```

Representação:

```text
fitness =
0,50 × Recall
+
0,30 × F1
+
0,20 × ROC-AUC
```

Foi dada maior importância ao Recall devido à natureza do problema de diagnóstico, em que falsos negativos possuem relevância especial.

## 5.4 Experimentos

Foram realizados pelo menos três experimentos com diferentes configurações de:

- tamanho da população;
- número de gerações;
- taxa de mutação;
- quantidade de indivíduos preservados pelo elitismo.

O melhor indivíduo foi selecionado utilizando somente os dados de treinamento e validação cruzada.

O conjunto de teste permaneceu separado até a avaliação final.

---

# 6. Threshold de Decisão

Além da otimização dos hiperparâmetros, foi analisado o threshold de classificação.

O valor padrão de uma Regressão Logística é:

```text
0,50
```

No projeto foi utilizado um processo de avaliação no conjunto de treinamento com validação cruzada para selecionar um threshold com maior sensibilidade para identificação de casos malignos.

O threshold selecionado foi aproximadamente:

```text
0,34
```

O fluxo de decisão é:

```text
Probabilidade de malignidade
           │
           ▼
Probabilidade >= 0,34
           │
      ┌────┴────┐
      │         │
     Sim       Não
      │         │
      ▼         ▼
   Maligno    Benigno
```

O threshold é armazenado separadamente em:

```text
results/threshold_otimizado.json
```

permitindo modificá-lo sem necessidade de treinar novamente o modelo.

---

# 7. Explicabilidade com SHAP

Foi utilizado **SHAP (SHapley Additive exPlanations)** para aumentar a interpretabilidade das previsões.

O SHAP permite identificar quais variáveis contribuíram para aumentar ou reduzir a tendência da previsão em direção à classe maligna.

Para cada paciente, são selecionados os cinco fatores com maior valor absoluto de contribuição SHAP.

Exemplo:

```text
symmetry_worst
radius_se
concave points_mean
perimeter_worst
perimeter_se
```

Cada fator retorna:

```text
variável
valor original
valor SHAP
efeito sobre a previsão
```

Exemplo:

```json
{
  "variavel": "radius_se",
  "valor": 1.095,
  "valor_shap": 1.1346,
  "efeito": "aumenta a tendência para malignidade"
}
```

A explicabilidade ajuda a compreender o comportamento do modelo, mas não representa causalidade médica.

---

# 8. Integração com LLM

Uma LLM pré-treinada é utilizada para transformar os resultados técnicos em uma explicação estruturada em linguagem natural.

O modelo de linguagem não realiza a classificação.

A classificação continua sendo responsabilidade exclusiva do modelo de Machine Learning.

O fluxo é:

```text
Modelo ML
   │
   ├── Probabilidade
   ├── Threshold
   └── Classificação
           │
           ▼
         SHAP
           │
           ▼
  Principais fatores
           │
           ▼
     Prompt estruturado
           │
           ▼
          LLM
           │
           ▼
Explicação em linguagem natural
```

---

# 9. Prompt Engineering

O prompt utilizado estabelece regras explícitas para reduzir interpretações inadequadas.

A LLM recebe instruções para:

- não realizar um novo diagnóstico;
- não alterar a classificação do modelo;
- não inventar informações clínicas;
- utilizar apenas as informações fornecidas;
- apresentar o resultado como apoio à decisão;
- incluir limitações;
- recomendar revisão por profissional de saúde.

A resposta é estruturada em:

```text
1. Resumo do resultado
2. Principais fatores
3. Interpretação
4. Limitações e recomendação de revisão clínica
```

---

# 10. Avaliação das Interpretações da LLM

Foram utilizados diferentes tipos de casos para avaliar o comportamento das explicações:

- alta probabilidade de malignidade;
- probabilidade próxima ao threshold;
- baixa probabilidade de malignidade.

Os critérios de avaliação utilizados foram:

- fidelidade;
- clareza;
- segurança;
- completude.

A classe real é mantida apenas para avaliação experimental e não é fornecida ao prompt utilizado pela LLM.

---

# 11. API REST

A solução é disponibilizada através de **FastAPI**.

Os principais endpoints são:

## GET `/`

Verifica se a aplicação está online.

## GET `/health`

Health check utilizado pela aplicação e pelas probes do Kubernetes.

Exemplo:

```json
{
  "status": "ok",
  "modelo": "carregado"
}
```

## GET `/features`

Retorna as 30 variáveis esperadas pelo modelo.

## POST `/predict`

Recebe os dados de um paciente e retorna:

```json
{
  "probabilidade_malignidade": 0.9999,
  "threshold": 0.34,
  "classificacao": "Maligno"
}
```

## POST `/interpret`

Executa:

```text
Predição
+
SHAP
+
LLM
```

e retorna a classificação, os principais fatores e a interpretação em linguagem natural.

## GET `/metrics`

Expõe métricas no formato utilizado pelo Prometheus.

---

# 12. Logging

Foi implementado logging centralizado da API.

Os registros são armazenados em:

```text
logs/api.log
```

Cada requisição registra informações como:

```text
Método
Rota
Status HTTP
Tempo de processamento
```

Exemplo:

```text
Método=POST |
Rota=/predict |
Status=200 |
Tempo=0.0152s
```

Eventos relacionados às previsões e erros também são registrados.

O sistema evita registrar informações sensíveis, como a chave da API da OpenAI.

---

# 13. Monitoramento

O monitoramento utiliza métricas compatíveis com Prometheus.

Foram implementadas métricas como:

```text
api_requisicoes_total
```

para contabilizar requisições e:

```text
api_tempo_requisicao_segundos
```

para observar o tempo de processamento.

As métricas são disponibilizadas através de:

```text
GET /metrics
```

---

# 14. Docker

A aplicação utiliza Docker para garantir um ambiente reproduzível.

Os principais serviços configurados no `docker-compose.yml` são:

```text
api
jupyter
postgres
n8n
```

A API utiliza Python e FastAPI dentro de um container.

O Jupyter permite executar e reproduzir os notebooks utilizando o mesmo ambiente de dependências.

---

# 15. Kubernetes

Kubernetes foi utilizado para demonstrar a capacidade de escalabilidade automática exigida pelo projeto.

Foram criados:

```text
Deployment
Service
HorizontalPodAutoscaler
Secret
```

## Deployment

Responsável por executar os pods da API.

## Service

Utiliza:

```text
LoadBalancer
```

para disponibilizar a API.

No ambiente local com Docker Desktop, a aplicação é acessada através de:

```text
http://localhost:8000
```

## Secret

As variáveis sensíveis são armazenadas através de Kubernetes Secret.

Exemplos:

```text
OPENAI_API_KEY
LLM_MODEL
```

O arquivo contendo valores reais não deve ser versionado.

---

# 16. Escalabilidade Automática

A escalabilidade foi implementada utilizando:

```text
Horizontal Pod Autoscaler
```

O HPA foi configurado com:

```text
Mínimo de réplicas: 1
Máximo de réplicas: 5
Meta de utilização de CPU: 60%
```

Fluxo:

```text
Carga normal
    │
    ▼
1 Pod
    │
    ▼
Aumento de requisições
    │
    ▼
CPU aumenta
    │
    ▼
HPA detecta utilização superior à meta
    │
    ▼
Kubernetes cria novos Pods
```

---

# 17. Metrics Server

Foi utilizado o Kubernetes Metrics Server para disponibilizar métricas de CPU e memória ao HPA.

Durante a configuração local no Docker Desktop foi necessário permitir a coleta das métricas do kubelet devido ao certificado utilizado pelo ambiente local.

A configuração foi aplicada apenas para o ambiente de desenvolvimento.

Após a configuração, o comando:

```bash
kubectl top nodes
```

passou a retornar métricas corretamente.

Exemplo observado:

```text
CPU: 252m
CPU%: 1%
Memória: 2872Mi
Memória%: 18%
```

---

# 18. Teste de Escalabilidade

Foi realizado um teste de carga contra o endpoint:

```text
POST /predict
```

O teste utilizou:

```text
50 trabalhadores simultâneos
120 segundos de duração
```

Durante o teste foram realizadas aproximadamente:

```text
42.100 requisições
```

O objetivo do teste foi gerar carga suficiente para observar o comportamento do HPA.

Antes do teste:

```text
CPU: aproximadamente 2%
Réplicas: 1
```

Durante a carga foram observados valores superiores à meta configurada:

```text
390% / 60%
426% / 60%
```

O Kubernetes respondeu aumentando automaticamente o número de réplicas:

```text
1 Pod
↓
2 Pods
↓
4 Pods
```

Isso demonstra que o mecanismo de escalabilidade automática estava funcional.

O número de respostas bem-sucedidas observado durante esse teste extremo foi inferior ao total de requisições, pois o objetivo da execução foi provocar saturação suficiente para verificar o comportamento do autoscaling, não realizar benchmarking de capacidade máxima estável.

---

# 19. Evidências do Autoscaling

As evidências foram armazenadas em:

```text
docs/evidencias/
```

## Evolução do HPA

```markdown
![Escalabilidade automática do HPA](evidencias/01_hpa_escalado_replicas.png)
```

A captura demonstra a utilização de CPU superior à meta e o aumento do número de réplicas.

## Criação automática de Pods

```markdown
![Criação automática de Pods](evidencias/02_kubernetes_creacion_pods.png)
```

A captura demonstra os novos pods passando pelos estados:

```text
Pending
↓
ContainerCreating
↓
Running
```

## Resultado do teste de carga

```markdown
![Resultado do teste de carga](evidencias/03_resultado_test_carga.png)
```

A captura demonstra o volume de requisições utilizado para provocar o escalamento.

---

# 20. Testes Automatizados

A solução utiliza `pytest`.

Foram criados testes para:

- health check;
- listagem de variáveis;
- modelo de previsão;
- probabilidade entre 0 e 1;
- classificação válida;
- geração de fatores SHAP;
- estrutura dos fatores SHAP;
- endpoint `/predict`;
- endpoint `/interpret`;
- endpoint `/metrics`.

Nas chamadas de teste da integração com LLM é utilizado `mock`.

Isso evita:

- consumo desnecessário da API externa;
- custos adicionais;
- dependência de conexão externa durante os testes.

As chamadas reais à LLM foram validadas separadamente através dos testes manuais e do notebook de integração.

---

# 21. Segurança e Configuração

Dados sensíveis não são armazenados diretamente no código.

As configurações são fornecidas através de variáveis de ambiente.

Exemplos:

```text
OPENAI_API_KEY
LLM_MODEL
DATABASE_URL
```

O arquivo:

```text
.env
```

não deve ser versionado.

Um exemplo sem credenciais reais pode ser disponibilizado através de:

```text
.env.example
```

Para Kubernetes, os valores sensíveis são armazenados em `Secret`.

---

# 22. Decisões de Implementação

## Regressão Logística

Foi mantida como modelo principal por apresentar:

- bom desempenho;
- treinamento rápido;
- probabilidades de classificação;
- boa compatibilidade com SHAP;
- interpretabilidade superior a modelos mais complexos.

## Algoritmo Genético

Foi escolhido para realizar a busca de hiperparâmetros utilizando uma abordagem evolutiva, atendendo diretamente aos requisitos da Fase 2.

## Recall como componente principal do fitness

No contexto de classificação de malignidade, reduzir falsos negativos possui relevância especial.

Por isso o Recall recebeu maior peso na função fitness.

## Threshold separado do modelo

O threshold foi armazenado externamente para permitir ajustes sem necessidade de novo treinamento.

## SHAP

Foi utilizado para fornecer explicabilidade global e individual.

## LLM

Foi utilizada somente como camada de interpretação e comunicação.

A LLM não substitui o modelo de Machine Learning.

## FastAPI

Foi escolhida devido a:

- simplicidade;
- desempenho;
- documentação automática com Swagger;
- boa integração com Python.

## Docker

Utilizado para aumentar a reprodutibilidade da solução.

## Kubernetes + HPA

Utilizados para demonstrar escalabilidade automática baseada em demanda.

## Prometheus

Utilizado como padrão de exposição das métricas da API.

---

# 23. Fluxo Completo da Solução

```text
Dados do paciente
       │
       ▼
FastAPI
       │
       ▼
Validação dos 30 atributos
       │
       ▼
Modelo otimizado por Algoritmo Genético
       │
       ▼
Probabilidade de malignidade
       │
       ▼
Threshold otimizado
       │
       ▼
Classificação
       │
       ▼
SHAP
       │
       ▼
Principais fatores
       │
       ▼
Prompt Engineering
       │
       ▼
LLM
       │
       ▼
Interpretação em linguagem natural
       │
       ▼
Resposta JSON
```

Paralelamente:

```text
Requisições
     │
     ├── Logging
     │
     └── Métricas Prometheus
              │
              ▼
        Metrics Server
              │
              ▼
             HPA
              │
              ▼
       Escalabilidade
        1 até 5 Pods
```

---

# 24. Limitações

A solução possui algumas limitações importantes.

O modelo foi treinado com um dataset público e não representa uma validação clínica para utilização em ambiente hospitalar real.

A classificação produzida pelo modelo não deve ser interpretada como diagnóstico médico definitivo.

A LLM também não realiza diagnóstico e não possui acesso a histórico clínico, exames adicionais ou contexto médico completo.

A infraestrutura Kubernetes utilizada foi implementada em ambiente local através do Docker Desktop.

A configuração demonstra tecnicamente o funcionamento do autoscaling, mas um ambiente de produção exigiria configurações adicionais de segurança, alta disponibilidade, observabilidade e gestão de infraestrutura.

---

# 25. Evoluções Futuras

A arquitetura foi preparada para futuras evoluções.

Entre elas:

- implantação em ambiente de nuvem;
- persistência de histórico de previsões;
- autenticação da API;
- dashboards de monitoramento;
- integração completa com Prometheus e Grafana;
- validação clínica com datasets adicionais;
- calibração avançada das probabilidades;
- integração com dados clínicos textuais;
- integração com prontuários eletrônicos;
- utilização de informações textuais na LLM no Módulo 3.

---

# 26. Conclusão

A arquitetura implementada integra Machine Learning, Algoritmos Genéticos, explicabilidade, LLM e recursos de infraestrutura em uma solução única.

O modelo desenvolvido na Fase 1 foi evoluído através da otimização de hiperparâmetros utilizando Algoritmo Genético.

O threshold de decisão foi analisado separadamente, enquanto o SHAP permitiu interpretar as variáveis que mais influenciaram cada previsão.

A LLM transformou essas informações técnicas em explicações estruturadas em linguagem natural, mantendo o modelo de Machine Learning como responsável pela classificação.

A aplicação foi disponibilizada através de FastAPI e containerizada com Docker.

Logging e métricas foram implementados para permitir acompanhamento do comportamento da solução.

Por fim, Kubernetes, Metrics Server e Horizontal Pod Autoscaler foram utilizados para demonstrar escalabilidade automática baseada na utilização de CPU.

Durante o teste de carga, o ambiente escalou automaticamente de uma para quatro réplicas, demonstrando na prática a capacidade da arquitetura de responder a variações de demanda.
