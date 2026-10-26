@echo off
setlocal EnableExtensions EnableDelayedExpansion
title FIAP Tech Challenge - Fase 2
cd /d "%~dp0"

:: Docker Compose usa a porta 8000 e o Kubernetes a porta 8010,
:: por isso os dois ambientes podem funcionar ao mesmo tempo.
set "API_URL=http://localhost:8000"
set "JUPYTER_URL=http://localhost:8888"
set "K8S_URL=http://localhost:8010"

goto MENU


:: ============================================================
:: MENU PRINCIPAL
:: ============================================================

:MENU
cls

echo.
echo ============================================================
echo            FIAP TECH CHALLENGE - FASE 2
echo     Otimizacao de Modelos de Diagnostico (AG + LLM)
echo ============================================================
echo.

call :VERIFICAR_DOCKER

if "!DOCKER_INSTALADO!"=="0" (
    echo Status: DOCKER NAO INSTALADO
) else if "!DOCKER_ATIVO!"=="0" (
    echo Status: DOCKER DESATIVADO
) else (
    curl -s -f "%API_URL%/health" >nul 2>&1
    if not errorlevel 1 (
        echo Docker Compose: EM EXECUCAO   ^(%API_URL%^)
    ) else (
        echo Docker Compose: PARADO
    )

    curl -s -f "%K8S_URL%/health" >nul 2>&1
    if not errorlevel 1 (
        echo Kubernetes:     EM EXECUCAO   ^(%K8S_URL%^)
    ) else (
        echo Kubernetes:     PARADO
    )
)

echo.
echo ------------------------------------------------------------
echo  PROJETO
echo   [1] Iniciar projeto (API + Jupyter)
echo   [2] Encerrar projeto
echo   [3] Verificar status
echo   [4] Executar testes automatizados
echo.
echo  KUBERNETES
echo   [5] Implantar no Kubernetes (HPA)
echo   [6] Executar teste de carga (autoscaling)
echo   [7] Remover do Kubernetes
echo.
echo   [8] Sair
echo ------------------------------------------------------------
echo.

set "OPCAO="
set /p "OPCAO=Escolha uma opcao: "

if "%OPCAO%"=="1" goto INICIAR
if "%OPCAO%"=="2" goto ENCERRAR
if "%OPCAO%"=="3" goto STATUS
if "%OPCAO%"=="4" goto TESTES
if "%OPCAO%"=="5" goto KUBERNETES
if "%OPCAO%"=="6" goto CARGA
if "%OPCAO%"=="7" goto REMOVER_K8S
if "%OPCAO%"=="8" goto SAIR

echo.
echo Opcao invalida.
timeout /t 2 /nobreak >nul
goto MENU


:: ============================================================
:: [1] INICIAR PROJETO
:: ============================================================

:INICIAR
cls

echo.
echo ============================================================
echo                   INICIANDO PROJETO
echo ============================================================
echo.

call :EXIGIR_DOCKER
if errorlevel 1 goto MENU

call :CRIAR_ENV

echo [1/3] Docker funcionando corretamente.
echo.

echo [2/3] Construindo e iniciando os containers...
echo       (a primeira execucao pode levar alguns minutos)
echo.

docker compose up -d --build

if errorlevel 1 (
    echo.
    echo ============================================================
    echo ERRO AO INICIAR OS CONTAINERS
    echo ============================================================
    echo.
    echo Verifique as mensagens acima.
    echo Se a porta 8000 ou 8888 estiver ocupada por outro programa,
    echo encerre-o e tente novamente.
    echo.
    pause
    goto MENU
)

echo.
echo [3/3] Aguardando a API ficar pronta...
echo.

call :AGUARDAR_URL "%API_URL%/health" 60
if errorlevel 1 (
    echo.
    echo ERRO: A API demorou muito para iniciar.
    echo.
    echo Para verificar os detalhes:
    echo docker compose logs api
    echo.
    pause
    goto MENU
)

echo API iniciada com sucesso.
echo.
echo Abrindo o navegador...

start "" "%API_URL%/docs"
timeout /t 1 /nobreak >nul
start "" "%JUPYTER_URL%"

echo.
echo ============================================================
echo               PROJETO INICIADO COM SUCESSO
echo ============================================================
echo.
echo API (Swagger):  %API_URL%/docs
echo Jupyter:        %JUPYTER_URL%
echo.
echo Execute os notebooks na ordem: 01 - 02 - 03 - 04
echo.
echo Voce pode minimizar esta janela.
echo Quando terminar, volte aqui e escolha:
echo.
echo [2] Encerrar projeto
echo.
echo ============================================================
echo.

pause
goto MENU


:: ============================================================
:: [2] ENCERRAR PROJETO
:: ============================================================

:ENCERRAR
cls

echo.
echo ============================================================
echo                   ENCERRANDO PROJETO
echo ============================================================
echo.

call :EXIGIR_DOCKER
if errorlevel 1 goto MENU

echo Encerrando os containers do docker compose...
echo.

docker compose down

if errorlevel 1 (
    echo.
    echo ERRO: Nao foi possivel encerrar o projeto.
) else (
    echo.
    echo Projeto encerrado com sucesso.
)

echo.
echo Observacao: a implantacao no Kubernetes (se existir) continua
echo ativa. Para remove-la, use a opcao [7].
echo.

pause
goto MENU


:: ============================================================
:: [3] STATUS
:: ============================================================

:STATUS
cls

echo.
echo ============================================================
echo                    STATUS DO PROJETO
echo ============================================================
echo.

call :VERIFICAR_DOCKER

if "!DOCKER_INSTALADO!"=="0" (
    echo Docker: NAO INSTALADO
    echo.
    pause
    goto MENU
)

echo Docker instalado: SIM

if "!DOCKER_ATIVO!"=="0" (
    echo Docker Desktop: DESATIVADO
    echo.
    pause
    goto MENU
)

echo Docker Desktop: ATIVO
echo.
echo --- Docker Compose ---
docker compose ps
echo.

curl -s -f "%API_URL%/health" >nul 2>&1
if errorlevel 1 (
    echo API ^(docker compose^): NAO RESPONDE
) else (
    echo API ^(docker compose^): ONLINE em %API_URL%/docs
)

curl -s -f "%K8S_URL%/health" >nul 2>&1
if errorlevel 1 (
    echo API ^(Kubernetes^):     NAO RESPONDE
) else (
    echo API ^(Kubernetes^):     ONLINE em %K8S_URL%/docs
)

where kubectl >nul 2>&1
if not errorlevel 1 (
    echo.
    echo --- Kubernetes ---
    kubectl get pods 2>nul
    echo.
    kubectl get hpa 2>nul
)

echo.
pause
goto MENU


:: ============================================================
:: [4] TESTES AUTOMATIZADOS
:: ============================================================

:TESTES
cls

echo.
echo ============================================================
echo                 TESTES AUTOMATIZADOS
echo ============================================================
echo.

call :EXIGIR_DOCKER
if errorlevel 1 goto MENU

curl -s -f "%API_URL%/health" >nul 2>&1
if errorlevel 1 (
    echo O projeto esta parado. Inicie-o primeiro com a opcao [1].
    echo.
    pause
    goto MENU
)

docker compose exec api python -m pytest tests/ -v

echo.
pause
goto MENU


:: ============================================================
:: [5] IMPLANTAR NO KUBERNETES
:: ============================================================

:KUBERNETES
cls

echo.
echo ============================================================
echo              IMPLANTACAO NO KUBERNETES (HPA)
echo ============================================================
echo.

call :EXIGIR_KUBERNETES
if errorlevel 1 goto MENU

call :CRIAR_ENV

echo [1/5] Construindo a imagem fiap-fase2-api:latest...
echo.
docker build -t fiap-fase2-api:latest .
if errorlevel 1 (
    echo.
    echo ERRO ao construir a imagem.
    echo.
    pause
    goto MENU
)
echo.

echo [2/5] Criando/atualizando o Secret a partir do .env...
kubectl create secret generic fiap-fase2-secrets --from-env-file=.env --dry-run=client -o yaml | kubectl apply -f -
echo.

echo [3/5] Aplicando Deployment, Service e HPA...
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/hpa.yaml
kubectl rollout restart deployment fiap-fase2-api >nul 2>&1
echo.

echo [4/5] Aguardando os pods ficarem prontos...
kubectl rollout status deployment fiap-fase2-api --timeout=180s
echo.

echo [5/5] Aguardando a API do Kubernetes responder...
call :AGUARDAR_URL "%K8S_URL%/health" 30
if errorlevel 1 (
    echo.
    echo AVISO: a API ainda nao respondeu em %K8S_URL%.
    echo Verifique com: kubectl get pods
    echo.
) else (
    echo API do Kubernetes online.
    echo.
)

kubectl get pods
echo.
kubectl get hpa
echo.

echo ============================================================
echo                 IMPLANTACAO CONCLUIDA
echo ============================================================
echo.
echo API no Kubernetes:  %K8S_URL%/docs
echo.
echo Para demonstrar o autoscaling, use a opcao [6].
echo.
echo Observacao: se o HPA mostrar ^<unknown^> na coluna TARGETS,
echo o Metrics Server precisa estar instalado (ver README).
echo.
echo ============================================================
echo.

pause
goto MENU


:: ============================================================
:: [6] TESTE DE CARGA
:: ============================================================

:CARGA
cls

echo.
echo ============================================================
echo           TESTE DE CARGA - AUTOSCALING (HPA)
echo ============================================================
echo.

call :EXIGIR_KUBERNETES
if errorlevel 1 goto MENU

curl -s -f "%K8S_URL%/health" >nul 2>&1
if errorlevel 1 (
    echo A API do Kubernetes nao esta respondendo em %K8S_URL%.
    echo Implante primeiro com a opcao [5].
    echo.
    pause
    goto MENU
)

where python >nul 2>&1
if errorlevel 1 (
    echo ERRO: Python nao encontrado neste computador.
    echo O teste de carga roda localmente e precisa de Python
    echo com as bibliotecas pandas e requests.
    echo.
    pause
    goto MENU
)

echo Abrindo duas janelas de acompanhamento:
echo  - HPA  (CPU e numero de replicas)
echo  - Pods (criacao de novos pods)
echo.

start "HPA - kubectl get hpa -w" cmd /k kubectl get hpa -w
start "Pods - kubectl get pods -w" cmd /k kubectl get pods -w

timeout /t 3 /nobreak >nul

echo Iniciando o teste de carga (120 segundos, 50 workers)...
echo.

python scripts/load_test.py %K8S_URL%

echo.
echo ------------------------------------------------------------
echo Eventos do HPA:
echo ------------------------------------------------------------
kubectl describe hpa fiap-fase2-api-hpa
echo.
echo O HPA reduz as replicas automaticamente apos cerca de
echo 5 minutos sem carga. Acompanhe na janela do HPA.
echo.

pause
goto MENU


:: ============================================================
:: [7] REMOVER DO KUBERNETES
:: ============================================================

:REMOVER_K8S
cls

echo.
echo ============================================================
echo               REMOVER DO KUBERNETES
echo ============================================================
echo.

call :EXIGIR_KUBERNETES
if errorlevel 1 goto MENU

kubectl delete -f k8s/hpa.yaml -f k8s/service.yaml -f k8s/deployment.yaml --ignore-not-found
kubectl delete secret fiap-fase2-secrets --ignore-not-found

echo.
echo Recursos do Kubernetes removidos.
echo.

pause
goto MENU


:: ============================================================
:: FUNCOES AUXILIARES
:: ============================================================

:VERIFICAR_DOCKER

set "DOCKER_INSTALADO=0"
set "DOCKER_ATIVO=0"

where docker >nul 2>&1
if errorlevel 1 exit /b

set "DOCKER_INSTALADO=1"

docker info >nul 2>&1
if errorlevel 1 exit /b

set "DOCKER_ATIVO=1"
exit /b


:EXIGIR_DOCKER

call :VERIFICAR_DOCKER

if "!DOCKER_INSTALADO!"=="0" (
    echo ERRO: Docker nao foi encontrado neste computador.
    echo.
    echo Instale o Docker Desktop e tente novamente.
    echo.
    pause
    exit /b 1
)

if "!DOCKER_ATIVO!"=="0" (
    echo ERRO: Docker Desktop nao esta em execucao.
    echo.
    echo Abra o Docker Desktop, aguarde iniciar e tente novamente.
    echo.
    pause
    exit /b 1
)

exit /b 0


:EXIGIR_KUBERNETES

call :EXIGIR_DOCKER
if errorlevel 1 exit /b 1

where kubectl >nul 2>&1
if errorlevel 1 (
    echo ERRO: kubectl nao encontrado.
    echo.
    echo Ative o Kubernetes no Docker Desktop:
    echo Settings - Kubernetes - Enable Kubernetes
    echo.
    pause
    exit /b 1
)

kubectl cluster-info >nul 2>&1
if errorlevel 1 (
    echo ERRO: o cluster Kubernetes nao esta respondendo.
    echo.
    echo Ative o Kubernetes no Docker Desktop e aguarde ele iniciar.
    echo.
    pause
    exit /b 1
)

exit /b 0


:CRIAR_ENV

if exist ".env" exit /b 0

if not exist ".env.example" exit /b 0

copy ".env.example" ".env" >nul
echo AVISO: o arquivo .env foi criado a partir do .env.example.
echo.
echo Para usar a LLM, preencha OPENAI_API_KEY e LLM_MODEL no .env.
echo Sem a chave, a API funciona normalmente, mas o /interpret
echo responde sem a explicacao da LLM.
echo.
pause
exit /b 0


:AGUARDAR_URL
:: %1 = URL, %2 = numero de tentativas (2 segundos cada)

for /L %%I in (1,1,%~2) do (
    curl -s -f "%~1" >nul 2>&1
    if not errorlevel 1 exit /b 0
    timeout /t 2 /nobreak >nul
)

exit /b 1


:: ============================================================
:: SAIR
:: ============================================================

:SAIR
endlocal
exit /b