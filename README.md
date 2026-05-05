# Churn Predictor

Pipeline de Machine Learning para predição de churn em clientes de telecom. Usa uma rede neural MLP em PyTorch como modelo principal, comparada com baselines em scikit-learn, com tracking via MLflow e API de inferência em FastAPI.

Dataset: [Telco Customer Churn (IBM)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7.043 clientes, 19 features, taxa de churn ~26%.

## Setup

Requer Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Baixe o dataset (manual ou via script):

```bash
python scripts/download_data.py
```

O CSV deve ficar em `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## Uso

```bash
make train      # treina baselines + MLP, registra no MLflow, salva artefatos em models/
make serve      # sobe API em http://localhost:8000 (docs em /docs)
make mlflow-ui  # interface MLflow em http://localhost:5000
make test       # roda 41 testes
make lint       # ruff check
```

## Estrutura

```
src/churn_predictor/
├── api/          FastAPI: schemas, middleware, model loading
├── data/         loader, schema Pandera, splits
├── evaluation/   métricas e análise de custo de negócio
├── features/     pipeline sklearn + transformadores customizados
├── models/       MLP PyTorch + baselines sklearn
├── training/     trainer com early stopping + script principal
└── utils/        config, logging, seeds

tests/            41 testes (smoke, schema, API, unit)
notebooks/        EDA, baselines, MLP
docs/             Model Card
scripts/          download de dados
```

## Decisões técnicas principais

**Pré-processamento.** Pipeline sklearn com dois transformadores customizados: `TenureBucketizer` (segmenta tempo de relacionamento em faixas) e `ChargesRatioFeature` (deriva razão TotalCharges/tenure). Validação de schema com Pandera antes de qualquer processamento.

**MLP.** Arquitetura (128, 64, 32) com BatchNorm + Dropout(0.3), inicialização Kaiming. Loss: `BCEWithLogitsLoss` com `pos_weight` calculado dinamicamente para lidar com desbalanceamento. Otimizador Adam com `weight_decay=1e-5`, `ReduceLROnPlateau` no scheduler, early stopping monitorando PR-AUC.

**Métrica primária: PR-AUC.** Em dados desbalanceados, ROC-AUC pode ser otimista. PR-AUC é mais honesta sobre a habilidade de identificar a classe minoritária.

**Threshold operacional.** Em vez de fixar em 0.5, fazemos busca pelo threshold que minimiza custo total de negócio, considerando FN=R$500 (LTV anual médio perdido) e FP=R$50 (custo da campanha de retenção).

**Reprodutibilidade.** Seeds fixadas em Python, NumPy, PyTorch e CUDA. Hash do dataset registrado em cada run do MLflow.

## API

Endpoints:

- `GET /health` — status do serviço e do modelo
- `POST /predict` — predição individual
- `POST /predict/batch` — até 1000 clientes por request
- `GET /docs` — Swagger UI

Exemplo:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.10,
    "TotalCharges": 1069.20
  }'
```

Resposta:

```json
{
  "churn_probability": 0.7823,
  "churn_prediction": 1,
  "risk_level": "high",
  "threshold_used": 0.42,
  "model_version": "0.1.0"
}
```

A API usa Pydantic para validação estrita (rejeita payloads inválidos com 422), middleware de logging estruturado em JSON com `X-Request-ID` por requisição e header `X-Process-Time-Ms` na resposta. O modelo é carregado uma única vez no startup via lifespan handler.

## Documentação adicional

- [Model Card](docs/model_card.md) — limitações, vieses conhecidos, cenários de falha.

## Stack

PyTorch, scikit-learn, MLflow, FastAPI, Pydantic v2, Pandera, structlog, ruff, pytest.

## Licença

MIT.
