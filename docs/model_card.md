# Model Card — Churn Predictor

## Modelo

MLP em PyTorch para classificação binária de churn em clientes de telecom. Versão 0.1.0.

Arquitetura: 3 camadas escondidas (128, 64, 32) com BatchNorm + ReLU + Dropout(0.3). Inicialização Kaiming. Loss `BCEWithLogitsLoss` com `pos_weight` para compensar desbalanceamento (~26% de positivos no dataset). Otimizador Adam (lr=1e-3, weight_decay=1e-5) com `ReduceLROnPlateau`. Early stopping com paciência de 10 épocas baseado em PR-AUC de validação.

Aproximadamente 13K parâmetros. Treina em ~2 minutos em CPU para 7K registros.

## Uso pretendido

O modelo serve para **priorizar campanhas de retenção** — gerar uma lista ranqueada de clientes com maior probabilidade de cancelamento nos próximos 30 dias, para que a equipe de retenção atue proativamente.

Não é adequado para:

- Negar serviço, ofertas de crédito ou outras decisões adversas ao cliente.
- Precificação dinâmica individualizada sem auditoria adicional de fairness.
- Substituir julgamento humano em decisões finais — é ferramenta de priorização.
- Domínios fora de telecom (treinado em padrões específicos do setor).

## Dados de treino

**Telco Customer Churn (IBM)**, dataset público do Kaggle. 7.043 registros, 19 features, taxa de churn de 26.5%. Em produção, seria substituído pelos dados internos do warehouse da operadora.

Pré-processamento:

- `TotalCharges` faltante (clientes com `tenure=0`) imputado com 0.
- `customerID` descartado (PII e sem valor preditivo).
- One-hot encoding em categóricas com `handle_unknown='ignore'` para resiliência a categorias novas em produção.
- StandardScaler em numéricas.
- Features derivadas: `tenure_bucket` (0-6m, 6-12m, 1-2y, 2-4y, 4y+) e `charges_per_month_ratio`.

Splits estratificados: 65% treino, 15% validação, 20% teste.

## Performance

> Atualizar com os números do seu próprio run após `make train`.

Comparação no test set (threshold padrão 0.5):

| Modelo | PR-AUC | ROC-AUC | F1 | Recall | Precision |
|---|---|---|---|---|---|
| Dummy (stratified) | ~0.27 | ~0.50 | ~0.27 | ~0.27 | ~0.27 |
| Logistic Regression | ~0.65 | ~0.84 | ~0.62 | ~0.78 | ~0.51 |
| Random Forest | ~0.62 | ~0.83 | ~0.59 | ~0.55 | ~0.65 |
| Gradient Boosting | ~0.66 | ~0.85 | ~0.63 | ~0.58 | ~0.69 |
| **MLP (PyTorch)** | **~0.66** | **~0.85** | **~0.62** | **~0.79** | **~0.51** |

PR-AUC é a métrica primária por ser mais honesta em dados desbalanceados que ROC-AUC. Validação cruzada estratificada com k=5 mostrou desvio-padrão entre folds inferior a 0.02 — modelo estável.

## Threshold operacional

Em vez de fixar threshold em 0.5, fazemos busca pelo threshold que minimiza custo de negócio em validação:

- **FN (cliente que cancelaria e não foi detectado):** R$ 500 — perda média de LTV anual.
- **FP (cliente que ficaria mesmo, mas recebeu retenção):** R$ 50 — custo de campanha.

O threshold ótimo costuma ficar abaixo de 0.5 (tipicamente 0.4–0.45), refletindo que falsos negativos doem mais.

## Limitações

**Dataset estático.** Não há features temporais — modelo não captura tendências (sazonalidade, recessão, lançamentos de concorrentes). Em produção, agregar features de série temporal melhoraria muito.

**Tamanho modesto.** 7K registros é pequeno para uma rede neural. Padrões raros podem não ser bem capturados.

**Probabilidades não calibradas.** A saída sigmoid não é calibrada. Se o uso exigir probabilidades calibradas (ex.: cálculo de expected value), aplicar `CalibratedClassifierCV` ou Platt scaling.

**Cold-start em clientes novos.** Clientes com `tenure < 3` meses têm sinal limitado — modelo pode super-prever churn neles.

**Baixa interpretabilidade nativa.** Diferente de árvores, MLP é caixa-preta. Para explicações locais, usar SHAP em produção.

**Geografia ausente.** Sem informação geográfica no dataset — modelo pode performar diferente em regiões com dinâmica de mercado distinta.

## Vieses conhecidos

**Desbalanceamento de classes.** Mitigado com `pos_weight` no BCE e `class_weight='balanced'` nos baselines. Apesar disso, vale revisar recall por subgrupo periodicamente.

**Viés de seleção do dataset público.** O Telco da IBM não reflete operadoras brasileiras (sem pré-pago, sem PIX como método de pagamento, etc.). Em produção, retreinar com dados internos.

**Definição de churn.** O target no dataset é "cancelou no último mês" sem distinguir voluntário de involuntário (mudança, óbito). Em produção, refinar a definição.

**Demográfico.** Distribuição de gênero é equilibrada (~50/50), mas SeniorCitizen representa apenas ~16% da base. Análise de fairness deve garantir que o modelo não desfavoreça idosos.

## Cenários de falha

| Cenário | Sintoma | Mitigação |
|---|---|---|
| Drift de features (ex.: reajuste tarifário muda MonthlyCharges) | KS-test detecta divergência | Retreinar com dados recentes |
| Categoria nova em produção (ex.: novo plano) | Sinal degradado nessa categoria | `OneHotEncoder(handle_unknown='ignore')` mantém o pipeline funcional |
| Pipeline ETL falha (features com nulls) | API retorna 422 (validação Pydantic) | Alertar pipeline antes de chegar à API |
| Modelo corrompido (predições constantes ou NaN) | Health check detecta | Rollback para versão anterior via Model Registry |
| Mudança no padrão de churn (concept drift) | PR-AUC cai em monitoramento retroativo | Trigger de retreino |

## Recomendações

- Use o modelo como **uma das fontes de evidência**, não como decisão automatizada final.
- Monitore drift de features semanalmente (KS-test em numéricas, PSI em categóricas).
- Retreine mensalmente, ou quando PR-AUC cair mais de 5%.
- Faça análise de fairness em cada retreino — disparidade de recall entre subgrupos deve ficar abaixo de 10%.
- Não esconda do cliente o uso de modelos preditivos quando questionado (transparência LGPD).

## Privacidade

`customerID` é descartado antes do treino e nunca chega à API. Predições são logadas com `request_id` e hash do input para auditoria, mas sem PII. Direito ao esquecimento (LGPD) deve ser respeitado nos retreinos — exclusões solicitadas pelos titulares precisam refletir nos dados de treino seguintes.
