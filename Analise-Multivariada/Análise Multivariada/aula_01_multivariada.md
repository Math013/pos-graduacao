# 📊 Análise Multivariada — Anotações de Aula
## Aula 1: Introdução, Conceitos Prévios e Aprendizado Estatístico vs. Machine Learning

> **Professor:** Vinícius Osterne, PhD  
> **Disciplina:** Estatística Multivariada  
> **Curso:** Pós-graduação em Ciência de Dados / Estatística

---

## 🗺️ Linha do Tempo da Disciplina

O curso segue uma progressão lógica, construindo cada bloco sobre o anterior:

```
Introdução & Conceitos → Álgebra Linear → Análise Exploratória
→ Probabilidade & Inferência → Modelos Multivariados
→ Estrutura dos Dados → Classificação & Agrupamento
```

| Encontro | Conteúdo |
|----------|----------|
| 1 | Introdução · Conceitos Prévios · Álgebra Linear · Análise Exploratória |
| 2 | Probabilidade · Inferência (T² de Hotelling, MANOVA) |
| 3 | Modelos Multivariados (regressão, correlação canônica) |
| 4 | Estrutura dos Dados (PCA, Análise Fatorial) |
| 5 | Técnicas de Classificação e Agrupamento (Clustering) |
| 6 | Apresentação dos Trabalhos (20 min/aluno) |

---

## 🎯 O que é Estatística Multivariada?

> **Analogia:** Imagine que você está tentando entender uma pessoa apenas olhando para um único traço — sua altura. Você teria uma visão muito limitada. A estatística multivariada é como tirar uma foto panorâmica: ela captura **várias dimensões ao mesmo tempo**, revelando padrões que uma análise univariada jamais enxergaria.

**Objetivos principais:**
- Estudar a **correlação e covariância** entre múltiplas variáveis simultaneamente
- **Reduzir dimensionalidade** sem perder informação relevante (ex.: PCA)
- **Classificar e agrupar** observações com base em múltiplas variáveis
- **Modelagem preditiva** com múltiplos preditores

**Aplicações práticas:**
- Bancos avaliando candidatos a crédito (renda + escolaridade + idade + histórico)
- Segmentação de clientes por comportamento de compra (Clustering)
- Análise de acidentes de trânsito por tipo/categoria (Análise de Correspondência)
- Modelos de score de fraude

---

## 🧱 Pré-requisitos da Disciplina

Para aproveitar bem o curso, é importante ter base em:

- **Análise Exploratória:** média, mediana, desvio padrão, gráficos
- **Probabilidade e Distribuições**
- **Álgebra Linear:** vetores, matrizes, autovalores
- **Inferência Estatística:** testes de hipóteses, intervalos de confiança
- **ANOVA:** comparação de médias entre grupos
- **Cálculo:** derivadas e integrais (usados na regressão múltipla)

---

## 📦 Tipos de Variáveis

> **Analogia:** Pense nos dados como ingredientes de uma receita. Alguns você pode medir em gramas (quantitativos), outros são apenas categorias — como "tipo de cozinha" (qualitativos). Confundir os dois é como tentar somar maçãs com laranjas.

```
Variáveis
├── Quantitativas (numéricas com sentido aritmético)
│   ├── Contínuas: altura, temperatura, salário
│   └── Discretas: número de filhos, contagem de defeitos
└── Qualitativas (categóricas)
    ├── Nominais: gênero, cor dos olhos, estado civil (sem ordem)
    └── Ordinais: grau de satisfação, nível de escolaridade (com ordem)
```

```r
# Criando um dataframe com diferentes tipos de variáveis em R
dados <- data.frame(
  nome        = c("Ana", "Bruno", "Carlos"),   # qualitativa nominal
  escolaridade = factor(c("Médio", "Superior", "Médio"),  # qualitativa ordinal
                        levels = c("Fundamental", "Médio", "Superior"),
                        ordered = TRUE),
  filhos      = c(2L, 0L, 1L),                 # quantitativa discreta
  salario     = c(3500.50, 8200.00, 5100.75)   # quantitativa contínua
)

# Verificando as classes de cada coluna
sapply(dados, class)

# Resumo estatístico
summary(dados)
```

---

## ⚖️ Por que a Escala Importa? Padronização

> **Analogia:** Imagine uma corrida em que um atleta mede a velocidade em km/h e outro em m/s. Se você comparar os números brutos, o de km/h vai parecer muito mais rápido — mas é só uma questão de unidade. Na estatística multivariada, variáveis em escalas diferentes distorcem qualquer análise baseada em distâncias.

**O problema:** Se X₁ = salário (ordem de 1.000) e X₂ = idade (ordem de 10), o salário "domina" o modelo só pela magnitude.

### Padronização Z-score (mais usada)

$$z_{ij} = \frac{x_{ij} - \bar{x}_j}{s_j}$$

Resultado: média 0 e desvio padrão 1. Torna a matriz de covariância equivalente à matriz de correlação.

### Normalização Min-Max

$$z_{ij} = \frac{x_{ij} - \min_j}{\max_j - \min_j}$$

Resultado: valores entre [0, 1].

```r
# Padronização Z-score em R
dados_num <- data.frame(
  salario = c(3500, 8200, 5100, 12000, 4300),
  idade   = c(25, 42, 33, 55, 29)
)

# scale() já aplica Z-score por padrão (subtrai média e divide por desvio padrão)
dados_padronizados <- scale(dados_num)

# Verificando que a média é ~0 e o desvio padrão é ~1
colMeans(dados_padronizados)   # deve ser próximo de 0
apply(dados_padronizados, 2, sd) # deve ser próximo de 1

# Normalização Min-Max (manual)
min_max <- function(x) (x - min(x)) / (max(x) - min(x))
dados_minmax <- as.data.frame(lapply(dados_num, min_max))
print(dados_minmax)
```

---

## 🤖 Aprendizado Estatístico vs. Aprendizado de Máquina

> **Analogia:** Pense em dois tipos de médicos. O **estatístico** quer entender *por que* o paciente está doente — ele investiga causas, faz exames, constrói um diagnóstico interpretável. O **engenheiro de ML** quer apenas *prever* se o paciente vai melhorar — ele não se importa com a causa, contanto que o modelo acerte. Ambos são úteis, mas para perguntas diferentes.

| Característica | Aprendizado Estatístico | Aprendizado de Máquina |
|---|---|---|
| **Foco** | Inferência (entender relações) | Predição (maximizar acurácia) |
| **Modelos** | Interpretáveis, com suposições sobre distribuição | Flexíveis, caixas-pretas |
| **Interesse** | Incerteza (IC, testes de hipótese) | Generalização (desempenho fora da amostra) |
| **Pergunta central** | **Por quê?** | **O quê?** |
| **Exemplo** | Regressão logística com odds ratios | Random Forest, XGBoost |
| **Tipo** | Paramétrico | Não paramétrico |

> ⚠️ **Nota do professor:** Na prática, as fronteiras são tênues. Os métodos que estudaremos transitam entre as duas abordagens.

### Paramétrico vs. Não Paramétrico

**Paramétrico (Aprendizado Estatístico):** preciso assumir uma distribuição para os dados.
- Regressão logística → resposta deve ser binomial
- Regressão Gamma → resposta deve ser positiva/assimétrica
- Regressão Normal → erros devem seguir N(0, σ²)

**Não Paramétrico (Machine Learning):** sem suposições sobre distribuição.
- Árvores de decisão, Random Forest → não assumem nada sobre a distribuição

---

## ⚠️ Armadilha Clássica: Acurácia em Dados Desbalanceados

> **Analogia do professor (exemplo de fraude):** Imagine que 99% das transações são legítimas e apenas 1% é fraude. Um modelo "burro" que classifica **tudo como legítimo** acerta 99% das vezes — mas é completamente inútil para detectar fraude!

### O problema da acurácia em classes desbalanceadas

```r
# Simulando o problema do modelo "burro"

# Base: 10.000 transações, 100 fraudes (1%)
set.seed(42)
n_total  <- 10000
n_fraude <- 100

y_real <- c(rep(1, n_fraude), rep(0, n_total - n_fraude))

# Modelo "burro": prevê tudo como não-fraude (0)
y_pred_burro <- rep(0, n_total)

# Calculando acurácia
acuracia <- mean(y_real == y_pred_burro)
cat("Acurácia do modelo burro:", round(acuracia * 100, 2), "%\n")
# Resultado: 99% — parece ótimo, mas é uma mentira!

# O modelo não captura NENHUMA fraude
fraudes_detectadas <- sum(y_pred_burro[y_real == 1] == 1)
cat("Fraudes detectadas:", fraudes_detectadas, "de", n_fraude, "\n")
# Resultado: 0 de 100
```

### Métricas mais adequadas para dados desbalanceados

Para problemas de fraude, o professor recomenda olhar para:

- **KS (Kolmogorov-Smirnov):** mede a separação entre a distribuição de scores dos fraudadores e não-fraudadores
- **AUC-ROC:** área sob a curva ROC
- **Avaliação de overfit:** comparar KS_treino vs. KS_teste (diferença grande = overfit)

```r
# Instalando pacotes necessários
# install.packages(c("ROCR", "pROC"))

library(pROC)

# Simulando scores de um modelo real
set.seed(42)
scores_fraude    <- rbeta(100, 2, 5)   # fraudadores tendem a ter scores mais baixos
scores_legitimo  <- rbeta(9900, 5, 2)  # legítimos tendem a ter scores mais altos

scores <- c(scores_fraude, scores_legitimo)
labels <- c(rep(1, 100), rep(0, 9900))

# Calculando AUC
roc_obj <- roc(labels, scores)
cat("AUC:", round(auc(roc_obj), 4), "\n")

# KS statistic (diferença máxima entre CDFs)
# O KS mede o quanto as distribuições se separam
ks_stat <- ks.test(scores_fraude, scores_legitimo)
cat("KS statistic:", round(ks_stat$statistic, 4), "\n")
cat("p-value:", ks_stat$p.value, "\n")

# Visualizando as distribuições
hist(scores_fraude, col = rgb(1,0,0,0.5), main = "Distribuição de Scores",
     xlab = "Score", xlim = c(0,1), breaks = 20)
hist(scores_legitimo, col = rgb(0,0,1,0.5), add = TRUE, breaks = 20)
legend("topright", c("Fraude", "Legítimo"),
       fill = c(rgb(1,0,0,0.5), rgb(0,0,1,0.5)))
```

### Distribuição ideal de scores (formato de sino)

O professor mencionou que a distribuição ideal dos scores (em escala de 0 a 1000) deve ter **formato de sino (normal)**: poucas rejeições, poucas aprovações automáticas, e a maior concentração no meio — onde o modelo precisa trabalhar para discriminar.

```r
# Visualizando a distribuição ideal de scores (0-1000)
scores_ideais <- rnorm(10000, mean = 500, sd = 100)
scores_ideais <- pmin(pmax(scores_ideais, 0), 1000) # limitando entre 0 e 1000

hist(scores_ideais,
     main = "Distribuição Ideal de Scores (formato de sino)",
     xlab = "Score (0-1000)",
     col  = "steelblue",
     breaks = 30)
abline(v = 300, col = "red",    lty = 2, lwd = 2)  # threshold de rejeição
abline(v = 700, col = "green",  lty = 2, lwd = 2)  # threshold de aprovação
legend("topright",
       c("Zona de rejeição", "Zona de aprovação"),
       col = c("red", "green"), lty = 2, lwd = 2)
```

---

## 💡 Insights do Professor (da transcrição)

> *"Quanto mais conhecimento você tem, mais senso crítico você tem."*

- Não foque apenas na métrica — foque no **negócio**
- Um modelo de fraude deve ser avaliado além da acurácia: KS, AUC, distribuição de scores, interpretabilidade dos atributos
- Modelos interpretáveis (aprendizado estatístico) permitem comunicar: "esse atributo tem 15% de participação no modelo"
- Overfit = diferença grande entre KS_treino e KS_teste → ideal que KS_treino ≥ KS_teste

---

## 📚 Referências e Próximos Passos

**Próxima aula (Encontro 1 — continuação):**
- Álgebra Linear: vetores, matrizes, autovalores
- Análise Exploratória: descritivas, distâncias, visualização

**Leituras recomendadas:**
- Johnson, R. A. & Wichern, D. W. — *Applied Multivariate Statistical Analysis*
- James, G. et al. — *An Introduction to Statistical Learning* (ISLR) — disponível gratuitamente em [https://www.statlearning.com](https://www.statlearning.com)
- Hair, J. F. et al. — *Multivariate Data Analysis*
