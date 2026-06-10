# 📊 Análise Multivariada — Anotações de Aula
## Aula 4: Modelos Multivariados — Regressão, Regressão Canônica e Correlação Canônica

> **Professor:** Vinícius Osterne, PhD  
> **Observação do professor:** *"Aula super densa — densa para vocês, densa para mim também. A ideia é que vocês adquiram pelo menos 30 a 40% do conhecimento: a formulação do problema, como resolver. A caixa preta por trás, a nomenclatura — isso vem com o tempo."*

---

## 🗺️ Onde Estamos na Linha do Tempo

```
✅ Introdução & Conceitos Prévios
✅ Álgebra Linear
✅ Análise Exploratória
✅ Probabilidade & Inferência (T², MANOVA, MANCOVA)
✅ Modelos Multivariados   ← HOJE
   ├── i.  Regressão Multivariada
   ├── ii. Regressão Canônica
   └── iii. Correlação Canônica
⬜ Estrutura dos Dados (PCA, Análise Fatorial)
⬜ Classificação & Agrupamento
```

---

## 🔗 A Família dos Modelos com Múltiplos Y

> **Analogia:** Pense nas técnicas de hoje como três câmeras fotografando a mesma cena, mas com lentes diferentes. A **Regressão Multivariada** é uma câmera com zoom em cada detalhe separadamente. A **Regressão Canônica** usa uma lente panorâmica para ver o melhor ângulo de predição. A **Correlação Canônica** é como dois fotógrafos olhando um para o outro — sem hierarquia, só associação.

**A progressão lógica das técnicas de regressão:**

```
Regressão Simples:    1 preditor X  →  1 desfecho Y
Regressão Múltipla:   p preditores X →  1 desfecho Y
Regressão Multivariada: p preditores X →  q desfechos Y  ← HOJE (i)
Regressão Canônica:   combinações de X  →  combinações de Y ← HOJE (ii)
Correlação Canônica:  associação simétrica entre X e Y     ← HOJE (iii)
MANOVA:               caso especial com X categórico (grupos)
```

---

# PARTE i — REGRESSÃO MULTIVARIADA

## O Modelo

> **Analogia:** Na regressão múltipla você tem um painel de instrumentos com vários botões (X) controlando uma única saída (Y) — como a velocidade de um carro. Na regressão multivariada, você controla **vários resultados ao mesmo tempo** — velocidade, temperatura do motor e consumo — todos com os mesmos botões.

**Modelo formal:**

$$\mathbf{Y}_{n \times q} = \mathbf{X}_{n \times p} \mathbf{B}_{p \times q} + \mathbf{E}_{n \times q}$$

Onde:
- **Y** (n × q): matriz de **q desfechos** (cada coluna = uma variável resposta)
- **X** (n × p): matriz de **p preditores** (com coluna de 1s para o intercepto)
- **B** (p × q): matriz de coeficientes — βjk = efeito do preditor Xj sobre o desfecho Yk
- **E** (n × q): erros com distribuição vec(E) ~ N(0, **Σ ⊗ Iₙ**)

> **O que é Σ ⊗ Iₙ?** Diz duas coisas ao mesmo tempo: **Σ** captura a correlação *entre os desfechos* do mesmo indivíduo (pressão sistólica e diastólica do mesmo paciente são correlacionadas). **Iₙ** garante que indivíduos diferentes são independentes entre si.

## Estimação

O estimador de mínimos quadrados é:

$$\hat{\mathbf{B}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{Y}$$

> **Insight crucial:** cada coluna de B̂ é **idêntica** ao estimador de uma regressão múltipla separada para aquele desfecho. A vantagem do modelo conjunto **não está nos coeficientes**, mas na **inferência** — que leva em conta a correlação entre os desfechos via Σ̂.

A matrix de covariância residual é estimada por:

$$\hat{\Sigma} = \frac{1}{n-p} \hat{\mathbf{E}}^\top \hat{\mathbf{E}}$$

## Suposições

1. **Linearidade** entre X e cada Yj
2. **Normalidade multivariada** dos erros: eᵢ ~ Nq(0, Σ)
3. **Homogeneidade de covariâncias**: Cov(eᵢ) = Σ para todo i
4. **Independência** entre as observações
5. **Ausência de multicolinearidade** entre os preditores

## Interpretação: dois níveis

1. **Nível multivariado (primário):** os testes (Wilks, Pillai, etc.) indicam se um preditor influencia o **conjunto de desfechos** como um todo. Só avance para o próximo nível se houver significância aqui.
2. **Nível univariado (secundário):** após confirmar a significância multivariada, examine cada equação separadamente — quais desfechos são afetados por quais preditores.

## Exemplo Prático — Dataset `state.x77` (censo EUA, 1977)

O professor usou este dataset em aula: 50 estados norte-americanos com variáveis socioeconômicas.

- **Desfechos Y:** `vida` (expectativa), `homicidio` (taxa), `analfabetismo`
- **Preditores X:** `renda`, `diploma` (% com ensino superior), `geadas`

```r
# Carregando e preparando os dados
dados <- as.data.frame(state.x77)

# Renomeando para português
names(dados) <- c("populacao", "renda", "analfabetismo",
                  "vida", "homicidio", "diploma", "geadas", "area")

# ---- Diagnóstico pré-regressão ----

# 1. Verificar linearidade: gráficos de dispersão
pairs(dados[, c("vida", "homicidio", "analfabetismo",
                "renda", "diploma", "geadas")],
      col = "steelblue", pch = 16, cex = 0.7)

# 2. Normalidade multivariada dos desfechos (distância de Mahalanobis)
vars_desfecho <- c("vida", "homicidio", "analfabetismo")
Y_desc <- as.matrix(dados[, vars_desfecho])
xbar   <- colMeans(Y_desc)
S_inv  <- solve(cov(Y_desc))
d2     <- apply(Y_desc, 1, function(x) t(x - xbar) %*% S_inv %*% (x - xbar))

# Teste KS: H0 = os dados seguem distribuição chi-quadrado(p)
ks_resultado <- ks.test(d2, "pchisq", df = length(vars_desfecho))
cat("KS p-valor:", round(ks_resultado$p.value, 4), "\n")
# Se p > 0.05: não rejeita H0 → normalidade multivariada plausível

# QQ-plot qui-quadrado
qqplot(qchisq(ppoints(nrow(dados)), df = 3), d2,
       main  = "QQ-Plot: normalidade multivariada dos desfechos",
       xlab  = "Quantis χ²(3)", ylab = "D² de Mahalanobis")
abline(0, 1, col = "red", lwd = 2)

# ---- Ajuste da Regressão Multivariada ----
# Sintaxe: cbind(Y1, Y2, Y3) ~ X1 + X2 + X3
fit <- lm(
  cbind(vida, homicidio, analfabetismo) ~ renda + diploma + geadas,
  data = dados
)

# Coeficientes: cada COLUNA corresponde a um desfecho
coef(fit)
# Exemplo de leitura: coef["renda", "vida"] = efeito da renda sobre expectativa de vida

# Resumo univariado (um modelo por desfecho)
summary(fit)
# Note: os coeficientes são os mesmos que em regressões separadas,
# mas os testes de hipótese podem diferir quando se usa a inferência conjunta

# ---- Testes de Hipótese Multivariados ----
# Usa as mesmas 4 estatísticas da MANOVA
fit_manova <- manova(
  cbind(vida, homicidio, analfabetismo) ~ renda + diploma + geadas,
  data = dados
)

summary(fit_manova, test = "Wilks")      # Lambda de Wilks (mais comum)
summary(fit_manova, test = "Pillai")     # Traço de Pillai (mais robusto)
summary(fit_manova, test = "Hotelling")  # Traço de Lawley-Hotelling
summary(fit_manova, test = "Roy")        # Maior raiz de Roy

# ---- Testando cada preditor individualmente ----
for (pred in c("renda", "diploma", "geadas")) {
  formula_str <- paste("cbind(vida, homicidio, analfabetismo) ~", pred)
  fit_uni <- manova(as.formula(formula_str), data = dados)
  cat("\nPreditor:", pred, "\n")
  print(summary(fit_uni, test = "Wilks"))
}

# ---- Diagnóstico dos resíduos ----
# Resíduos multivariados: cada observação tem um VETOR de resíduos
residuos <- residuals(fit)

# Normalidade dos resíduos por desfecho
par(mfrow = c(1, 3))
for (j in 1:3) {
  qqnorm(residuos[, j], main = paste("QQ-Plot:", vars_desfecho[j]))
  qqline(residuos[, j], col = "red")
}
par(mfrow = c(1, 1))
```

---

# PARTE ii — REGRESSÃO CANÔNICA

## Motivação

> **Analogia:** Imagine um aluno com 3 hábitos (estudo, frequência, sono) e 4 notas (Matemática, Português, Ciências, História). As notas são todas correlacionadas — um aluno esforçado tende a ir bem em tudo. Modelar cada nota separadamente **desperdiça essa estrutura**. A regressão canônica pergunta: *"existe uma combinação dos hábitos que resume bem a performance geral?"*

**Diferença da regressão multivariada:**

| Aspecto | Regressão Multivariada | Regressão Canônica |
|---|---|---|
| Opera em | Variáveis originais | Combinações lineares (canônicas) |
| Prediz | Cada Yj diretamente | Combinações de Y a partir de combinações de X |
| Útil quando | Desfechos relativamente independentes | Desfechos muito correlacionados entre si |

## Variáveis Canônicas

$$U_k = \mathbf{a}_k^\top \mathbf{X} \quad \text{e} \quad V_k = \mathbf{b}_k^\top \mathbf{Y}$$

Os vetores **aₖ** e **bₖ** são encontrados **maximizando** Cor(Uₖ, Vₖ) = ρₖ, sujeito a Var(Uₖ) = Var(Vₖ) = 1 e não correlação com pares anteriores.

A solução vem dos **autovalores** da matriz: Σ⁻¹_XX Σ_XY Σ⁻¹_YY Σ_YX

> **Conexão com autovalores (Aula 2)!** Mais uma vez os autovalores aparecem — eles são o denominador comum de toda a estatística multivariada.

## Número de Pares e Qualidade

- São obtidos **r = min(p, q)** pares canônicos com ρ₁ ≥ ρ₂ ≥ ... ≥ ρᵣ ≥ 0
- O **1º par** captura a maior predição possível
- O **2º par** captura a maior predição residual, ortogonal ao 1º
- Na prática, **apenas os primeiros pares são significativos**

> ⚠️ **Armadilha — ρₖ alto pode enganar!** Uma correlação canônica alta entre Uₖ e Vₖ não garante que essas combinações representem bem as variáveis originais. Use sempre o **índice de redundância** para medir a qualidade real:

$$Rd(Y|X) = \frac{1}{q} \sum_{k=1}^{r} \rho_k^2 \cdot \sum_{j=1}^{q} r^2(Y_j, V_k)$$

## Teste de Bartlett (quantos pares são significativos?)

O teste sequencial avalia par a par:

$$\chi^2_k \approx -\left[n - 1 - \frac{1}{2}(p+q+1)\right] \ln \prod_{i=k}^{r}(1 - \rho_i^2)$$

Procedimento: testa todos → se rejeita, remove ρ₁ e testa o restante → continua até não rejeitar.

---

# PARTE iii — CORRELAÇÃO CANÔNICA

## A Diferença Fundamental: Simetria

> **Analogia:** A regressão canônica é como uma rua de mão única — X prediz Y, há hierarquia. A correlação canônica é uma **praça pública** — X e Y têm papel equivalente, nenhum prediz o outro, apenas se associam.

**Enquanto a regressão canônica pergunta:** *"quanto X prediz Y?"*  
**A correlação canônica pergunta:** *"como X e Y se associam?"*

Na prática, os **cálculos são idênticos** — a distinção está no objetivo e na interpretação.

## O que muda ao trocar X e Y?

| | Muda? |
|---|---|
| Correlações canônicas ρₖ | ❌ Não mudam |
| Testes de significância | ❌ Não mudam |
| Vetores aₖ e bₖ | ✅ Mudam (mas de forma análoga) |
| Interpretação das cargas | ✅ Muda |

## Três Níveis de Interpretação

1. **Correlações canônicas (ρₖ):** força da associação entre o k-ésimo par. ρ²ₖ = proporção de variância compartilhada entre Uₖ e Vₖ.

2. **Cargas canônicas:** correlação entre variáveis originais e variáveis canônicas do **mesmo conjunto**. Permitem nomear as dimensões latentes ("este par representa o fator de dedicação").

3. **Cargas cruzadas:** correlação entre variáveis de X e variáveis canônicas de Y (e vice-versa). São a medida **mais direta** de associação entre os conjuntos.

## Comparativo Final das Três Técnicas

| Aspecto | Reg. Multivariada | Reg. Canônica | Corr. Canônica |
|---|---|---|---|
| **Relação** | Assimétrica | Assimétrica | **Simétrica** |
| **Objetivo** | Predição direta | Predição em dimensões | **Associação** |
| **Opera em** | Variáveis originais | Vars. canônicas | Vars. canônicas |
| **Pergunta** | Quais X afetam quais Y? | Como X resume Y? | Como X e Y se relacionam? |
| **Medida de qualidade** | R² por desfecho | Índice de redundância | Correlação canônica ρₖ |

## Código R — Correlação Canônica

```r
# install.packages("CCA")  # pacote especializado para correlação canônica
library(CCA)

# Usando o dataset state.x77 — mesmas variáveis do exemplo anterior
# X: condições socioeconômicas
# Y: indicadores de saúde/educação
X <- scale(dados[, c("renda", "diploma", "geadas")])     # padronizando
Y <- scale(dados[, c("vida", "homicidio", "analfabetismo")])

# Ajuste da correlação canônica
cc_result <- cc(X, Y)

# Correlações canônicas (ρₖ)
cat("Correlações canônicas:\n")
print(round(cc_result$cor, 4))
# ρ₁ ≥ ρ₂ ≥ ρ₃ (r = min(3,3) = 3 pares)

# Coeficientes canônicos (vetores aₖ e bₖ)
cat("\nCoeficientes canônicos de X (aₖ):\n")
print(round(cc_result$xcoef, 4))
cat("\nCoeficientes canônicos de Y (bₖ):\n")
print(round(cc_result$ycoef, 4))

# Cargas canônicas e cruzadas
cc_comp <- comput(X, Y, cc_result)

cat("\nCargas canônicas de X:\n")
print(round(cc_comp$corr.X.xscores, 4))

cat("\nCargas cruzadas (X com scores de Y):\n")
print(round(cc_comp$corr.X.yscores, 4))

# ---- Teste de Bartlett (quantos pares são significativos?) ----
# install.packages("CCP")
library(CCP)

n <- nrow(X)
p <- ncol(X)
q <- ncol(Y)

# Teste sequencial de Wilks para os pares canônicos
p_vals <- p.asym(rho = cc_result$cor, n = n, p = p, q = q, tstat = "Wilks")
print(p_vals)
# Interpreta: qual o menor k tal que o p-valor > 0.05?
# Pares anteriores a esse k são significativos

# ---- Visualizando os pares canônicos ----
# Scores canônicos
U <- X %*% cc_result$xcoef   # scores de X
V <- Y %*% cc_result$ycoef   # scores de Y

par(mfrow = c(1, min(2, ncol(U))))
for (k in 1:min(2, ncol(U))) {
  plot(U[, k], V[, k],
       main  = paste0("Par canônico ", k, " (ρ = ", round(cc_result$cor[k], 2), ")"),
       xlab  = paste0("U", k, " (combinação de X)"),
       ylab  = paste0("V", k, " (combinação de Y)"),
       col   = "steelblue", pch = 16, cex = 0.8)
  abline(lm(V[, k] ~ U[, k]), col = "red", lwd = 2)
  # Nuvem alongada = alta correlação; nuvem circular = baixa correlação
}
par(mfrow = c(1, 1))

# ---- Índice de redundância ----
# Mede quanto de Y é predito pelo conjunto X (melhor que ρ sozinho)
# Rd = (1/q) * Σ ρₖ² * Σ r²(Yⱼ, Vₖ)
cargas_Y <- cc_comp$corr.Y.yscores   # cargas canônicas de Y
rho2     <- cc_result$cor^2          # quadrado das correlações canônicas

Rd <- mean(sapply(1:length(rho2), function(k) {
  rho2[k] * mean(cargas_Y[, k]^2)
}))
cat("\nÍndice de Redundância Rd(Y|X):", round(Rd, 4), "\n")
cat("Interpretação: X explica em média", round(Rd*100, 1),
    "% da variância de Y\n")
```

---

## 💡 Insights do Professor (da transcrição)

> *"Isolation Forest é uma abordagem baseada em árvore para detectar outliers. Mas uma distância de Mahalanobis é capaz de identificar o mesmo — com suas restrições, claro, como a suposição de normalidade. Existem técnicas mais simples que podem detectar o mesmo."*

- A distância de Mahalanobis (vista na Aula 2) é a fundação da detecção de outliers multivariados — mas tem restrições (normalidade). Isolation Forest e deep learning ampliam isso sem suposições.
- A regressão multivariada e a MANOVA são **casos da mesma família** — a MANOVA é regressão multivariada com preditores categóricos.
- A sugestão do professor para o trabalho final: **comparar** o modelo de regressão multivariada com regressões univariadas separadas — os coeficientes são idênticos, mas os testes de hipótese podem diferir por conta da estrutura de correlação Σ.
- *"Tudo que eu tinha no meu caderno na graduação, mercado, doutorado — digitei em LaTeX. Essa é a forma que eu encontrei de aprender melhor: anotar não só o que o professor escreve no quadro, mas o que ele fala."*

---

## 🔮 Próxima Aula — Estrutura dos Dados

O professor anunciou que entraremos em:

- **PCA (Análise de Componentes Principais):** redução de dimensionalidade — onde os **autovalores e autovetores da Aula 2** finalmente ganham seu papel central
- **Análise Fatorial:** estruturas latentes por trás das variáveis observadas

---

## 📚 Referências

- Johnson & Wichern — *Applied Multivariate Statistical Analysis*, Cap. 7 (Reg. Multivariada) e Cap. 10 (Correlação Canônica)
- Rencher, A. — *Methods of Multivariate Analysis*, Cap. 8 e 11
- Pacote `CCA`: https://cran.r-project.org/package=CCA
- Pacote `CCP` (testes de significância para correlação canônica): https://cran.r-project.org/package=CCP
- Dataset `state.x77`: nativo do R — `?state.x77` no console
