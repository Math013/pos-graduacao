# 📊 Análise Multivariada — Anotações de Aula
## Aula 3: Probabilidade, Inferência Multivariada, T² de Hotelling, MANOVA e MANCOVA

> **Professor:** Vinícius Osterne, PhD  
> **Observação do professor:** *"A terceira aula é mais carregada. Mas quem escolhe entender de fato a coisa se diferencia. As coisas estão cada vez mais fáceis e cada vez mais complicadas — ao mesmo tempo."*

---

## 🗺️ Onde Estamos na Linha do Tempo

```
✅ Introdução & Conceitos Prévios
✅ Álgebra Linear
✅ Análise Exploratória
✅ Probabilidade & Inferência  ← HOJE
   ├── Teorema Central do Limite (Multivariado)
   ├── Distribuição Normal Multivariada
   ├── T² de Hotelling (1 e 2 populações)
   ├── MANOVA
   └── MANCOVA
⬜ Modelos Multivariados
⬜ Estrutura dos Dados (PCA, Fatorial)
⬜ Classificação & Agrupamento
```

---

## 🧩 O Papel do Teorema Central do Limite (TLC) na Multivariada

> **Analogia:** Imagine que você tem uma urna cheia de bolas com números aleatórios de qualquer formato de distribuição — pode ser assimétrica, bimodal, o que for. Se você pegar amostras dessa urna repetidamente e calcular a média de cada amostra, a distribuição dessas médias vai se parecer cada vez mais com uma curva de sino (normal) conforme você aumenta o tamanho das amostras. O TLC é essa garantia mágica.

### Por que o TLC é fundamental para a multivariada?

O TLC garante que, independente da distribuição original dos dados, **as médias amostrais** convergem para a distribuição normal quando **n é grande**. Isso nos permite:

- Construir **intervalos de confiança** para o vetor de médias
- Realizar **testes de hipóteses** sobre médias
- Fundamentar a **estatística T² de Hotelling**
- Sustentar toda a **teoria assintótica de modelos lineares e multivariados**

> ⚠️ **Nuance importante (apontada pelo professor):** O TLC não dispensa a suposição de normalidade — ele **viabiliza a construção** do teste mesmo quando a população não é perfeitamente normal. Para amostras grandes (n ≥ 30), o teste funciona de forma robusta. Para amostras pequenas, a normalidade da população ainda é necessária.

```r
# Demonstração do TLC na prática:
# Mesmo com uma distribuição exponencial (assimétrica), as médias se normalizam

set.seed(42)
n_simulacoes <- 5000
n_amostra    <- 50   # tamanho de cada amostra

# Distribuição original: exponencial (assimétrica para a direita)
medias <- replicate(n_simulacoes, mean(rexp(n_amostra, rate = 1)))

par(mfrow = c(1, 2))

# Distribuição original
hist(rexp(10000, rate = 1),
     main = "Distribuição Original (Exponencial)",
     xlab = "X", col = "steelblue", breaks = 50)

# Distribuição das médias amostrais → fica normal!
hist(medias,
     main = paste("Distribuição das Médias (n =", n_amostra, ")"),
     xlab = "Média Amostral", col = "darkorange", breaks = 50,
     freq = FALSE)
curve(dnorm(x, mean = mean(medias), sd = sd(medias)),
      add = TRUE, col = "darkred", lwd = 2)

par(mfrow = c(1, 1))
```

---

## 🔔 Distribuição Normal Multivariada (DNM)

> **Analogia:** A normal univariada é uma curva de sino em 1D. A normal multivariada é como uma "barraca de acampamento" (elipsoide) em múltiplas dimensões — a barraca se deforma de acordo com as correlações entre as variáveis.

**Notação:** X ~ Nₚ(μ, Σ)

Onde:
- **μ** = vetor de médias (p × 1) — o centro da distribuição
- **Σ** = matriz de covariância (p × p) — a forma e orientação da distribuição

### Propriedades Fundamentais (que sustentam toda a inferência)

**1. Fechada sob marginalização:**
Se (X, Y) ~ N multivariada, então X sozinha também segue uma N multivariada. Você pode "esquecer" variáveis e a estrutura normal se mantém.

**2. Independência ⟺ Correlação zero:**
Na DNM e **apenas na DNM**: se σ_jk = 0 (sem correlação), as variáveis são independentes. Isso não vale para outras distribuições!

**3. Fechada sob transformações lineares:**
Se X ~ Nₚ(μ, Σ) e A é uma matriz q × p, então **AX ~ Nq(Aμ, AΣAᵀ)**.
Isso fundamenta o PCA, a MANOVA e os modelos lineares multivariados.

```r
# Simulando dados de uma Normal Multivariada
# install.packages("MASS")
library(MASS)

# Definindo parâmetros
mu    <- c(5, 10)           # vetor de médias
Sigma <- matrix(c(4, 3,     # matriz de covariância
                  3, 9), nrow = 2)
# Cov(X1, X2) = 3 → correlação positiva

set.seed(42)
dados_dnm <- mvrnorm(n = 500, mu = mu, Sigma = Sigma)
colnames(dados_dnm) <- c("X1", "X2")

# Visualizando
plot(dados_dnm, col = adjustcolor("steelblue", 0.5),
     main = "Distribuição Normal Bivariada",
     xlab = "X1", ylab = "X2", pch = 16)

# Testando normalidade multivariada com distâncias de Mahalanobis
# Sob normalidade, D² ~ Qui-quadrado(p)
xbar  <- colMeans(dados_dnm)
S     <- cov(dados_dnm)
d2    <- mahalanobis(dados_dnm, center = xbar, cov = S)

# QQ-plot qui-quadrado: pontos alinhados = normalidade
qqplot(qchisq(ppoints(nrow(dados_dnm)), df = 2), d2,
       main = "QQ-Plot Qui-quadrado (teste de normalidade multivariada)",
       xlab = "Quantis Teóricos χ²(2)",
       ylab = "Distâncias de Mahalanobis ao quadrado")
abline(0, 1, col = "red", lwd = 2)
```

---

## 📐 Por que não fazer Vários Testes t Separados?

Antes de entrar no T², é fundamental entender **por que não podemos simplesmente fazer um teste t para cada variável separadamente**.

> **Analogia:** Imagine que você está jogando cara ou coroa com uma moeda honesta. A chance de errar uma vez é 5%. Mas se você jogar 20 vezes, a chance de errar pelo menos uma vez é muito maior. Fazer múltiplos testes t é a mesma coisa — o erro se acumula.

### O problema da inflação do erro Tipo I

Com múltiplos testes a α = 0,05:

| Número de Testes | Chance de ao menos 1 falso positivo |
|---|---|
| 1 | 5,0% |
| 3 | 14,3% |
| 5 | 22,6% |
| 10 | 40,1% |
| 20 | 64,2% |

Além disso, testes t separados **ignoram a correlação entre as variáveis** — o que é exatamente o que a análise multivariada quer estudar!

```r
# Demonstrando a inflação do erro Tipo I
set.seed(42)
n_sim <- 10000
p     <- 5    # número de variáveis

# Dados onde H0 é VERDADEIRA (não há diferença real)
rejeitou_ao_menos_um <- replicate(n_sim, {
  dados_grupo1 <- matrix(rnorm(30 * p), nrow = 30)
  dados_grupo2 <- matrix(rnorm(30 * p), nrow = 30)
  
  # t-test separado para cada variável
  pvals <- sapply(1:p, function(j) {
    t.test(dados_grupo1[, j], dados_grupo2[, j])$p.value
  })
  any(pvals < 0.05)  # TRUE se rejeitou H0 em pelo menos 1 teste
})

cat("Taxa de erro real com", p, "testes t separados:",
    round(mean(rejeitou_ao_menos_um) * 100, 1), "%\n")
cat("Taxa esperada (teoria): ~", round((1 - 0.95^p) * 100, 1), "%\n")
# Deveria ser 5% — mas está inflado!
```

---

## 🎯 T² de Hotelling — O Teste t Multivariado

> **Analogia:** O teste t univariado pergunta: "a média da minha amostra está longe demais do valor hipotético µ₀?" O T² de Hotelling pergunta a mesma coisa, mas para um **vetor de médias inteiro** — e ele usa a **distância de Mahalanobis** para medir esse afastamento, levando em conta a correlação entre todas as variáveis ao mesmo tempo.

### A Progressão Lógica: de t² ao T²

**Univariado (1 variável):**
$$t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}} \quad \Rightarrow \quad t^2 = n(\bar{x} - \mu_0)(s^2)^{-1}(\bar{x} - \mu_0)$$

**Multivariado (p variáveis) — T² de Hotelling:**
$$T^2 = n(\bar{\mathbf{x}} - \boldsymbol{\mu}_0)^\top \mathbf{S}^{-1}(\bar{\mathbf{x}} - \boldsymbol{\mu}_0)$$

É **exatamente** a distância de Mahalanobis ao quadrado entre x̄ e μ₀, ponderada por n.

### Tabela Comparativa: t vs. T²

| Aspecto | Univariado (t) | Multivariado (T²) |
|---|---|---|
| Estatística | t² = n(x̄ − µ₀)²/s² | T² = n(x̄ − µ₀)ᵀ S⁻¹ (x̄ − µ₀) |
| Distribuição | F(1, n−1) | (p(n−1)/(n−p)) × F(p, n−p) |
| Rejeita H₀ se | t² > F₁,ₙ₋₁;α | T² > p(n−1)/(n−p) × Fₚ,ₙ₋ₚ;α |

> **Por que a distribuição envolve F se o nome é T²?** Porque no caso univariado t² = F(1, n-1). A generalização segue a mesma lógica: numerador normal + denominador Wishart = distribuição F reescalonada.

### Cenários de Aplicação

```
1 população  → T² de uma amostra (H₀: μ = μ₀)
2 populações independentes → T² de duas amostras (H₀: μ₁ = μ₂)
2 populações pareadas → T² pareado (H₀: μ_D = 0)
g > 2 populações → MANOVA
```

### Código R — T² de Hotelling (1 população)

```r
# install.packages("DescTools")
library(DescTools)

# Simulando dados: 3 variáveis de desempenho cognitivo
set.seed(123)
n     <- 40
dados <- data.frame(
  memoria    = rnorm(n, mean = 52, sd = 8),
  atencao    = rnorm(n, mean = 48, sd = 6),
  raciocinio = rnorm(n, mean = 55, sd = 10)
)

# Vetor hipotético (norma esperada para a população)
mu_0 <- c(50, 50, 50)

# Calculando T² manualmente
xbar <- colMeans(dados)
S    <- cov(dados)
p    <- ncol(dados)

# T² = n * (xbar - mu0)' * S^-1 * (xbar - mu0)
diff <- xbar - mu_0
T2   <- n * t(diff) %*% solve(S) %*% diff
cat("T² calculado manualmente:", round(T2, 4), "\n")

# Convertendo para estatística F
F_stat <- (n - p) / (p * (n - 1)) * T2
gl1    <- p
gl2    <- n - p
p_val  <- pf(F_stat, df1 = gl1, df2 = gl2, lower.tail = FALSE)

cat("Estatística F:", round(F_stat, 4), "\n")
cat("Graus de liberdade:", gl1, "e", gl2, "\n")
cat("p-valor:", round(p_val, 4), "\n")
cat("Decisão:", ifelse(p_val < 0.05, "Rejeita H0 (μ ≠ μ₀)", "Não rejeita H0"), "\n")

# Usando a função do pacote DescTools
HotellingsT2Test(dados, mu = mu_0)
```

### Código R — T² de Hotelling (2 populações independentes)

```r
# Comparando dois grupos: tratamento vs controle
set.seed(42)
grupo_A <- data.frame(
  memoria    = rnorm(30, mean = 55, sd = 8),
  atencao    = rnorm(30, mean = 52, sd = 6),
  raciocinio = rnorm(30, mean = 58, sd = 10)
)
grupo_B <- data.frame(
  memoria    = rnorm(30, mean = 48, sd = 9),
  atencao    = rnorm(30, mean = 45, sd = 7),
  raciocinio = rnorm(30, mean = 50, sd = 11)
)

# Usando HotellingsT2Test para 2 grupos
HotellingsT2Test(grupo_A, grupo_B)
# Se p < 0.05 → os vetores de médias dos dois grupos são diferentes
```

### Região de Confiança (vs. Intervalo de Confiança)

> **Analogia:** No caso univariado, um intervalo de confiança é um segmento de reta. No caso multivariado, ele vira uma **elipse** (em 2D) ou um **elipsoide** (em p dimensões). Todos os valores de μ dentro dessa elipse são "plausíveis" dado os dados.

```r
# Visualizando a região de confiança 95% no caso bivariado
library(car)  # install.packages("car")

dados_biv <- dados[, c("memoria", "atencao")]
mu_0_biv  <- c(50, 50)

# dataEllipse plota a nuvem de pontos + elipse de confiança
dataEllipse(dados_biv$memoria, dados_biv$atencao,
            levels = 0.95,
            col    = c("steelblue", "darkblue"),
            pch    = 16, cex = 0.7,
            xlab   = "Memória", ylab   = "Atenção",
            main   = "Região de Confiança 95% para (μ₁, μ₂)")
points(mu_0_biv[1], mu_0_biv[2], pch = 4, col = "red", cex = 2, lwd = 2)
legend("topright", c("μ₀ hipotético", "Nuvem de dados"),
       col = c("red", "steelblue"), pch = c(4, 16))
```

---

## 🔬 MANOVA — Análise Multivariada de Variância

> **Analogia:** A ANOVA pergunta: "as médias de **uma** variável são diferentes entre os grupos?" (como comparar o peso médio de pacientes em 3 dietas diferentes).  
> A MANOVA pergunta: "os **vetores de médias** de **múltiplas** variáveis são diferentes entre os grupos?" (como comparar peso + colesterol + pressão ao mesmo tempo).

**A hierarquia dos testes:**

```
Uma variável  → t de Student  (2 grupos)
Uma variável  → ANOVA         (g > 2 grupos)
    ↓ generaliza para p variáveis
p variáveis   → T² de Hotelling  (2 grupos)
p variáveis   → MANOVA           (g > 2 grupos)
```

### Hipóteses

- **H₀:** μ₁ = μ₂ = ... = μg (todos os vetores de médias são iguais)
- **H₁:** pelo menos um vetor de médias é diferente

### Decomposição da Variabilidade

Assim como na ANOVA divide SQtotal = SQentre + SQdentro, a MANOVA faz o mesmo com **matrizes** (não escalares):

$$\mathbf{T} = \mathbf{H} + \mathbf{E}$$

- **T** = variabilidade total (todos os pontos em torno da média geral)
- **H** = variabilidade **H**ipótese / entre grupos (médias dos grupos vs. média geral)
- **E** = variabilidade do **E**rro / dentro dos grupos (indivíduos vs. média do seu grupo)

> **Intuição:** Se os grupos forem muito diferentes, H será grande em relação a E. Se forem parecidos, H será pequena.

### Lambda de Wilks — A Estatística de Teste Principal

$$\Lambda^* = \frac{|\mathbf{E}|}{|\mathbf{H} + \mathbf{E}|} \in (0, 1]$$

> **Analogia:** O Lambda de Wilks é como uma "pontuação de semelhança" entre os grupos. Λ* próximo de **1** → grupos parecidos (H₀ plausível). Λ* próximo de **0** → grupos muito diferentes (rejeita H₀).

### As 4 Estatísticas Multivariadas

Todas são funções dos autovalores λᵢ de E⁻¹H:

| Estatística | Fórmula | Melhor quando... |
|---|---|---|
| **Wilks (Λ*)** | ∏(1 + λᵢ)⁻¹ | Uso geral — mais comum na literatura |
| **Pillai (V)** | Σλᵢ/(1+λᵢ) | Mais robusto a violações de normalidade |
| **Lawley-Hotelling (U)** | Σλᵢ | Diferenças em todas as dimensões |
| **Roy (θ)** | λ₁/(1+λ₁) | Separação máxima em uma única direção |

> 💡 Na prática, se as 4 estatísticas concordam → conclusão sólida. Se divergem → investigue as suposições.

### Suposições da MANOVA

1. **Normalidade multivariada** dentro de cada grupo
2. **Homogeneidade das matrizes de covariância** (Σ₁ = Σ₂ = ... = Σg) — testada com o **Teste de Box M**
3. **Independência** das observações

```r
# install.packages(c("heplots", "biotools"))
library(heplots)   # para boxM
library(ggplot2)

# Usando o dataset clássico de siris de caranguejos (pacote MASS)
data(crabs, package = "MASS")

# Variáveis morfológicas: FL, RW, CL, CW, BD
# Grupos: sp (espécie) × sex (sexo) → aqui usamos só sp
dados_manova <- crabs[, c("sp", "FL", "RW", "CL", "CW", "BD")]

# ----- 1. Verificar normalidade multivariada por grupo -----
# (QQ-plot Mahalanobis — já vimos na seção anterior)

# ----- 2. Teste de Box M (homogeneidade das covariâncias) -----
box_m <- boxM(dados_manova[, 2:6], dados_manova$sp)
print(box_m)
# H0: Σ_B = Σ_O (covariâncias iguais)
# p < 0.05 → covariâncias diferentes → use Pillai (mais robusto)

# ----- 3. Ajustar a MANOVA -----
fit_manova <- manova(
  cbind(FL, RW, CL, CW, BD) ~ sp,
  data = dados_manova
)

# Testando com as 4 estatísticas
summary(fit_manova, test = "Wilks")
summary(fit_manova, test = "Pillai")
summary(fit_manova, test = "Hotelling")
summary(fit_manova, test = "Roy")

# ----- 4. ANOVAs univariadas como follow-up (post-hoc) -----
# Se a MANOVA rejeita H0, investigamos qual(is) variável(is) diferem
summary.aov(fit_manova)
```

### Decomposição Manual: Matrizes H e E

```r
# Calculando H e E manualmente para entender a estrutura

Y           <- as.matrix(dados_manova[, 2:6])
grupos      <- unique(dados_manova$sp)
media_geral <- colMeans(Y)
N           <- nrow(Y)
p           <- ncol(Y)
g           <- length(grupos)

# Matriz E (within groups — erro)
E_mat <- Reduce("+", lapply(grupos, function(gr) {
  sub  <- as.matrix(dados_manova[dados_manova$sp == gr, 2:6])
  xbar <- colMeans(sub)
  # Soma dos produtos cruzados dos desvios em torno da média do grupo
  t(sweep(sub, 2, xbar)) %*% sweep(sub, 2, xbar)
}))

# Matriz H (between groups — hipótese)
n_por_grupo <- table(dados_manova$sp)
H_mat <- Reduce("+", lapply(grupos, function(gr) {
  ni   <- n_por_grupo[gr]
  xbar <- colMeans(as.matrix(dados_manova[dados_manova$sp == gr, 2:6]))
  diff <- xbar - media_geral
  # Produto externo ponderado pelo tamanho do grupo
  ni * outer(diff, diff)
}))

# Lambda de Wilks manual
Lambda <- det(E_mat) / det(H_mat + E_mat)
cat("Lambda de Wilks manual:", round(Lambda, 4), "\n")

# Estatística qui-quadrado aproximada
qui2 <- -(N - 1 - (p + g) / 2) * log(Lambda)
gl   <- p * (g - 1)
pval <- pchisq(qui2, df = gl, lower.tail = FALSE)

cat("Estatística χ²:", round(qui2, 4), "\n")
cat("GL:", gl, "\n")
cat("p-valor:", round(pval, 6), "\n")
```

---

## 🔧 MANCOVA — MANOVA com Covariáveis

> **Analogia:** Imagine que você quer comparar o desempenho cognitivo de pessoas que meditam vs. que fazem exercício. Mas o grupo de meditação é mais velho e mais escolarizado. Se você fizer a MANOVA direto, não sabe se a diferença vem da prática ou da idade/escolaridade. A MANCOVA **remove o efeito da idade e escolaridade primeiro** e então compara os grupos — como nivelar o campo de jogo antes da partida.

### Modelo

$$X_{kj} = \mu + \tau_k + \Gamma z_{kj} + \epsilon_{kj}$$

- **μ** = média geral
- **τk** = efeito do grupo k (o que queremos testar)
- **Γ** = coeficientes das covariáveis (efeito a ser removido)
- **zkj** = vetor de covariáveis da observação j do grupo k

### Hipóteses

- **H₀:** τ₁ = τ₂ = ... = τg = 0 (não há efeito de grupo após remover covariáveis)
- **H₁:** pelo menos um τk ≠ 0

### Quando usar MANOVA vs. MANCOVA?

| Situação | Use |
|---|---|
| Grupos aleatorizados, sem variáveis de confusão | MANOVA |
| Grupos diferem em variáveis contínuas relevantes (idade, escore basal, etc.) | MANCOVA |
| Quer aumentar o poder controlando variabilidade extra | MANCOVA |

> ⚠️ **Suposição crítica da MANCOVA:** a relação entre a covariável e as variáveis dependentes deve ser **a mesma em todos os grupos** (homogeneidade das inclinações). Sempre teste isso antes!

```r
# Exemplo de MANCOVA em R
set.seed(42)
n_grupo <- 30

# Simulando dados com covariável confundidora (idade)
dados_mancova <- data.frame(
  grupo     = rep(c("Meditacao", "Exercicio", "Controle"), each = n_grupo),
  idade     = c(rnorm(n_grupo, 45, 8), rnorm(n_grupo, 35, 7), rnorm(n_grupo, 40, 9)),
  memoria   = NA,
  atencao   = NA,
  raciocinio = NA
)

# Desempenho influenciado pelo grupo E pela idade
for (i in 1:nrow(dados_mancova)) {
  efeito_grupo <- switch(dados_mancova$grupo[i],
                         Meditacao = 5, Exercicio = 2, Controle = 0)
  dados_mancova$memoria[i]    <- 50 + efeito_grupo + 0.3 * dados_mancova$idade[i] + rnorm(1, 0, 5)
  dados_mancova$atencao[i]    <- 48 + efeito_grupo + 0.2 * dados_mancova$idade[i] + rnorm(1, 0, 4)
  dados_mancova$raciocinio[i] <- 52 + efeito_grupo + 0.4 * dados_mancova$idade[i] + rnorm(1, 0, 6)
}

# ----- MANOVA sem controlar a covariável (pode ser enganosa) -----
fit_manova <- manova(
  cbind(memoria, atencao, raciocinio) ~ grupo,
  data = dados_mancova
)
summary(fit_manova, test = "Wilks")

# ----- MANCOVA controlando a idade -----
# Basta adicionar a covariável na fórmula com +
fit_mancova <- manova(
  cbind(memoria, atencao, raciocinio) ~ grupo + idade,
  data = dados_mancova
)
summary(fit_mancova, test = "Wilks")

# Comparando resultados:
# Lambda de Wilks deve mudar após controlar a idade
# Se a covariável era importante, o p-valor do grupo muda!
```

---

## 💡 Insights do Professor (da transcrição)

> *"Rodar uma função qualquer IA te passa. Entender a construção do teste é o que diferencia. 90% das pessoas ficam só no primeiro nível — me passa a função. Poucos chegam no segundo: como eu obtenho essa estatística? Que suposições foram feitas?"*

- **A regressão logística não foi criada agora** — foi proposta em 1972. Conhecer a história evita reinventar a roda e permite contextualizar os modelos
- O TLC é o que nos permite **usar a matemática da distribuição normal no mundo real**, onde as populações raramente são normais
- MANOVA e T² são **equivalentes** para g = 2 grupos — o T² é um caso particular da MANOVA
- Para p = 1 variável, a MANOVA se reduz à ANOVA e o Lambda de Wilks se reduz ao F da ANOVA clássica

---

## 🔮 Próxima Aula

O professor avisou que na próxima aula entraremos em **Inferência sobre o vetor de médias de uma população** com mais profundidade:

- Região de confiança elipsoidal (versão multivariada do IC)
- T² pareado (observações dependentes — ex.: antes/depois)
- Início dos **Modelos Multivariados** (regressão multivariada)

---

## 📚 Referências

- Johnson & Wichern — *Applied Multivariate Statistical Analysis*, Cap. 5 e 6
- Rencher, A. — *Methods of Multivariate Analysis*, Cap. 3 e 5
- Pacote `DescTools` — `HotellingsT2Test()`: https://cran.r-project.org/package=DescTools
- Pacote `heplots` — `boxM()` e `heplot()`: https://cran.r-project.org/package=heplots
- Pacote `car` — `Manova()` com diagnósticos: https://cran.r-project.org/package=car
