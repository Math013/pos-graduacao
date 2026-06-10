# 📊 Análise Multivariada — Anotações de Aula
## Aula 2: Álgebra Linear & Análise Exploratória Multivariada

> **Professor:** Vinícius Osterne, PhD  
> **Observação do professor:** *"A segunda aula é sempre um pouco mais complicada porque a gente fala de álgebra linear — cansa um pouco. Mas tentei trazer o máximo possível para o nosso contexto para que não fique tão cansativo."*

---

## 🗺️ Onde Estamos na Linha do Tempo

```
✅ Introdução & Conceitos Prévios
✅ Álgebra Linear         ← HOJE
✅ Análise Exploratória   ← HOJE
⬜ Probabilidade & Inferência (próxima aula)
⬜ Modelos Multivariados
⬜ Estrutura dos Dados
⬜ Classificação & Agrupamento
```

---

# PARTE 1 — ÁLGEBRA LINEAR

> **Por que isso importa?** Toda a estatística multivariada é escrita em linguagem de matrizes. Covariância, correlação, PCA, MANOVA — tudo depende de operações matriciais. É como aprender o alfabeto antes de ler: chato, mas indispensável.

---

## 📐 Escalares, Vetores e Matrizes

> **Analogia:** Pense assim: um **escalar** é uma nota musical única (ex.: Lá = 440 Hz). Um **vetor** é uma melodia — uma sequência de notas. Uma **matriz** é uma partitura inteira — múltiplas melodias ao mesmo tempo, organizadas em linhas e colunas.

```r
# Escalar: um único número
altura <- 1.75

# Vetor: uma sequência de valores (uma observação com p variáveis)
pessoa_1 <- c(altura = 1.75, peso = 70, idade = 30)

# Matriz: n observações × p variáveis (a estrutura base dos nossos dados)
# Cada linha = uma observação; cada coluna = uma variável
X <- matrix(
  c(1.75, 70, 30,
    1.60, 55, 25,
    1.80, 90, 45),
  nrow = 3,      # 3 pessoas
  ncol = 3,      # 3 variáveis: altura, peso, idade
  byrow = TRUE,  # preenche linha por linha
  dimnames = list(
    c("p1", "p2", "p3"),
    c("Altura", "Peso", "Idade")
  )
)

print(X)
dim(X)  # retorna c(n_linhas, n_colunas) = c(3, 3)
```

---

## 🔄 Operações Matriciais Essenciais

### Transposta

> **Analogia:** Transpor é como virar a partitura de cabeça para baixo: as linhas viram colunas e as colunas viram linhas.

```r
A <- matrix(c(1, 2, 3,
              4, 5, 6), nrow = 2, byrow = TRUE)
cat("A:\n"); print(A)
cat("t(A) — transposta:\n"); print(t(A))
```

### Multiplicação de Matrizes

> ⚠️ **Atenção:** multiplicação de matrizes **não é** multiplicar elemento por elemento. É uma operação de produto interno entre linhas e colunas. Em R, usa-se `%*%` (não `*`).

```r
A <- matrix(c(1, 2,
              3, 4), nrow = 2, byrow = TRUE)
B <- matrix(c(5, 6,
              7, 8), nrow = 2, byrow = TRUE)

cat("Multiplicação matricial A %*% B:\n"); print(A %*% B)
cat("Multiplicação elemento-a-elemento A * B:\n"); print(A * B)
# Os resultados são DIFERENTES!
```

### Inversa de uma Matriz

> **Analogia:** A inversa de uma matriz é como o "número recíproco" no mundo das matrizes. Assim como 5 × (1/5) = 1, uma matriz A multiplicada pela sua inversa A⁻¹ resulta na **matriz identidade** I (que faz o papel do número 1).

```r
A <- matrix(c(4, 7,
              2, 6), nrow = 2, byrow = TRUE)

A_inv <- solve(A)        # solve() calcula a inversa em R
cat("A:\n");     print(A)
cat("A^-1:\n");  print(A_inv)
cat("A %*% A^-1 deve ser a identidade:\n")
print(round(A %*% A_inv, 10))  # round para eliminar erros numéricos
```

### Matriz Identidade e Diagonal

```r
# Identidade: 1s na diagonal, 0s no resto — o "neutro" da multiplicação matricial
I3 <- diag(3)
print(I3)

# Diagonal: só tem valores na diagonal principal
D  <- diag(c(3, 7, 2))
print(D)

# A inversa de uma diagonal é trivial: basta inverter cada elemento
print(diag(1 / c(3, 7, 2)))
```

---

## 🔁 Matriz Simétrica e Matriz de Covariância

> **Conexão fundamental:** A matriz de covariância **sempre é simétrica**, porque Cov(X, Y) = Cov(Y, X). Isso significa que todas as propriedades de matrizes simétricas se aplicam diretamente a ela!

```r
# Uma matriz simétrica: A = t(A)
S <- matrix(c(4, 2, 1,
              2, 5, 3,
              1, 3, 6), nrow = 3, byrow = TRUE)

cat("S == t(S)?", all.equal(S, t(S)), "\n")  # deve ser TRUE

# A covariância de dados reais também é simétrica
set.seed(42)
dados   <- matrix(rnorm(50), nrow = 10)
cov_mat <- cov(dados)
cat("Cov(dados) é simétrica?", all.equal(cov_mat, t(cov_mat)), "\n")

# Propriedade: A^T * A sempre produz uma matriz simétrica!
A3      <- matrix(1:6, nrow = 3)
simetrica <- t(A3) %*% A3
cat("t(A) %*% A é simétrica?", all.equal(simetrica, t(simetrica)), "\n")
```

---

## 🌀 Autovalores e Autovetores — O Coração da Multivariada

> **Analogia (a melhor delas!):** Imagine que você tem uma máquina de espirografar vetores — ela pega qualquer seta e, em geral, muda tanto o tamanho quanto a direção. Os **autovetores** são as setas especiais que essa máquina **não consegue torcer** — ela só muda o tamanho. O quanto ela alonga ou encolhe essa seta especial é o **autovalor**.

Matematicamente: **Av = λv**

Onde:
- **A** = a matriz (ex: matriz de covariância S)
- **v** = autovetor (direção especial que não rotaciona)
- **λ** = autovalor (fator de escala)

### Intuição geométrica dos autovalores

| Autovalor λ | O que acontece com o vetor |
|---|---|
| λ > 1 | Estica (fica mais longo) |
| 0 < λ < 1 | Encolhe |
| λ < 0 | Inverte a direção |
| λ = 1 | Não muda nada |

### Conexão com os dados

> **Analogia:** Os dados formam uma **nuvem de pontos** no espaço. Pense nessa nuvem como uma bola de futebol americano (elipse). Os **autovetores da covariância** são os eixos dessa bola — as direções em que ela mais se alonga. Os **autovalores** são o comprimento de cada eixo — o quanto os dados se espalharam em cada direção.

```r
# Calculando autovalores e autovetores em R
A_eig <- matrix(c(3, 1,
                  1, 3), nrow = 2, byrow = TRUE)

resultado <- eigen(A_eig)  # função nativa do R

cat("Autovalores:\n");            print(resultado$values)
cat("Autovetores (colunas):\n");  print(resultado$vectors)

# VERIFICAÇÃO: A %*% v deve ser igual a lambda * v
v1      <- resultado$vectors[, 1]   # primeiro autovetor (coluna 1)
lambda1 <- resultado$values[1]      # primeiro autovalor

cat("\nVerificação Av = λv:\n")
cat("A %*% v1:   ", round(A_eig %*% v1, 6), "\n")
cat("lambda1*v1: ", round(lambda1 * v1, 6), "\n")
# Devem ser iguais!

# Exemplo do slide: S = [[3,1],[1,3]] → λ1 = 4, λ2 = 2
S_ex <- matrix(c(3, 1, 1, 3), nrow = 2)
cat("\nAutovalores de S =[[3,1],[1,3]]:", eigen(S_ex)$values, "\n")
# Esperado: 4 e 2
```

### Decomposição Espectral

Toda matriz simétrica pode ser reescrita como: **A = P Λ Pᵀ**

Onde P é a matriz de autovetores e Λ é a matriz diagonal de autovalores.

```r
# Reconstruindo a matriz a partir de autovalores e autovetores
P      <- resultado$vectors        # matriz de autovetores
Lambda <- diag(resultado$values)   # diagonal de autovalores

A_reconstruida <- P %*% Lambda %*% t(P)
cat("A original:\n");      print(A_eig)
cat("A reconstruída:\n");  print(round(A_reconstruida, 10))
# Deve ser idêntica!
```

> 💡 **Por que isso é importante?** A PCA (Análise de Componentes Principais) é literalmente a decomposição espectral da matriz de covariância. Entender autovalores = entender PCA.

---

# PARTE 2 — ANÁLISE EXPLORATÓRIA MULTIVARIADA

> **Citação do professor:** *"Nem tudo se resume a rede neural, nem tudo se resume a IA, nem tudo se resume a LLM. Os gráficos revelam estruturas que as estatísticas resumidas escondem."*

---

## 📏 Distâncias Multivariadas

Antes de visualizar, precisamos entender como **medir diferença** entre observações no espaço multivariado.

> **Analogia:** Imagine que você quer saber o quão "diferente" dois pacientes são com base em múltiplos exames. A distância euclidiana seria como medir em linha reta — ingênua. A distância de Mahalanobis é como um GPS inteligente que leva em conta o tráfego (a correlação entre as variáveis).

| Distância | Fórmula | Considera correlação? |
|---|---|---|
| **Manhattan** | Σ\|xj − yj\| | Não |
| **Euclidiana** | √Σ(xj − yj)² | Não |
| **Mahalanobis** | √(x−y)ᵀ S⁻¹ (x−y) | ✅ Sim |

A **Mahalanobis** é a mais importante na multivariada: ela pondera pelo inverso da variância, evitando que variáveis com maior escala dominem a distância.

```r
# Calculando as três distâncias em R
x <- c(1, 2)
y <- c(4, 6)
S <- matrix(c(2, 1, 1, 3), nrow = 2)  # matriz de covariância

# Manhattan
d_manhattan <- sum(abs(x - y))
cat("Manhattan:", d_manhattan, "\n")  # esperado: 7

# Euclidiana
d_euclidiana <- sqrt(sum((x - y)^2))
cat("Euclidiana:", round(d_euclidiana, 4), "\n")  # esperado: 5

# Mahalanobis (manual)
diff_vec <- x - y
S_inv    <- solve(S)
d_maha   <- sqrt(t(diff_vec) %*% S_inv %*% diff_vec)
cat("Mahalanobis:", round(d_maha, 4), "\n")  # esperado: ~2.65

# Mahalanobis via função nativa do R
# mahalanobis() calcula distâncias de PONTOS ao centroide da distribuição
set.seed(42)
dados_ex <- data.frame(
  x1 = rnorm(50, mean = 5, sd = 2),
  x2 = rnorm(50, mean = 10, sd = 3)
)
centro    <- colMeans(dados_ex)
cov_dados <- cov(dados_ex)

d_mahal   <- mahalanobis(dados_ex, center = centro, cov = cov_dados)

# Distâncias elevadas = possíveis outliers multivariados
hist(d_mahal, main = "Distâncias de Mahalanobis",
     xlab = "D²", col = "steelblue", breaks = 15)
# Sob normalidade, D² ~ Qui-quadrado(p)
abline(v = qchisq(0.975, df = 2), col = "red", lty = 2, lwd = 2)
legend("topright", "Limiar 97.5% (χ²)", col = "red", lty = 2, lwd = 2)
```

---

## 📊 Ferramentas Gráficas da Análise Exploratória Multivariada

> **Regra de ouro do professor:** *"Use mais de um gráfico, mas que tragam informações distintas. Não use gráficos diferentes para mostrar a mesma coisa — isso é encher espaço, não informar."*

### Guia de Uso: Qual Gráfico para Qual Situação?

| Gráfico | Objetivo | Nº de variáveis |
|---|---|---|
| Histograma / Densidade | Distribuição marginal de uma variável | p = 1 |
| Dispersão simples | Relação entre duas variáveis | p = 2 |
| Boxplots paralelos | Comparar distribuições entre grupos | p ≥ 2 |
| Matriz de dispersão | Ver todos os pares simultaneamente | 2 ≤ p ≤ 6 |
| Heatmap de correlação | Estrutura de dependência global | p ≥ 3 |
| Perfis médios | Comparação multivariada de grupos | p ≥ 3 |
| Coordenadas paralelas | Padrões com muitas variáveis | **p ≥ 4** |

```r
# Instalando pacotes necessários (execute uma vez)
# install.packages(c("GGally", "reshape2", "MASS", "ggplot2"))

library(ggplot2)
library(GGally)
library(reshape2)
library(MASS)

# Gerando dados de exemplo: 2 grupos, 4 variáveis
set.seed(123)
n <- 60
dados <- data.frame(
  Altura = c(rnorm(n/2, 170, 8),  rnorm(n/2, 165, 7)),
  Peso   = c(rnorm(n/2, 75,  12), rnorm(n/2, 60,  10)),
  Idade  = c(rnorm(n/2, 40,  10), rnorm(n/2, 30,  8)),
  Renda  = c(rnorm(n/2, 8000, 2000), rnorm(n/2, 5000, 1500)),
  Grupo  = rep(c("A", "B"), each = n/2)
)
```

### 1. Histograma + Densidade

```r
# Revela: assimetria, multimodalidade, outliers univariados
ggplot(dados, aes(x = Altura)) +
  geom_histogram(aes(y = after_stat(density)),
                 bins = 12, fill = "gray60", color = "white", alpha = 0.8) +
  geom_density(color = "darkorange", linewidth = 1) +
  labs(title = "Distribuição de Altura", x = "Altura (cm)", y = "Densidade") +
  theme_minimal()
```

### 2. Boxplots Paralelos

```r
# Compara distribuições por grupo — ótimo para ver diferenças entre grupos
dados_long <- melt(dados[, c("Altura","Peso","Idade","Renda","Grupo")],
                   id.vars = "Grupo")

ggplot(dados_long, aes(x = variable, y = value, fill = Grupo)) +
  geom_boxplot(alpha = 0.7, outlier.size = 1.5) +
  scale_fill_manual(values = c("A" = "darkorange", "B" = "gray60")) +
  labs(title = "Boxplots Paralelos por Grupo", x = NULL, y = "Valor") +
  theme_minimal()
```

### 3. Matriz de Dispersão (scatterplot matrix)

```r
# Mostra todos os pares de variáveis ao mesmo tempo
# Na diagonal: distribuição de cada variável
# No triângulo superior: correlação numérica
# No triângulo inferior: gráfico de dispersão
ggpairs(
  dados[, c("Altura","Peso","Idade","Renda")],
  lower = list(continuous = wrap("points", alpha = 0.4, size = 1.2, color = "gray40")),
  upper = list(continuous = wrap("cor",    size = 3.5)),
  diag  = list(continuous = wrap("densityDiag", fill = "gray70", alpha = 0.6)),
  title = "Matriz de Dispersão"
) + theme_minimal()
```

### 4. Heatmap de Correlação

> **Professor em aula:** *"Laranja indica correlação positiva forte. Azul indica correlação negativa. De cara, imediato, você já enxerga: X2 e X1 têm correlação 0.85 — as demais não me importam porque a correlação é muito baixa."*

```r
# Muito mais informativo que olhar uma matriz de números crua
R_mat  <- cor(dados[, c("Altura","Peso","Idade","Renda")])
R_long <- melt(R_mat)

ggplot(R_long, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(value, 2)), size = 3.5, fontface = "bold") +
  scale_fill_gradient2(
    low      = "steelblue",   # correlação negativa
    mid      = "white",       # sem correlação
    high     = "darkorange",  # correlação positiva
    midpoint = 0,
    limits   = c(-1, 1),
    name     = "Correlação"
  ) +
  labs(title = "Heatmap de Correlação") +
  theme_minimal() +
  theme(axis.title = element_blank())
```

### 5. Gráfico de Perfis Médios

> **Analogia:** Imagine dois times de futebol (Grupo A e Grupo B). O gráfico de perfis médios é como comparar o desempenho médio de cada time em várias categorias (velocidade, força, técnica, resistência) ao mesmo tempo — conectando os pontos para ver qual time se destaca em quê.

> **Conexão com MANOVA:** este gráfico é a "prévia visual" do teste MANOVA — que formalmente vai testar se os vetores de médias dos grupos são estatisticamente diferentes.

```r
# Padronizando para colocar todas as variáveis na mesma escala
dados_pad <- dados
dados_pad[, c("Altura","Peso","Idade","Renda")] <-
  scale(dados[, c("Altura","Peso","Idade","Renda")])

medias_grupo <- aggregate(. ~ Grupo,
                          data = dados_pad[, c("Altura","Peso","Idade","Renda","Grupo")],
                          FUN  = mean)

medias_long <- melt(medias_grupo, id.vars = "Grupo")

ggplot(medias_long, aes(x = variable, y = value, color = Grupo, group = Grupo)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  scale_color_manual(values = c("A" = "darkorange", "B" = "gray50")) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray70") +
  labs(title    = "Perfis Médios por Grupo (variáveis padronizadas)",
       x = NULL, y = "Média padronizada") +
  theme_minimal()
```

### 6. Coordenadas Paralelas

> **Quando usar:** quando você tem **mais de 4 variáveis** e a matriz de dispersão fica grande demais. Cada linha é uma observação; cada eixo vertical é uma variável. Linhas que se cruzam entre dois eixos = correlação negativa. Linhas paralelas = correlação positiva.

```r
# Regra do professor:
# p <= 3 variáveis → use gráfico de perfis médios
# p >= 4 variáveis → use coordenadas paralelas

cores <- ifelse(dados$Grupo == "A", "darkorange", "gray60")

parcoord(
  dados[, c("Altura","Peso","Idade","Renda")],
  col  = adjustcolor(cores, alpha.f = 0.4),  # transparência para ver sobreposição
  lty  = 1,
  main = "Coordenadas Paralelas"
)
legend("topright",
       legend = c("Grupo A", "Grupo B"),
       col    = c("darkorange", "gray60"),
       lty = 1, lwd = 2, bty = "n")
```

---

## 💡 Insights do Professor (da transcrição)

> *"Nem tudo se resume a rede neural, IA ou LLM. Isso é uma doença. Fico indignado."*

- Dominar álgebra linear + análise exploratória é o que diferencia quem entende estatística de quem apenas roda modelos
- Use gráficos distintos no mesmo relatório — cada um revela um aspecto diferente dos dados
- O heatmap de correlação é muito mais eficiente para comunicar com clientes do que uma matriz de números no console do R
- Gráfico de perfis médios é a base visual para testes como MANOVA — que veremos na próxima aula

---

## 🔮 Próxima Aula — O que vem pela frente

Na **Aula 3**, o professor prometeu:

- **Distribuições multivariadas:** distribuição Normal Multivariada (generalização da curva normal para vetores)
- **T² de Hotelling:** a versão multivariada do teste t de Student — compara vetores de médias
- **MANOVA:** comparação de médias entre grupos com múltiplas variáveis simultaneamente

> *"A gente começa a falar sobre diferença. Não é fácil, mas se você dominar essa ideia, você domina qualquer coisa."*

---

## 📚 Referências

- Johnson & Wichern — *Applied Multivariate Statistical Analysis*, Cap. 1-4
- Hair et al. — *Multivariate Data Analysis*, Cap. 2 (Análise Exploratória)
- Documentação do pacote `GGally`: https://ggobi.github.io/ggally/
- Documentação do pacote `MASS` (função `parcoord`): https://cran.r-project.org/package=MASS
