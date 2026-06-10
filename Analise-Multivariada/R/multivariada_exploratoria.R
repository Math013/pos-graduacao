# ==============================================================================
# Análise Exploratória Multivariada
# Prof. Vinícius Osterne, PhD
# ==============================================================================

# Pacotes necessário

library(ggplot2)
library(GGally)
library(reshape2)
library(MASS)

# ==============================================================================
# 1. VETOR DE MÉDIAS
# ==============================================================================

# Dados: n=4 indivíduos, p=3 variáveis (Altura, Peso, Idade)
X <- matrix(
  c(170, 65, 25,
    182, 80, 30,
    158, 55, 22,
    174, 72, 28),
  nrow = 4, byrow = TRUE,
  dimnames = list(
    paste("Indivíduo", 1:4),
    c("Altura", "Peso", "Idade")
  )
)

# Vetor de médias amostral
x_barra <- colMeans(X)
x_barra

# Equivalente em notação matricial: (1/n) * t(X) %*% 1_n
n <- nrow(X)
uns <- rep(1, n)
x_barra_matricial <- (1/n) * t(X) %*% uns
x_barra_matricial


# ==============================================================================
# 2. MATRIZ DE VARIÂNCIAS E COVARIÂNCIAS
# ==============================================================================

# Dados: n=4 indivíduos, p=2 variáveis (Altura, Peso)
X2 <- X[, c("Altura", "Peso")]

# Matriz de covariâncias amostral (divisor n-1)
S <- cov(X2)
S

# Calculando manualmente cada elemento
X2_centrado <- sweep(X2, 2, colMeans(X2))   # matriz centrada

s11 <- sum(X2_centrado[, 1]^2) / (n - 1)
s22 <- sum(X2_centrado[, 2]^2) / (n - 1)
s12 <- sum(X2_centrado[, 1] * X2_centrado[, 2]) / (n - 1)

round(s11, 1) # variância de Altura
round(s22, 1) # variância de Peso
round(s12, 1) # covariância entre Altura e Peso

# Em notação matricial: S = (1/(n-1)) * t(X_til) %*% X_til
S_matricial <- (1/(n-1)) * t(X2_centrado) %*% X2_centrado
S_matricial



# ==============================================================================
# 3. MATRIZ DE CORRELAÇÃO
# ==============================================================================

R <- cor(X2)
R

# Calculando manualmente
s1 <- sqrt(s11)
s2 <- sqrt(s22)
r12 <- s12 / (s1 * s2)

round(s1, 2)  # desvio-padrão de Altura
round(s2, 2)  # desvio-padrão de Peso
round(r12, 3) # correlação entre Altura e Peso

# Relação entre S e R via matriz diagonal de desvios-padrão
D    <- diag(c(s1, s2))
D_inv <- solve(D)

R_via_formula <- D_inv %*% S %*% D_inv
R_via_formula

# Inversa: recuperando S a partir de R e D
S_recuperada <- D %*% R %*% D
S_recuperada


# ==============================================================================
# 4. DISTÂNCIAS MULTIVARIADAS
# ==============================================================================

x <- c(1, 2)
y <- c(4, 6)

# Distância de Manhattan
d_manhattan <- sum(abs(x - y))
d_manhattan            # distância de Manhattan

# Distância Euclidiana
d_euclidiana <- sqrt(sum((x - y)^2))
d_euclidiana # distância Euclidiana

# Distância de Mahalanobis
Sigma     <- matrix(c(2, 1, 1, 3), nrow = 2)
Sigma_inv <- solve(Sigma) #inversa da matriz de covariancia

dif          <- x - y
d_mahal_quad <- t(dif) %*% Sigma_inv %*% dif
d_mahal      <- sqrt(d_mahal_quad)
round(d_mahal, 2)      # distância de Mahalanobis


# ==============================================================================
# 5. ANÁLISES GRÁFICAS
# ==============================================================================

# Base de dados um pouco maior para os gráficos ficarem mais interessantes
set.seed(42)
n_graf <- 80

altura <- rnorm(n_graf, mean = 171, sd = 10)
peso   <- 0.8 * altura - 65 + rnorm(n_graf, sd = 4)
idade  <- rnorm(n_graf, mean = 35, sd = 8)
renda  <- -0.4 * idade + rnorm(n_graf, mean = 60, sd = 12)
grupo  <- factor(rep(c("A", "B"), each = n_graf / 2))

dados <- data.frame(Altura = altura, Peso = peso,
                    Idade  = idade,  Renda = renda,
                    Grupo  = grupo)

# ------------------------------------------------------------------------------
# 5.1 Gráfico de Dispersão
# ------------------------------------------------------------------------------
ggplot(dados, aes(x = Altura, y = Peso)) +
  geom_point(color = "gray50", alpha = 0.7, size = 2) +
  geom_smooth(method = "lm", se = FALSE, color = "darkorange", linewidth = 0.8) +
  labs(title = "Dispersão: Altura × Peso",
       x = "Altura (cm)", y = "Peso (kg)") +
  theme_minimal(base_size = 13)



# ------------------------------------------------------------------------------
# 5.2 Histograma com curva de densidade
# ------------------------------------------------------------------------------

ggplot(dados, aes(x = Altura)) +
  geom_histogram(aes(y = after_stat(density)),
                 bins = 12, fill = "gray60", color = "white", alpha = 0.8) +
  geom_density(color = "darkorange", linewidth = 1) +
  labs(title = "Distribuição marginal: Altura",
       x = "Altura (cm)", y = "Densidade") +
  theme_minimal(base_size = 13)


# ------------------------------------------------------------------------------
# 5.3 Matriz de Dispersão (scatterplot matrix)
# ------------------------------------------------------------------------------

ggpairs(
  dados[, c("Altura", "Peso", "Idade", "Renda")],
  lower = list(continuous = wrap("points", alpha = 0.4, size = 1.2, color = "gray40")),
  upper = list(continuous = wrap("cor", size = 3.5)),
  diag  = list(continuous = wrap("densityDiag", fill = "gray70", alpha = 0.6)),
  title = "Matriz de Dispersão"
) +
  theme_minimal(base_size = 11)

# ------------------------------------------------------------------------------
# 5.4 Boxplots Paralelos
# ------------------------------------------------------------------------------

dados_long <- reshape2::melt(
  dados[, c("Altura", "Peso", "Idade", "Renda", "Grupo")],
  id.vars = "Grupo"
)

ggplot(dados_long, aes(x = variable, y = value, fill = Grupo)) +
  geom_boxplot(alpha = 0.7, outlier.size = 1.5) +
  scale_fill_manual(values = c("A" = "darkorange", "B" = "gray60")) +
  labs(title = "Boxplots Paralelos por Grupo",
       x = NULL, y = "Valor") +
  theme_minimal(base_size = 13)

# ------------------------------------------------------------------------------
# 5.5 Heatmap de Correlação
# ------------------------------------------------------------------------------

R_graf <- cor(dados[, c("Altura", "Peso", "Idade", "Renda")])
R_long  <- reshape2::melt(R_graf)

ggplot(R_long, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(value, 2)), size = 3.5, fontface = "bold") +
  scale_fill_gradient2(low = "steelblue", mid = "white", high = "darkorange",
                       midpoint = 0, limits = c(-1, 1),
                       name = "Correlação") +
  labs(title = "Heatmap de Correlação") +
  theme_minimal(base_size = 13) +
  theme(axis.title = element_blank())

# ------------------------------------------------------------------------------
# 5.6 Gráfico de Perfis Médios
# ------------------------------------------------------------------------------

# Médias por grupo (variáveis padronizadas para comparação na mesma escala)
dados_pad <- dados
dados_pad[, c("Altura","Peso","Idade","Renda")] <-
  scale(dados[, c("Altura","Peso","Idade","Renda")])

medias_grupo <- aggregate(. ~ Grupo,
                          data  = dados_pad[, c("Altura","Peso","Idade","Renda","Grupo")],
                          FUN   = mean)

medias_long <- reshape2::melt(medias_grupo, id.vars = "Grupo")

ggplot(medias_long, aes(x = variable, y = value,
                        color = Grupo, group = Grupo)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  scale_color_manual(values = c("A" = "darkorange", "B" = "gray50")) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray70") +
  labs(title = "Perfis Médios por Grupo (variáveis padronizadas)",
       x = NULL, y = "Média padronizada") +
  theme_minimal(base_size = 13)

# ------------------------------------------------------------------------------
# 5.7 Coordenadas Paralelas
# ------------------------------------------------------------------------------

# MASS::parcoord para uma versão rápida
cores <- ifelse(dados$Grupo == "A", "darkorange", "gray60")

parcoord(
  dados[, c("Altura", "Peso", "Idade", "Renda")],
  col = adjustcolor(cores, alpha.f = 0.4),
  lty = 1,
  main = "Coordenadas Paralelas"
)
legend("topright", legend = c("Grupo A", "Grupo B"),
       col = c("darkorange", "gray60"), lty = 1, lwd = 2, bty = "n")














  
  
  
  
  

