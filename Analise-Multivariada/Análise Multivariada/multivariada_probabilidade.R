# ==============================================================================
# Probabilidade Multivariada
# Prof. Vinícius Osterne, PhD
# ==============================================================================

library(ggplot2)
library(mvtnorm)   # dnorm multivariada e simulações
library(reshape2)
library(MASS)


# ==============================================================================
# 1. DISTRIBUIÇÃO NORMAL UNIVARIADA: REVISÃO
# ==============================================================================

# Curvas normais com diferentes parâmetros
x <- seq(-6, 10, length.out = 500)

dnorm_df <- data.frame(
  x    = rep(x, 3),
  y    = c(dnorm(x, 0, 1), dnorm(x, 2, 1), dnorm(x, 0, 2)),
  curva = rep(c("mu=0, sigma=1", "mu=2, sigma=1", "mu=0, sigma=2"), each = 500)
)

ggplot(dnorm_df, aes(x = x, y = y, color = curva, linetype = curva)) +
  geom_line(linewidth = 0.9) +
  scale_color_manual(values = c("steelblue", "darkorange", "gray60")) +
  scale_linetype_manual(values = c("solid", "dashed", "dotted")) +
  labs(title = "Curvas da Distribuição Normal Univariada",
       x = "x", y = "f(x)", color = NULL, linetype = NULL) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "top")


# ==============================================================================
# 2. DISTRIBUIÇÃO NORMAL MULTIVARIADA
# ==============================================================================

# Parâmetros da normal bivariada
mu    <- c(0, 0)
Sigma <- matrix(c(1, 0.6, 0.6, 1), nrow = 2)

# Grade de pontos no plano (X1, X2)
grid <- expand.grid(
  x1 = seq(-3, 3, length.out = 80),
  x2 = seq(-3, 3, length.out = 80)
)
grid$densidade <- dmvnorm(as.matrix(grid), mean = mu, sigma = Sigma)

# ------------------------------------------------------------------------------
# 2.1 Superfície 3D (via base R — persp)
# ------------------------------------------------------------------------------

z_mat <- matrix(grid$densidade, nrow = 80)

persp(
  x    = seq(-3, 3, length.out = 80),
  y    = seq(-3, 3, length.out = 80),
  z    = z_mat,
  theta = 40, phi = 30,
  col   = "orange",
  shade = 0.4,
  border = NA,
  xlab = "X1", ylab = "X2", zlab = "f",
  main = "Superfície da Normal Bivariada"
)

# ------------------------------------------------------------------------------
# 2.2 Curvas de nível
# ------------------------------------------------------------------------------

ggplot(grid, aes(x = x1, y = x2, z = densidade)) +
  geom_contour_filled(bins = 8, alpha = 0.85) +
  scale_fill_manual(
    values = colorRampPalette(c("white", "orange", "darkorange4"))(8)
  ) +
  geom_point(aes(x = 0, y = 0), color = "white", size = 3) +
  labs(title = "Curvas de Nível — Normal Bivariada (rho = 0.6)",
       x = expression(X[1]), y = expression(X[2]), fill = "Densidade") +
  coord_fixed() +
  theme_minimal(base_size = 13)


# ==============================================================================
# 3. PROPRIEDADES DA DNM
# ==============================================================================

set.seed(7)
n <- 500

# Simulando X ~ N3(mu, Sigma) para ilustrar as três propriedades
mu3    <- c(2, 5, 8)
Sigma3 <- matrix(c(4, 2, 0,
                   2, 9, 0,
                   0, 0, 3), nrow = 3)

X3 <- mvrnorm(n, mu = mu3, Sigma = Sigma3)
colnames(X3) <- c("X1", "X2", "X3")

# ------------------------------------------------------------------------------
# Propriedade 1: Fechada sob marginalização
# A marginal de (X1, X2) deve ser N2(mu[1:2], Sigma[1:2, 1:2])
# ------------------------------------------------------------------------------

# Marginal amostral de X1 e X2
marginal_12 <- as.data.frame(X3[, c("X1", "X2")])

# Grade para a densidade teórica marginal
grid_12 <- expand.grid(
  X1 = seq(-3, 8,  length.out = 60),
  X2 = seq(-2, 14, length.out = 60)
)
grid_12$dens <- dmvnorm(
  as.matrix(grid_12),
  mean  = mu3[1:2],
  sigma = Sigma3[1:2, 1:2]
)

ggplot() +
  geom_point(data = marginal_12, aes(x = X1, y = X2),
             color = "gray50", alpha = 0.3, size = 1) +
  geom_contour(data = grid_12, aes(x = X1, y = X2, z = dens),
               color = "darkorange", bins = 6, linewidth = 0.8) +
  labs(title = "Propriedade 1: Marginal (X1, X2) ainda é Normal",
       subtitle = "Curvas de nível teóricas sobrepostas aos pontos simulados",
       x = expression(X[1]), y = expression(X[2])) +
  theme_minimal(base_size = 13)

# ------------------------------------------------------------------------------
# Propriedade 2: Independência equivale a correlação zero na DNM
# X3 foi construída independente de X1 e X2 (Sigma[1:2, 3] = 0)
# ------------------------------------------------------------------------------

cor(X3)   # deve mostrar ~0 nas entradas (1,3) e (2,3)

# Visualmente: dispersão X1 vs X3 (sem padrão) × X1 vs X2 (com padrão)
par(mfrow = c(1, 2))

plot(X3[, "X1"], X3[, "X3"],
     pch = 16, col = adjustcolor("gray50", 0.4), cex = 0.8,
     xlab = "X1", ylab = "X3",
     main = "X1 vs X3  (cov = 0, independentes)")

plot(X3[, "X1"], X3[, "X2"],
     pch = 16, col = adjustcolor("darkorange", 0.4), cex = 0.8,
     xlab = "X1", ylab = "X2",
     main = "X1 vs X2  (cov ≠ 0, dependentes)")

par(mfrow = c(1, 1))

# ------------------------------------------------------------------------------
# Propriedade 3: Fechada sob transformações lineares
# Y = A %*% X ~ N_q(A*mu, A*Sigma*t(A))
# ------------------------------------------------------------------------------

# Transformação: soma e diferença de X1 e X2
A <- matrix(c(1, 1, 0,
              1,-1, 0), nrow = 2, byrow = TRUE)

Y <- t(A %*% t(X3))   # cada linha de X3 é um vetor
colnames(Y) <- c("X1+X2", "X1-X2")

mu_Y    <- A %*% mu3
Sigma_Y <- A %*% Sigma3 %*% t(A)

mu_Y    # média teórica de Y
Sigma_Y # covariância teórica de Y

colMeans(Y)  # deve ser próximo de mu_Y
cov(Y)       # deve ser próximo de Sigma_Y


# ==============================================================================
# 4. DISTRIBUIÇÃO t DE STUDENT: REVISÃO
# ==============================================================================

x_t <- seq(-5, 5, length.out = 500)

t_df <- data.frame(
  x   = rep(x_t, 4),
  y   = c(dt(x_t, 1), dt(x_t, 5), dt(x_t, 30), dnorm(x_t)),
  dist = rep(c("t (nu=1, Cauchy)", "t (nu=5)", "t (nu=30)", "Normal"), each = 500)
)

ggplot(t_df, aes(x = x, y = y, color = dist, linetype = dist)) +
  geom_line(linewidth = 0.9) +
  scale_color_manual(values = c("darkorange", "gray60", "steelblue", "black")) +
  scale_linetype_manual(values = c("dashed", "solid", "dotted", "longdash")) +
  labs(title = "Distribuição t de Student vs. Normal",
       x = "x", y = "f(x)", color = NULL, linetype = NULL) +
  coord_cartesian(ylim = c(0, 0.42)) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "top")

# Caudas mais pesadas: probabilidade além de 2 para t_5 vs Normal
pt(-2, df = 5)   # P(t_5 < -2)
pnorm(-2)        # P(Z  < -2)


# ==============================================================================
# 5. ESTATÍSTICA T² DE HOTELLING: DEFINIÇÃO E GEOMETRIA
# ==============================================================================

# No univariado: t² = n * (x_bar - mu0)² / s²
# No multivariado: T² = n * (x_bar - mu0)' %*% S^{-1} %*% (x_bar - mu0)
# O miolo é exatamente a distância de Mahalanobis ao quadrado entre x_bar e mu0

# Exemplo com p=2 para visualizar o que T² está medindo
set.seed(42)
n  <- 50
p  <- 2

mu_sim    <- c(3, 6)
Sigma_sim <- matrix(c(4, 2, 2, 3), nrow = 2)

amostra <- mvrnorm(n, mu = mu_sim, Sigma = Sigma_sim)
x_bar   <- colMeans(amostra)
S       <- cov(amostra)

x_bar  # vetor de médias amostral
S      # matriz de covariâncias amostral

# Distância de Mahalanobis entre x_bar e dois candidatos a mu0
mu0_perto <- c(3.2, 6.1)   # próximo da média amostral
mu0_longe <- c(0.0, 0.0)   # longe da média amostral

dif_perto <- x_bar - mu0_perto
dif_longe <- x_bar - mu0_longe

d2_perto <- t(dif_perto) %*% solve(S) %*% dif_perto
d2_longe <- t(dif_longe) %*% solve(S) %*% dif_longe

round(d2_perto, 3) # distância de Mahalanobis² (mu0 perto)
round(d2_longe, 3) # distância de Mahalanobis² (mu0 longe)

# T² correspondentes
n * d2_perto  # T² para mu0 perto — pequeno, pouca evidência contra H0
n * d2_longe  # T² para mu0 longe — grande, forte evidência contra H0

# ------------------------------------------------------------------------------
# Visualização: onde caem os dois mu0 em relação à nuvem
# ------------------------------------------------------------------------------

dados_plot <- as.data.frame(amostra)
colnames(dados_plot) <- c("X1", "X2")

# Elipse de concentração amostral (95%)
elipse <- as.data.frame(
  ellipse::ellipse(S, centre = x_bar, level = 0.95)
)
colnames(elipse) <- c("X1", "X2")

ggplot(dados_plot, aes(x = X1, y = X2)) +
  geom_point(color = "gray50", alpha = 0.4, size = 1.8) +
  geom_path(data = elipse, aes(x = X1, y = X2),
            color = "darkorange", linewidth = 0.9, linetype = "dashed") +
  geom_point(aes(x = x_bar[1], y = x_bar[2]),
             color = "darkorange", size = 4) +
  annotate("text", x = x_bar[1] + 0.15, y = x_bar[2] + 0.2,
           label = "x_bar", color = "darkorange", size = 3.5) +
  geom_point(aes(x = mu0_perto[1], y = mu0_perto[2]),
             color = "steelblue", size = 4, shape = 17) +
  annotate("text", x = mu0_perto[1] + 0.15, y = mu0_perto[2] + 0.2,
           label = "mu0 (perto)", color = "steelblue", size = 3.5) +
  geom_point(aes(x = mu0_longe[1], y = mu0_longe[2]),
             color = "firebrick", size = 4, shape = 17) +
  annotate("text", x = mu0_longe[1] + 0.15, y = mu0_longe[2] + 0.2,
           label = "mu0 (longe)", color = "firebrick", size = 3.5) +
  labs(title = expression("T"^2 * " de Hotelling: o que a estatística mede"),
       subtitle = "Elipse de 95% — mu0 dentro vs. fora da nuvem",
       x = expression(X[1]), y = expression(X[2])) +
  theme_minimal(base_size = 13)

# Relação com a distribuição F (conversão — a ser explorada em Inferência)
# F = ((n - p) / (p * (n - 1))) * T²  ~  F_{p, n-p}
