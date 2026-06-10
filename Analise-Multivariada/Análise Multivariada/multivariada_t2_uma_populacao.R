# ============================================================
# Inferencia Multivariada — Metodo 1
# T2 de Hotelling: uma populacao
# Exemplo REAL: perfil metabolico vs valores de referencia
# Fonte: Reaven & Miller (1979) via pacote mclust
#        dataset "diabetes" — 145 individuos nao diabeticos
# Variaveis: glicose, insulina e SSPG (resistencia insulinica)
# Prof. Vinicius Osterne
# ============================================================

# install.packages(c("mclust", "ggplot2", "MVN", "DescTools"))

library(mclust)
library(ggplot2)
library(MVN)
library(DescTools)

# ------------------------------------------------------------
# Dados reais: subconjunto de individuos normais (classe "Normal")
# Reaven & Miller (1979), Diabetologia 16: 17-24
# ------------------------------------------------------------

data(diabetes)

dados_raw <- subset(diabetes, class == "Normal")[, c("glucose", "insulin", "sspg")]
dados     <- as.matrix(dados_raw)
colnames(dados) <- c("glicose", "insulina", "sspg")
head(dados)

n <- nrow(dados)   # 76 individuos normais
p <- ncol(dados)

# Valor de referencia clinico (hipotese nula)
# glicose em jejum: 90 mg/dL | insulina basal: 15 uU/mL | sspg: 120 mg/dL
mu0 <- c(90, 15, 120)


# Estimativas amostrais
Xbarra <- colMeans(dados)
S      <- cov(dados)

Xbarra                 # vetor de medias amostrais
S                      # matriz de covariancias amostral
round(cor(dados), 3)   # correlacoes entre as variaveis

# ------------------------------------------------------------
# Estatistica T2 de Hotelling (manual)
# H0: mu = mu0  (perfil igual ao de referencia clinica)
# H1: mu != mu0
# ------------------------------------------------------------

diff_vec <- Xbarra - mu0
T2_obs   <- as.numeric(n * t(diff_vec) %*% solve(S) %*% diff_vec)

fator    <- ((n - 1) * p) / (n - p)
F_obs    <- T2_obs / fator
pvalor   <- pf(F_obs, df1 = p, df2 = n - p, lower.tail = FALSE)

T2_obs   # estatistica T2 observada
F_obs    # estatistica F equivalente
pvalor   # p-valor

# Valor critico a 5%
T2_critico <- fator * qf(0.05, df1 = p, df2 = n - p, lower.tail = FALSE)
T2_critico # rejeita H0 se T2_obs > T2_critico

# ------------------------------------------------------------
# Usando pacotes
# ------------------------------------------------------------

# Testando normalidade multivariada
result <- mvn(dados)
result

# Confirmacao via DescTools
HotellingsT2Test(as.data.frame(dados), mu = mu0)

# ------------------------------------------------------------
# Diagnostico: Q-Q plot de Mahalanobis
# ------------------------------------------------------------

S_inv <- solve(S)
d2 <- apply(dados, 1, function(x) {
  v <- x - Xbarra
  as.numeric(t(v) %*% S_inv %*% v)
})

df_qq <- data.frame(
  theoretical = qchisq(ppoints(n), df = p),
  observed    = sort(d2)
)

ggplot(df_qq, aes(x = theoretical, y = observed)) +
  geom_point(color = "#E87722", alpha = 0.7, size = 2) +
  geom_abline(slope = 1, intercept = 0,
              color = "white", linewidth = 0.8, linetype = "dashed") +
  labs(
    title    = "Q-Q plot de distancias de Mahalanobis",
    subtitle = "Pontos proximos a linha indicam normalidade multivariada",
    x        = expression("Quantis " ~ chi[p]^2),
    y        = "Distancias de Mahalanobis observadas"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.background  = element_rect(fill = "#1a1a1a", color = NA),
    panel.background = element_rect(fill = "#1a1a1a", color = NA),
    panel.grid       = element_line(color = "#2e2e2e"),
    text             = element_text(color = "gray80"),
    axis.text        = element_text(color = "gray60"),
    plot.title       = element_text(color = "gray95", face = "bold"),
    plot.subtitle    = element_text(color = "gray60", size = 9)
  )

# ------------------------------------------------------------
# Visualizacao: regiao de confianca elipsoidal (projecao p = 2)
# glicose x insulina
# ------------------------------------------------------------

theta_seq <- seq(0, 2*pi, length.out = 300)
Xbarra_2  <- Xbarra[1:2]
S_2       <- S[1:2, 1:2]
mu0_2     <- mu0[1:2]

eigen_S <- eigen(S_2)
raio    <- sqrt(((n - 1) * 2 / (n - 2)) * qf(0.95, 2, n - 2))
eixos   <- raio * eigen_S$vectors %*% diag(sqrt(eigen_S$values))

elipse <- t(apply(cbind(cos(theta_seq), sin(theta_seq)), 1,
                  function(v) Xbarra_2 + eixos %*% v))
df_elipse <- data.frame(x = elipse[, 1], y = elipse[, 2])

ggplot() +
  geom_path(data = df_elipse, aes(x = x, y = y),
            color = "orange", linewidth = 1.0) +
  geom_point(aes(x = Xbarra_2[1], y = Xbarra_2[2]),
             color = "orange", size = 3) +
  geom_point(aes(x = mu0_2[1], y = mu0_2[2]),
             color = "white", size = 3, shape = 4, stroke = 1.5) +
  annotate("text", x = mu0_2[1] + 1.5, y = mu0_2[2],
           label = expression(mu[0]), color = "gray70", size = 3.5) +
  annotate("text", x = Xbarra_2[1] + 1.5, y = Xbarra_2[2],
           label = expression(bar(X)), color = "orange", size = 3.5) +
  labs(
    title    = "Regiao de confianca 95% para mu (glicose x insulina)",
    subtitle = "Elipse laranja = regiao de confianca | Cruz branca = mu0 clinico",
    x        = "Glicose (mg/dL)",
    y        = "Insulina (uU/mL)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.background  = element_rect(fill = "#1a1a1a", color = NA),
    panel.background = element_rect(fill = "#1a1a1a", color = NA),
    panel.grid       = element_line(color = "#2e2e2e"),
    text             = element_text(color = "gray80"),
    axis.text        = element_text(color = "gray60"),
    plot.title       = element_text(color = "gray95", face = "bold", size = 10),
    plot.subtitle    = element_text(color = "gray60", size = 8)
  )