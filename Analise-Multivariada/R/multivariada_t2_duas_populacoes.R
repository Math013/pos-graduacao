# ============================================================
# Inferencia Multivariada — Metodo 2
# T2 de Hotelling: duas populacoes independentes
# Exemplo REAL: perfil metabolico de normais vs pre-diabeticos
# Fonte: Reaven & Miller (1979) via pacote mclust
#        dataset "diabetes" — grupos Normal e Chemical
# Variaveis: glicose, insulina e SSPG
# Prof. Vinicius Osterne
# ============================================================

install.packages(c("mclust", "ggplot2", "Hotelling", "biotools", "tidyr", "dplyr"))

library(mclust)
library(ggplot2)
library(Hotelling)
library(biotools)
library(tidyr)
library(dplyr)

# ------------------------------------------------------------
# Dados reais: Normal vs Chemical (pre-diabetico)
# Reaven & Miller (1979), Diabetologia 16: 17-24
# ------------------------------------------------------------

data(diabetes)

vars <- c("glucose", "insulin", "sspg")

dados_g1 <- as.matrix(subset(diabetes, class == "Normal")[, vars])
dados_g2 <- as.matrix(subset(diabetes, class == "Chemical")[, vars])

head(dados_g1)
head(dados_g2)

colnames(dados_g1) <- colnames(dados_g2) <- c("glicose", "insulina", "sspg")

n1 <- nrow(dados_g1)   # 76 individuos normais
n2 <- nrow(dados_g2)   # 36 pre-diabeticos
p2 <- ncol(dados_g1)

# Estimativas por grupo
Xbarra1 <- colMeans(dados_g1)
Xbarra2 <- colMeans(dados_g2)
S1      <- cov(dados_g1)
S2      <- cov(dados_g2)

Xbarra1                          # medias grupo Normal
Xbarra2                          # medias grupo Chemical
round(Xbarra1 - Xbarra2, 2)      # diferenca observada entre grupos

# Matriz de covariancias pooled
S_pool <- ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)

S_pool                       # matriz pooled
round(cov2cor(S_pool), 3)    # correlacoes pooled

# ------------------------------------------------------------
# Estatistica T2 (manual)
# H0: mu1 = mu2  (perfis iguais entre os grupos)
# H1: mu1 != mu2
# ------------------------------------------------------------

delta   <- Xbarra1 - Xbarra2
T2_obs  <- as.numeric(
  t(delta) %*% solve((1/n1 + 1/n2) * S_pool) %*% delta
)

gl_den  <- n1 + n2 - p2 - 1
fator   <- ((n1 + n2 - 2) * p2) / gl_den
F_obs   <- T2_obs / fator
pvalor  <- pf(F_obs, df1 = p2, df2 = gl_den, lower.tail = FALSE)

T2_obs  # estatistica T2
F_obs   # estatistica F equivalente
pvalor  # p-valor

# Valor critico a 5%
T2_crit <- fator * qf(0.05, df1 = p2, df2 = gl_den, lower.tail = FALSE)
T2_crit # rejeita H0 se T2_obs > T2_crit

# Confirmacao via pacote Hotelling
result = hotelling.test(
  as.data.frame(dados_g1),
  as.data.frame(dados_g2)
)
result


# ------------------------------------------------------------
# Diagnostico: teste de Box para homocedasticidade
# ------------------------------------------------------------

dados_todos <- rbind(
  data.frame(dados_g1, grupo = "normal"),
  data.frame(dados_g2, grupo = "chemical")
)

box_test <- boxM(dados_todos[, 1:3], dados_todos$grupo)
box_test  # H0: Sigma1 = Sigma2

# ------------------------------------------------------------
# Visualizacao: distribuicoes marginais por grupo
# ------------------------------------------------------------

df_long <- dados_todos |>
  pivot_longer(cols = 1:3, names_to = "variavel", values_to = "valor")

ggplot(df_long, aes(x = valor, fill = grupo)) +
  geom_density(alpha = 0.5, color = NA) +
  scale_fill_manual(values = c(
    "chemical" = "#E87722",
    "normal"   = "gray60"
  )) +
  facet_wrap(~variavel, scales = "free") +
  labs(
    title    = "Distribuicao das variaveis por grupo",
    subtitle = "Laranja = pre-diabeticos (Chemical) | Cinza = normais",
    x        = "Valor", y = "Densidade", fill = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.background  = element_rect(fill = "#1a1a1a", color = NA),
    panel.background = element_rect(fill = "#1a1a1a", color = NA),
    panel.grid       = element_line(color = "#2e2e2e"),
    text             = element_text(color = "gray80"),
    strip.text       = element_text(color = "gray90", face = "bold"),
    axis.text        = element_text(color = "gray60"),
    legend.position  = "bottom",
    plot.title       = element_text(color = "gray95", face = "bold"),
    plot.subtitle    = element_text(color = "gray60", size = 9)
  )

# ------------------------------------------------------------
# Visualizacao: regiao de confianca para a diferenca (projecao p = 2)
# glicose x insulina
# ------------------------------------------------------------

theta_seq <- seq(0, 2*pi, length.out = 300)
delta_2   <- delta[1:2]
S_pool_2  <- S_pool[1:2, 1:2]
fator_c   <- 1/n1 + 1/n2

eigen_pool <- eigen(fator_c * S_pool_2)
raio2      <- sqrt(fator * qf(0.95, p2, gl_den))
eixos2     <- raio2 * eigen_pool$vectors %*%
  diag(sqrt(eigen_pool$values))

elipse2 <- t(apply(cbind(cos(theta_seq), sin(theta_seq)), 1,
                   function(v) delta_2 + eixos2 %*% v))
df_elipse2 <- data.frame(x = elipse2[, 1], y = elipse2[, 2])

ggplot() +
  geom_path(data = df_elipse2, aes(x = x, y = y),
            color = "orange", linewidth = 1.0) +
  geom_point(aes(x = delta_2[1], y = delta_2[2]),
             color = "orange", size = 3) +
  geom_point(aes(x = 0, y = 0),
             color = "white", size = 3.5, shape = 4, stroke = 1.5) +
  annotate("text", x = 1.5, y = -3,
           label = "delta[0] == 0",
           color = "gray70", size = 3.5, parse = TRUE) +
  labs(
    title    = "Regiao de confianca 95% para mu1 - mu2",
    subtitle = "Cruz branca = delta0 = 0 | Ponto laranja = diferenca observada",
    x        = "Diferenca em glicose (mg/dL)",
    y        = "Diferenca em insulina (uU/mL)"
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