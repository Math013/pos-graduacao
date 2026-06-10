# ============================================================
# Inferencia Multivariada — Metodo 4
# MANOVA
# Exemplo REAL: morfometria de caranguejos por especie e sexo
# Fonte: Campbell & Mahon (1974) via pacote MASS
#        dataset "crabs" — 200 individuos, 4 grupos
# Grupos: Azul-Macho, Azul-Femea, Laranja-Macho, Laranja-Femea
# Variaveis: FL (lob frontal), RW (largura rear), CL (comp carapaca),
#            CW (larg carapaca), BD (profundidade corpo)
# Prof. Vinicius Osterne
# ============================================================

# install.packages(c("MASS", "ggplot2", "biotools", "MVN", "tidyr"))

library(MASS)
library(ggplot2)
library(biotools)
library(tidyr)

# ------------------------------------------------------------
# Dados reais: dataset crabs
# Campbell & Mahon (1974)
# 200 caranguejos — 4 grupos de 50 individuos cada
# FL = lob frontal (mm)   | RW = largura rear (mm)
# CL = comp carapaca (mm) | CW = larg carapaca (mm)
# BD = profundidade corpo (mm)
# ------------------------------------------------------------

data(crabs)

dados <- data.frame(
  grupo = factor(paste(
    ifelse(crabs$sp  == "B", "Azul",    "Laranja"),
    ifelse(crabs$sex == "M", "Macho",   "Femea")
  )),
  FL = crabs$FL,
  RW = crabs$RW,
  CL = crabs$CL,
  CW = crabs$CW,
  BD = crabs$BD
)

# Visualizando a base
head(dados, 52)        # primeiras 20 linhas
str(dados)             # estrutura
table(dados$grupo)     # tamanho de cada grupo

grupos <- levels(dados$grupo)
N      <- nrow(dados)
g      <- length(grupos)
p      <- 5

# ------------------------------------------------------------
# Analise descritiva
# ------------------------------------------------------------

# Medias por grupo
aggregate(cbind(FL, RW, CL, CW, BD) ~ grupo, data = dados,
          FUN = function(x) round(mean(x), 2))

# Desvios padrao por grupo
aggregate(cbind(FL, RW, CL, CW, BD) ~ grupo, data = dados,
          FUN = function(x) round(sd(x), 2))

# Minimo, mediana e maximo por variavel
summary(dados[, c("FL", "RW", "CL", "CW", "BD")])

# Boxplots por variavel e grupo
vars_plot <- c("FL", "RW", "CL", "CW", "BD")

df_long <- do.call(rbind, lapply(vars_plot, function(v) {
  data.frame(grupo = dados$grupo, variavel = v, valor = dados[[v]])
}))

ggplot(df_long, aes(x = grupo, y = valor, fill = grupo)) +
  geom_boxplot(alpha = 0.7, outlier.color = "white",
               outlier.size = 1.2, linewidth = 0.4) +
  scale_fill_manual(values = c(
    "Azul Femea"    = "blue",
    "Azul Macho"    = "lightblue",
    "Laranja Femea" = "darkorange",
    "Laranja Macho" = "orange"
  )) +
  facet_wrap(~variavel, scales = "free_y") +
  labs(
    title    = "Distribuicao das variaveis morfometricas por grupo",
    subtitle = "Fonte: Campbell & Mahon (1974) — dataset crabs (MASS)",
    x = NULL, y = "Valor (mm)", fill = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.background  = element_rect(fill = "black", color = NA),
    panel.background = element_rect(fill = "gray", color = NA),
    panel.grid       = element_line(color = "#2e2e2e"),
    text             = element_text(color = "gray80"),
    strip.text       = element_text(color = "gray90", face = "bold"),
    axis.text.x      = element_text(color = "gray60", angle = 20, hjust = 1),
    axis.text.y      = element_text(color = "gray60"),
    legend.position  = "none",
    plot.title       = element_text(color = "gray95", face = "bold"),
    plot.subtitle    = element_text(color = "gray60", size = 9)
  )

# Medias com IC 95% — perfil por grupo
df_medias <- do.call(rbind, lapply(vars_plot, function(v) {
  do.call(rbind, lapply(grupos, function(gr) {
    x <- dados[dados$grupo == gr, v]
    data.frame(grupo = gr, variavel = v,
               m  = mean(x),
               se = sd(x) / sqrt(length(x)))
  }))
}))

ggplot(df_medias, aes(x = variavel, y = m,
                      color = grupo, group = grupo)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.5) +
  geom_errorbar(aes(ymin = m - 1.96*se, ymax = m + 1.96*se),
                width = 0.15, linewidth = 0.5) +
  scale_color_manual(values = c(
    "Azul Femea"    = "blue",
    "Azul Macho"    = "lightblue",
    "Laranja Femea" = "darkorange",
    "Laranja Macho" = "orange"
  )) +
  labs(
    title    = "Perfil de medias por grupo (IC 95%)",
    subtitle = "Permite visualizar quais variaveis separam mais os grupos",
    x = "Variavel", y = "Media (mm)", color = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.background  = element_rect(fill = "#1a1a1a", color = NA),
    panel.background = element_rect(fill = "#1a1a1a", color = NA),
    panel.grid       = element_line(color = "#2e2e2e"),
    text             = element_text(color = "gray80"),
    axis.text        = element_text(color = "gray60"),
    legend.position  = "bottom",
    plot.title       = element_text(color = "gray95", face = "bold"),
    plot.subtitle    = element_text(color = "gray60", size = 9)
  )

# ------------------------------------------------------------
# Diagnosticos pre-MANOVA
# ------------------------------------------------------------

# Normalidade multivariada por grupo (distancia de Mahalanobis)
# H0: dados seguem distribuicao normal multivariada
lapply(grupos, function(gr) {
  sub    <- as.matrix(dados[dados$grupo == gr, c("FL", "RW", "CL", "CW", "BD")])
  xbar   <- colMeans(sub)
  S_inv  <- solve(cov(sub))
  d2     <- apply(sub, 1, function(x) t(x - xbar) %*% S_inv %*% (x - xbar))
  ks     <- ks.test(d2, "pchisq", df = ncol(sub))
  data.frame(grupo = gr, estatistica = round(ks$statistic, 4),
             pvalor = round(ks$p.value, 4))
})

# Homocedasticidade: teste de Box
box_manova <- boxM(dados[, 2:6], dados$grupo)
box_manova   # H0: Sigma1 = Sigma2 = Sigma3 = Sigma4

# Teste de Kruskal-Wallis (abordagem não paramétrica multivariada)

# ------------------------------------------------------------
# Ajuste da MANOVA
# ------------------------------------------------------------

fit <- manova(
  cbind(FL, RW, CL, CW, BD) ~ grupo,
  data = dados
)

summary(fit, test = "Wilks")      # Lambda de Wilks
summary(fit, test = "Pillai")     # Traco de Pillai
summary(fit, test = "Hotelling")  # Traco de Lawley-Hotelling
summary(fit, test = "Roy")        # Raiz de Roy

# ANOVAs univariadas como follow-up
summary.aov(fit)

# ------------------------------------------------------------
# Decomposicao manual: matrizes H e E
# ------------------------------------------------------------

Y           <- as.matrix(dados[, 2:6])
media_geral <- colMeans(Y)
n_por_grupo <- table(dados$grupo)

E_mat <- Reduce("+", lapply(grupos, function(gr) {
  sub  <- as.matrix(dados[dados$grupo == gr, 2:6])
  xbar <- colMeans(sub)
  t(sweep(sub, 2, xbar)) %*% sweep(sub, 2, xbar)
}))

H_mat <- Reduce("+", lapply(grupos, function(gr) {
  ni   <- n_por_grupo[gr]
  xbar <- colMeans(as.matrix(dados[dados$grupo == gr, 2:6]))
  diff <- xbar - media_geral
  ni * outer(diff, diff)
}))

Lambda <- det(E_mat) / det(H_mat + E_mat)
Lambda   # lambda de Wilks calculado manualmente

# Aproximacao qui-quadrado
qui2 <- -(N - 1 - (p + g) / 2) * log(Lambda)
gl   <- p * (g - 1)
pval <- pchisq(qui2, df = gl, lower.tail = FALSE)

qui2   # estatistica qui-quadrado
pval   # p-valor aproximado

# Autovalores de E^{-1} H
autos <- Re(eigen(solve(E_mat) %*% H_mat)$values)
autos <- autos[autos > 1e-10]
autos  # autovalores relevantes

# Quatro estatisticas calculadas manualmente
Pillai_stat <- sum(autos / (1 + autos))
LH_stat     <- sum(autos)
Roy_stat    <- max(autos) / (1 + max(autos))

Pillai_stat
LH_stat
Roy_stat

# ------------------------------------------------------------
# Visualizacao: medias e IC 95% por grupo
# ------------------------------------------------------------

vars_plot <- c("FL", "RW", "CL", "CW", "BD")

df_medias <- do.call(rbind, lapply(vars_plot, function(v) {
  do.call(rbind, lapply(grupos, function(gr) {
    x  <- dados[dados$grupo == gr, v]
    data.frame(
      grupo    = gr,
      variavel = v,
      m        = mean(x),
      se       = sd(x) / sqrt(length(x))
    )
  }))
}))

ggplot(df_medias, aes(x = grupo, y = m, color = grupo)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = m - 1.96*se, ymax = m + 1.96*se),
                width = 0.2, linewidth = 0.7) +
  scale_color_manual(values = c(
    "Azul Femea"    = "gray60",
    "Azul Macho"    = "gray80",
    "Laranja Femea" = "#E87722",
    "Laranja Macho" = "#f0a050"
  )) +
  facet_wrap(~variavel, scales = "free_y") +
  labs(
    title    = "Medias e IC 95% por grupo e variavel morfometrica",
    subtitle = "Barras = intervalo de confianca 95% para a media",
    x = NULL, y = "Media (mm)", color = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.background  = element_rect(fill = "#1a1a1a", color = NA),
    panel.background = element_rect(fill = "#1a1a1a", color = NA),
    panel.grid       = element_line(color = "#2e2e2e"),
    text             = element_text(color = "gray80"),
    strip.text       = element_text(color = "gray90", face = "bold"),
    axis.text.x      = element_text(color = "gray60", angle = 15, hjust = 1),
    axis.text.y      = element_text(color = "gray60"),
    legend.position  = "none",
    plot.title       = element_text(color = "gray95", face = "bold"),
    plot.subtitle    = element_text(color = "gray60", size = 9)
  )