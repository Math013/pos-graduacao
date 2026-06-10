# ============================================================
# Modelos Multivariados
# Regressao Multivariada
# Exemplo REAL: indicadores sociais dos estados americanos
# Fonte: U.S. Bureau of the Census (1977) via R base
#        dataset "state.x77" — 50 estados, anos 1970
# Desfechos:  expectativa de vida, taxa de homicidio, analfabetismo
# Preditores: renda per capita, diploma ens. medio, geadas/ano
# Prof. Vinicius Osterne
# ============================================================

# install.packages(c("ggplot2", "car"))

library(ggplot2)
library(car)

# ------------------------------------------------------------
# Dados reais: state.x77
# U.S. Bureau of the Census (1977)
# 50 observacoes (estados), 8 variaveis
# ------------------------------------------------------------

dados <- as.data.frame(state.x77)
colnames(dados) <- c("populacao", "renda", "analfabetismo",
                     "vida", "homicidio", "diploma",
                     "geadas", "area")

# Visualizando a base
head(dados, 20)
str(dados)
summary(dados[, c("vida", "homicidio", "analfabetismo",
                  "renda", "diploma", "geadas")])

# ------------------------------------------------------------
# Analise descritiva
# ------------------------------------------------------------

# Correlacoes entre desfechos: justifica o modelo conjunto
round(cor(dados[, c("vida", "homicidio", "analfabetismo")]), 3)

# Correlacoes entre preditores: checa multicolinearidade
round(cor(dados[, c("renda", "diploma", "geadas")]), 3)

# Boxplots dos desfechos
vars_desfecho <- c("vida", "homicidio", "analfabetismo")

df_long_d <- do.call(rbind, lapply(vars_desfecho, function(v) {
  data.frame(variavel = v, valor = dados[[v]])
}))

ggplot(df_long_d, aes(x = variavel, y = valor, fill = variavel)) +
  geom_boxplot(alpha = 0.7, outlier.color = "white",
               outlier.size = 1.5, linewidth = 0.4) +
  scale_fill_manual(values = c(
    "vida"          = "#E87722",
    "homicidio"     = "gray60",
    "analfabetismo" = "#f0a050"
  )) +
  facet_wrap(~variavel, scales = "free") +
  labs(
    title    = "Distribuicao dos desfechos por estado",
    subtitle = "Fonte: U.S. Bureau of the Census (1977) — 50 estados americanos",
    x = NULL, y = "Valor", fill = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.background  = element_rect(fill = "#1a1a1a", color = NA),
    panel.background = element_rect(fill = "#1a1a1a", color = NA),
    panel.grid       = element_line(color = "#2e2e2e"),
    text             = element_text(color = "gray80"),
    strip.text       = element_text(color = "gray90", face = "bold"),
    axis.text        = element_text(color = "gray60"),
    legend.position  = "none",
    plot.title       = element_text(color = "gray95", face = "bold"),
    plot.subtitle    = element_text(color = "gray60", size = 9)
  )

# Dispersao: preditores vs desfechos
vars_pred <- c("renda", "diploma", "geadas")

df_scatter <- do.call(rbind, lapply(vars_pred, function(px) {
  do.call(rbind, lapply(vars_desfecho, function(py) {
    data.frame(preditor = px, desfecho = py,
               x = dados[[px]], y = dados[[py]])
  }))
}))

ggplot(df_scatter, aes(x = x, y = y)) +
  geom_point(color = "#E87722", alpha = 0.6, size = 1.8) +
  geom_smooth(method = "lm", color = "white",
              linewidth = 0.7, se = FALSE) +
  facet_grid(desfecho ~ preditor, scales = "free") +
  labs(
    title    = "Relacao entre preditores e desfechos",
    subtitle = "Linha branca = tendencia linear",
    x = NULL, y = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.background  = element_rect(fill = "#1a1a1a", color = NA),
    panel.background = element_rect(fill = "#1a1a1a", color = NA),
    panel.grid       = element_line(color = "#2e2e2e"),
    text             = element_text(color = "gray80"),
    strip.text       = element_text(color = "gray90", face = "bold", size = 9),
    axis.text        = element_text(color = "gray60", size = 8),
    plot.title       = element_text(color = "gray95", face = "bold"),
    plot.subtitle    = element_text(color = "gray60", size = 9)
  )

# ------------------------------------------------------------
# Diagnosticos pre-regressao
# ------------------------------------------------------------

# Normalidade multivariada dos desfechos (Mahalanobis)
Y_desc <- as.matrix(dados[, vars_desfecho])
xbar   <- colMeans(Y_desc)
S_inv  <- solve(cov(Y_desc))
d2     <- apply(Y_desc, 1, function(x) t(x - xbar) %*% S_inv %*% (x - xbar))
ks.test(d2, "pchisq", df = length(vars_desfecho))

# ------------------------------------------------------------
# Ajuste da regressao multivariada
# ------------------------------------------------------------

fit <- lm(
  cbind(vida, homicidio, analfabetismo) ~ renda + diploma + geadas,
  data = dados
)

coef(fit)      # coeficientes estimados: cada coluna e um desfecho
summary(fit)   # resumo por equacao (nivel univariado)

# ------------------------------------------------------------
# Testes de hipotese multivariados
# ------------------------------------------------------------

fit_manova <- manova(
  cbind(vida, homicidio, analfabetismo) ~ renda + diploma + geadas,
  data = dados
)

summary(fit_manova, test = "Wilks")      # Lambda de Wilks
summary(fit_manova, test = "Pillai")     # Traco de Pillai
summary(fit_manova, test = "Hotelling")  # Traco de Lawley-Hotelling
summary(fit_manova, test = "Roy")        # Raiz de Roy

# Teste por preditor
summary(manova(cbind(vida, homicidio, analfabetismo) ~ renda,
               data = dados), test = "Wilks")
summary(manova(cbind(vida, homicidio, analfabetismo) ~ diploma,
               data = dados), test = "Wilks")
summary(manova(cbind(vida, homicidio, analfabetismo) ~ geadas,
               data = dados), test = "Wilks")

# ------------------------------------------------------------
# Decomposicao manual: matrizes H e E
# ------------------------------------------------------------

X_mat <- model.matrix(fit)
Y_mat <- as.matrix(dados[, vars_desfecho])
n     <- nrow(dados)

B_hat  <- solve(t(X_mat) %*% X_mat) %*% t(X_mat) %*% Y_mat
B_hat  # comparar com coef(fit)

Y_hat   <- X_mat %*% B_hat
E_resid <- Y_mat - Y_hat

p_pred    <- ncol(X_mat)
Sigma_hat <- (t(E_resid) %*% E_resid) / (n - p_pred)
Sigma_hat  # estimativa da matriz de covariancia dos erros

# ------------------------------------------------------------
# Diagnostico dos residuos multivariados
# ------------------------------------------------------------

residuos <- as.data.frame(E_resid)
colnames(residuos) <- c("res_vida", "res_homicidio", "res_analfabetismo")

# Distancia de Mahalanobis para detectar outliers multivariados
maha <- mahalanobis(residuos,
                    center = colMeans(residuos),
                    cov    = cov(residuos))

corte    <- qchisq(0.975, df = length(vars_desfecho))
outliers <- which(maha > corte)
outliers   # indices das observacoes suspeitas
rownames(dados)[outliers]   # nomes dos estados suspeitos

# ------------------------------------------------------------
# Visualizacao: residuos multivariados
# ------------------------------------------------------------

df_resid          <- residuos
df_resid$maha     <- maha
df_resid$outlier  <- maha > corte
df_resid$estado   <- rownames(dados)

ggplot(df_resid, aes(x = res_vida, y = res_homicidio,
                     color = outlier)) +
  geom_point(size = 2.5, alpha = 0.8) +
  geom_text(data = df_resid[df_resid$outlier, ],
            aes(label = estado), vjust = -0.8,
            size = 3, color = "#E87722") +
  stat_ellipse(aes(group = 1), color = "#E87722",
               linetype = "dashed", linewidth = 0.8) +
  geom_hline(yintercept = 0, color = "gray50", linetype = "dotted") +
  geom_vline(xintercept = 0, color = "gray50", linetype = "dotted") +
  scale_color_manual(values = c("FALSE" = "gray60", "TRUE" = "#E87722"),
                     labels = c("Normal", "Outlier multivariado")) +
  labs(
    title    = "Residuos multivariados: vida x homicidio",
    subtitle = "Elipse de 95% baseada na distancia de Mahalanobis",
    x = "Residuo expectativa de vida",
    y = "Residuo taxa de homicidio",
    color = NULL
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
# Visualizacao: coeficientes estimados por desfecho
# ------------------------------------------------------------

df_coef <- as.data.frame(coef(fit))
df_coef$preditor <- rownames(df_coef)
df_coef <- df_coef[df_coef$preditor != "(Intercept)", ]

df_coef_long <- do.call(rbind, lapply(vars_desfecho, function(v) {
  data.frame(preditor    = df_coef$preditor,
             desfecho    = v,
             coeficiente = df_coef[[v]])
}))

ggplot(df_coef_long,
       aes(x = preditor, y = coeficiente, fill = preditor)) +
  geom_col(width = 0.6) +
  geom_hline(yintercept = 0, color = "gray50", linetype = "dashed") +
  facet_wrap(~desfecho, scales = "free_y",
             labeller = labeller(desfecho = c(
               vida          = "Expect. de vida",
               homicidio     = "Taxa de homicidio",
               analfabetismo = "Analfabetismo"
             ))) +
  scale_fill_manual(values = c(
    "renda"   = "#E87722",
    "diploma" = "#f0a050",
    "geadas"  = "#ffd080"
  )) +
  labs(
    title    = "Coeficientes estimados por preditor e desfecho",
    subtitle = "Cada painel corresponde a um desfecho social",
    x = NULL, y = "Coeficiente estimado", fill = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.background  = element_rect(fill = "#1a1a1a", color = NA),
    panel.background = element_rect(fill = "#1a1a1a", color = NA),
    panel.grid       = element_line(color = "#2e2e2e"),
    text             = element_text(color = "gray80"),
    strip.text       = element_text(color = "gray90", face = "bold"),
    axis.text        = element_text(color = "gray60"),
    legend.position  = "none",
    plot.title       = element_text(color = "gray95", face = "bold"),
    plot.subtitle    = element_text(color = "gray60", size = 9)
  )