# ============================================================
# Apresentacao Final - Regressao Multivariada
# Aplicacao: dados judiciais CNJ 2026 - direito penal
# Autor: Matheus
# Curso: Analise Multivariada - Prof. Vinicius Osterne
# ============================================================

# -----------------------
# Pacotes
# -----------------------
# install.packages(c("arrow","dplyr","ggplot2","MVTests","car","tidyr"))

library(arrow)      # ler parquet
library(dplyr)
library(ggplot2)
library(tidyr)
library(MVTests)
library(car)

# ============================================================
# 1. CARGA E PREPARACAO
# ============================================================

dados_brutos <- arrow::read_parquet("direito_penal.parquet")

cat("N bruto:", nrow(dados_brutos), "\n")

dados <- dados_brutos |>
  dplyr::filter(ramo_justica == "direito_penal" | TRUE) |>
  # Mantive a linha acima para documentar; o filtro de ramo
  # ja foi aplicado upstream (arquivo ja filtrado).
  dplyr::filter(
    casos_novos        > 0,
    pendentes          > 0,
    baixados           > 0,
    tp_julgamento_dias > 0,
    julgados           > 0,
    !is.na(tp_baixa_dias),
    tp_baixa_dias      > 0
  ) |>
  dplyr::mutate(
    log_tp_julg     = log1p(tp_julgamento_dias),
    log_tp_baixa    = log1p(tp_baixa_dias),
    log_casos       = log1p(casos_novos),
    log_pendentes   = log1p(pendentes),
    taxa_baixa      = baixados / casos_novos,
    congestionamento = pendentes / (casos_novos + baixados)
  ) |>
  dplyr::filter(taxa_baixa <= 5, congestionamento <= 50)

cat("N apos filtros:", nrow(dados), "\n")

# ============================================================
# 2. ANALISE EXPLORATORIA
# ============================================================

# Correlacao entre desfechos
cat("\n--- Correlacao entre desfechos (justifica modelo conjunto) ---\n")
print(round(cor(dados[, c("log_tp_julg", "log_tp_baixa")]), 3))
# Esperado: 0,726

# Correlacao entre preditores (checa multicolinearidade)
cat("\n--- Correlacao entre preditores ---\n")
print(round(cor(dados[, c("log_casos", "log_pendentes", "congestionamento")]), 3))

# ============================================================
# 3. VERIFICACAO DOS PRESSUPOSTOS
# ============================================================

Y_mat <- as.matrix(dados[, c("log_tp_julg", "log_tp_baixa")])

# ============================================================
# 4. AJUSTE DO MODELO MULTIVARIADO
# ============================================================

fit <- lm(
  cbind(log_tp_julg, log_tp_baixa) ~
    log_casos + log_pendentes + congestionamento,
  data = dados
)

cat("\n--- Matriz B_hat ---\n")
print(round(coef(fit), 4))
# Esperado:
#                   log_tp_julg  log_tp_baixa
# (Intercept)            3.8566        4.2942
# log_casos             -0.4798       -0.5401
# log_pendentes          1.2501        1.4072
# congestionamento      -0.0577       -0.1429

cat("\n--- Resumo por equacao ---\n")
print(summary(fit))

# ============================================================
# 5. TESTES DE HIPOTESE MULTIVARIADOS
# ============================================================

fit_mv <- manova(
  cbind(log_tp_julg, log_tp_baixa) ~
    log_casos + log_pendentes + congestionamento,
  data = dados
)

cat("\n--- Wilks (esperado L=0,343) ---\n")
print(summary(fit_mv, test = "Wilks"))

cat("\n--- Pillai (esperado V=0,692) ---\n")
print(summary(fit_mv, test = "Pillai"))

cat("\n--- Hotelling-Lawley (esperado T=1,816) ---\n")
print(summary(fit_mv, test = "Hotelling-Lawley"))

cat("\n--- Roy (esperado theta=1,759) ---\n")
print(summary(fit_mv, test = "Roy"))

# ============================================================
# 6. DECOMPOSICAO MANUAL (DIDATICO)
# ============================================================

X_mat <- model.matrix(fit)
n     <- nrow(dados)
p     <- ncol(X_mat)
q     <- ncol(Y_mat)

# B_hat = (X'X)^-1 X'Y
B_hat <- solve(t(X_mat) %*% X_mat) %*% t(X_mat) %*% Y_mat
cat("\nB_hat manual:\n"); print(round(B_hat, 4))

# Sigma_hat
E_resid   <- Y_mat - X_mat %*% B_hat
Sigma_hat <- t(E_resid) %*% E_resid / (n - p)
cat("\nSigma_hat:\n"); print(round(Sigma_hat, 4))

cat("\nCorrelacao residual entre desfechos (esperado 0,437):\n")
print(round(cov2cor(Sigma_hat), 4))

# Matriz H (hipotese: todos os preditores = 0)
C <- diag(p)[-1, , drop = FALSE]   # remove linha do intercepto
M <- diag(q)
CB <- C %*% B_hat %*% M
mid <- solve(C %*% solve(t(X_mat) %*% X_mat) %*% t(C))
H <- t(CB) %*% mid %*% CB
E_mat <- t(M) %*% (t(E_resid) %*% E_resid) %*% M

cat("\nAutovalores de H E^-1 (esperado 1,76 e 0,057):\n")
eigvals <- sort(Re(eigen(H %*% solve(E_mat))$values), decreasing = TRUE)
print(round(eigvals, 4))
# Concentracao em 1 dimensao confirma alta correlacao entre desfechos

# ============================================================
# 7. UNI vs MULTI - O ACHADO CENTRAL
# ============================================================

fit_uni_julg  <- lm(log_tp_julg  ~ log_casos + log_pendentes + congestionamento, data = dados)
fit_uni_baixa <- lm(log_tp_baixa ~ log_casos + log_pendentes + congestionamento, data = dados)

cat("\n--- UNIVARIADO: log_tp_julg ---\n")
print(round(coef(fit_uni_julg), 4))
cat("\n--- UNIVARIADO: log_tp_baixa ---\n")
print(round(coef(fit_uni_baixa), 4))
cat("\n--- MULTIVARIADO ---\n")
print(round(coef(fit), 4))
cat("\n>>> COEFICIENTES SAO IDENTICOS. A diferenca esta na inferencia. <<<\n")

# R^2 por equacao
cat("\nR^2 por equacao:\n")
cat("  log_tp_julg :", round(summary(fit)[[1]]$r.squared, 4), "(esperado 0,490)\n")
cat("  log_tp_baixa:", round(summary(fit)[[2]]$r.squared, 4), "(esperado 0,607)\n")

# ============================================================
# 8. DIAGNOSTICO DOS RESIDUOS MULTIVARIADOS
# ============================================================

resid_df <- as.data.frame(residuals(fit))
colnames(resid_df) <- c("res_julg", "res_baixa")

maha <- mahalanobis(resid_df,
                    center = colMeans(resid_df),
                    cov    = cov(resid_df))

corte <- qchisq(0.975, df = q)
outliers <- which(maha > corte)
cat("\nOutliers multivariados (97,5%):", length(outliers),
    "de", n, "(", round(100*length(outliers)/n, 1), "%)\n")
# Esperado: ~745 (5,1%)

# ============================================================
# 9. VISUALIZACOES
# ============================================================

# Tema dark consistente
tema_dark <- theme_minimal(base_size = 12) +
  theme(
    plot.background  = element_rect(fill = "#1a1a1a", color = NA),
    panel.background = element_rect(fill = "#1a1a1a", color = NA),
    panel.grid       = element_line(color = "#2e2e2e"),
    text             = element_text(color = "gray80"),
    strip.text       = element_text(color = "gray90", face = "bold"),
    axis.text        = element_text(color = "gray60"),
    plot.title       = element_text(color = "gray95", face = "bold"),
    plot.subtitle    = element_text(color = "gray60", size = 9),
    legend.position  = "bottom"
  )

# 9.1 Elipse de residuos multivariados
df_plot <- resid_df
df_plot$maha    <- maha
df_plot$outlier <- maha > corte

p1 <- ggplot(df_plot, aes(x = res_julg, y = res_baixa, color = outlier)) +
  geom_point(alpha = 0.4, size = 1) +
  stat_ellipse(aes(group = 1), color = "#E87722",
               linetype = "dashed", linewidth = 0.8) +
  geom_hline(yintercept = 0, color = "gray50", linetype = "dotted") +
  geom_vline(xintercept = 0, color = "gray50", linetype = "dotted") +
  scale_color_manual(values = c("FALSE" = "gray60", "TRUE" = "#E87722"),
                     labels = c("Normal", "Outlier multivariado")) +
  labs(
    title    = "Residuos multivariados",
    subtitle = paste0("Elipse de 95% - correlacao residual = 0,437 - n = ", n),
    x = "Residuo: log(tempo de julgamento)",
    y = "Residuo: log(tempo de baixa)",
    color = NULL
  ) + tema_dark
print(p1)

# 9.2 Coeficientes por desfecho
df_coef <- as.data.frame(coef(fit))
df_coef$preditor <- rownames(df_coef)
df_coef <- df_coef[df_coef$preditor != "(Intercept)", ]

df_long <- df_coef |>
  tidyr::pivot_longer(cols = -preditor,
                      names_to  = "desfecho",
                      values_to = "coef")

p2 <- ggplot(df_long, aes(x = preditor, y = coef, fill = preditor)) +
  geom_col(width = 0.6) +
  geom_hline(yintercept = 0, color = "gray50", linetype = "dashed") +
  facet_wrap(~desfecho, scales = "free_y") +
  scale_fill_manual(values = c(
    "log_casos"        = "#E87722",
    "log_pendentes"    = "#f0a050",
    "congestionamento" = "#ffd080"
  )) +
  labs(
    title    = "Coeficientes por preditor e desfecho",
    subtitle = "Cada painel = um Y; cada barra = um X",
    x = NULL, y = "Coeficiente estimado", fill = NULL
  ) + tema_dark + theme(legend.position = "none",
                        axis.text.x = element_text(angle = 15, hjust = 1))
print(p2)

# 9.3 Dispersao bruta dos desfechos (motivacao)
p3 <- ggplot(dados, aes(x = log_tp_julg, y = log_tp_baixa)) +
  geom_point(color = "#E87722", alpha = 0.15, size = 0.8) +
  geom_smooth(method = "lm", color = "white",
              linewidth = 0.5, se = FALSE) +
  labs(
    title    = "Correlacao entre desfechos: r = 0,726",
    subtitle = paste0("Justifica o modelo conjunto - n = ", n),
    x = "log(tempo de julgamento + 1)",
    y = "log(tempo de baixa + 1)"
  ) + tema_dark
print(p3)

# 9.4 Q-Q plot Mahalanobis (normalidade multivariada)
df_qq <- data.frame(
  teoricos = qchisq(ppoints(n) , df = q),
  observados = sort(maha)
)
p4 <- ggplot(df_qq, aes(x = teoricos, y = observados)) +
  geom_point(color = "gray60", alpha = 0.4, size = 0.8) +
  geom_abline(slope = 1, intercept = 0,
              color = "#E87722", linetype = "dashed", linewidth = 1) +
  labs(
    title    = "Q-Q plot: normalidade multivariada dos residuos",
    subtitle = "Desvio da diagonal nas caudas indica violacao",
    x = "Quantis teoricos chi^2(2)",
    y = "Distancia de Mahalanobis"
  ) + tema_dark
print(p4)

# ============================================================
# 10. EXPORTAR FIGURAS (opcional)
# ============================================================

# ggsave("fig1_residuos.png", p1, width = 9, height = 7, dpi = 130, bg = "#1a1a1a")
# ggsave("fig2_coeficientes.png", p2, width = 11, height = 5, dpi = 130, bg = "#1a1a1a")
# ggsave("fig3_motivacao.png", p3, width = 8, height = 7, dpi = 130, bg = "#1a1a1a")
# ggsave("fig4_qqplot.png", p4, width = 8, height = 7, dpi = 130, bg = "#1a1a1a")

# ============================================================
# FIM
# ============================================================