# ============================================================
# Modelos Multivariados
# Correlacao Canonica
# Prof. Vinicius Osterne
# ============================================================

# install.packages(c("ggplot2", "CCA", "gridExtra"))

library(ggplot2)
library(CCA)
library(gridExtra)

# ------------------------------------------------------------
# Dados reais: state.x77
# U.S. Bureau of the Census (1977) — 50 estados americanos
# ------------------------------------------------------------

dados <- as.data.frame(state.x77)
colnames(dados) <- c("populacao", "renda", "analfabetismo",
                     "vida", "homicidio", "diploma",
                     "geadas", "area")

X <- dados[, c("renda", "diploma", "geadas")]
Y <- dados[, c("vida", "homicidio", "analfabetismo")]

# Visao geral
head(dados[, c("renda", "diploma", "geadas",
               "vida", "homicidio", "analfabetismo")], 20)
summary(cbind(X, Y))

# ------------------------------------------------------------
# Analise descritiva
# ------------------------------------------------------------

# Correlacoes dentro de cada conjunto
round(cor(X), 3)   # correlacoes dentro de X (socioeconomico)
round(cor(Y), 3)   # correlacoes dentro de Y (qualidade de vida)

# Correlacoes cruzadas X-Y: estrutura a ser capturada
round(cor(X, Y), 3)

# ------------------------------------------------------------
# Diagnostico pre-analise: normalidade multivariada
# ------------------------------------------------------------

XY    <- as.matrix(cbind(X, Y))
xbar  <- colMeans(XY)
S_inv <- solve(cov(XY))
d2    <- apply(XY, 1, function(x) t(x - xbar) %*% S_inv %*% (x - xbar))
ks.test(d2, "pchisq", df = ncol(XY))

# ------------------------------------------------------------
# Correlacao canonica
# ------------------------------------------------------------

fit_cc <- cc(X, Y)

fit_cc$cor     # correlacoes canonicas rho_k: r = min(3,3) = 3 pares
# 1º par canônico: correlação = 0.808 (U_1 x V_1)
# 2º par canônico: correlação = 0.520 (U_2 x V_2)
# 3º par canônico: correlação = 0.161 (U_3 x V_3)


fit_cc$xcoef   # vetores a_k para X
# U1 = -0.000053*renda - 0.067758*diploma - 0.012326*geadas

fit_cc$ycoef   # vetores b_k para Y
# V1 = 0.036140*vida + 0.045523*homicidio + 1.485537*analfabetismo



# ------------------------------------------------------------
# Scores canonicos
# ------------------------------------------------------------

scores <- comput(X, Y, fit_cc)

U <- scores$xscores   # scores do conjunto socioeconomico (U_k = a^tX)
V <- scores$yscores   # scores do conjunto qualidade de vida (V_k = b^tY)

round(diag(cor(U, V)), 3)   # diagonal deve ser igual a fit_cc$cor
round(cor(U), 3)             # deve ser identidade: ortogonalidade dentro de X
round(cor(V), 3)             # deve ser identidade: ortogonalidade dentro de Y
# os eixos canônicos dentro do bloco X e do bloco Y são ortogonais


# ------------------------------------------------------------
# Cargas canonicas
# Quais variáveis estão realmente mais associadas a cada dimensão canônica
# ------------------------------------------------------------

cargas_X <- cor(X, U) # mede o quanto a variável original X_j está associada ao eixo canônico U_k
cargas_Y <- cor(Y, V) # mede o quanto a variável original Y_j está associada ao eixo canônico V_k

round(cargas_X, 3)   # cargas canonicas de X
# primeiro eixo socioeconômico está fortemente relacionado principalmente com: geadas e diploma

round(cargas_Y, 3)   # cargas canonicas de Y
# primeiro eixo qualidade de vida está fortemente relacionado principalmente com: analfabetismo

# Cargas cruzadas
cargas_cruzadas_X <- cor(X, V) # mede o quanto a variável original X_j está associada ao eixo canônico V_k
cargas_cruzadas_Y <- cor(Y, U) # mede o quanto a variável original Y_j está associada ao eixo canônico U_k

round(cargas_cruzadas_X, 3)
# primeiro eixo qualidade de vida está fortemente relacionado principalmente com: geadas e diploma

round(cargas_cruzadas_Y, 3)
# primeiro eixo socioeconômico está fortemente relacionado principalmente com: analfabetismo




# ------------------------------------------------------------
# Simetria: trocar X e Y nao muda rho_k
# ------------------------------------------------------------

fit_cc_inv <- cc(Y, X)

round(fit_cc$cor, 4)      # rho_k com X primeiro
round(fit_cc_inv$cor, 4)  # rho_k com Y primeiro

# ------------------------------------------------------------
# Teste sequencial de Bartlett
# ------------------------------------------------------------

n    <- nrow(dados)
p_X  <- ncol(X)
p_Y  <- ncol(Y)
r    <- min(p_X, p_Y)
rhos <- fit_cc$cor

resultado_bartlett <- do.call(rbind, lapply(1:r, function(k) {
  lambda_prod <- prod(1 - rhos[k:r]^2)
  qui2 <- -(n - 1 - (p_X + p_Y + 1) / 2) * log(lambda_prod)
  gl   <- (p_X - k + 1) * (p_Y - k + 1)
  pval <- pchisq(qui2, df = gl, lower.tail = FALSE)
  data.frame(par = k, rho = round(rhos[k], 3),
             qui2 = round(qui2, 2), gl = gl,
             pvalor = round(pval, 4))
}))
resultado_bartlett

# ------------------------------------------------------------
# Visualizacao: correlacoes canonicas
# ------------------------------------------------------------

df_rho <- data.frame(
  par  = factor(paste0("Par ", 1:r)),
  rho  = rhos,
  rho2 = rhos^2
)

ggplot(df_rho, aes(x = par, y = rho2, fill = par)) +
  geom_col(width = 0.5) +
  geom_text(aes(label = round(rho2, 3)),
            vjust = -0.5, color = "gray80", size = 4) +
  scale_fill_manual(values = c("#E87722", "#f0a050", "gray50")) +
  scale_y_continuous(limits = c(0, 1.1)) +
  labs(
    title    = "Quadrado das correlacoes canonicas por par",
    subtitle = "A simetria garante que trocar X e Y nao altera esses valores",
    x = NULL, y = expression(rho[k]^2), fill = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.background  = element_rect(fill = "#1a1a1a", color = NA),
    panel.background = element_rect(fill = "#1a1a1a", color = NA),
    panel.grid       = element_line(color = "#2e2e2e"),
    text             = element_text(color = "gray80"),
    axis.text        = element_text(color = "gray60"),
    legend.position  = "none",
    plot.title       = element_text(color = "gray95", face = "bold"),
    plot.subtitle    = element_text(color = "gray60", size = 9)
  )

# ------------------------------------------------------------
# Visualizacao: pares canonicos U_k vs V_k
# ------------------------------------------------------------

df_scores <- data.frame(U1 = U[,1], V1 = V[,1],
                        U2 = U[,2], V2 = V[,2],
                        estado = rownames(dados))

p1 <- ggplot(df_scores, aes(x = U1, y = V1)) +
  geom_point(color = "#E87722", alpha = 0.7, size = 2) +
  geom_smooth(method = "lm", se = FALSE,
              color = "gray70", linetype = "dashed", linewidth = 0.8) +
  labs(
    title = paste0("Par 1: rho = ", round(rhos[1], 3)),
    x = expression(U[1]~"(socioeconomico)"),
    y = expression(V[1]~"(qualidade de vida)")
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.background  = element_rect(fill = "#1a1a1a", color = NA),
    panel.background = element_rect(fill = "#1a1a1a", color = NA),
    panel.grid       = element_line(color = "#2e2e2e"),
    text             = element_text(color = "gray80"),
    axis.text        = element_text(color = "gray60"),
    plot.title       = element_text(color = "gray95", face = "bold")
  )

p2 <- ggplot(df_scores, aes(x = U2, y = V2)) +
  geom_point(color = "gray55", alpha = 0.7, size = 2) +
  geom_smooth(method = "lm", se = FALSE,
              color = "gray45", linetype = "dashed", linewidth = 0.8) +
  labs(
    title = paste0("Par 2: rho = ", round(rhos[2], 3)),
    x = expression(U[2]~"(socioeconomico)"),
    y = expression(V[2]~"(qualidade de vida)")
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.background  = element_rect(fill = "#1a1a1a", color = NA),
    panel.background = element_rect(fill = "#1a1a1a", color = NA),
    panel.grid       = element_line(color = "#2e2e2e"),
    text             = element_text(color = "gray80"),
    axis.text        = element_text(color = "gray60"),
    plot.title       = element_text(color = "gray95", face = "bold")
  )

grid.arrange(p1, p2, ncol = 2)

# ------------------------------------------------------------
# Visualizacao: heatmap das cargas cruzadas
# ------------------------------------------------------------

df_cz <- as.data.frame(round(cargas_cruzadas_Y, 3))
df_cz$variavel <- rownames(df_cz)
colnames(df_cz)[1:r] <- paste0("U", 1:r)

df_cz_long <- do.call(rbind, lapply(paste0("U", 1:r), function(u) {
  data.frame(par            = u,
             variavel       = df_cz$variavel,
             carga_cruzada  = df_cz[[u]])
}))

ggplot(df_cz_long, aes(x = par, y = variavel, fill = carga_cruzada)) +
  geom_tile(color = "#1a1a1a", linewidth = 0.5) +
  geom_text(aes(label = round(carga_cruzada, 2)),
            color = "gray95", size = 4) +
  scale_fill_gradient2(low = "gray30", mid = "#1a1a1a",
                       high = "#E87722", midpoint = 0) +
  labs(
    title    = "Cargas cruzadas: qualidade de vida no espaco socioeconomico",
    subtitle = "Correlacao entre cada variavel de Y e os scores U_k de X",
    x = "Dimensao canonica socioeconomica", y = NULL,
    fill = "Carga\ncruzada"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.background  = element_rect(fill = "#1a1a1a", color = NA),
    panel.background = element_rect(fill = "#1a1a1a", color = NA),
    panel.grid       = element_blank(),
    text             = element_text(color = "gray80"),
    axis.text        = element_text(color = "gray60"),
    plot.title       = element_text(color = "gray95", face = "bold"),
    plot.subtitle    = element_text(color = "gray60", size = 9)
  )

