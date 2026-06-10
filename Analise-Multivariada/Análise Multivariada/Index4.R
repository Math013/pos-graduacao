# ============================================================
# PREPARAÇÃO DA BASE — Regressão Multivariada / Dados CNJ
# ============================================================
library(arrow)    # ler arquivos .parquet
library(dplyr)    # manipulação de dados

# --- Carga ---
# Ajuste o caminho para onde está seu arquivo
pen <- arrow::read_parquet("C:/Users/Matheus/Desktop/Arquivos_PUC/direito_penal.parquet")

# --- Filtros + agregação ---
# Filtramos conhecimento criminal + Justiça Estadual,
# depois agregamos por órgão × tipo de crime × grau,
# SOMANDO numeradores e denominadores (não as razões).
base <- pen |>
  dplyr::filter(
    procedimento == "Conhecimento criminal",
    ramo_justica == "Justiça Estadual"
  ) |>
  dplyr::group_by(id_orgao_julgador, sub_area, sigla_grau) |>
  dplyr::summarise(
    tp_julg_dias  = sum(tp_julgamento_dias, na.rm = TRUE),
    tp_julg_proc  = sum(tp_julgamento_proc, na.rm = TRUE),
    tp_baixa_dias = sum(tp_baixa_dias,      na.rm = TRUE),
    tp_baixa_proc = sum(tp_baixa_proc,      na.rm = TRUE),
    baixados      = sum(baixados,           na.rm = TRUE),
    casos_novos   = sum(casos_novos,        na.rm = TRUE),
    pendentes     = sum(pendentes,          na.rm = TRUE),
    .groups = "drop"   # evita warning de agrupamento residual
  )

# --- Filtros de validade: precisamos dos 3 desfechos calculáveis ---
base <- base |>
  dplyr::filter(
    tp_julg_proc > 0,
    tp_baixa_proc > 0,
    casos_novos > 0,
    !grepl("Bullying|Intimida", sub_area)   # descarta as 2 células de bullying
  )

# --- Construção dos desfechos (log1p das razões) ---
# log1p(x) = log(1 + x): estável para zeros, comprime a cauda
base <- base |>
  dplyr::mutate(
    Y1_log_tjulg  = log1p(tp_julg_dias  / tp_julg_proc),
    Y2_log_tbaixa = log1p(tp_baixa_dias / tp_baixa_proc),
    Y3_log_iad    = log1p(baixados      / casos_novos),
    log_pendentes = log1p(pendentes),
    # fator com Furto como referência (primeiro nível = referência no R)
    sub_area   = factor(sub_area, levels = c("Furto","Roubo","Ameaça","Feminicídio")),
    sigla_grau = factor(sigla_grau)
  )

# Conferência rápida
cat("Células:", nrow(base), "| Órgãos:", dplyr::n_distinct(base$id_orgao_julgador), "\n")
table(base$sub_area)

# ============================================================
# PRIMEIRO MODELO MULTIVARIADO
# ============================================================
# A sintaxe-chave: cbind(...) no lado esquerdo = regressão MULTIVARIADA.
# Sem o cbind(), R ajustaria só o primeiro Y (univariada).
fit <- lm(
  cbind(Y1_log_tjulg, Y2_log_tbaixa, Y3_log_iad) ~ sub_area + sigla_grau + log_pendentes,
  data = base
)

# coef(fit) devolve a matriz B-hat: linhas = preditores, colunas = desfechos
coef(fit)

library(car)

cat("===== TRAÇO DE PILLAI (principal) =====\n")
car::Anova(fit, test.statistic = "Pillai")

# Teste de Wilks (clássico — para comparação)
cat("\n===== LAMBDA DE WILKS (comparação) =====\n")
car::Anova(fit, test.statistic = "Wilks")

# ============================================================
# ETAPA UNIVARIADA (autorizada: multivariado deu significativo)
# ============================================================
summary(fit)

# ============================================================
# MATRIZ DE COVARIÂNCIA/CORRELAÇÃO RESIDUAL (Σ̂)
# ============================================================
# Resíduos do modelo multivariado: uma coluna por desfecho
res <- residuals(fit)   # matriz n × 3

# Σ̂ — covariância residual (divide por n - p, estimador não-viesado)
n <- nrow(base)
p <- length(coef(fit)[, 1])   # nº de parâmetros por equação
Sigma_hat <- crossprod(res) / (n - p)   # crossprod(res) = t(res) %*% res
cat("===== Σ̂ — Covariância residual =====\n")
print(round(Sigma_hat, 4))

# Correlação residual (mais fácil de interpretar que covariância)
cat("\n===== Correlação residual =====\n")
cor_res <- cov2cor(Sigma_hat)   # converte covariância em correlação
print(round(cor_res, 3))

# Comparação: correlação BRUTA dos desfechos (antes do modelo)
cat("\n===== Correlação BRUTA (para comparar) =====\n")
cor_bruta <- cor(base[, c("Y1_log_tjulg","Y2_log_tbaixa","Y3_log_iad")])
print(round(cor_bruta, 3))

# ============================================================
# DIAGNÓSTICO DE PRESSUPOSTOS
# ============================================================
# ============================================================
# TESTE DE MARDIA — implementação direta (sem pacotes)
# ============================================================
mardia_manual <- function(X) {
  X <- as.matrix(X)
  n <- nrow(X); p <- ncol(X)
  Xc <- scale(X, center = TRUE, scale = FALSE)   # centraliza
  S  <- cov(X) * (n - 1) / n                       # covariância (MLE)
  Sinv <- solve(S)
  
  # matriz de distâncias de Mahalanobis entre todos os pares
  D <- Xc %*% Sinv %*% t(Xc)
  
  # Assimetria multivariada (b1p)
  b1p <- sum(D^3) / (n^2)
  # Curtose multivariada (b2p)
  b2p <- sum(diag(D)^2) / n
  
  # Estatísticas de teste
  # Assimetria → qui-quadrado
  gl_skew <- p * (p + 1) * (p + 2) / 6
  estat_skew <- n * b1p / 6
  p_skew <- pchisq(estat_skew, df = gl_skew, lower.tail = FALSE)
  
  # Curtose → normal padrão
  z_kurt <- (b2p - p * (p + 2)) / sqrt(8 * p * (p + 2) / n)
  p_kurt <- 2 * pnorm(abs(z_kurt), lower.tail = FALSE)
  
  cat("===== TESTE DE MARDIA (manual) =====\n")
  cat("Assimetria multivariada (b1p):", round(b1p, 4), "\n")
  cat("  Estatística:", round(estat_skew, 2),
      "| gl:", gl_skew, "| p-valor:", format.pval(p_skew), "\n")
  cat("Curtose multivariada (b2p):", round(b2p, 4),
      "(esperado sob normal:", p*(p+2), ")\n")
  cat("  Z:", round(z_kurt, 2), "| p-valor:", format.pval(p_kurt), "\n")
  
  invisible(list(b1p = b1p, b2p = b2p, p_skew = p_skew, p_kurt = p_kurt))
}

res <- residuals(fit)
mardia_manual(res)

# ---- 2. Outliers multivariados (Mahalanobis) ----
# d² = distância de cada resíduo ao centro, ponderada pela covariância.
# Sob normalidade, d² ~ qui-quadrado com q (=3) graus de liberdade.
centro <- colMeans(res)
S      <- cov(res)
d2     <- mahalanobis(res, center = centro, cov = S)

# Ponto de corte: quantil 97,5% da qui-quadrado com 3 g.l.
corte <- qchisq(0.975, df = ncol(res))
n_out <- sum(d2 > corte)

cat("\n===== OUTLIERS MULTIVARIADOS (Mahalanobis) =====\n")
cat("Ponto de corte (χ²_0.975, 3 gl):", round(corte, 3), "\n")
cat("Outliers detectados:", n_out, "de", nrow(res),
    "(", round(100 * n_out / nrow(res), 1), "%)\n")

# Sob normalidade perfeita, esperaríamos ~2,5% acima do corte.
# Muito mais que isso = caudas pesadas (típico de dados reais).

# ============================================================
# ANÁLISE DE ROBUSTEZ — modelo sem outliers multivariados
# ============================================================
# d2 e corte já existem do diagnóstico anterior.
# Marcamos quais linhas NÃO são outliers e refazemos o modelo.

base_sem_out <- base[d2 <= corte, ]   # mantém só os não-outliers
cat("Observações originais:", nrow(base), "\n")
cat("Após remover outliers:", nrow(base_sem_out),
    "(", nrow(base) - nrow(base_sem_out), "removidas )\n\n")

# Reajuste do modelo multivariado na base limpa
fit_rob <- lm(
  cbind(Y1_log_tjulg, Y2_log_tbaixa, Y3_log_iad) ~ sub_area + sigla_grau + log_pendentes,
  data = base_sem_out
)

# ---- Comparação dos coeficientes (foco no tipo de crime) ----
cat("===== COMPARAÇÃO DOS COEFICIENTES =====\n")
comp <- data.frame(
  Original  = round(coef(fit)[, 1], 4),       # Y1 do modelo original
  Sem_Outl  = round(coef(fit_rob)[, 1], 4)    # Y1 do modelo robusto
)
comp$Dif <- round(comp$Sem_Outl - comp$Original, 4)
cat("\n--- Y1 (tempo julgamento) ---\n")
print(comp)

# Mesma comparação para Y2 e Y3
for (j in 2:3) {
  nome <- colnames(coef(fit))[j]
  cat("\n---", nome, "---\n")
  cmp <- data.frame(
    Original = round(coef(fit)[, j], 4),
    Sem_Outl = round(coef(fit_rob)[, j], 4)
  )
  cmp$Dif <- round(cmp$Sem_Outl - cmp$Original, 4)
  print(cmp)
}

# ---- Inferência multivariada na base limpa ----
cat("\n===== PILLAI — modelo sem outliers =====\n")
car::Anova(fit_rob, test.statistic = "Pillai")