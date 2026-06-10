# =============================================================================
# ANÁLISE DE ASSUNTOS JURÍDICOS — 2026
# =============================================================================

# -----------------------------------------------------------------------------
# 0. BIBLIOTECAS
# -----------------------------------------------------------------------------
library(duckdb)
library(DBI)
library(arrow)
library(dplyr)

# -----------------------------------------------------------------------------
# 1. CONEXÃO E EXPLORAÇÃO INICIAL DO PARQUET
# -----------------------------------------------------------------------------

PARQUET_PATH <- "C:/Users/Matheus/Desktop/Arquivos_PUC/tbl_fato_assuntos_R.parquet"

con <- dbConnect(duckdb::duckdb())

# Schema: tipos e nomes das colunas
schema <- dbGetQuery(con, glue::glue("
  DESCRIBE SELECT * FROM read_parquet('{PARQUET_PATH}')
"))
print(schema)

# Sumário estatístico por coluna (min, max, média, nulos etc.)
sumario <- dbGetQuery(con, glue::glue("
  SUMMARIZE SELECT * FROM read_parquet('{PARQUET_PATH}')
"))
View(sumario)

# -----------------------------------------------------------------------------
# 2. EXTRAÇÃO — APENAS O ANO DE INTERESSE
# -----------------------------------------------------------------------------

dados_raw <- dbGetQuery(con, glue::glue("
  SELECT *
  FROM read_parquet('{PARQUET_PATH}')
  WHERE ano = 2026
"))

dbDisconnect(con, shutdown = TRUE)

# -----------------------------------------------------------------------------
# 3. RENOMEAÇÃO DE COLUNAS
# -----------------------------------------------------------------------------

dados <- dados_raw %>%
  rename(
    # Indicadores principais
    casos_novos          = ind1,
    pendentes            = ind2,
    baixados             = ind3,
    pendentes_liquidos   = ind5,
    julgados             = ind8a,
    
    # Tempos médios (dias e nº de processos de referência)
    tp_baixa_dias        = ind16_dias,
    tp_baixa_proc        = ind16_proc,
    tp_julgamento_dias   = ind17_dias,
    tp_julgamento_proc   = ind17_proc,
    tp_pendente_liq_dias = ind18_dias,
    tp_pendente_liq_proc = ind18_proc,
    tp_pendente_dias     = ind19_dias,
    tp_pendente_proc     = ind19_proc,
    
    # 5% de processos mais antigos
    cinco_pct_total      = ind15total,
    cinco_pct_data_min   = ind15min,
    cinco_pct_data_max   = ind15max,
    
    # Redistribuições
    redistrib_entrada    = ind20,
    redistrib_saida      = ind21,
    
    # Pendentes com corte de 15 anos
    pendentes_15anos     = cp.15anos,
    pendentes_liq_15anos = cpl.15anos
  )

# -----------------------------------------------------------------------------
# 4. MAPEAMENTO DE ASSUNTOS
# -----------------------------------------------------------------------------

mapa_assunto <- c(
  "9607"  = "DIREITO CIVIL",
  "7698"  = "DIREITO CIVIL",
  "9587"  = "DIREITO CIVIL",
  "3402"  = "DIREITO PENAL",
  "3419"  = "DIREITO PENAL",
  "3416"  = "DIREITO PENAL",
  "12091" = "DIREITO PENAL",
  "15483" = "DIREITO PENAL",
  "15482" = "DIREITO PENAL"
)

mapa_sub_assunto <- c(
  "9607"  = "Contratos Bancários",
  "7698"  = "Perdas e Danos",
  "9587"  = "Compra e Venda",
  "3402"  = "Ameaça",
  "3419"  = "Roubo",
  "3416"  = "Furto",
  "12091" = "Feminicídio",
  "15483" = "Intimidação Sistemática Virtual (Cyberbullying)",
  "15482" = "Intimidação Sistemática (Bullying)"
)

dados_classificados <- dados %>%
  mutate(
    area_principal = mapa_assunto[as.character(id_assunto)],
    sub_area       = mapa_sub_assunto[as.character(id_assunto)]
  ) %>%
  relocate(area_principal, sub_area, .after = id_assunto)

# -----------------------------------------------------------------------------
# 5. DIAGNÓSTICO — IDs SEM CLASSIFICAÇÃO
# -----------------------------------------------------------------------------

ids_nao_mapeados <- dados_classificados %>%
  filter(is.na(area_principal)) %>%
  count(id_assunto, sort = TRUE)

print(ids_nao_mapeados)

# -----------------------------------------------------------------------------
# 6. SUBSETS POR ÁREA JURÍDICA
# -----------------------------------------------------------------------------

direito_civil <- dados_classificados %>%
  filter(area_principal == "DIREITO CIVIL")

arrow::write_parquet(direito_civil, "direito_civil.parquet")

direito_penal <- dados_classificados %>%
  filter(area_principal == "DIREITO PENAL")

arrow::write_parquet(direito_penal, "direito_penal.parquet")

# -----------------------------------------------------------------------------
# 7. LENDO ARQUIVOS CLUSTERIZADOS - DIREITO PENAL
# -----------------------------------------------------------------------------

dados_penal <- arrow::read_parquet("direito_penal.parquet")

dim(dados_penal)
glimpse(dados_penal)

# -----------------------------------------------------------------------------
# 8. DIAGNÓSTICO - DIREITO PENAL
# -----------------------------------------------------------------------------

library(tidyverse)

dados_penal %>%
  distinct(id_orgao_julgador, sigla_grau) %>%
  count(sigla_grau)

dados_penal %>%
  count(ano)

dados_penal %>%
  summarise(across(where(is.numeric), ~ mean(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "variavel", values_to = "pct_na") %>%
  arrange(desc(pct_na)) %>%
  print(n = Inf)

dados_g1 <- dados_penal %>% filter(sigla_grau == "G1")

dados_g1 %>%
  summarise(across(where(is.numeric), ~ mean(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "variavel", values_to = "pct_na") %>%
  arrange(desc(pct_na)) %>%
  print(n = Inf)

dados_g1 %>%
  mutate(baixados_na = is.na(baixados)) %>%
  group_by(baixados_na) %>%
  summarise(
    n = n(),
    casos_novos_mediana = median(casos_novos, na.rm = TRUE),
    casos_novos_p_na    = mean(is.na(casos_novos)),
    pendentes_mediana   = median(pendentes, na.rm = TRUE)
  )

# -----------------------------------------------------------------------------
# 8. DIREITO PENAL - FILTRO G1
# -----------------------------------------------------------------------------

vars_contagem <- c(
  "casos_novos", "pendentes", "pendentes_liquidos",
  "baixados", "julgados",
  "redistrib_entrada", "redistrib_saida"
)

dados_g1_trat <- dados_g1 %>%
  mutate(across(all_of(vars_contagem), ~ replace_na(., 0)))

dados_orgao <- dados_g1_trat %>%
  group_by(id_orgao_julgador, sigla_tribunal) %>%
  summarise(
    # Volumes (somar)
    casos_novos_tot   = sum(casos_novos),
    pendentes_tot     = sum(pendentes),
    pend_liq_tot      = sum(pendentes_liquidos),
    baixados_tot      = sum(baixados),
    julgados_tot      = sum(julgados),
    redistrib_in_tot  = sum(redistrib_entrada),
    redistrib_out_tot = sum(redistrib_saida),
    
    # Tempos (mediana, ignorando NA)
    tp_baixa_med      = median(tp_baixa_dias,      na.rm = TRUE),
    tp_julg_med       = median(tp_julgamento_dias, na.rm = TRUE),
    tp_pend_med       = median(tp_pendente_dias,   na.rm = TRUE),
    
    # Contexto
    n_assuntos        = n_distinct(id_assunto),
    n_linhas          = n(),
    
    .groups = "drop"
  )

dados_orgao <- dados_orgao %>%
  mutate(
    taxa_baixa   = baixados_tot / (casos_novos_tot + pendentes_tot + 1),
    taxa_congest = 1 - (baixados_tot / (casos_novos_tot + pendentes_tot + 1)),
    prod_julg    = julgados_tot / (casos_novos_tot + 1),
    saldo_proc   = casos_novos_tot - baixados_tot
  )

glimpse(dados_orgao)
cat("\nÓrgãos únicos:", nrow(dados_orgao), "\n\n")

# Quantos NAs sobraram?
dados_orgao %>%
  summarise(across(where(is.numeric), ~ mean(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "variavel", values_to = "pct_na") %>%
  arrange(desc(pct_na)) %>%
  print(n = Inf)

# Distribuição das variáveis numéricas (pra avaliar assimetria)
dados_orgao %>%
  select(where(is.numeric), -id_orgao_julgador) %>%
  summary()

# ============================================================
# LIMPEZA: filtros defensivos antes do PCA
# ============================================================

# Antes de filtrar, salvar diagnóstico do que vai ser removido
n_inicial <- nrow(dados_orgao)
cat("Órgãos antes da limpeza:", n_inicial, "\n")

dados_orgao_clean <- dados_orgao %>%
  # FILTRO 1: remover IDs sentinelas (não são varas reais)
  filter(id_orgao_julgador > 0) %>%
  
  # FILTRO 2: remover órgãos com volume muito baixo
  # Critério: pelo menos 10 casos (novos + pendentes) no recorte
  # Justificativa: abaixo disso, taxas viram ruído puro
  filter((casos_novos_tot + pendentes_tot) >= 10) %>%
  
  # FILTRO 3: remover órgãos sem nenhum tempo de baixa registrado
  # Justificativa: sem tempos, não conseguimos analisar morosidade
  filter(!is.na(tp_baixa_med)) %>%
  
  # FILTRO 4: remover taxas absurdas (>1.5 = artefato)
  # Justificativa: taxa de baixa biológica máxima é ~1 (com alguma folga)
  filter(taxa_baixa <= 1.5, taxa_baixa >= 0)

cat("Órgãos após limpeza:", nrow(dados_orgao_clean), "\n")
cat("Removidos:", n_inicial - nrow(dados_orgao_clean),
    "(", round(100*(1 - nrow(dados_orgao_clean)/n_inicial), 1), "%)\n\n")

# Diagnóstico pós-limpeza
dados_orgao_clean %>%
  select(where(is.numeric), -id_orgao_julgador) %>%
  summary()

n_inicial <- nrow(dados_orgao)

dados_orgao_clean <- dados_orgao %>%
  filter(id_orgao_julgador > 0) %>%
  filter((casos_novos_tot + pendentes_tot) >= 10) %>%
  filter(!is.na(tp_baixa_med)) %>%
  filter(taxa_baixa <= 1.5, taxa_baixa >= 0)

cat("Órgãos antes:", n_inicial, "\n")
cat("Órgãos depois:", nrow(dados_orgao_clean), "\n")
cat("Removidos:", n_inicial - nrow(dados_orgao_clean), "\n\n")

dados_orgao_clean %>%
  select(where(is.numeric), -id_orgao_julgador) %>%
  summary()

# ============================================================
# PASSO 1: Selecionar variáveis para o PCA
# ============================================================
vars_pca <- c(
  "casos_novos_tot", "pendentes_tot", "baixados_tot", "julgados_tot",
  "redistrib_in_tot", "redistrib_out_tot",
  "tp_baixa_med", "tp_pend_med",
  "taxa_baixa"
)

# Submatriz só com as variáveis numéricas + id como rótulo de linha
mat_pca <- dados_orgao_clean %>%
  select(id_orgao_julgador, all_of(vars_pca)) %>%
  # Aplica log1p em tudo MENOS taxa_baixa
  mutate(across(all_of(setdiff(vars_pca, "taxa_baixa")), ~ log1p(.))) %>%
  # Confere que não sobrou NA
  drop_na()

cat("Matriz para PCA:", nrow(mat_pca), "órgãos x", ncol(mat_pca) - 1, "variáveis\n")

# ============================================================
# PASSO 2: Rodar PCA sobre a matriz de CORRELAÇÃO
# ============================================================
# scale. = TRUE faz a padronização automaticamente (PCA sobre R, não Σ)
# center = TRUE remove a média de cada variável

pca <- prcomp(
  mat_pca %>% select(-id_orgao_julgador),
  center = TRUE,
  scale. = TRUE
)

# ============================================================
# PASSO 3: Diagnóstico básico do PCA
# ============================================================
# Resumo: variância explicada por componente
summary(pca)

# Scree plot — variância de cada componente
# (pode usar base R ou factoextra para uma versão mais bonita)
plot(pca, type = "l", main = "Scree plot — varas criminais G1")

# Autovalores: aplicar critério de Kaiser (lambda > 1)
autovalores <- pca$sdev^2
cat("\nAutovalores (eigen):\n")
print(round(autovalores, 3))
cat("\nCritério de Kaiser (lambda > 1) sugere k =",
    sum(autovalores > 1), "componentes\n")

# Variância acumulada
var_acum <- cumsum(autovalores) / sum(autovalores)
cat("\nVariância acumulada:\n")
print(round(var_acum, 3))