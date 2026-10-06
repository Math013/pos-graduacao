# =============================================================================
# prep_rhc.R — Base analítica de sobrevivência a partir do RHC/SP (FOSP)
# Entrada : RHC_SP_AAAA.DBF (um arquivo por ano de diagnóstico)
# Saída   : rhc_analitica.rds  (lida pelo .Rmd do relatório)
# Códigos : Dicionário de Dados RHC/SP — FOSP, revisão 20/07/2026
# =============================================================================

library(foreign)
library(dplyr)
library(purrr)
library(survival)

dir_dados <- "C:/Users/Matheus/Desktop/Arquivos_PUC/Análise de Sobrevivência/ProjetoFinal"
TOPOS     <- c("C18", "C19", "C20")   # colorretal: cólon, junção retossigmoide, reto

# -----------------------------------------------------------------------------
# 1. Leitura e empilhamento (bronze)
# -----------------------------------------------------------------------------
arquivos <- list.files(dir_dados, pattern = "^RHC_SP_\\d{4}\\.DBF$",
                       full.names = TRUE, ignore.case = TRUE)
stopifnot(length(arquivos) > 0)

base_completa <- arquivos |>
  set_names(basename(arquivos)) |>
  map(\(f) read.dbf(f, as.is = TRUE)) |>
  map(\(d) mutate(d, across(everything(), as.character))) |>
  bind_rows(.id = "arquivo") |>
  mutate(across(where(is.character), \(x) iconv(x, from = "latin1", to = "UTF-8")))

# -----------------------------------------------------------------------------
# 2. Checagem dos domínios antes de filtrar (compare com o dicionário)
# -----------------------------------------------------------------------------
checar <- c("ULTINFO", "ERRO", "DIAGPREV", "ECGRUP", "PERDASEG", "CATEATEND", "SEXO")
walk(checar, \(v) { cat("\n==", v, "==\n"); print(table(base_completa[[v]], useNA = "ifany")) })

# -----------------------------------------------------------------------------
# 3. Funil de seleção (silver) — cada passo registra o n restante
# -----------------------------------------------------------------------------
funil <- tibble::tibble(etapa = character(), n = integer())

contar <- function(df, rotulo) {
  n <- nrow(df)                                        # força o pipe interno antes de ler o funil
  funil <<- tibble::add_row(funil, etapa = rotulo, n = n)
  message(sprintf("%-45s %8d", rotulo, n))
  df
}

rhc <- base_completa |>
  contar("Base empilhada") |>
  filter(TOPOGRUP %in% TOPOS) |>                     contar("Topografia colorretal (C18-C20)") |>
  filter(ERRO == "0") |>                             contar("Admissão sem erro") |>
  filter(DIAGPREV == "1") |>                         contar("Caso analítico (sem diag./trat. prévio)") |>
  mutate(IDADE = as.integer(IDADE)) |>
  filter(IDADE >= 18) |>                             contar("Idade >= 18") |>
  filter(ECGRUP %in% c("I", "II", "III", "IV")) |>   contar("Estádio I-IV conhecido") |>
  mutate(
    DTDIAG     = as.Date(DTDIAG),
    DTULTINFO  = as.Date(DTULTINFO),
    ULTINFO    = as.integer(ULTINFO),
    ANODIAG    = as.integer(ANODIAG),
    tempo_dias = as.numeric(DTULTINFO - DTDIAG),
    # NA em ULTINFO deve continuar NA: `%in%` sozinho transformaria NA em 0 (censura)
    status     = if_else(is.na(ULTINFO), NA_integer_,
                         as.integer(ULTINFO %in% c(3L, 4L)))   # 3 = óbito câncer, 4 = outras causas
  ) |>
  filter(!is.na(tempo_dias), !is.na(status)) |>     contar("Tempo e status não ausentes") |>
  filter(tempo_dias >= 0) |>                         contar("Tempo não negativo")

cat("\nObservações com tempo = 0:", sum(rhc$tempo_dias == 0), "\n")

# -----------------------------------------------------------------------------
# 4. Variáveis de análise
# -----------------------------------------------------------------------------
rhc <- rhc |>
  mutate(
    tempo_dias = pmax(tempo_dias, 0.5),   # distribuições paramétricas exigem t > 0
    tempo_anos = tempo_dias / 365.25,
    estadio    = factor(ECGRUP, levels = c("I", "II", "III", "IV")),
    sexo       = factor(SEXO, levels = c("1", "2"), labels = c("Masculino", "Feminino")),
    categoria  = factor(CATEATEND, levels = c("2", "1", "3"),
                        labels = c("SUS", "Convênio", "Particular")),   # SUS como referência
    cirurgia   = factor(CIRURGIA, levels = c("0", "1"), labels = c("Não", "Sim")),
    quimio     = factor(QUIMIO,   levels = c("0", "1"), labels = c("Não", "Sim")),
    radio      = factor(RADIO,    levels = c("0", "1"), labels = c("Não", "Sim"))
  )

# -----------------------------------------------------------------------------
# 5. Diagnóstico de seguimento por ano de diagnóstico
# -----------------------------------------------------------------------------
print(table(status = rhc$status))

rhc |>
  group_by(ANODIAG) |>
  summarise(n = n(),
            obitos = sum(status),
            pct_obito = round(100 * mean(status), 1),
            seg_max_anos = round(max(tempo_anos), 1)) |>
  print()

# KM reverso (Schemper & Smith, 1996): censura vira "evento" -> mediana de seguimento potencial
print(survfit(Surv(tempo_anos, 1 - status) ~ ANODIAG, data = rhc))

# -----------------------------------------------------------------------------
# 6. Persistência
# -----------------------------------------------------------------------------
saveRDS(rhc,   file.path(dir_dados, "rhc_analitica.rds"))
saveRDS(funil, file.path(dir_dados, "funil_selecao.rds"))   # fluxograma de seleção para o relatório
