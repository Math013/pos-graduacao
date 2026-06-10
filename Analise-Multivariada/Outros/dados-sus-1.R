#Em 26/02 Ajustes do Script criado por Ana Lorena Ferreira e Nahari - Consultora Técnica CGAN - MS
#Objetivo: Atualizar indicadores de estado nutricional infantil (RIPSA).

library(anthro)
library(epiDisplay)
library(dplyr)
library(anthro)
library(survey)
library(srvyr)
library(haven)
library(survey)
library(jtools)
library(remotes)
library(svrepmisc)

banco <- read_dta("C:/Users/Matheus/Desktop/Arquivos_PUC/Análise Multivariada/PNSN1989-PNDS1996e2006-.dta")
View(banco)

summary(banco$idademes)
summary(banco$peso)
summary(banco$altura)

## Selecionar variaveis ####
names(banco)
data = banco %>% dplyr::select("ident","anopesq","idademes","peso","altura","sex","M111_COR", "bmi", "baz2006",
                        "baz06", "waz06", "haz06", "regiao", "quintil", "pond", "est_geo", "sample06")

#Para usar o anthro
#sex: 1 for males and 2 for females or "m" or "M" for males and "f" or "F" for females. 
#age: age can be in either days or months (if optional argument is_age_in_month is set to TRUE). 
#weight: must be in kilograms. 
#lenhei: (length or height) must be in centimeters. 

data = data %>% mutate(sex = case_when(sex ==  1 ~ "M",sex ==2~"F"))

## Selecionando apenas as plausiveis
data <- data[which(data$sample06 == 1),]

#classificação do CV: Referência: https://bvsms.saude.gov.br/bvs/pnds/img/Minicurso_PNDS2006_4_Expansao_ponderacaoIBlavatsky.pdf
# 0-5: ótimo
# 5-15: bom
# 15-30: razoável
# 30-50: pouco precisa
# >50: impreciso

#Corrigir variável de raça cor
data <- data %>% 
  dplyr::mutate(
  raca_cor = case_when(
  M111_COR == 1 ~ 1,
  M111_COR == 2 ~ 2,
  M111_COR == 3 ~ 3,
  M111_COR == 4 ~ 4,
  M111_COR == 5 ~ 5,
  M111_COR >=6 ~ NA_integer_), 
  raca_cor = haven::labelled(raca_cor, labels = c(
    "Branco" =1, "Preta" =2, "Parda" =3, "Amarela"=4,"Indígena"=5)),
  raca_cor = haven::as_factor(raca_cor, levels = "labels")) 
  


### Separar por ano de pesquisa
pnds2006_ <- data[data$anopesq == 2006,]
pnds1996_ <- data[data$anopesq == 1996,]
pnsn_ <- data[data$anopesq == 1989,]


## Expansão banco ####

pnsn<- svydesign(id=~1,
                     weights = ~pond,
                     strata = ~est_geo,
                     nest=TRUE,
                     data = pnsn_)

pnds1996<- svydesign(id=~1,
                     weights = ~pond,
                     strata = ~est_geo,
                     nest=TRUE,
                     data = pnds1996_)

pnds2006<- svydesign(id=~1,
                     weights = ~pond,
                     strata = ~est_geo,
                     nest=TRUE,
                     data = pnds2006_)


################PNSN-1989#################
##########################################
#FRP.5.3 – Prevalência de excesso de peso segundo IMC para idade em crianças menores de 5 anos(baz06)
#Geral
svymean(~baz06, pnsn, na = TRUE)
cv(svymean(~baz06, pnsn, na = TRUE))
confint(svymean(~baz06, pnsn, na = TRUE))

#Por sexo (sex)
survey::svyby(~baz06, ~sex, design = pnsn, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~baz06, ~sex, design = pnsn, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~baz06, ~sex, design = pnsn, na = TRUE,na.rm=TRUE,svymean))

#Por região (regiao)
survey::svyby(~baz06, ~regiao,design = pnsn, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~baz06, ~regiao,design = pnsn, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~baz06, ~regiao,design = pnsn, na = TRUE,na.rm=TRUE,svymean))

#Por região (regiao) e sexo (sex)
survey::svyby(~baz06, ~regiao+sex,design = pnsn, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~baz06, ~regiao+sex,design = pnsn, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~baz06, ~regiao+sex,design = pnsn, na = TRUE,na.rm=TRUE,svymean))

#Por renda (quintil de IEN)
survey::svyby(~baz06, ~quintil,design = pnsn, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~baz06, ~quintil,design = pnsn, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~baz06, ~quintil,design = pnsn, na = TRUE,na.rm=TRUE,svymean))

################PNDS-1996#################
##########################################
##FRP.5.3 – Prevalência de excesso de peso segundo IMC para idade em crianças menores de 5 anos(baz06)

#Geral
svymean(~baz06, pnds1996, na = TRUE)
cv(svymean(~baz06, pnds1996, na = TRUE))
confint(svymean(~baz06, pnds1996, na = TRUE))

#Por sexo (sex)
survey::svyby(~baz06, ~sex, design = pnds1996, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~baz06, ~sex, design = pnds1996, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~baz06, ~sex, design = pnds1996, na = TRUE,na.rm=TRUE,svymean))

#Por região (regiao)
survey::svyby(~baz06, ~regiao,design = pnds1996, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~baz06, ~regiao,design = pnds1996, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~baz06, ~regiao,design = pnds1996, na = TRUE,na.rm=TRUE,svymean))

#Por região (regiao) e sexo (sex)
survey::svyby(~baz06, ~regiao+sex,design = pnds1996, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~baz06, ~regiao+sex,design = pnds1996, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~baz06, ~regiao+sex,design = pnds1996, na = TRUE,na.rm=TRUE,svymean))

#Por renda (quintil de IEN)
survey::svyby(~baz06, ~quintil,design = pnds1996, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~baz06, ~quintil,design = pnds1996, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~baz06, ~quintil,design = pnds1996, na = TRUE,na.rm=TRUE,svymean))

#Por raça-cor (raca_cor)
survey::svyby(~baz06, ~raca_cor,design = pnds1996, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~baz06, ~raca_cor,design = pnds1996, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~baz06, ~raca_cor,design = pnds1996, na = TRUE,na.rm=TRUE,svymean))


################PNDS-2006#################
##########################################
##FRP.5.3 – Prevalência de excesso de peso segundo IMC para idade em crianças menores de 5 anos(baz06)
#Geral
svymean(~baz06, pnds2006, na = TRUE)
cv(svymean(~baz06, pnds2006, na = TRUE))
confint(svymean(~baz06, pnds2006, na = TRUE))

#Por sexo (sex)
survey::svyby(~baz06, ~sex, design = pnds2006, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~baz06, ~sex, design = pnds2006, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~baz06, ~sex, design = pnds2006, na = TRUE,na.rm=TRUE,svymean))

#Por região (regiao)
survey::svyby(~baz06, ~regiao,design = pnds2006, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~baz06, ~regiao,design = pnds2006, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~baz06, ~regiao,design = pnds2006, na = TRUE,na.rm=TRUE,svymean))

#Por região (regiao) e sexo (sex)
survey::svyby(~baz06, ~regiao+sex,design = pnds2006, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~baz06, ~regiao+sex,design = pnds2006, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~baz06, ~regiao+sex,design = pnds2006, na = TRUE,na.rm=TRUE,svymean))

#Por renda (quintil de IEN)
survey::svyby(~baz06, ~quintil,design = pnds2006, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~baz06, ~quintil,design = pnds2006, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~baz06, ~quintil,design = pnds2006, na = TRUE,na.rm=TRUE,svymean))

#Por raca_cor da mãe (raca_cor)
survey::svyby(~baz06, ~raca_cor,design = pnds2006, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~baz06, ~raca_cor,design = pnds2006, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~baz06, ~raca_cor,design = pnds2006, na = TRUE,na.rm=TRUE,svymean))


#_+_#_+_##_+_#_+_##_+_#_+_##_+_#_+_##_+_#_+_##_+_#_+_#
#_+_#_+_##_+_#_+_##_+_#_+_##_+_#_+_##_+_#_+_##_+_#_+_#
#_+_#_+_##_+_#_+_##_+_#_+_##_+_#_+_##_+_#_+_##_+_#_+_#

######Banco do ENANI 2019##############
library(base)
#Se você estiver usando o banco em formato .rds ele já contêm os microdados e as informações do desenho amostral,
#assim como a calibração.
#Caso esteja usando o banco em .csv deve-se  declarar o objeto de desenho e submeter à aplicação da calibração dos
#pesos amostrais básicos usando pós-estratificação 

#Banco em .rds (formato recomendado)
banco.completo <- readRDS("ENANI.rds")
enani <- banco.completo

#Banco em .csv
#enani<- svydesign(id=~id_upa_anon,
#weights = ~peso_crianca,
#strata = ~estrato_sel_anon,
#nest=TRUE,
#data = Enani_final)


enani$variables <- enani$variables %>%
  dplyr::mutate(zhaz_categ2 = dplyr::case_when(is.na(vd_zhaz) ~ NA_character_,
                                               vd_zhaz < -2 ~ "12",
                                               vd_zhaz >= -2 ~ "3"),
                zhaz_categ2 = haven::labelled(zhaz_categ2, labels = c("Baixa altura\nZ < -2" = "12",
                                                                      "Altura adequada\nZ ≥ -2"= "3")),
                zhaz_categ2 = haven::as_factor(zhaz_categ2, levels = "labels")) %>%
  
  dplyr::mutate(zhaz_categ2_new = dplyr::case_when(vd_zhaz < -2 ~ "12",
                                                   vd_zhaz >= -2 ~ "3", 
                                                   TRUE ~ "12"),
                zhaz_categ2_new = haven::labelled(zhaz_categ2_new, labels = c("Baixa altura\nZ < -2" = "12",
                                                                              "Altura adequada\nZ ≥ -2"= "3")),
                zhaz_categ2_new = haven::as_factor(zhaz_categ2_new, levels = "labels")) %>%
  
  dplyr::mutate(zhaz_dummy = ifelse(zhaz_categ2 == "Baixa altura\nZ < -2", 1, 0),
                zhaz_dummy_categ = factor(zhaz_dummy, levels = c(0,1), labels = c("No", "Yes"))) %>%
  
 
   dplyr::mutate(zwaz_categ2 = dplyr::case_when(is.na(vd_zwaz) ~ NA_character_,
                                               vd_zwaz < -2 ~ "12",
                                               vd_zwaz >= -2 ~ "3"),
                zwaz_categ2 = haven::labelled(zwaz_categ2, labels = c("Baixo peso\nZ < -2" = "12",
                                                                      "Peso adequado ou elevado\nZ ≥ -2"= "3")),
                zwaz_categ2 = haven::as_factor(zwaz_categ2, levels = "labels")) %>%
  
  dplyr::mutate(zwaz_categ2_new = dplyr::case_when(vd_zwaz < -2 ~ "12",
                                                   vd_zwaz >= -2 ~ "3", 
                                                   TRUE ~ "12"),
                zwaz_categ2_new = haven::labelled(zwaz_categ2_new, labels = c("Baixo peso\nZ < -2" = "12",
                                                                              "Peso adequado ou elevado\nZ ≥ -2"= "3")),
                zwaz_categ2_new = haven::as_factor(zwaz_categ2_new, levels = "labels")) %>%
  
  dplyr::mutate(zwaz_dummy = ifelse(zwaz_categ2 == "Baixo peso\nZ < -2", 1, 0),
                zwaz_dummy_categ = factor(zwaz_dummy, levels = c(0,1), labels = c("No", "Yes"))) %>%
  
  dplyr::mutate(zimc_categ2 = dplyr::case_when(is.na(vd_zimc) ~ NA_character_,
                                               vd_zimc < -2 ~ "12",
                                               (vd_zimc >= -2 & vd_zimc <= 1) ~ "3",
                                               (vd_zimc > 1 & vd_zimc <= 2) ~ "4",
                                               (vd_zimc > 2) ~ "56"),
                zimc_categ2 = haven::labelled(zimc_categ2, labels = c("Magreza\nZ < -2" = "12",
                                                                      "Eutrofia\n -2 ≤ Z ≤ 1" = "3",
                                                                      "Risco de sobrepeso\n 1 < Z ≤ 2" = "4",
                                                                      "Overweight\n Z > 2" = "56")),
                zimc_categ2 = haven::as_factor(zimc_categ2, levels = "labels")) %>%
  
  
  dplyr::mutate(zimc_categ2_new = dplyr::case_when(vd_zimc < -2 ~ "12",
                                                   (vd_zimc >= -2 & vd_zimc <= 1) ~ "3",
                                                   (vd_zimc > 1 & vd_zimc <= 2) ~ "4",
                                                   (vd_zimc > 2) ~ "56",
                                                   TRUE ~ "12"),
                zimc_categ2_new = haven::labelled(zimc_categ2_new, labels = c("Magreza\nZ < -2" = "12",
                                                                              "Eutrofia\n -2 ≤ Z ≤ 1" = "3",
                                                                              "Risco de sobrepeso\n 1 < Z ≤ 2" = "4",
                                                                              "Overweight\n Z > 2" = "56")),
                zimc_categ2_new = haven::as_factor(zimc_categ2_new, levels = "labels"))%>%

dplyr::mutate(zimc_dummy = ifelse(zimc_categ2 == "Overweight\n Z > 2", 1, 0),
              zimc_dummy_categ = factor(zimc_dummy, levels = c(0,1), labels = c("No", "Yes")))

enani$variables$zhaz_dummy_categ
enani$variables$zwaz_dummy_categ
enani$variables$zimc_dummy_categ


###########ENANI - 2019 #################
#FRP.5.3 – Prevalência de excesso de peso segundo IMC para idade em crianças menores de 5 anos 
#Geral

svymean(~zimc_dummy_categ, enani, na = TRUE)
cv(svymean(~zimc_dummy_categ, enani, na = TRUE))
confint(svymean(~zimc_dummy_categ, enani, na = TRUE))

#Por sexo (b02_sexo)
survey::svyby(~zimc_dummy_categ, ~b02_sexo, design = enani, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~zimc_dummy_categ, ~b02_sexo, design = enani, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~zimc_dummy_categ, ~b02_sexo, design = enani, na = TRUE,na.rm=TRUE,svymean))

#Por região (a00_regiao)
survey::svyby(~zimc_dummy_categ, ~a00_regiao,design = enani, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~zimc_dummy_categ, ~a00_regiao,design = enani, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~zimc_dummy_categ, ~a00_regiao,design = enani, na = TRUE,na.rm=TRUE,svymean))

#Por região (a00_regiao) e b02_sexoo (b02_sexo)
survey::svyby(~zimc_dummy_categ, ~a00_regiao+b02_sexo,design = enani, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~zimc_dummy_categ, ~a00_regiao+b02_sexo,design = enani, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~zimc_dummy_categ, ~a00_regiao+b02_sexo,design = enani, na = TRUE,na.rm=TRUE,svymean))

#Por IEN
survey::svyby(~zimc_dummy_categ, ~vd_ien_quintos,design = enani, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~zimc_dummy_categ, ~vd_ien_quintos,design = enani, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~zimc_dummy_categ, ~vd_ien_quintos,design = enani, na = TRUE,na.rm=TRUE,svymean))

#Por raça-cor da mãe
survey::svyby(~zimc_dummy_categ, ~j03_cor,design = enani, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~zimc_dummy_categ, ~j03_cor,design = enani, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~zimc_dummy_categ, ~j03_cor,design = enani, na = TRUE,na.rm=TRUE,svymean))


#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*

################PNSN-1989#################
##########################################
### FRP.5.4 – Prevalência de déficit ponderal para a idade em crianças menores de 5 anos de idade (waz06)
#Geral
svymean(~waz06, pnsn, na = TRUE)
cv(svymean(~waz06, pnsn, na = TRUE))
confint(svymean(~waz06, pnsn, na = TRUE))

#Por sexo (sex)
survey::svyby(~waz06, ~sex, design = pnsn, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~waz06, ~sex, design = pnsn, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~waz06, ~sex, design = pnsn, na = TRUE,na.rm=TRUE,svymean))

#Por região (regiao)
survey::svyby(~waz06, ~regiao,design = pnsn, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~waz06, ~regiao,design = pnsn, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~waz06, ~regiao,design = pnsn, na = TRUE,na.rm=TRUE,svymean))

#Por região (regiao) e sexo (sex)
survey::svyby(~waz06, ~regiao+sex,design = pnsn, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~waz06, ~regiao+sex,design = pnsn, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~waz06, ~regiao+sex,design = pnsn, na = TRUE,na.rm=TRUE,svymean))

#Por renda (quintil de IEN)
survey::svyby(~waz06, ~quintil,design = pnsn, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~waz06, ~quintil,design = pnsn, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~waz06, ~quintil,design = pnsn, na = TRUE,na.rm=TRUE,svymean))

################PNDS-1996#################
##########################################
### FRP.5.4 – Prevalência de déficit ponderal para a idade em crianças menores de 5 anos de idade (waz06)
#Geral
svymean(~waz06, pnds1996, na = TRUE)
cv(svymean(~waz06, pnds1996, na = TRUE))
confint(svymean(~waz06, pnds1996, na = TRUE))

#Por sexo (sex)
survey::svyby(~waz06, ~sex, design = pnds1996, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~waz06, ~sex, design = pnds1996, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~waz06, ~sex, design = pnds1996, na = TRUE,na.rm=TRUE,svymean))

#Por região (regiao)
survey::svyby(~waz06, ~regiao,design = pnds1996, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~waz06, ~regiao,design = pnds1996, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~waz06, ~regiao,design = pnds1996, na = TRUE,na.rm=TRUE,svymean))

#Por região (regiao) e sexo (sex)
survey::svyby(~waz06, ~regiao+sex,design = pnds1996, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~waz06, ~regiao+sex,design = pnds1996, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~waz06, ~regiao+sex,design = pnds1996, na = TRUE,na.rm=TRUE,svymean))

#Por renda (quintil de IEN)
survey::svyby(~waz06, ~quintil,design = pnds1996, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~waz06, ~quintil,design = pnds1996, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~waz06, ~quintil,design = pnds1996, na = TRUE,na.rm=TRUE,svymean))

#Por raça-cor (raca_cor)
survey::svyby(~waz06, ~raca_cor,design = pnds1996, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~waz06, ~raca_cor,design = pnds1996, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~waz06, ~raca_cor,design = pnds1996, na = TRUE,na.rm=TRUE,svymean))


################PNDS-2006#################
##########################################
### FRP.5.4 – Prevalência de déficit ponderal para a idade em crianças menores de 5 anos de idade (waz06)
#Geral
svymean(~waz06, pnds2006, na = TRUE)
cv(svymean(~waz06, pnds2006, na = TRUE))
confint(svymean(~waz06, pnds2006, na = TRUE))

#Por sexo (sex)
survey::svyby(~waz06, ~sex, design = pnds2006, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~waz06, ~sex, design = pnds2006, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~waz06, ~sex, design = pnds2006, na = TRUE,na.rm=TRUE,svymean))

#Por região (regiao)
survey::svyby(~waz06, ~regiao,design = pnds2006, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~waz06, ~regiao,design = pnds2006, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~waz06, ~regiao,design = pnds2006, na = TRUE,na.rm=TRUE,svymean))

#Por região (regiao) e sexo (sex)
survey::svyby(~waz06, ~regiao+sex,design = pnds2006, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~waz06, ~regiao+sex,design = pnds2006, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~waz06, ~regiao+sex,design = pnds2006, na = TRUE,na.rm=TRUE,svymean))

#Por renda (quintil de IEN)
survey::svyby(~waz06, ~quintil,design = pnds2006, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~waz06, ~quintil,design = pnds2006, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~waz06, ~quintil,design = pnds2006, na = TRUE,na.rm=TRUE,svymean))

#Por raça-cor (raca_cor)
survey::svyby(~waz06, ~raca_cor,design = pnds2006, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~waz06, ~raca_cor,design = pnds2006, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~waz06, ~raca_cor,design = pnds2006, na = TRUE,na.rm=TRUE,svymean))


#_+_#_+_##_+_#_+_##_+_#_+_##_+_#_+_##_+_#_+_##_+_#_+_#
#_+_#_+#######Banco do ENANI 2019##############+_#_+_#
#_+_#_+_##_+_#_+_##_+_#_+_##_+_#_+_##_+_#_+_##_+_#_+_#

##########################################
### FRP.5.4 – Prevalência de déficit ponderal para a idade em crianças menores de 5 anos de idade 
#Geral

svymean(~zwaz_dummy_categ, enani, na = TRUE)
cv(svymean(~zwaz_dummy_categ, enani, na = TRUE))
confint(svymean(~zwaz_dummy_categ, enani, na = TRUE))

#Por sexo (b02_sexo)
survey::svyby(~zwaz_dummy_categ, ~b02_sexo, design = enani, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~zwaz_dummy_categ, ~b02_sexo, design = enani, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~zwaz_dummy_categ, ~b02_sexo, design = enani, na = TRUE,na.rm=TRUE,svymean))

#Por região (a00_regiao)
survey::svyby(~zwaz_dummy_categ, ~a00_regiao,design = enani, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~zwaz_dummy_categ, ~a00_regiao,design = enani, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~zwaz_dummy_categ, ~a00_regiao,design = enani, na = TRUE,na.rm=TRUE,svymean))

#Por região (a00_regiao) e b02_sexoo (b02_sexo)
survey::svyby(~zwaz_dummy_categ, ~a00_regiao+b02_sexo,design = enani, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~zwaz_dummy_categ, ~a00_regiao+b02_sexo,design = enani, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~zwaz_dummy_categ, ~a00_regiao+b02_sexo,design = enani, na = TRUE,na.rm=TRUE,svymean))

#Por IEN
survey::svyby(~zwaz_dummy_categ, ~vd_ien_quintos,design = enani, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~zwaz_dummy_categ, ~vd_ien_quintos,design = enani, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~zwaz_dummy_categ, ~vd_ien_quintos,design = enani, na = TRUE,na.rm=TRUE,svymean))

#Por raça-cor da mãe
survey::svyby(~zwaz_dummy_categ, ~j03_cor,design = enani, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~zwaz_dummy_categ, ~j03_cor,design = enani, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~zwaz_dummy_categ, ~j03_cor,design = enani, na = TRUE,na.rm=TRUE,svymean))

#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*

################PNSN-1989#################
##########################################
### FRP.5.5 – Prevalência de déficit estatural para a idade em crianças menores de 5 anos de idade (haz06)
#Geral
svymean(~haz06, pnsn, na = TRUE)
cv(svymean(~haz06, pnsn, na = TRUE))
confint(svymean(~haz06, pnsn, na = TRUE))

#Por sexo (sex)
survey::svyby(~haz06, ~sex, design = pnsn, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~haz06, ~sex, design = pnsn, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~haz06, ~sex, design = pnsn, na = TRUE,na.rm=TRUE,svymean))

#Por região (regiao)
survey::svyby(~haz06, ~regiao,design = pnsn, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~haz06, ~regiao,design = pnsn, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~haz06, ~regiao,design = pnsn, na = TRUE,na.rm=TRUE,svymean))

#Por região (regiao) e sexo (sex)
survey::svyby(~haz06, ~regiao+sex,design = pnsn, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~haz06, ~regiao+sex,design = pnsn, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~haz06, ~regiao+sex,design = pnsn, na = TRUE,na.rm=TRUE,svymean))

#Por renda (quintil de IEN)
survey::svyby(~haz06, ~quintil,design = pnsn, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~haz06, ~quintil,design = pnsn, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~haz06, ~quintil,design = pnsn, na = TRUE,na.rm=TRUE,svymean))


################PNDS-1996#################
##########################################
### FRP.5.5 – Prevalência de déficit estatural para a idade em crianças menores de 5 anos de idade (haz06)
#Geral
svymean(~haz06, pnds1996, na = TRUE)
cv(svymean(~haz06, pnds1996, na = TRUE))
confint(svymean(~haz06, pnds1996, na = TRUE))

#Por sexo (sex)
survey::svyby(~haz06, ~sex, design = pnds1996, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~haz06, ~sex, design = pnds1996, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~haz06, ~sex, design = pnds1996, na = TRUE,na.rm=TRUE,svymean))

#Por região (regiao)
survey::svyby(~haz06, ~regiao,design = pnds1996, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~haz06, ~regiao,design = pnds1996, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~haz06, ~regiao,design = pnds1996, na = TRUE,na.rm=TRUE,svymean))

#Por região (regiao) e sexo (sex)
survey::svyby(~haz06, ~regiao+sex,design = pnds1996, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~haz06, ~regiao+sex,design = pnds1996, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~haz06, ~regiao+sex,design = pnds1996, na = TRUE,na.rm=TRUE,svymean))

#Por renda (quintil de IEN)
survey::svyby(~haz06, ~quintil,design = pnds1996, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~haz06, ~quintil,design = pnds1996, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~haz06, ~quintil,design = pnds1996, na = TRUE,na.rm=TRUE,svymean))

#Por raça-cor da mãe (raca_cor)
survey::svyby(~haz06, ~raca_cor,design = pnds1996, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~haz06, ~raca_cor,design = pnds1996, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~haz06, ~raca_cor,design = pnds1996, na = TRUE,na.rm=TRUE,svymean))

################PNDS-2006#################
##########################################
### FRP.5.5 – Prevalência de déficit estatural para a idade em crianças menores de 5 anos de idade (haz06)
#Geral
svymean(~haz06, pnds2006, na = TRUE)
cv(svymean(~haz06, pnds2006, na = TRUE))
confint(svymean(~haz06, pnds2006, na = TRUE))

#Por sexo (sex)
survey::svyby(~haz06, ~sex, design = pnds2006, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~haz06, ~sex, design = pnds2006, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~haz06, ~sex, design = pnds2006, na = TRUE,na.rm=TRUE,svymean))

#Por região (regiao)
survey::svyby(~haz06, ~regiao,design = pnds2006, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~haz06, ~regiao,design = pnds2006, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~haz06, ~regiao,design = pnds2006, na = TRUE,na.rm=TRUE,svymean))

#Por região (regiao) e sexo (sex)
survey::svyby(~haz06, ~regiao+sex,design = pnds2006, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~haz06, ~regiao+sex,design = pnds2006, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~haz06, ~regiao+sex,design = pnds2006, na = TRUE,na.rm=TRUE,svymean))

#Por renda (quintil de IEN)
survey::svyby(~haz06, ~quintil,design = pnds2006, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~haz06, ~quintil,design = pnds2006, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~haz06, ~quintil,design = pnds2006, na = TRUE,na.rm=TRUE,svymean))

#Por raça-cor da mãe (raca_cor)
survey::svyby(~haz06, ~raca_cor,design = pnds2006, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~haz06, ~raca_cor,design = pnds2006, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~haz06, ~raca_cor,design = pnds2006, na = TRUE,na.rm=TRUE,svymean))

#_+_#_+_##_+_#_+_##_+_#_+_##_+_#_+_##_+_#_+_##_+_#_+_#
#_+_#_+#######Banco do ENANI 2019##############+_#_+_#
#_+_#_+_##_+_#_+_##_+_#_+_##_+_#_+_##_+_#_+_##_+_#_+_#

##########################################
### FRP.5.5 – Prevalência de déficit estatural para a idade em crianças menores de 5 anos de idade 
#Geral

svymean(~zhaz_dummy_categ, enani, na = TRUE)*100
cv(svymean(~zhaz_dummy_categ, enani, na = TRUE))
confint(svymean(~zhaz_dummy_categ, enani, na = TRUE))*100

#Por sexo (b02_sexo)
survey::svyby(~zhaz_dummy_categ, ~b02_sexo, design = enani, na = TRUE,na.rm=TRUE,svymean)*100
cv(survey::svyby(~zhaz_dummy_categ, ~b02_sexo, design = enani, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~zhaz_dummy_categ, ~b02_sexo, design = enani, na = TRUE,na.rm=TRUE,svymean))*100

#Por região (a00_regiao)
survey::svyby(~zhaz_dummy_categ, ~a00_regiao,design = enani, na = TRUE,na.rm=TRUE,svymean)*100
cv(survey::svyby(~zhaz_dummy_categ, ~a00_regiao,design = enani, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~zhaz_dummy_categ, ~a00_regiao,design = enani, na = TRUE,na.rm=TRUE,svymean))*100

#Por região (a00_regiao) e b02_sexoo (b02_sexo)
survey::svyby(~zhaz_dummy_categ, ~a00_regiao+b02_sexo,design = enani, na = TRUE,na.rm=TRUE,svymean)*100
cv(survey::svyby(~zhaz_dummy_categ, ~a00_regiao+b02_sexo,design = enani, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~zhaz_dummy_categ, ~a00_regiao+b02_sexo,design = enani, na = TRUE,na.rm=TRUE,svymean))*100

#Por IEN
survey::svyby(~zhaz_dummy_categ, ~vd_ien_quintos,design = enani, na = TRUE,na.rm=TRUE,svymean)*100
cv(survey::svyby(~zhaz_dummy_categ, ~vd_ien_quintos,design = enani, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~zhaz_dummy_categ, ~vd_ien_quintos,design = enani, na = TRUE,na.rm=TRUE,svymean))*100

#Por raça-cor da mãe
survey::svyby(~zhaz_dummy_categ, ~j03_cor,design = enani, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~zhaz_dummy_categ, ~j03_cor,design = enani, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~zhaz_dummy_categ, ~j03_cor,design = enani, na = TRUE,na.rm=TRUE,svymean))

#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~#
#################################################################################################
#~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~#
#FRP.5.6 Prevalência de dupla carga da má nutrição em crianças menores de 5 anos
#Calculo da má nutrição PNSN - 1989
pnsn$variables <- pnsn$variables %>%
  dplyr::mutate(zimc_categ2 = dplyr::case_when(is.na(baz2006) ~ NA_character_,
                                               baz2006 < -2 ~ "12",
                                               (baz2006 >= -2 & baz2006 <= 1) ~ "3",
                                               (baz2006 > 1 & baz2006 <= 2) ~ "4",
                                               (baz2006 > 2) ~ "56"),
                zimc_categ2 = haven::labelled(zimc_categ2, labels = c("Magreza\nZ < -2" = "12",
                                                                      "Eutrofia\n -2 ≤ Z ≤ 1" = "3",
                                                                      "Risco de sobrepeso\n 1 < Z ≤ 2" = "4",
                                                                      "Overweight\n Z > 2" = "56")),
                zimc_categ2 = haven::as_factor(zimc_categ2, levels = "labels"))%>%
  
  dplyr::mutate(ma_nutri_dummy = case_when((zimc_categ2 == "Overweight\n Z > 2" | zimc_categ2=="Magreza\nZ < -2") ~ 1,
                                           zimc_categ2== "Eutrofia\n -2 ≤ Z ≤ 1" ~ 0,
                                           zimc_categ2== "Risco de sobrepeso\n 1 < Z ≤ 2" ~ 0))

svymean(~ma_nutri_dummy, pnsn, na = TRUE)
cv(svymean(~ma_nutri_dummy, pnsn, na = TRUE))
confint(svymean(~ma_nutri_dummy, pnsn, na = TRUE))

#Calculo da má nutrição pnds1996
#FRP.5.6 Prevalência de dupla carga da má nutrição em crianças menores de 5 anos

pnds1996$variables <- pnds1996$variables %>%
  dplyr::mutate(zimc_categ2 = dplyr::case_when(is.na(baz2006) ~ NA_character_,
                                               baz2006 < -2 ~ "12",
                                               (baz2006 >= -2 & baz2006 <= 1) ~ "3",
                                               (baz2006 > 1 & baz2006 <= 2) ~ "4",
                                               (baz2006 > 2) ~ "56"),
                zimc_categ2 = haven::labelled(zimc_categ2, labels = c("Magreza\nZ < -2" = "12",
                                                                      "Eutrofia\n -2 ≤ Z ≤ 1" = "3",
                                                                      "Risco de sobrepeso\n 1 < Z ≤ 2" = "4",
                                                                      "Overweight\n Z > 2" = "56")),
                zimc_categ2 = haven::as_factor(zimc_categ2, levels = "labels"))%>%
  
  dplyr::mutate(ma_nutri_dummy = case_when((zimc_categ2 == "Overweight\n Z > 2" | zimc_categ2=="Magreza\nZ < -2") ~ 1,
                                           zimc_categ2== "Eutrofia\n -2 ≤ Z ≤ 1" ~ 0,
                                           zimc_categ2== "Risco de sobrepeso\n 1 < Z ≤ 2" ~ 0))

svymean(~ma_nutri_dummy, pnds1996, na = TRUE)
cv(svymean(~ma_nutri_dummy, pnds1996, na = TRUE))
confint(svymean(~ma_nutri_dummy, pnds1996, na = TRUE))

#Calculo da má nutrição pnds2006
#FRP.5.6 Prevalência de dupla carga da má nutrição em crianças menores de 5 anos

pnds2006$variables <- pnds2006$variables %>%
  dplyr::mutate(zimc_categ2 = dplyr::case_when(is.na(baz2006) ~ NA_character_,
                                               baz2006 < -2 ~ "12",
                                               (baz2006 >= -2 & baz2006 <= 1) ~ "3",
                                               (baz2006 > 1 & baz2006 <= 2) ~ "4",
                                               (baz2006 > 2) ~ "56"),
                zimc_categ2 = haven::labelled(zimc_categ2, labels = c("Magreza\nZ < -2" = "12",
                                                                      "Eutrofia\n -2 ≤ Z ≤ 1" = "3",
                                                                      "Risco de sobrepeso\n 1 < Z ≤ 2" = "4",
                                                                      "Overweight\n Z > 2" = "56")),
                zimc_categ2 = haven::as_factor(zimc_categ2, levels = "labels"))%>%
  
  dplyr::mutate(ma_nutri_dummy = case_when((zimc_categ2 == "Overweight\n Z > 2" | zimc_categ2=="Magreza\nZ < -2") ~ 1,
                                           zimc_categ2== "Eutrofia\n -2 ≤ Z ≤ 1" ~ 0,
                                           zimc_categ2== "Risco de sobrepeso\n 1 < Z ≤ 2" ~ 0))

svymean(~ma_nutri_dummy, pnds2006, na = TRUE)
cv(svymean(~ma_nutri_dummy, pnds2006, na = TRUE))
confint(svymean(~ma_nutri_dummy, pnds2006, na = TRUE))

#Calculo da má nutrição enani
#FRP.5.6 Prevalência de dupla carga da má nutrição em crianças menores de 5 anos
enani$variables <- enani$variables %>%
  dplyr::mutate(zimc_categ2 = dplyr::case_when(is.na(vd_zimc) ~ NA_character_,
                                               vd_zimc < -2 ~ "12",
                                               (vd_zimc >= -2 & vd_zimc <= 1) ~ "3",
                                               (vd_zimc > 1 & vd_zimc <= 2) ~ "4",
                                               (vd_zimc > 2) ~ "56"),
                zimc_categ2 = haven::labelled(zimc_categ2, labels = c("Magreza\nZ < -2" = "12",
                                                                      "Eutrofia\n -2 ≤ Z ≤ 1" = "3",
                                                                      "Risco de sobrepeso\n 1 < Z ≤ 2" = "4",
                                                                      "Overweight\n Z > 2" = "56")),
                zimc_categ2 = haven::as_factor(zimc_categ2, levels = "labels"))%>%
  
  dplyr::mutate(ma_nutri_dummy = case_when((zimc_categ2 == "Overweight\n Z > 2" | zimc_categ2=="Magreza\nZ < -2") ~ 1,
                                           zimc_categ2== "Eutrofia\n -2 ≤ Z ≤ 1" ~ 0,
                                           zimc_categ2== "Risco de sobrepeso\n 1 < Z ≤ 2" ~ 0))

svymean(~ma_nutri_dummy, enani, na = TRUE)
cv(svymean(~ma_nutri_dummy, enani, na = TRUE))
confint(svymean(~ma_nutri_dummy, enani, na = TRUE))

####### Os indicadores para tabela
#FRP.5.6 Prevalência de dupla carga da má nutrição em crianças menores de 5 anos
# PNSN-Brasil
svymean(~ma_nutri_dummy, pnsn, na = TRUE)
cv(svymean(~ma_nutri_dummy, pnsn, na = TRUE))
confint(svymean(~ma_nutri_dummy, pnsn, na = TRUE))
# Sexo
survey::svyby(~ma_nutri_dummy, ~sex, design = pnsn, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~ma_nutri_dummy, ~sex, design = pnsn, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~ma_nutri_dummy, ~sex, design = pnsn, na = TRUE,na.rm=TRUE,svymean))
# Região
survey::svyby(~ma_nutri_dummy, ~regiao, design = pnsn, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~ma_nutri_dummy, ~regiao, design = pnsn, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~ma_nutri_dummy, ~regiao, design = pnsn, na = TRUE,na.rm=TRUE,svymean))
# Sexo e Região
survey::svyby(~ma_nutri_dummy, ~regiao+sex, design = pnsn, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~ma_nutri_dummy, ~regiao+sex, design = pnsn, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~ma_nutri_dummy, ~regiao+sex, design = pnsn, na = TRUE,na.rm=TRUE,svymean))
#Por renda (quintil de IEN)
survey::svyby(~ma_nutri_dummy, ~quintil,design = pnsn, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~ma_nutri_dummy, ~quintil,design = pnsn, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~ma_nutri_dummy, ~quintil,design = pnsn, na = TRUE,na.rm=TRUE,svymean))

#################################################################################################

# pnds1996-Brasil
svymean(~ma_nutri_dummy, pnds1996, na = TRUE)
cv(svymean(~ma_nutri_dummy, pnds1996, na = TRUE))
confint(svymean(~ma_nutri_dummy, pnds1996, na = TRUE))
# Sexo
survey::svyby(~ma_nutri_dummy, ~sex, design = pnds1996, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~ma_nutri_dummy, ~sex, design = pnds1996, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~ma_nutri_dummy, ~sex, design = pnds1996, na = TRUE,na.rm=TRUE,svymean))
# Região
survey::svyby(~ma_nutri_dummy, ~regiao, design = pnds1996, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~ma_nutri_dummy, ~regiao, design = pnds1996, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~ma_nutri_dummy, ~regiao, design = pnds1996, na = TRUE,na.rm=TRUE,svymean))
# Sexo e Região
survey::svyby(~ma_nutri_dummy, ~regiao+sex, design = pnds1996, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~ma_nutri_dummy, ~regiao+sex, design = pnds1996, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~ma_nutri_dummy, ~regiao+sex, design = pnds1996, na = TRUE,na.rm=TRUE,svymean))
#Por renda (quintil de IEN)
survey::svyby(~ma_nutri_dummy, ~quintil,design = pnds1996, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~ma_nutri_dummy, ~quintil,design = pnds1996, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~ma_nutri_dummy, ~quintil,design = pnds1996, na = TRUE,na.rm=TRUE,svymean))
#Por raça-cor (raca_cor)
survey::svyby(~ma_nutri_dummy, ~raca_cor,design = pnds1996, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~ma_nutri_dummy, ~raca_cor,design = pnds1996, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~ma_nutri_dummy, ~raca_cor,design = pnds1996, na = TRUE,na.rm=TRUE,svymean))



# pnds2006-Brasil
svymean(~ma_nutri_dummy, pnds2006, na = TRUE)
cv(svymean(~ma_nutri_dummy, pnds2006, na = TRUE))
confint(svymean(~ma_nutri_dummy, pnds2006, na = TRUE))
# Sexo
survey::svyby(~ma_nutri_dummy, ~sex, design = pnds2006, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~ma_nutri_dummy, ~sex, design = pnds2006, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~ma_nutri_dummy, ~sex, design = pnds2006, na = TRUE,na.rm=TRUE,svymean))
# Região
survey::svyby(~ma_nutri_dummy, ~regiao, design = pnds2006, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~ma_nutri_dummy, ~regiao, design = pnds2006, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~ma_nutri_dummy, ~regiao, design = pnds2006, na = TRUE,na.rm=TRUE,svymean))
# Sexo e Região
survey::svyby(~ma_nutri_dummy, ~regiao+sex, design = pnds2006, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~ma_nutri_dummy, ~regiao+sex, design = pnds2006, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~ma_nutri_dummy, ~regiao+sex, design = pnds2006, na = TRUE,na.rm=TRUE,svymean))
#Por renda (quintil de IEN)
survey::svyby(~ma_nutri_dummy, ~quintil,design = pnds2006, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~ma_nutri_dummy, ~quintil,design = pnds2006, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~ma_nutri_dummy, ~quintil,design = pnds2006, na = TRUE,na.rm=TRUE,svymean))
#Por raça-cor (raca_cor)
survey::svyby(~ma_nutri_dummy, ~raca_cor,design = pnds2006, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~ma_nutri_dummy, ~raca_cor,design = pnds2006, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~ma_nutri_dummy, ~raca_cor,design = pnds2006, na = TRUE,na.rm=TRUE,svymean))


# enani-Brasil
svymean(~ma_nutri_dummy, enani, na = TRUE)
cv(svymean(~ma_nutri_dummy, enani, na = TRUE))
confint(svymean(~ma_nutri_dummy, enani, na = TRUE))
# b02_sexoo
survey::svyby(~ma_nutri_dummy, ~b02_sexo, design = enani, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~ma_nutri_dummy, ~b02_sexo, design = enani, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~ma_nutri_dummy, ~b02_sexo, design = enani, na = TRUE,na.rm=TRUE,svymean))
# Região
survey::svyby(~ma_nutri_dummy, ~a00_regiao, design = enani, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~ma_nutri_dummy, ~a00_regiao, design = enani, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~ma_nutri_dummy, ~a00_regiao, design = enani, na = TRUE,na.rm=TRUE,svymean))
# b02_sexoo e Região
survey::svyby(~ma_nutri_dummy, ~a00_regiao+b02_sexo, design = enani, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~ma_nutri_dummy, ~a00_regiao+b02_sexo, design = enani, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~ma_nutri_dummy, ~a00_regiao+b02_sexo, design = enani, na = TRUE,na.rm=TRUE,svymean))
#Por IEN
survey::svyby(~ma_nutri_dummy, ~vd_ien_quintos,design = enani, na = TRUE,na.rm=TRUE,svymean)*100
cv(survey::svyby(~ma_nutri_dummy, ~vd_ien_quintos,design = enani, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~ma_nutri_dummy, ~vd_ien_quintos,design = enani, na = TRUE,na.rm=TRUE,svymean))*100

#Por raça-cor da mãe
survey::svyby(~ma_nutri_dummy, ~j03_cor,design = enani, na = TRUE,na.rm=TRUE,svymean)
cv(survey::svyby(~ma_nutri_dummy, ~j03_cor,design = enani, na = TRUE,na.rm=TRUE,svymean))
confint(survey::svyby(~ma_nutri_dummy, ~j03_cor,design = enani, na = TRUE,na.rm=TRUE,svymean))

