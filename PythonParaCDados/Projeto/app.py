# ============================================================
# 🦠 Painel Epidemiológico – caso_full (Streamlit + Plotly)
# ============================================================

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import requests, gzip
from sklearn.linear_model import LinearRegression

# ---------- Config inicial ----------
st.set_page_config(
    page_title="Painel Epidemiológico | caso_full",
    page_icon="🦠",
    layout="wide"
)

st.title("🦠 Painel Epidemiológico – COVID-19 (caso_full)")
st.caption("Dados do Brasil.io, com análises interativas de mortalidade, contaminação e dispersão.")

# ============================================================
# Utils
# ============================================================
REGIOES = {
    "Norte": ["AC","AM","AP","PA","RO","RR","TO"],
    "Nordeste": ["AL","BA","CE","MA","PB","PE","PI","RN","SE"],
    "Centro-Oeste": ["DF","GO","MT","MS"],
    "Sudeste": ["ES","MG","RJ","SP"],
    "Sul": ["PR","RS","SC"],
}
MAPA_REGIOES = {uf: reg for reg,ufs in REGIOES.items() for uf in ufs}

EXPECTED_COLS = [
    "city","city_ibge_code","date","epidemiological_week",
    "estimated_population","estimated_population_2019","is_last",
    "is_repeated","last_available_confirmed",
    "last_available_confirmed_per_100k_inhabitants","last_available_date",
    "last_available_death_rate","last_available_deaths","order_for_place",
    "place_type","state","new_confirmed","new_deaths"
]

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    num_cols = [
        "estimated_population","estimated_population_2019",
        "last_available_confirmed","last_available_confirmed_per_100k_inhabitants",
        "last_available_death_rate","last_available_deaths",
        "new_confirmed","new_deaths"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "is_last" in df: 
        df["is_last"] = df["is_last"].astype(bool, errors="ignore")
    if "is_repeated" in df:
        df["is_repeated"] = df["is_repeated"].astype(bool, errors="ignore")
    df["region"] = df["state"].map(MAPA_REGIOES)
    df["taxa_mortalidade"] = (df["last_available_death_rate"] * 100).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["taxa_contaminacao"] = (df["last_available_confirmed"] / df["estimated_population"]) * 100
    return df

def split_city_state_latest(df: pd.DataFrame):
    df = standardize_columns(df)
    df_state_nr = df[(df["is_repeated"] == False) & (df["place_type"] == "state")].copy()
    df_state_last = df[(df["is_last"] == True) & (df["place_type"] == "state")].copy()
    df_city = df[(df["place_type"] != "state") & df["city"].notna()].copy()
    df_city = df_city[df_city["city"] != "Importados/Indefinidos"]
    df_city_nr = df_city[df_city["is_repeated"] == False].copy()
    df_city_last = df_city[df_city["is_last"] == True].copy()
    return df_state_nr, df_state_last, df_city_nr, df_city_last

def stats_dispersion(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return 0, 0, 0
    media = s.mean()
    dp = s.std(ddof=1)
    cv = dp / media * 100 if media != 0 else 0
    return media, dp, cv

# ============================================================
# 📥 Fonte de Dados Automática – Brasil.io
# ============================================================
st.subheader("📁 Fonte de Dados – Brasil.io")

url = "https://data.brasil.io/dataset/covid19/caso_full.csv.gz"
with st.spinner("Baixando e descompactando dados do Brasil.io..."):
    response = requests.get(url)
    response.raise_for_status()
    with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz:
        df = pd.read_csv(gz)

df = standardize_columns(df)
df_state_nr, df_state_last, df_city_nr, df_city_last = split_city_state_latest(df)

st.success("✅ Dados carregados com sucesso a partir do Brasil.io!")

# ============================================================
# KPIs
# ============================================================
st.subheader("🔎 Visão Geral")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Estados", f"{df_state_last['state'].nunique()}")
c2.metric("Cidades", f"{df_city_last['city'].nunique()}")
total_casos = int(df_state_nr.groupby("state")["last_available_confirmed"].max().sum())
total_mortes = int(df_state_nr.groupby("state")["last_available_deaths"].max().sum())
c3.metric("Casos Totais", f"{total_casos:,}".replace(",", "."))
c4.metric("Mortes Totais", f"{total_mortes:,}".replace(",", "."))

st.divider()

# ============================================================
# Mortes por Estado / Cidades
# ============================================================
col1, col2 = st.columns(2)
with col1:
    mortes_estado = (
        df_state_nr.groupby("state")["last_available_deaths"].max().reset_index()
        .sort_values("last_available_deaths", ascending=True)
    )
    fig = px.bar(
        mortes_estado,
        x="last_available_deaths",
        y="state",
        orientation="h",
        color="last_available_deaths",
        color_continuous_scale="Blues",
        title="💀 Mortes por Estado",
        text=mortes_estado["last_available_deaths"].apply(lambda x: f"{x:,}".replace(",", ".")),
    )
    fig.update_layout(template="plotly_dark", height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    mortes_cidade = (
        df_city_last.groupby(["state","city"])["last_available_deaths"]
        .max().reset_index().sort_values(by="last_available_deaths", ascending=False).head(20)
    )
    mortes_cidade["cidade_estado"] = mortes_cidade["city"] + " (" + mortes_cidade["state"] + ")"
    fig2 = px.bar(
        mortes_cidade.sort_values("last_available_deaths", ascending=True),
        x="last_available_deaths",
        y="cidade_estado",
        orientation="h",
        color="last_available_deaths",
        color_continuous_scale="Reds",
        title="🏙️ Top 20 Cidades com Mais Mortes",
        text=mortes_cidade["last_available_deaths"].apply(lambda x: f"{x:,}".replace(",", ".")),
    )
    fig2.update_layout(template="plotly_dark", height=500, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ============================================================
# Cidade × Estado (Taxa Mortalidade %)
# ============================================================
st.subheader("📍 Comparação Cidade × Estado (Taxa de Mortalidade %)")

estados = sorted(df_city_last["state"].unique())
c1, c2 = st.columns(2)
estado_sel = c1.selectbox("Estado", estados, index=estados.index("SP"))
cidades = sorted(df_city_last[df_city_last["state"] == estado_sel]["city"].unique())
cidade_sel = c2.selectbox("Cidade", cidades)

top_n = st.slider("Quantidade de cidades", 5, 50, 20, step=5)
base_estado = (
    df_city_last[df_city_last["state"] == estado_sel]
    .sort_values("taxa_mortalidade", ascending=False)
    .head(top_n)
)
media_est = (
    df_state_last.loc[df_state_last["state"] == estado_sel, "last_available_death_rate"].iloc[0] * 100
)
valor_cidade = df_city_last.loc[
    (df_city_last["state"] == estado_sel) & (df_city_last["city"] == cidade_sel),
    "taxa_mortalidade"
].iloc[0]

m1, m2, m3 = st.columns(3)
m1.metric("Taxa da Cidade (%)", f"{valor_cidade:.2f}")
m2.metric("Média Estadual (%)", f"{media_est:.2f}",
          delta=f"{valor_cidade - media_est:.2f}",
          delta_color="inverse" if valor_cidade < media_est else "normal")
m3.metric("Cidades", f"{len(cidades)}")

base_estado["cor"] = base_estado["city"].apply(lambda x: "salmon" if x == cidade_sel else "gray")
fig3 = px.bar(
    base_estado.sort_values("taxa_mortalidade", ascending=True),
    x="taxa_mortalidade",
    y="city",
    orientation="h",
    color="cor",
    color_discrete_map="identity",
    text=base_estado["taxa_mortalidade"].apply(lambda x: f"{x:.2f}%"),
    title=f"Taxa de Mortalidade – {estado_sel} (Top {top_n})",
)
fig3.add_vline(x=media_est, line_dash="dash", line_color="white")
fig3.add_annotation(x=media_est, y=-0.5, text=f"Média Estadual: {media_est:.2f}%", showarrow=False, font=dict(color="white"))
fig3.update_traces(textposition="outside", marker_line_color="white", marker_line_width=1.2)
fig3.update_layout(template="plotly_dark", height=700, showlegend=False)
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ============================================================
# Análises Regionais
# ============================================================
st.subheader("🗺️ Análises Regionais")

taxas_regiao = (
    df_city_last.groupby("region").agg({
        "estimated_population":"sum",
        "last_available_confirmed":"sum",
        "last_available_deaths":"sum"
    }).reset_index()
)
taxas_regiao["taxa_contaminacao"] = taxas_regiao["last_available_confirmed"] / taxas_regiao["estimated_population"] * 100
taxas_regiao["taxa_mortalidade"] = taxas_regiao["last_available_deaths"] / taxas_regiao["last_available_confirmed"] * 100

taxas_regiao = taxas_regiao.sort_values(by='taxa_contaminacao', ascending=True)

fig4 = px.bar(
    taxas_regiao,
    x="taxa_contaminacao", y="region", orientation="h",
    color="taxa_contaminacao", color_continuous_scale="Purples",
    title="Taxa de Contaminação por Região (%)",
    text=taxas_regiao["taxa_contaminacao"].apply(lambda x: f"{x:.2f}%"),
)
fig4.update_layout(template="plotly_dark", height=500, showlegend=False)
st.plotly_chart(fig4, use_container_width=True)

taxas_regiao = taxas_regiao.sort_values(by='taxa_mortalidade', ascending=True)

fig5 = px.bar(
    taxas_regiao,
    x="taxa_mortalidade", y="region", orientation="h",
    color="taxa_mortalidade", color_continuous_scale="Blues",
    title="Taxa de Mortalidade por Região (%)",
    text=taxas_regiao["taxa_mortalidade"].apply(lambda x: f"{x:.2f}%"),
)
fig5.update_layout(template="plotly_dark", height=500, showlegend=False)
st.plotly_chart(fig5, use_container_width=True)

st.dataframe(taxas_regiao.round(3))

st.divider()

# ============================================================
# Regressões Lineares
# ============================================================
st.subheader("📈 Regressões Lineares")

df_reg = df_city_last[(df_city_last["last_available_confirmed"] > 0) &
                      (df_city_last["last_available_deaths"] > 0) &
                      (df_city_last["estimated_population"] > 0)]
x_vars = {
    "Casos Confirmados":"last_available_confirmed",
    "População Estimada":"estimated_population",
    "Casos por 100 mil hab.":"last_available_confirmed_per_100k_inhabitants"
}
y_vars = {
    "Mortes":"last_available_deaths",
    "Taxa de Mortalidade (%)":"taxa_mortalidade",
    "Taxa de Contaminação (%)":"taxa_contaminacao"
}
cX, cY = st.columns(2)
x_label = cX.selectbox("Eixo X", list(x_vars.keys()))
y_label = cY.selectbox("Eixo Y", list(y_vars.keys()))
X = df_reg[[x_vars[x_label]]].values
y = df_reg[y_vars[y_label]].values
model = LinearRegression().fit(X, y)
y_pred = model.predict(X)
r2 = model.score(X, y)
fig6 = px.scatter(df_reg, x=x_vars[x_label], y=y_vars[y_label], opacity=0.6,
                  title=f"{y_label} × {x_label} (R²={r2:.3f})")
fig6.add_traces(go.Scatter(x=df_reg[x_vars[x_label]], y=y_pred, mode="lines",
                           line=dict(color="red", width=2), name="Reta de Regressão"))
fig6.update_layout(template="plotly_dark", height=600)
st.plotly_chart(fig6, use_container_width=True)

st.divider()

# ============================================================
# Dispersão
# ============================================================
st.subheader("📊 Dispersão da Taxa de Mortalidade – Cidades × Estados")

taxa_estado = df_state_nr.groupby("state")["taxa_mortalidade"].max().reset_index()
media_cid, dp_cid, cv_cid = stats_dispersion(df_city_last["taxa_mortalidade"])
media_est, dp_est, cv_est = stats_dispersion(taxa_estado["taxa_mortalidade"])

col = st.columns(3)
col[0].metric("Média Cidades (%)", f"{media_cid:.2f}")
col[1].metric("Desvio Padrão Cidades", f"{dp_cid:.2f}")
col[2].metric("CV Cidades", f"{cv_cid:.2f}")
col2 = st.columns(3)
col2[0].metric("Média Estados (%)", f"{media_est:.2f}")
col2[1].metric("Desvio Padrão Estados", f"{dp_est:.2f}")
col2[2].metric("CV Estados", f"{cv_est:.2f}")

df_disp = pd.DataFrame({
    "Grupo":["Cidades","Estados"],
    "Média":[media_cid, media_est],
    "Desvio Padrão":[dp_cid, dp_est],
    "CV":[cv_cid, cv_est]
})

fig7 = px.bar(df_disp.melt(id_vars="Grupo"), x="Grupo", y="value", color="variable",
              barmode="group", text_auto=".2f", title="Dispersão Estatística – Taxa de Mortalidade (%)",
              color_discrete_sequence=px.colors.qualitative.Dark24)
fig7.update_layout(template="plotly_dark", height=500)
st.plotly_chart(fig7, use_container_width=True)

# ============================================================
# 🔥 Heatmap de Correlação – Variáveis Epidemiológicas
# ============================================================
st.divider()
st.subheader("🧠 Correlação entre Variáveis Epidemiológicas")

st.markdown("""
O mapa de calor abaixo mostra o **grau de correlação** entre as principais variáveis do dataset.
Valores próximos de **1** indicam correlação positiva forte (as duas variáveis crescem juntas),  
valores próximos de **-1** indicam correlação inversa (uma cresce enquanto a outra diminui).
""")

# Selecionar colunas relevantes numéricas
colunas_corr = [
    "estimated_population",
    "last_available_confirmed",
    "last_available_deaths",
    "last_available_death_rate",
    "last_available_confirmed_per_100k_inhabitants",
    "taxa_contaminacao",
    "taxa_mortalidade",
]

df_corr = df_city_last[colunas_corr].copy().dropna()
corr_matrix = df_corr.corr().round(2)

# Criar heatmap Plotly
fig_corr = px.imshow(
    corr_matrix,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    title="Mapa de Correlação – Indicadores Epidemiológicos",
    aspect="auto",
    zmin=-1, zmax=1
)

fig_corr.update_layout(
    template="plotly_dark",
    title_font=dict(size=16, color="white"),
    height=700,
    xaxis_title="Variáveis",
    yaxis_title="Variáveis",
    coloraxis_colorbar=dict(title="Correlação", tickvals=[-1, -0.5, 0, 0.5, 1]),
    margin=dict(l=50, r=50, t=100, b=50)
)

st.plotly_chart(fig_corr, use_container_width=True)

# Interpretação textual dinâmica
st.markdown("### 🧩 Interpretação Automática")
corr_mortes_casos = corr_matrix.loc["last_available_deaths", "last_available_confirmed"]
corr_taxas = corr_matrix.loc["taxa_mortalidade", "taxa_contaminacao"]

st.write(f"""
- **Mortes × Casos Confirmados:** correlação de `{corr_mortes_casos:.2f}` → forte relação direta: mais casos → mais mortes.  
- **Taxa de Mortalidade × Taxa de Contaminação:** correlação de `{corr_taxas:.2f}` → relação inversa, ou fraca, indicando que alta contaminação não implica necessariamente maior letalidade.  
- **População × Casos Confirmados:** normalmente também alta, refletindo maior volume em centros urbanos.
""")

