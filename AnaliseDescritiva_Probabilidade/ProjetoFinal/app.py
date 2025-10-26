import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import skew, kurtosis, norm

# -------------------------------------------------------------
# CONFIGURAÇÃO GERAL DO APP
# -------------------------------------------------------------
st.set_page_config(
    page_title="📊 Análise Descritiva ENADE 2017 - Engenharia Civil",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Análise Descritiva - Engenharia Civil (ENADE 2017)")

st.sidebar.markdown("📂 **Carregamento dos Dados**")

# -------------------------------------------------------------
# UPLOAD INTERATIVO
# -------------------------------------------------------------
uploaded_file = st.sidebar.file_uploader(
    "Envie o arquivo 'MICRODADOS_ENADE_2017.txt'",
    type=["txt", "csv"]
)

@st.cache_data(show_spinner="🔄 Lendo o arquivo... isso pode levar alguns segundos.")
def load_data(file):
    """Lê o arquivo ENADE e converte colunas numéricas."""
    def convert_num(x):
        try:
            return float(x.replace(',', '.').strip())
        except Exception:
            return pd.NA

    df = pd.read_csv(
        file,
        sep=';',
        encoding='latin1',
        low_memory=False,
        converters={
            'NT_OBJ_CE': convert_num,
            'NT_GER': convert_num,
            'NT_OBJ_FG': convert_num
        }
    )
    return df

# -------------------------------------------------------------
# PROCESSAMENTO E VISUALIZAÇÃO
# -------------------------------------------------------------
if uploaded_file is not None:
    df_base = load_data(uploaded_file)
    st.success("✅ Arquivo carregado com sucesso!")

    st.write(f"**{len(df_base):,} linhas** e **{len(df_base.columns)} colunas** encontradas.")
    st.dataframe(df_base.head())

    # Exemplo de estatísticas descritivas
    st.subheader("📈 Estatísticas Descritivas")
    st.write(df_base.describe(include='all').T)

    # Exemplo de gráfico interativo
    st.subheader("🎨 Distribuição de Notas (NT_GER)")
    if "NT_GER" in df_base.columns:
        fig = px.histogram(df_base, x="NT_GER", nbins=30, title="Distribuição da Nota Geral")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Coluna 'NT_GER' não encontrada no dataset.")
else:
    st.info("👆 Envie o arquivo 'MICRODADOS_ENADE_2017.txt' para iniciar a análise.")
# -------------------------------------------------------------
# LIMPEZA E TRANSFORMAÇÕES
# -------------------------------------------------------------
engenharia_civil = df_base[df_base["CO_GRUPO"] == 5710].copy()

# Mapas de categorias
de_para_regiao = {1: "Norte", 2: "Nordeste", 3: "Sul", 4: "Sudeste", 5: "Centro-Oeste"}
de_para_cor = {
    "A": "Branca",
    "B": "Preta",
    "C": "Amarela",
    "D": "Parda",
    "E": "Indígena",
    "F": "Sem Informação",
}
de_para_turno = {1: "Matutino", 2: "Vespertino", 3: "Integral", 4: "Noturno"}

engenharia_civil["REGIAO_CURSO"] = engenharia_civil["CO_REGIAO_CURSO"].map(de_para_regiao)
engenharia_civil["COR"] = engenharia_civil["QE_I02"].map(de_para_cor)
engenharia_civil["TURNO"] = engenharia_civil["CO_TURNO_GRADUACAO"].map(de_para_turno)

engenharia_civil["NT_OBJ_FG"] = pd.to_numeric(
    engenharia_civil["NT_OBJ_FG"], errors="coerce"
)
engenharia_civil = engenharia_civil.dropna(subset=["NT_OBJ_FG"])

# -------------------------------------------------------------
# FILTROS INTERATIVOS
# -------------------------------------------------------------
st.sidebar.subheader("🎚️ Filtros de análise")

regiao_sel = st.sidebar.selectbox(
    "Selecione a Região",
    ["Todas"] + sorted(engenharia_civil["REGIAO_CURSO"].dropna().unique().tolist()),
)

cor_sel = st.sidebar.multiselect(
    "Selecione os grupos de COR",
    sorted(engenharia_civil["COR"].dropna().unique().tolist()),
)

turno_sel = st.sidebar.multiselect(
    "Selecione o Turno",
    sorted(engenharia_civil["TURNO"].dropna().unique().tolist()),
)

# Aplicando filtros
df_filtered = engenharia_civil.copy()
if regiao_sel != "Todas":
    df_filtered = df_filtered[df_filtered["REGIAO_CURSO"] == regiao_sel]
if cor_sel:
    df_filtered = df_filtered[df_filtered["COR"].isin(cor_sel)]
if turno_sel:
    df_filtered = df_filtered[df_filtered["TURNO"].isin(turno_sel)]

# -------------------------------------------------------------
# ESTATÍSTICAS DESCRITIVAS
# -------------------------------------------------------------
moda_series = df_filtered["NT_OBJ_FG"].mode()
moda_value = moda_series.iloc[0] if not moda_series.empty else np.nan

estat = {
    "Quantidade": df_filtered["NT_OBJ_FG"].count(),
    "Média": df_filtered["NT_OBJ_FG"].mean(),
    "Mediana": df_filtered["NT_OBJ_FG"].median(),
    "Moda": moda_value,
    "Máximo": df_filtered["NT_OBJ_FG"].max(),
    "Mínimo": df_filtered["NT_OBJ_FG"].min(),
    "Desvio Padrão": df_filtered["NT_OBJ_FG"].std(),
    "CV (%)": df_filtered["NT_OBJ_FG"].std() / df_filtered["NT_OBJ_FG"].mean() * 100,
    "Assimetria": skew(df_filtered["NT_OBJ_FG"]),
    "Curtose": kurtosis(df_filtered["NT_OBJ_FG"]),
}

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("📈 Estatísticas Resumidas")
    st.table(pd.DataFrame(estat, index=["Valores"]).T.round(3))

with col2:
    st.subheader("📊 Distribuição de Notas")
    fig = px.histogram(
        df_filtered,
        x="NT_OBJ_FG",
        nbins=40,
        color="COR",
        marginal="box",
        title="Distribuição das Notas por Grupo de Cor",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_layout(bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# BOX PLOTS COMPARATIVOS
# -------------------------------------------------------------
st.subheader("🎨 Boxplots Comparativos")

tab1, tab2, tab3 = st.tabs(["Por Cor", "Por Região", "Por Turno"])

with tab1:
    fig1 = px.box(
        df_filtered,
        x="COR",
        y="NT_OBJ_FG",
        color="COR",
        title="Distribuição de Notas por Cor",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    fig2 = px.box(
        df_filtered,
        x="REGIAO_CURSO",
        y="NT_OBJ_FG",
        color="REGIAO_CURSO",
        title="Distribuição de Notas por Região",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    fig3 = px.box(
        df_filtered,
        x="TURNO",
        y="NT_OBJ_FG",
        color="TURNO",
        title="Distribuição de Notas por Turno",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig3, use_container_width=True)

# -------------------------------------------------------------
# DISTRIBUIÇÃO NORMAL - PROBABILIDADE
# -------------------------------------------------------------
st.subheader("📉 Distribuição Normal (Probabilidade entre intervalos)")

media = df_filtered["NT_OBJ_FG"].mean()
dp = df_filtered["NT_OBJ_FG"].std()

valores = np.linspace(0, 100, 200)
dist = norm(loc=media, scale=dp)
pdf = dist.pdf(valores)

intervalo = st.slider("Selecione o intervalo de notas:", 0, 100, (35, 50))
prob = norm.cdf(intervalo[1], loc=media, scale=dp) - norm.cdf(intervalo[0], loc=media, scale=dp)

st.info(f"📊 Probabilidade de uma nota estar entre **{intervalo[0]}** e **{intervalo[1]}**: **{prob:.4f}**")

fig_pdf = px.line(
    x=valores,
    y=pdf,
    labels={"x": "Nota", "y": "Densidade"},
    title="Função Densidade de Probabilidade (Normal)",
)
fig_pdf.add_vrect(
    x0=intervalo[0],
    x1=intervalo[1],
    fillcolor="lightgreen",
    opacity=0.3,
    annotation_text="Intervalo Selecionado",
)
st.plotly_chart(fig_pdf, use_container_width=True)

# -------------------------------------------------------------
# DOWNLOAD
# -------------------------------------------------------------
st.sidebar.subheader("📥 Exportar resultados")
csv = df_filtered.to_csv(index=False).encode("utf-8")
st.sidebar.download_button("Baixar dados filtrados (CSV)", csv, "enade_filtrado.csv", "text/csv")
