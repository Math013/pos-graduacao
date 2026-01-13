import pandas as pd

# -------------------------------------------------------------
# Caminhos de entrada e saída
# -------------------------------------------------------------
input_path = r"C:\Users\Matheus\Desktop\PUC\AnaliseDescritiva_Probabilidade\ProjetoFinal\MICRODADOS_ENADE_2017.txt"
output_path = "ProjetoFinal/MICRODADOS_ENADE_2017_SAMPLE.txt"

# -------------------------------------------------------------
# Função para converter valores numéricos (com vírgula -> ponto)
# -------------------------------------------------------------
def convert_num(x):
    """Converte valores numéricos com vírgula para float."""
    try:
        return float(x.replace(",", ".").strip())
    except Exception:
        return pd.NA

# -------------------------------------------------------------
# Leitura do arquivo original (ENADE completo)
# -------------------------------------------------------------
print("🔄 Lendo arquivo original (pode levar alguns minutos)...")

df = pd.read_csv(
    input_path,
    sep=";",                 # separador padrão do ENADE
    encoding="latin1",       # mantém acentuação correta
    low_memory=False,        # evita alertas de tipo
    converters={             # trata colunas numéricas
        "NT_OBJ_CE": convert_num,
        "NT_GER": convert_num,
        "NT_OBJ_FG": convert_num,
    },
)

# -------------------------------------------------------------
# Filtra apenas o curso de Engenharia Civil (código 5710)
# -------------------------------------------------------------
print("🎯 Filtrando curso de Engenharia Civil (CO_GRUPO = 5710)...")
df = df[df["CO_GRUPO"] == 5710].copy()

# -------------------------------------------------------------
# Salva arquivo reduzido no formato TXT
# -------------------------------------------------------------
print("💾 Salvando amostra reduzida...")
df.to_csv(
    output_path,
    sep=";",           # mantém padrão do ENADE
    index=False,       # remove índice
    encoding="latin1"  # mesma codificação
)

# print(f"✅ Amostra salva com sucesso em: {output_path}")
# print(f"📊 Linhas: {df.shape[0]:,} | Colunas: {df.shape[1]}")
