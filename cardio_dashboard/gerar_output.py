import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import shapiro, mannwhitneyu, chi2_contingency
import json
from warnings import filterwarnings

filterwarnings("ignore")

# ==============================
# CONFIGURAÇÃO
# ==============================

BASE_DIR = os.path.dirname(__file__)

DATA_PATH = os.path.join(BASE_DIR, "dados.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "static", "output")

TEMPLATE_PATH = os.path.join(BASE_DIR, "templates", "dashboard.html")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# CARREGAR DADOS
# ==============================

print("Carregando dados...")

df = pd.read_csv(DATA_PATH, sep=';')
# ==============================
# LIMPEZA DE DADOS
# ==============================
if 'imc' in df.columns:
    df['imc'] = df['imc'].astype(str).str.replace(',', '.', regex=False)
    df['imc'] = pd.to_numeric(df['imc'], errors='coerce')
# ==============================
# NORMALIZAR COLUNAS BINÁRIAS
# ==============================

colunas_binarias = [
    "diabetes_gestacional",
    "hipertensao",
    "hipertensao_pre_eclampsia",
    "obesidade_pre_gestacional",
    "tabagismo",
    "alcoolismo"
]

for col in colunas_binarias:
    if col in df.columns:

        df[col] = (
            df[col]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        df[col] = df[col].replace({
            "sim": 1,
            "yes": 1,
            "true": 1,
            "1": 1,

            "nao": 0,
            "não": 0,
            "no": 0,
            "false": 0,
            "0": 0
        })

        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
df["chd_confirmada"] = df["chd_confirmada"].astype(int)

# ==============================
# CRIAR VARIÁVEL FATORES DE RISCO
# ==============================

# Definir fatores de risco médicos
fatores_risco = [
"diabetes_gestacional",
"hipertensao",
"hipertensao_pre_eclampsia",
"obesidade_pre_gestacional",
]

# Definir fatores de estilo de vida
fatores_estilo_vida = [
"tabagismo",
"alcoolismo"
]

# Filtra a lista de fatores de risco para incluir apenas as colunas que existem no DataFrame
fatores_risco_existentes = [col for col in fatores_risco if col in df.columns]

df["total_fatores_risco"] = df[fatores_risco_existentes].sum(axis=1)

# Filtra a lista de fatores de estilo de vida para incluir apenas as colunas que existem no DataFrame
fatores_estilo_vida_existentes = [col for col in fatores_estilo_vida if col in df.columns]


# ==============================
# ANÁLISE POR GRUPO DE RISCO
# ==============================

# Definir os grupos de risco de forma dinâmica e segura
max_risk = df['total_fatores_risco'].max()

bins = [-1, 1]
labels = ['Baixo Risco (0-1)']

if max_risk >= 2:
    bins.append(3)
    labels.append('Médio Risco (2-3)')

if max_risk > 3:
    bins.append(max_risk + 1)
    labels.append('Alto Risco (>3)')

df['grupo_de_risco'] = pd.cut(df['total_fatores_risco'], bins=bins, labels=labels, right=True, ordered=False)

# Variáveis para a análise descritiva por grupo
variaveis_analise_risco = ['idade', 'imc', 'pressao_sistolica', 'frequencia_cardiaca_fetal', 'idade_gestacional']
variaveis_existentes_analise_risco = [col for col in variaveis_analise_risco if col in df.columns]

# Gerar estatística descritiva por grupo de risco
analise_descritiva_risco = df.groupby('grupo_de_risco')[variaveis_existentes_analise_risco].describe()

# ==============================
# ANÁLISE POR COMORBIDADE
# ==============================

print("Analisando por comorbidades...")

analises_comorbidades = {}
variaveis_para_analise_comorbidade = ['idade', 'imc', 'pressao_sistolica', 'frequencia_cardiaca_fetal', 'idade_gestacional']
variaveis_existentes_comorbidade = [col for col in variaveis_para_analise_comorbidade if col in df.columns]

for comorbidade in fatores_risco_existentes + fatores_estilo_vida_existentes: # Inclui estilo de vida na análise por comorbidade
    desc_comorbidade = df.groupby(comorbidade)[variaveis_existentes_comorbidade].describe()
    analises_comorbidades[comorbidade] = desc_comorbidade


# ==============================
# ESTATÍSTICA DESCRITIVA
# ==============================

numericas = df.select_dtypes(include=np.number)

# Remove colunas de ID que não são relevantes para a descrição geral
cols_to_drop_from_desc = ['gestante_id', 'consulta_numero', 'num_chd_codigos']
desc_numericas = numericas.drop(columns=[col for col in cols_to_drop_from_desc if col in numericas.columns])

desc = numericas.describe().T

# ==============================
# SEPARAR GRUPOS
# ==============================

chd = df[df["chd_confirmada"]==1]

sem_chd = df[df["chd_confirmada"]==0]

# ==============================
# FUNÇÃO SALVAR GRÁFICOS
# ============================== 

def salvar_fig(nome):

    path = os.path.join(OUTPUT_DIR,nome)

    plt.savefig(path,bbox_inches="tight")

    plt.close()

    return nome

# ==============================
# FUNÇÕES DE GERAÇÃO DE GRÁFICOS E TABELAS
# ==============================

print("Gerando gráficos...")

def gerar_graficos_html(df, numericas):
    html = ""

    # Distribuição sinais vitais
    sinais = ["pressao_sistolica", "bpm_materno", "saturacao", "temperatura_corporal"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax = ax.flatten()
    for i, col in enumerate(sinais):
        if col in df.columns:
            sns.histplot(df[col], ax=ax[i], kde=True)
            ax[i].set_title(col)
    plt.tight_layout()
    salvar_fig("sinais_vitais.png")
    html += """
    <div class='card'>
    <h3>Sinais Vitais</h3>
    <img src="/static/output/sinais_vitais.png">
    </div>
    """

    # Boxplots
    vars_box = ["idade", "imc", "pressao_sistolica", "frequencia_cardiaca_fetal", "idade_gestacional"]
    fig, ax = plt.subplots(1, len(vars_box), figsize=(20, 4))
    ax = ax.flatten()
    for i, col in enumerate(vars_box):
        if col in df.columns:
            sns.boxplot(data=df, x="chd_confirmada", y=col, ax=ax[i])
            ax[i].set_title(col)
    plt.tight_layout()
    salvar_fig("boxplots.png")
    html += """
    <div class='card'>
    <h3>Analise dos Sinais Vitais</h3>
    <img src="/static/output/boxplots.png">
    </div>
    """

    # Heatmap correlação
    corr = numericas.corr(method="spearman")
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Heatmap de Correlação de Spearman")
    salvar_fig("heatmap.png")
    html += """
    <div class='card'>
    <h3>Correlação</h3>
    <img src="/static/output/heatmap.png">
    </div>
    """
    return html
    
def gerar_tabelas_html(desc_df, analise_risco_df, analises_comorbidades_dict):
    
    comorbidades_html = ""
    for nome, df_comorbidade in analises_comorbidades_dict.items():
        comorbidades_html += f"""
        <h3>{nome.replace('_', ' ').title()}</h3>
        {df_comorbidade.to_html(classes='table table-striped', float_format='{:.2f}'.format)}
        """
    
    return f"""
    <div class="card">
    <h2>Estatística Descritiva Geral</h2>
    {desc_df.to_html(classes='table table-striped', float_format='{:.2f}'.format)}
    </div>
    <div class="card">
    <h2>Análise Descritiva por Grupo de Risco</h2>
    {analise_risco_df.to_html(classes='table table-striped', float_format='{:.2f}'.format)}
    </div>
    <div class="card">
    <h2>Análise por Comorbidade</h2>
    {comorbidades_html}
    </div>
    """

def gerar_analise_focada_html(df, fator_foco, todos_fatores_risco):
    """
    Gera uma análise comparativa para um fator de risco específico.
    Mostra a prevalência de outras comorbidades nos grupos com e sem o fator em foco.
    """
    if fator_foco not in df.columns:
        return ""

    print(f"Gerando análise focada em {fator_foco}...")
    
    outros_fatores = [f for f in todos_fatores_risco if f != fator_foco and f in df.columns]
    
    # Calcula a prevalência (média, pois são 0/1) de outros fatores para cada grupo
    prevalencia_df = df.groupby(fator_foco)[outros_fatores].mean().mul(100).reset_index()
    
    # Reorganiza o dataframe para o formato "longo" para o gráfico
    melted_df = prevalencia_df.melt(id_vars=fator_foco, var_name='Comorbidade', value_name='Prevalência (%)')
    
    plt.figure(figsize=(12, 7))
    sns.barplot(data=melted_df, x='Comorbidade', y='Prevalência (%)', hue=fator_foco)
    
    titulo_fator = fator_foco.replace('_', ' ').title()
    plt.title(f"Comparativo de Prevalência de Comorbidades em Grupos com e sem {titulo_fator}")
    plt.ylabel("Pacientes (%)")
    plt.xlabel("Fator de Risco Adicional")
    plt.xticks(rotation=45, ha='right')
    
    nome_figura = f"comparativo_{fator_foco}.png"
    salvar_fig(nome_figura)
    print("Gerando gráficos de CHD por comorbidade...")

    num_comorbidades = len(fatores_risco)
    if num_comorbidades == 0:
        return ""

    # Determina o layout da grade de subplots (ex: 2x3 para 6 comorbidades)
    num_cols = 3
    num_rows = (num_comorbidades + num_cols - 1) // num_cols # Arredonda para cima

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 4))
    axes = axes.flatten() # Transforma a matriz de eixos em um array 1D para fácil iteração

    for i, comorbidade in enumerate(fatores_risco):
        sns.countplot(data=df, x=comorbidade, hue='chd_confirmada', ax=axes[i])
        axes[i].set_title(f"CHD por {comorbidade.replace('_', ' ').title()}")
        axes[i].set_ylabel("Contagem")
        axes[i].set_xlabel("") # Remove o label do eixo X para não poluir

    plt.tight_layout() # Ajusta o layout para evitar sobreposição
    nome_figura_combinada = "chd_por_comorbidades_combinado.png"
    salvar_fig(nome_figura_combinada)
    return f"<div class='card'><h2>CHD por Comorbidade</h2><img src='/static/output/{nome_figura_combinada}'></div>";

def gerar_graficos_chd_por_estilo_vida_html(df, fatores_estilo_vida):
    print("Gerando gráficos de CHD por Estilo de Vida...")

    num_fatores = len(fatores_estilo_vida)
    if num_fatores == 0:
        return ""

    num_cols = 2 # Pode ajustar conforme necessário
    num_rows = (num_fatores + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 4))
    axes = axes.flatten()

    for i, fator in enumerate(fatores_estilo_vida):
        sns.countplot(data=df, x=fator, hue='chd_confirmada', ax=axes[i])
        axes[i].set_title(f"CHD por {fator.replace('_', ' ').title()}")
        axes[i].set_ylabel("Contagem")
        axes[i].set_xlabel("")

    plt.tight_layout()
    nome_figura_combinada = "chd_por_estilo_vida_combinado.png"
    salvar_fig(nome_figura_combinada)
    return f"<div class='card'><h2>CHD por Estilo de Vida</h2><img src='/static/output/{nome_figura_combinada}'></div>"

def gerar_analise_intercorrencias_fetais_html(df, variaveis_numericas):
    intercorrencias = ['doppler_ducto_venoso', 'eixo_cardiaco', 'quatro_camaras']
    intercorrencias_existentes = [i for i in intercorrencias if i in df.columns]

    if not intercorrencias_existentes:
        return ""

    print("Gerando análise de intercorrências fetais...")
    html = ""
    
    num_intercorrencias = len(intercorrencias_existentes)
    if num_intercorrencias > 0:
        num_cols = 3 # Define o número de colunas para os subplots
        num_rows = (num_intercorrencias + num_cols - 1) // num_cols # Calcula o número de linhas

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 4))
        axes = axes.flatten() # Transforma a matriz de eixos em um array 1D

        for i, intercorrencia in enumerate(intercorrencias_existentes):
            sns.countplot(data=df, x=intercorrencia, hue='chd_confirmada', ax=axes[i])
            axes[i].set_title(f"CHD por {intercorrencia.replace('_', ' ').title()}")
            axes[i].set_ylabel("Contagem")
            axes[i].set_xlabel("")
        
        # Remove subplots vazios, se houver
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        nome_figura_combinada = "chd_por_intercorrencias_fetais_combinado.png"
        salvar_fig(nome_figura_combinada)
        html += f"<h3>Relação entre Intercorrências Fetais e CHD</h3>"
        html += f"<img src='/static/output/{nome_figura_combinada}'>"
        html += "<hr>"

    # Adiciona as tabelas descritivas para cada intercorrência
    for intercorrencia in intercorrencias_existentes:
        # Tabela: Média das variáveis numéricas por categoria da intercorrência
        titulo = intercorrencia.replace('_', ' ').title()
        desc_df = df.groupby(intercorrencia)[variaveis_numericas].mean()
        html += f"<p>Média das variáveis para cada categoria de {titulo}:</p>"
        html += desc_df.to_html(classes='table table-striped', float_format='{:.2f}'.format)
        html += "<hr>"
    return f"<div class='card'><h2>Análise de Intercorrências Fetais</h2>{html}</div>"
def gerar_analise_perfil_materno_por_comorbidade_html(df, comorbidades_para_analise, variaveis_perfil_materno):
    print("Gerando análise do perfil materno por comorbidade...")
    html_content = ""

    for comorbidade_foco in comorbidades_para_analise:
        if comorbidade_foco not in df.columns:
            continue

        nome_legivel_comorbidade = comorbidade_foco.replace("_", " ").title()
        
        # Calcula as estatísticas descritivas para as variáveis do perfil materno
        # agrupadas por ter ou não a comorbidade em foco
        desc_perfil = df.groupby(comorbidade_foco)[variaveis_perfil_materno].describe()
        
        # Formata o DataFrame para HTML
        html_content += f"""
        <h3>Perfil Materno para {nome_legivel_comorbidade}</h3>
        <p>Estatísticas descritivas do perfil materno para gestantes que possuem (1) e não possuem (0) {nome_legivel_comorbidade}.</p>
        {desc_perfil.to_html(classes='table table-striped', float_format='{:.2f}'.format)}
        <hr>
        """
    return f"<div class='card'><h2>Análise do Perfil Materno por Comorbidade</h2>{html_content}</div>"

# ==============================
# HTML FINAL
# ============================== 

graficos_html = gerar_graficos_html(df, desc_numericas) # Contém Sinais Vitais, Boxplots e Heatmap

analise_perfil_materno_html = gerar_analise_perfil_materno_por_comorbidade_html(
    df,
    fatores_risco_existentes + fatores_estilo_vida_existentes,
    variaveis_existentes_comorbidade
)

graficos_chd_estilo_vida_html = gerar_graficos_chd_por_estilo_vida_html(df, fatores_estilo_vida_existentes)
tabelas_descritivas_content = gerar_tabelas_html(desc, analise_descritiva_risco, analises_comorbidades)

html_dashboard = f""" 

<html>

<head>

<title>Dashboard Cardio</title>

<style>

body{{font-family:Arial;background:#f5f6fa}}

.container{{width:100%;margin:auto}}

.card{{background:white;padding:20px;margin:20px;border-radius:8px;overflow-x:auto;}}

img{{width:100%}}

table {{width: 100%; border-collapse: collapse;}}
th, td {{padding: 8px; text-align: left; border-bottom: 1px solid #ddd;}}
thead {{background-color: #f2f2f2;}}


</style>

</head>

<body>

<div class="container">

<h1>Relatório de Análise Cardio</h1>


{graficos_chd_estilo_vida_html}


{analise_perfil_materno_html}

{graficos_html}

</div>

</body>

</html>

"""

with open(TEMPLATE_PATH,"w",encoding="utf-8") as f:

    f.write(html_dashboard)

print("Relatório gerado com sucesso!")
