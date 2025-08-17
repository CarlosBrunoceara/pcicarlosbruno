# app.py

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import math

# ===================================================================
# 1. CONFIGURAÇÃO DA PÁGINA E CARREGAMENTO DE MODELOS
# ===================================================================
st.set_page_config(
    page_title="Ferramenta PCI - Análise de Pavimentos",
    page_icon="ରା",
    layout="wide"
)

@st.cache_resource
def carregar_modelos():
    try:
        model = tf.keras.models.load_model("modelo_pci.keras")
        preprocessor = joblib.load("preprocessor_pci.joblib")
        scaler_y = joblib.load("scaler_y_pci.joblib")
        return model, preprocessor, scaler_y
    except Exception as e:
        st.error(f"Erro CRÍTICO ao carregar arquivos do modelo: {e}")
        st.stop()

loaded_model, loaded_preprocessor, loaded_scaler_y = carregar_modelos()

if 'amostras' not in st.session_state:
    st.session_state.amostras = {}

# ===================================================================
# 2. FUNÇÕES DE CÁLCULO (AMOSTRAGEM E PCI)
# ===================================================================
def calcular_amostras_params(CV, W, e, s):
    AREA_PADRAO = 225.0
    if CV <= 0 or W <= 0: return {"error": "CV e W devem ser > 0."}
    area = AREA_PADRAO
    ca = area / W
    if (CV/ca) <= 1: n_cont = 1.0
    else: n_cont = ( ( (CV/ca) * s**2 ) / ( ((e**2)/4) * ( (CV/ca) - 1 ) + s**2 ) )
    n_min = max(1, math.ceil(n_cont))
    espacamento = (CV / n_min) if n_min > 0 else 0
    posicoes = [math.floor((i * espacamento) * 10) / 10 for i in range(n_min)]
    return {"n_minimo": n_min, "Area_m2": area, "Posicoes_m": posicoes}

def classify_pci_and_get_color(pci_value):
    if pd.isna(pci_value): return "Não Calculado", "#808080"
    if 85 <= pci_value <= 100: return "Bom", "darkgreen"
    if 70 <= pci_value < 85: return "Satisfatório", "limegreen"
    if 55 <= pci_value < 70: return "Regular", "gold"
    if 40 <= pci_value < 55: return "Ruim", "deeppink"
    if 25 <= pci_value < 40: return "Muito ruim", "red"
    if 10 <= pci_value < 25: return "Péssimo", "saddlebrown"
    if 0 <= pci_value < 10: return "Perda funcional", "dimgray"
    return "Fora do Intervalo", "black"

def calcular_pci_para_amostra(df_amostra):
    dv_col_name = 'VALOR DEDUZIDO'
    if df_amostra.empty or dv_col_name not in df_amostra.columns: return np.nan
    df_amostra[dv_col_name] = pd.to_numeric(df_amostra[dv_col_name], errors='coerce')
    dv_validos = df_amostra[dv_col_name].dropna()
    if dv_validos.empty: return 100
    hdv = dv_validos.max()
    m_calc = 1 + (9/98) * (100 - hdv)
    m = round(min(10, m_calc))
    df_filtrada = df_amostra.nlargest(m, dv_col_name)
    q = sum(1 for v in df_filtrada[dv_col_name] if v > 2)
    if q <= 1:
        cdv = df_filtrada[dv_col_name].sum()
    else:
        vdt_total = df_filtrada[dv_col_name].sum()
        cdv_calc = -0.0018 * (vdt_total**2) + 0.9187 * vdt_total - 18.047
        cdv = max(cdv_calc, hdv)
    return max(0, 100 - cdv)

def prever_valor_deduzido(defeito_fmt, severidade_code, densidade):
    try:
        defeito = defeito_fmt.split(' - ', 1)[1]
        dados = pd.DataFrame([[defeito, severidade_code, densidade]], columns=['DEFEITO', 'SEVERIDADE', 'DENSIDADE (%)'])
        pred_scaled = loaded_model.predict(loaded_preprocessor.transform(dados), verbose=0)
        valor_final = loaded_scaler_y.inverse_transform(pred_scaled)
        return round(float(valor_final[0][0]), 2)
    except Exception: return np.nan

# ===================================================================
# 3. INTERFACE GRÁFICA (SIDEBAR)
# ===================================================================
with st.sidebar:
    st.header("1. Parâmetros da Via")
    cv = st.number_input('Comprimento da Via (CV, m)', value=1000.0, format="%.2f")
    largura = st.number_input('Largura da VIA (m)', value=7.0, format="%.2f")
    erro = st.number_input('Erro aceitável (e)', value=5.0, format="%.2f")
    desvio_padrao = st.number_input('Desvio padrão (s)', value=10.0, format="%.2f")

    if st.button("Calcular Amostragem e Gerar Tabelas", type="primary", use_container_width=True):
        res = calcular_amostras_params(cv, largura, erro, desvio_padrao)
        if "error" in res:
            st.error(res["error"])
        else:
            st.session_state.amostras.clear()
            n_amostras, area, posicoes = res['n_minimo'], res['Area_m2'], res['Posicoes_m']
            for i in range(n_amostras):
                amostra_id = f"Amostra_{i+1}"
                st.session_state.amostras[amostra_id] = {
                    "df": pd.DataFrame(columns=['DEFEITO', 'SEVERIDADE', 'SEVERIDADE_CODE', 'Q1', 'Q2', 'Q3', 'Q4', 'TOTAL', 'DENSIDADE', 'VALOR DEDUZIDO']),
                    "posicao": posicoes[i],
                    "area": area,
                    "pci": np.nan
                }
            st.success(f"{n_amostras} amostras geradas com área padrão de {area:.2f} m².")
            st.rerun()

    st.header("2. Gerenciar Amostras")
    if st.button("Adicionar Amostra Extra", use_container_width=True):
        idx = len(st.session_state.amostras) + 1
        amostra_id = f"Amostra_Extra_{idx}"
        st.session_state.amostras[amostra_id] = {
            "df": pd.DataFrame(columns=['DEFEITO', 'SEVERIDADE', 'SEVERIDADE_CODE', 'Q1', 'Q2', 'Q3', 'Q4', 'TOTAL', 'DENSIDADE', 'VALOR DEDUZIDO']),
            "posicao": 0.0, "area": 225.0, "pci": np.nan
        }
        st.rerun()

    if st.session_state.amostras:
        amostra_a_excluir = st.selectbox("Excluir Amostra:", options=[""] + list(st.session_state.amostras.keys()))
        if st.button("Confirmar Exclusão", use_container_width=True):
            if amostra_a_excluir and amostra_a_excluir in st.session_state.amostras:
                del st.session_state.amostras[amostra_a_excluir]
                st.rerun()

# ===================================================================
# 4. ÁREA PRINCIPAL (EXIBIÇÃO DAS TABELAS E RESULTADOS)
# ===================================================================
st.title("Ferramenta Integrada de Análise de Pavimentos")

if not st.session_state.amostras:
    st.info("⬅️ Utilize o painel à esquerda para calcular o número de amostras.")
else:
    pcis_validos = [data['pci'] for data in st.session_state.amostras.values() if pd.notna(data['pci'])]
    if pcis_validos:
        pci_medio = np.mean(pcis_validos)
        classificacao, cor = classify_pci_and_get_color(pci_medio)
        st.header("Resultado Final da Via")
        col1, col2 = st.columns(2)
        col1.metric(label="PCI Médio da Via", value=f"{pci_medio:.2f}")
        col2.markdown(f"#### Classificação: <span style='color:{cor};'>{classificacao}</span>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.header("3. Coleta de Dados e Análise por Amostra")
    
    mapa_defeitos = {'BLOCOS DANIFICADOS': 1, 'DEPRESSÕES': 2, 'DANO DE CONTENÇÃO': 3, 'ESPAÇAMENTO EXCESSIVO DAS JUNTAS': 4, 'DIFERENÇA DE ALTURA DO BLOCO': 5, 'ONDULAÇÃO': 6, 'DESLOCAMENTO HORIZONTAL': 7, 'PERDA DE MATERIAL DE REJUNTAMENTO': 8, 'PERDA DE BLOCOS': 9, 'REMENDO': 10, 'DEFORMAÇÃO DE TRILHA DE RODA': 11}
    itens_defeitos_ordenados = sorted(mapa_defeitos.items(), key=lambda item: item[1])
    opcoes_defeito = [''] + [f"{num} - {defeito}" for defeito, num in itens_defeitos_ordenados]
    opcoes_severidade = [('', ''), ('Alto (H)', 'A'), ('Médio (M)', 'M'), ('Baixo (L)', 'L')]

    for amostra_id, amostra_data in st.session_state.amostras.items():
        pos, area, df = amostra_data['posicao'], amostra_data['area'], amostra_data['df']
        
        with st.expander(f"**{amostra_id.replace('_', ' ')}** (Posição: {pos:.1f} m)", expanded=True):
            
            area_atual = st.number_input("Área da Amostra (m²)", value=area, min_value=0.1, format="%.2f", key=f"area_{amostra_id}")
            if area_atual != area:
                st.session_state.amostras[amostra_id]['area'] = area_atual
                if not df.empty:
                    df['DENSIDADE'] = (df['TOTAL'] / area_atual) * 100
                    for i, row in df.iterrows():
                        df.loc[i, 'VALOR DEDUZIDO'] = prever_valor_deduzido(row['DEFEITO'], row['SEVERIDADE_CODE'], df.loc[i, 'DENSIDADE'])
                    st.session_state.amostras[amostra_id]['df'] = df
                st.rerun()

            # --- ALTERADO E CORRIGIDO: Formatação aplicada apenas a colunas numéricas ---
            df_para_exibir = df.drop(columns=['SEVERIDADE_CODE'], errors='ignore') # Oculta a coluna de código
            numeric_cols = df_para_exibir.select_dtypes(include=np.number).columns
            format_dict = {col: '{:.2f}' for col in numeric_cols}
            st.dataframe(df_para_exibir.style.format(format_dict, na_rep=""), use_container_width=True)
            
            with st.form(key=f"form_{amostra_id}", clear_on_submit=True):
                st.markdown("**Adicionar / Excluir Linha de Defeito**")
                c1,c2,c3,c4,c5,c6 = st.columns([3, 2, 1, 1, 1, 1])
                defeito = c1.selectbox("Defeito", options=opcoes_defeito, label_visibility="collapsed")
                severidade_tupla = c2.selectbox("Severidade", options=opcoes_severidade, format_func=lambda x: x[0], label_visibility="collapsed")
                q1 = c3.number_input("Q1", min_value=0.0, format="%.2f", label_visibility="collapsed")
                q2 = c4.number_input("Q2", min_value=0.0, format="%.2f", label_visibility="collapsed")
                q3 = c5.number_input("Q3", min_value=0.0, format="%.2f", label_visibility="collapsed")
                q4 = c6.number_input("Q4", min_value=0.0, format="%.2f", label_visibility="collapsed")
                
                c_submit_1, c_submit_2, c_submit_3 = st.columns([2, 1, 1])
                add_button = c_submit_1.form_submit_button("Adicionar Linha")
                idx_excluir = c_submit_2.number_input("Índice p/ Excluir", min_value=0, max_value=max(0, len(df)-1), step=1)
                del_button = c_submit_3.form_submit_button("Excluir Linha")

                if add_button and defeito and severidade_tupla[1]:
                    quantidades = [q1, q2, q3, q4]
                    total = sum(quantidades)
                    densidade = (total / area_atual) * 100
                    valor = prever_valor_deduzido(defeito, severidade_tupla[1], densidade)
                    nova_linha = {'DEFEITO': defeito, 'SEVERIDADE': severidade_tupla[0], 'SEVERIDADE_CODE': severidade_tupla[1], 'Q1': q1, 'Q2': q2, 'Q3': q3, 'Q4': q4, 'TOTAL': total, 'DENSIDADE': densidade, 'VALOR DEDUZIDO': valor}
                    st.session_state.amostras[amostra_id]['df'] = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)
                    st.rerun()
                
                if del_button and 0 <= idx_excluir < len(df):
                    st.session_state.amostras[amostra_id]['df'] = df.drop(index=idx_excluir).reset_index(drop=True)
                    st.rerun()

            col_b1, col_b2 = st.columns([1, 3])
            if col_b1.button("Calcular PCI desta Amostra", type="primary", key=f"pci_btn_{amostra_id}", use_container_width=True):
                pci_calculado = calcular_pci_para_amostra(df)
                st.session_state.amostras[amostra_id]['pci'] = pci_calculado
                st.rerun()
            
            pci_individual = amostra_data['pci']
            if pd.notna(pci_individual):
                classificacao_ind, cor_ind = classify_pci_and_get_color(pci_individual)
                col_b2.metric(label=f"PCI da Amostra", value=f"{pci_individual:.2f}", help=f"Classificação: {classificacao_ind}")
