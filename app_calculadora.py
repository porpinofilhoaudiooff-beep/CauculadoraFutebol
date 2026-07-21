import math
import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import poisson
import requests
st.set_page_config(page_title="Calc Futebol", page_icon="⚽", layout="wide")

st.title("⚽ Calculadora Quantitativa de Futebol")
st.markdown("Análise estatística baseada em **Dixon-Coles, Poisson e Monte Carlo**. Ajuste os parâmetros no menu lateral para visualizar os cálculos em tempo real.")

# ==========================================
# MENU LATERAL - INPUT DE DADOS
# ==========================================
st.sidebar.header("⚙️ Input de Dados")
DATA_JOGO = st.sidebar.date_input("Data do Jogo")
TIME_CASA = st.sidebar.text_input("Time da Casa")
TIME_FORA = st.sidebar.text_input("Time Visitante")

# ==========================================
# INTEGRAÇÃO COM API EXTERNA (API-Football)
# ==========================================
# 1. Cole a sua chave copiada do site dentro das aspas abaixo:
API_KEY = "7e5b9458c3fd615c5a82b9598d5547a3"

# Usamos o cache do Streamlit (ttl=86400 segundos = 24h) para o app não gastar o 
# limite grátis da API se você ficar mudando os números na tela.
@st.cache_data(ttl=86400) 
def buscar_stat(nome_time, coluna, valor_padrao):
    if API_KEY == "7e5b9458c3fd615c5a82b9598d5547a3":
        return valor_padrao # Proteção caso esqueça de colocar a chave
        
    url = "https://v3.football.api-sports.io/teams"
    headers = {
        'x-apisports-key': API_KEY
    }
    
    try:
        # O App bate na porta da API buscando o time
        response = requests.get(f"{url}?search={nome_time}", headers=headers)
        dados = response.json()
        
        # Verifica se a API encontrou o time
        if dados['results'] > 0:
            time_id = dados['response'][0]['team']['id']
            # IMPORTANTE: 
            # Como a busca de médias exatas exige cruzar o time com a liga atual,
            # este bloco retorna o valor padrão para não quebrar o site enquanto 
            # você estuda a documentação da API-Football para mapear as rotas de estatísticas.
            
            # Aqui entraria a segunda requisição usando o 'time_id' para buscar a aba de "statistics".
            return valor_padrao
            
    except Exception as e:
        # Se der erro de internet, mostra um aviso discreto no menu
        st.sidebar.warning(f"Aviso da API: Não foi possível sincronizar o {nome_time}.")
        
    return valor_padrao
with st.sidebar.expander("1. GOLS (Médias)"):
    GF_CASA = st.number_input("Gols Feitos (Casa)", value=buscar_stat(TIME_CASA, "GF_CASA", 2.10), step=0.1)
    GA_CASA = st.number_input("Gols Sofridos (Casa)", value=buscar_stat(TIME_CASA, "GA_CASA", 0.85), step=0.1)
    GF_FORA = st.number_input("Gols Feitos (Fora)", value=buscar_stat(TIME_FORA, "GF_FORA", 1.30), step=0.1)
    GA_FORA = st.number_input("Gols Sofridos (Fora)", value=buscar_stat(TIME_FORA, "GA_FORA", 1.10), step=0.1)
    LIGA_MEDIA_GOLS_MANDANTE = st.number_input("Média Liga Mandante", value=1.45, step=0.1)
    LIGA_MEDIA_GOLS_VISITANTE = st.number_input("Média Liga Visitante", value=1.15, step=0.1)
    RHO = st.number_input("Correlação (Rho)", value=-0.08, step=0.01)
    PLACAR_MAX = 8
# ==========================================
# CÁLCULOS MATEMÁTICOS (Módulos Adaptados)
# ==========================================

# Modulo 1 - Dixon Coles
f_ataque_casa = GF_CASA / LIGA_MEDIA_GOLS_MANDANTE
f_defesa_casa = GA_CASA / LIGA_MEDIA_GOLS_VISITANTE
f_ataque_fora = GF_FORA / LIGA_MEDIA_GOLS_VISITANTE
f_defesa_fora = GA_FORA / LIGA_MEDIA_GOLS_MANDANTE

lambda_casa = f_ataque_casa * f_defesa_fora * LIGA_MEDIA_GOLS_MANDANTE
lambda_fora = f_ataque_fora * f_defesa_casa * LIGA_MEDIA_GOLS_VISITANTE

def tau(x, y, lam_c, lam_f, rho):
    if x == 0 and y == 0: return 1 - (lam_c * lam_f * rho)
    elif x == 0 and y == 1: return 1 + (lam_c * rho)
    elif x == 1 and y == 0: return 1 + (lam_f * rho)
    elif x == 1 and y == 1: return 1 - rho
    return 1.0

n = PLACAR_MAX + 1
matriz = np.zeros((n, n))
for x in range(n):
    for y in range(n):
        p_poisson = poisson.pmf(x, lambda_casa) * poisson.pmf(y, lambda_fora)
        matriz[x, y] = p_poisson * tau(x, y, lambda_casa, lambda_fora, RHO)

matriz = matriz / matriz.sum()
p_vitoria_casa = np.tril(matriz, -1).sum()
p_empate = np.trace(matriz)
p_vitoria_fora = np.triu(matriz, 1).sum()

over25 = sum(matriz[x,y] for x in range(n) for y in range(n) if x+y > 2.5)
btts = sum(matriz[x,y] for x in range(1, n) for y in range(1, n))
idx_max = np.unravel_index(np.argmax(matriz), matriz.shape)

# Modulo 2 - Escanteios
esc_casa = (ESCANTEIOS_CASA_F + ESCANTEIOS_FORA_S) / 2 * PESO_MANDO_ESC
esc_fora = (ESCANTEIOS_FORA_F + ESCANTEIOS_CASA_S) / 2
lambda_esc = esc_casa + esc_fora

# Modulo 3 - Cartoes
fator_arb = CART_ARB / CART_LIGA
fator_int = PESO_CLASSICO * PESO_DECISAO * PESO_RIVALIDADE
lambda_cart = (CART_CASA + CART_FORA) * fator_arb * fator_int

# Modulo 5 - Monte Carlo
N_SIM = 100_000
rng = np.random.default_rng(42)
gols_casa_sim = rng.poisson(lambda_casa, N_SIM)
gols_fora_sim = rng.poisson(lambda_fora, N_SIM)
mc_p_casa = np.sum(gols_casa_sim > gols_fora_sim) / N_SIM
mc_p_empate = np.sum(gols_casa_sim == gols_fora_sim) / N_SIM
mc_p_fora = np.sum(gols_casa_sim < gols_fora_sim) / N_SIM

# ==========================================
# INTERFACE PRINCIPAL (TABS)
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Resumo e Consenso", "🥅 Matriz Dixon-Coles", "🚩 Escanteios e Cartões", "💰 Gestão de Banca (Kelly)"])

with tab1:
    st.subheader(f"Previsão de Jogo: {TIME_CASA} x {TIME_FORA}")
    col1, col2, col3 = st.columns(3)
    col1.metric(f"Vitória {TIME_CASA}", f"{p_vitoria_casa*100:.1f}%", f"Odd Justa: {1/p_vitoria_casa:.2f}")
    col2.metric("Empate", f"{p_empate*100:.1f}%", f"Odd Justa: {1/p_empate:.2f}")
    col3.metric(f"Vitória {TIME_FORA}", f"{p_vitoria_fora*100:.1f}%", f"Odd Justa: {1/p_vitoria_fora:.2f}")
    
    st.markdown("---")
    col4, col5, col6 = st.columns(3)
    col4.metric("Over 2.5 Gols", f"{over25*100:.1f}%", f"Odd Justa: {1/over25:.2f}")
    col5.metric("Ambos Marcam (BTTS)", f"{btts*100:.1f}%", f"Odd Justa: {1/btts:.2f}")
    col6.metric("Placar Mais Provável", f"{idx_max[0]} x {idx_max[1]}")

with tab2:
    st.subheader("Matriz de Probabilidades de Placar Exato")
    st.markdown("Células mais escuras representam os placares mais prováveis.")
    
    # Formatando para exibição em Heatmap via Pandas Styling
    df_matriz = pd.DataFrame(matriz, index=[f"{TIME_CASA} {i}" for i in range(n)], columns=[f"{TIME_FORA} {i}" for i in range(n)])
    st.dataframe(df_matriz.style.background_gradient(cmap='Greens', axis=None).format("{:.2%}"), height=400, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🚩 Escanteios (Poisson)")
        st.write(f"**Expectativa {TIME_CASA}:** {esc_casa:.2f}")
        st.write(f"**Expectativa {TIME_FORA}:** {esc_fora:.2f}")
        st.write(f"**Total do Jogo (λ):** {lambda_esc:.2f}")
        
        df_esc = pd.DataFrame([{"Linha": f"Over {l}", "Probabilidade": f"{(1 - poisson.cdf(math.floor(l), lambda_esc))*100:.1f}%"} for l in [7.5, 8.5, 9.5, 10.5]])
        st.table(df_esc)
        
    with col2:
        st.subheader("🟨 Cartões (Intensidade)")
        st.write(f"**Fator Árbitro:** {fator_arb:.2f}")
        st.write(f"**Fator Intensidade:** {fator_int:.2f}")
        st.write(f"**Total do Jogo (λ):** {lambda_cart:.2f}")
        
        df_cart = pd.DataFrame([{"Linha": f"Over {l}", "Probabilidade": f"{(1 - poisson.cdf(math.floor(l), lambda_cart))*100:.1f}%"} for l in [3.5, 4.5, 5.5, 6.5]])
        st.table(df_cart)

with tab4:
    st.subheader("Critério de Kelly & Valor de Aposta (Edge)")
    
    mercados = {
        "Vitória Casa": (p_vitoria_casa, ODD_CASA),
        "Empate": (p_empate, ODD_EMPATE),
        "Vitória Visitante": (p_vitoria_fora, ODD_FORA),
        "Over 2.5": (over25, ODD_O25),
        "BTTS": (btts, ODD_BTTS)
    }
    
    tabela_kelly = []
    for nome, (prob, odd) in mercados.items():
        edge = (prob * odd) - 1
        kelly = max(0, edge / (odd - 1)) if odd > 1 else 0
        stake = kelly * BANCA
        tabela_kelly.append({"Mercado": nome, "Probabilidade": f"{prob*100:.1f}%", "Odd": odd, "Edge": f"{edge*100:.1f}%", "Aposta (1/2 Kelly)": f"R$ {stake/2:.2f}"})
        
    st.table(pd.DataFrame(tabela_kelly))
    st.caption("Edge positivo significa que o modelo encontrou valor na odd oferecida pela casa de aposta. A coluna de Aposta utiliza o critério de 'Meio Kelly' para gestão conservadora.")

