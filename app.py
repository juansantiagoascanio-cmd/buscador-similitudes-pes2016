import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import ftfy
import json

# 1. Configuración de la interfaz web
st.set_page_config(page_title="PES 2016 - Buscador de Similitudes", layout="centered")
st.title("PES 2016 – Buscador de similitudes entre jugadores")

# 2. Carga y limpieza adaptativa de la base de datos (Estructura Firebase JSON)
@st.cache_data
def cargar_datos_pes2016():
    try:
        with open("data.json", "r", encoding="utf-8-sig") as f:
            datos_json = json.load(f)
        
        if "players" in datos_json:
            lista_jugadores = list(datos_json["players"].values())
            df = pd.DataFrame(lista_jugadores)
        else:
            df = pd.DataFrame(datos_json)
            
    except (ValueError, json.JSONDecodeError):
        try:
            with open("data.json", "r", encoding="latin-1") as f:
                datos_json = json.load(f)
            if "players" in datos_json:
                lista_jugadores = list(datos_json["players"].values())
                df = pd.DataFrame(lista_jugadores)
            else:
                df = pd.DataFrame(datos_json)
        except FileNotFoundError:
            st.error("⚠️ No se encontró el archivo 'data.json'. Asegúrate de que esté en la misma carpeta.")
            st.stop()
    except FileNotFoundError:
        st.error("⚠️ No se encontró el archivo 'data.json'. Asegúrate de que esté en la misma carpeta.")
        st.stop()
        
    if "Player Name" not in df.columns:
        st.error("⚠️ No se encontró la columna 'Player Name' en el archivo JSON. Verifica su estructura.")
        st.stop()

    df["Player Name"] = df["Player Name"].fillna("").astype(str).apply(ftfy.fix_text)
    if "Team Name" in df.columns:
        df["Team Name"] = df["Team Name"].fillna("").astype(str).apply(ftfy.fix_text)
        
    columnas_limpieza = ['Player Name', 'Age', 'Overall Rating', 'Low Pass', 'Dribbling']
    columnas_presentes = [col for col in columnas_limpieza if col in df.columns]
    df = df.dropna(subset=columnas_presentes)
    
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(25)
    df['Overall Rating'] = pd.to_numeric(df['Overall Rating'], errors='coerce').fillna(70)
    
    return df

# Inicialización de la variable global de datos
df = cargar_datos_pes2016()

# 3. Formulario de selección y filtros
with st.container():
    # MODIFICACIÓN: Ordenar el DataFrame por valoración general (de mejor a peor) antes de extraer los nombres único
    df_ordenado_por_rating = df.sort_values(by="Overall Rating", ascending=False)
    
    # Usamos dict.fromkeys() para mantener el orden descendente eliminando duplicados si los hubiera
    jugadores_disponibles = list(dict.fromkeys(df_ordenado_por_rating["Player Name"]))
    
    jugador_seleccionado = st.selectbox("Seleccionar jugador:", jugadores_disponibles)
    
    edad_maxima = st.slider("Edad máxima:", min_value=15, max_value=45, value=25, step=1)
    calificacion_maxima = st.slider("Calificación máxima general:", min_value=40, max_value=109, value=80, step=1)
    
    n_resultados = st.number_input("Los N mejores partidos:", min_value=1, max_value=30, value=5)
    buscar = st.button("Encuentra jugadores similares")

# 4. Procesamiento e índice de similitud matemática
if buscar:
    fila_objetivo = df[df["Player Name"] == jugador_seleccionado].iloc[0]
    id_objetivo = fila_objetivo.name 
    
    df_filtrado = df[
        (df["Age"] <= edad_maxima) & 
        (df["Overall Rating"] <= calificacion_maxima) & 
        (df["Player Name"] != jugador_seleccionado)
    ]
    
    if df_filtrado.empty:
        st.warning("No hay jugadores en tu base de datos que cumplan simultáneamente con los filtros de Edad y General establecidos.")
    else:
        columnas_metricas = [
            'Attacking Prowess', 'Ball Control', 'Dribbling', 'Low Pass', 'Lofted Pass',
            'Finishing', 'Place Kicking', 'Swerve', 'Header', 'Defensive Prowess',
            'Ball Winning', 'Kicking Power', 'Speed', 'Explosive Power', 'Body Balance',
            'Jump', 'Stamina', 'Goalkeeping', 'Catching', 'Clearing', 'Reflexes', 'Coverage'
        ]
        
        columnas_metricas = [col for col in columnas_metricas if col in df.columns]
        
        df_num = df.copy()
        for col in columnas_metricas:
            df_num[col] = pd.to_numeric(df_num[col], errors='coerce').fillna(40)
            
        scaler = MinMaxScaler()
        df_metricas_norm = scaler.fit_transform(df_num[columnas_metricas])
        df_norm_completo = pd.DataFrame(df_metricas_norm, columns=columnas_metricas, index=df_num.index)
        
        vector_objetivo = df_norm_completo.loc[[id_objetivo]]
        vectores_filtrados = df_norm_completo.loc[df_filtrado.index]
        
        similitudes = cosine_similarity(vectores_filtrados, vector_objetivo)
        
        df_filtrado = df_filtrado.copy()
        df_filtrado["Similitud (%)"] = (similitudes.flatten() * 100).round(2)
        
        resultados = df_filtrado.sort_values(by="Similitud (%)", ascending=False).head(int(n_resultados))
        
        st.subheader(f"Jugadores más similares a {jugador_seleccionado}:")
        
        columnas_visibles = ["Player Name", "Age", "Overall Rating"]
        if "Team Name" in df.columns:
            columnas_visibles.append("Team Name")
        if "Position" in df.columns:
            columnas_visibles.append