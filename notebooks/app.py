import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

# ==============================================================================
# 1. CONFIGURACIÓ DE LA PÀGINA
# ==============================================================================
st.set_page_config(page_title="Simulador Vulnerabilitat BCN", layout="wide", page_icon="🌍")

st.title("🌍 Simulador de Vulnerabilitat Energètica de Barcelona")
st.markdown("""
Aquesta eina permet avaluar l'impacte de diferents escenaris climàtics i demogràfics 
sobre l'Índex de Vulnerabilitat (CVI) utilitzant el model d'Intel·ligència Artificial (Random Forest) desenvolupat al TFG.
""")

# ==============================================================================
# 2. DICCIONARI D'ESCENARIS (Sincronitzat amb el teu codi d'entrenament)
# ==============================================================================
DICCIONARI_ESCENARIS = {
    '0': [
        'renda_disponible_Import_Euros', 'pad_dom_Llars_1_Avi_Sol', 
        'edificacions_any_Any_Mitja_Ponderat', 'heating_thermal_demand_intensity__2025-03-06', 
        'torrid_nights__2024-01-01'
    ],
    '1': [
        'pad_dom_Llars_1_Avi_Sol', 'edificacions_any_Any_Mitja_Ponderat', 
        'heating_thermal_demand_intensity__2025-03-06', 'torrid_nights__2024-01-01'
    ],
    '3A': [
        'renda_disponible_Import_Euros', 'pad_dom_Llars_1_Avi_Sol', 
        'edificacions_any_Any_Mitja_Ponderat', 'heating_degree_days__2024-01-01', 
        'heating_thermal_demand_intensity__2025-03-06'
    ],
    '3B': [
        'renda_disponible_Import_Euros', 'pad_dom_Llars_1_Avi_Sol', 
        'edificacions_any_Any_Mitja_Ponderat', 'torrid_nights__2024-01-01', 
        'vegetation_index_avg__2022-01-01'
    ],
    '4A': [
        'edificacions_any_Any_Mitja_Ponderat', 'edificacions_superficie_Superficie_m2', 
        'percentage_population_over_65__2022-01-01', 'percentage_single_person_households__2022-01-01', 
        'heating_thermal_demand_intensity__2025-03-06', 'torrid_nights__2024-01-01', 
        'vegetation_index_avg__2022-01-01'
    ]
}

opcions_menu = {
    "Escenari 0: Baseline (Amb Renda)": "0",
    "Escenari 1: Estudi d'Ablació (Sense Renda)": "1",
    "Escenari 3A: Model d'Hivern": "3A",
    "Escenari 3B: Model d'Estiu": "3B",
    "Escenari 4A: Predictor Urbà Complet": "4A"
}

# ==============================================================================
# 3. PANELL LATERAL: SELECCIÓ I CONTROLS
# ==============================================================================
st.sidebar.header("⚙️ Configuració del Model")
nom_escenari = st.sidebar.selectbox("📊 Selecciona l'Escenari a visualitzar:", list(opcions_menu.keys()))
escenari_actual = opcions_menu[nom_escenari]
columnes_X = DICCIONARI_ESCENARIS[escenari_actual]

st.sidebar.markdown("---")
st.sidebar.header("🛠️ Simulació (What-If)")
st.sidebar.markdown("Modifica els paràmetres urbans per veure com reacciona la ciutat en aquest escenari:")

# Només mostrem els sliders si la variable corresponent existeix a l'escenari triat
mod_renda = st.sidebar.slider("💶 Variació de la Renda (%)", -30, 30, 0, step=5) if 'renda_disponible_Import_Euros' in columnes_X else 0
mod_calor = st.sidebar.slider("🌡️ Increment Nits Tòrrides (%)", 0, 100, 0, step=5) if 'torrid_nights__2024-01-01' in columnes_X else 0
mod_avis_sols = st.sidebar.slider("🧓 Augment Avis Sols (%)", 0, 50, 0, step=5) if 'pad_dom_Llars_1_Avi_Sol' in columnes_X else 0
mod_gent_gran = st.sidebar.slider("👴 Augment Densitat Gent Gran (%)", 0, 50, 0, step=5) if 'percentage_population_over_65__2022-01-01' in columnes_X else 0
mod_verd = st.sidebar.slider("🌳 Augment del Verd Urbà (%)", 0, 50, 0, step=5) if 'vegetation_index_avg__2022-01-01' in columnes_X else 0

# ==============================================================================
# 4. FUNCIONS DE CÀRREGA DE DADES
# ==============================================================================
@st.cache_resource
def carregar_model(escenari):
    ruta = f'../models/model_vulnerabilitat_RF_Escenari_{escenari}.joblib'
    if os.path.exists(ruta):
        return joblib.load(ruta)
    return None

@st.cache_data
def carregar_dades(escenari):
    ruta = f'../data/processed/mapa_prediccions_Escenari_{escenari}.csv'
    if os.path.exists(ruta):
        return pd.read_csv(ruta, sep=';', decimal=',')
    return None

# ==============================================================================
# 5. EXECUCIÓ DEL CÀLCUL
# ==============================================================================
model = carregar_model(escenari_actual)
df_base = carregar_dades(escenari_actual)

if model is None or df_base is None:
    st.warning(f"⚠️ **Dades no trobades per a l'{nom_escenari}.**\nSi us plau, executa primer el teu script de Python d'entrenament posant `ESCENARI_ACTUAL = '{escenari_actual}'` per generar els arxius necessaris.")
else:
    # Fem una còpia per aplicar-hi les variacions
    df_simulat = df_base.copy()

    # Apliquem matemàticament les variacions només si la variable està a l'escenari
    if 'renda_disponible_Import_Euros' in columnes_X:
        df_simulat['renda_disponible_Import_Euros'] *= (1 + (mod_renda / 100))
    if 'torrid_nights__2024-01-01' in columnes_X:
        df_simulat['torrid_nights__2024-01-01'] *= (1 + (mod_calor / 100))
    if 'pad_dom_Llars_1_Avi_Sol' in columnes_X:
        df_simulat['pad_dom_Llars_1_Avi_Sol'] *= (1 + (mod_avis_sols / 100))
    if 'percentage_population_over_65__2022-01-01' in columnes_X:
        df_simulat['percentage_population_over_65__2022-01-01'] *= (1 + (mod_gent_gran / 100))
    if 'vegetation_index_avg__2022-01-01' in columnes_X:
        df_simulat['vegetation_index_avg__2022-01-01'] *= (1 + (mod_verd / 100))

    # Tornem a predir amb el Random Forest
    df_simulat['CVI_Simulat'] = model.predict(df_simulat[columnes_X])

    # ==============================================================================
    # 6. VISUALITZACIÓ DEL MAPA (TRADUCTOR WKT A GEOJSON)
    # ==============================================================================
    import json
    import shapely.wkt
    import shapely.geometry
    
    ruta_geojson = '../data/raw/BarcelonaCiutat_SeccionsCensals.json' 
    
    try:
        # 1. Llegim l'arxiu de l'Ajuntament
        with open(ruta_geojson, 'r', encoding='utf-8') as f:
            dades_brutes = json.load(f)
            
        # 2. Reconstruïm el mapa sencer sobre la marxa
        geojson_bcn = {"type": "FeatureCollection", "features": []}
        
        for element in dades_brutes:
            # Generem el codi de 9 xifres per connectar el pont
            dist = str(element.get('codi_districte', '')).strip().zfill(2)
            sec = str(element.get('codi_seccio_censal', '')).strip().zfill(3)
            codi_mapa = "8019" + dist + sec
            
            # TRADUCTOR MÀGIC: Converteix el text 'POLYGON' a coordenades reals
            geom_text = element.get('geometria_wgs84')
            if isinstance(geom_text, str):
                try:
                    # Desempaquetem el WKT
                    shapely_geom = shapely.wkt.loads(geom_text)
                    geom_obj = shapely.geometry.mapping(shapely_geom)
                except Exception:
                    geom_obj = None
            else:
                geom_obj = None
                
            feature = {
                "type": "Feature",
                "id": codi_mapa, 
                "properties": {"Nom_Barri": element.get('nom_barri', '')},
                "geometry": geom_obj
            }
            if geom_obj is not None:
                geojson_bcn["features"].append(feature)
            
        # 3. Assegurem que l'Excel no tingui ".0" decimals per emparellar bé
        df_simulat['CODI_UNIC'] = df_simulat['CODI_UNIC'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            
        # 4. Dibuixem el mapa
        fig = px.choropleth_mapbox(
            df_simulat,
            geojson=geojson_bcn,
            locations='CODI_UNIC', 
            featureidkey='id', 
            color='CVI_Simulat',
            color_continuous_scale="viridis",
            mapbox_style="carto-positron",
            zoom=11.5, 
            center={"lat": 41.38879, "lon": 2.15899}, 
            opacity=0.6,
            labels={'CVI_Simulat': 'Vulnerabilitat'}
        )
        
        # Ajust de la llegenda petita
        fig.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            coloraxis_colorbar=dict(title="Índex CVI", thickness=15, len=0.6, yanchor="middle", y=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"⚠️ Error carregant el mapa: {e}")
        st.dataframe(df_simulat[['CODI_UNIC', 'CVI_Simulat']].sort_values(by='CVI_Simulat', ascending=False).head(20), use_container_width=True)