import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os
import json
import shapely.wkt
import shapely.geometry

# ==============================================================================
# 1. CONFIGURACIÓ DE LA PÀGINA
# ==============================================================================
st.set_page_config(page_title="Simulador Vulnerabilitat BCN", layout="wide")

st.title("Simulador de Vulnerabilitat Energètica de Barcelona")
st.markdown("""
Plataforma interactiva per a l'anàlisi de la vulnerabilitat energètica a Barcelona. Aquesta eina permet simular escenaris climàtics i avaluar polítiques públiques mitjançant el model predictiu Random Forest desenvolupat en aquest Treball de Final de Grau.
""")

# ==============================================================================
# 2. DICCIONARI D'ESCENARIS
# ==============================================================================
DICCIONARI_ESCENARIS = {
    '0': ['renda_disponible_Import_Euros', 'pad_dom_Llars_1_Avi_Sol', 'edificacions_any_Any_Mitja_Ponderat', 'heating_thermal_demand_intensity__2025-03-06', 'torrid_nights__2024-01-01'],
    '1': ['pad_dom_Llars_1_Avi_Sol', 'edificacions_any_Any_Mitja_Ponderat', 'heating_thermal_demand_intensity__2025-03-06', 'torrid_nights__2024-01-01'],
    '3A': ['renda_disponible_Import_Euros', 'pad_dom_Llars_1_Avi_Sol', 'edificacions_any_Any_Mitja_Ponderat', 'heating_degree_days__2024-01-01', 'heating_thermal_demand_intensity__2025-03-06'],
    '3B': ['renda_disponible_Import_Euros', 'pad_dom_Llars_1_Avi_Sol', 'edificacions_any_Any_Mitja_Ponderat', 'torrid_nights__2024-01-01', 'vegetation_index_avg__2022-01-01'],
    '4': ['pad_dom_Llars_1_Avi_Sol', 'edificacions_any_Any_Mitja_Ponderat', 'edificacions_superficie_Superficie_m2', 'percentage_population_over_65__2022-01-01', 'percentage_single_person_households__2022-01-01', 'heating_thermal_demand_intensity__2025-03-06', 'torrid_nights__2024-01-01', 'vegetation_index_avg__2022-01-01'],
    '4A': ['edificacions_any_Any_Mitja_Ponderat', 'edificacions_superficie_Superficie_m2', 'percentage_population_over_65__2022-01-01', 'percentage_single_person_households__2022-01-01', 'heating_thermal_demand_intensity__2025-03-06', 'torrid_nights__2024-01-01', 'vegetation_index_avg__2022-01-01']
}

opcions_menu = {
    "Escenari 0: Baseline (Amb Renda)": "0",
    "Escenari 1: Estudi d'Ablació (Sense Renda)": "1",
    "Escenari 3A: Model d'Hivern": "3A",
    "Escenari 3B: Model d'Estiu": "3B",
    "Escenari 4: Predictor Urbà Complet": "4",
    "Escenari 4A: Predictor Urbà Simplificat": "4A"
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
st.sidebar.markdown("Avalua l'impacte de polítiques i xocs climàtics:")

mod_renda = st.sidebar.slider("💶 Variació de la Renda (%)", -30, 30, 0, step=5, help="Simula l'impacte de polítiques de subsidis o una caiguda del poder adquisitiu.") if 'renda_disponible_Import_Euros' in columnes_X else 0
mod_calor = st.sidebar.slider("🌡️ Impacte d'Onada de Calor (%)", 0, 100, 0, step=5, help="Simula un xoc climàtic basat en l'increment de la temperatura nocturna. Per exemple, fixar el lliscador al 50% implica un augment del 50% en la freqüència de Nits Tòrrides respecte a la situació base de l'escenari (multiplica el valor inicial per 1,5)") if 'torrid_nights__2024-01-01' in columnes_X else 0
mod_verd = st.sidebar.slider("🌳 Polítiques de Verd Urbà (%)", 0, 50, 0, step=5, help="Simula polítiques de mitigació (ex. Eixos Verds).") if 'vegetation_index_avg__2022-01-01' in columnes_X else 0
mod_demanda = st.sidebar.slider("🏠 Rehabilitació Energètica (%)", 0, 50, 0, step=5, help="Simula obres estructurals. Un 20% significa una reducció del 20% en pèrdues de calor.") if 'heating_thermal_demand_intensity__2025-03-06' in columnes_X else 0

# ==============================================================================
# 4. FUNCIONS DE CÀRREGA DE DADES
# ==============================================================================
@st.cache_resource
def carregar_model(escenari):
    ruta = f'../models/model_vulnerabilitat_RF_Escenari_{escenari}.joblib'
    if os.path.exists(ruta): return joblib.load(ruta)
    return None

@st.cache_data
def carregar_dades(escenari):
    ruta = f'../data/processed/mapa_prediccions_Escenari_{escenari}.csv'
    if os.path.exists(ruta): return pd.read_csv(ruta, sep=';', decimal=',')
    return None

model = carregar_model(escenari_actual)
df_base_original = carregar_dades(escenari_actual)

if model is None or df_base_original is None:
    st.warning(f"⚠️ **Dades no trobades per a l'{nom_escenari}.**")
else:
    df_base = df_base_original.copy()
    
    # Punt de partida del propi escenari (sense tocar els sliders)
    df_base['CVI_Base_Escenari'] = model.predict(df_base[columnes_X])
    
    df_simulat = df_base.copy()

    # Apliquem modificacions dels sliders
    if 'renda_disponible_Import_Euros' in columnes_X: df_simulat['renda_disponible_Import_Euros'] *= (1 + (mod_renda / 100))
    if 'torrid_nights__2024-01-01' in columnes_X: df_simulat['torrid_nights__2024-01-01'] *= (1 + (mod_calor / 100))
    if 'vegetation_index_avg__2022-01-01' in columnes_X: df_simulat['vegetation_index_avg__2022-01-01'] *= (1 + (mod_verd / 100))
    if 'heating_thermal_demand_intensity__2025-03-06' in columnes_X: df_simulat['heating_thermal_demand_intensity__2025-03-06'] *= (1 - (mod_demanda / 100))

    # Predicció simulada de l'escenari modificat
    df_simulat['CVI_Simulat'] = model.predict(df_simulat[columnes_X])
    
    # 1. Delta original: Model vs Realitat (CRB 2023)
    df_simulat['Diferencia_vs_CRB'] = df_simulat['CVI_Simulat'] - df_base['CVI__2023-01-01']
    df_simulat['Impacte_Absolut_CRB'] = df_simulat['Diferencia_vs_CRB'].abs()
    
    # 2. Delta nou: Efecte exclusiu dels sliders (Simulació vs Base de l'Escenari)
    df_simulat['Efecte_Politiques'] = df_simulat['CVI_Simulat'] - df_base['CVI_Base_Escenari']

    # ==============================================================================
    # 5. EXTRACCIÓ DE GEOMETRIES (JSON) I CÀLCUL DE CENTRES
    # ==============================================================================
    ruta_geojson = '../data/raw/BarcelonaCiutat_SeccionsCensals.json' 
    with open(ruta_geojson, 'r', encoding='utf-8') as f:
        dades_brutes = json.load(f)
        
    geojson_bcn = {"type": "FeatureCollection", "features": []}
    diccionari_barris = {}
    diccionari_centres = {}
    
    for element in dades_brutes:
        dist = str(element.get('codi_districte', '')).strip().zfill(2)
        sec = str(element.get('codi_seccio_censal', '')).strip().zfill(3)
        codi_mapa = "8019" + dist + sec
        nom_barri = element.get('nom_barri', 'Desconegut')
        diccionari_barris[codi_mapa] = nom_barri
        
        geom_text = element.get('geometria_wgs84')
        if isinstance(geom_text, str):
            try:
                shapely_geom = shapely.wkt.loads(geom_text)
                geom_obj = shapely.geometry.mapping(shapely_geom)
                centre_geom = shapely_geom.centroid
                diccionari_centres[codi_mapa] = (centre_geom.y, centre_geom.x)
            except Exception:
                geom_obj = None
        else:
            geom_obj = None
        
        if geom_obj is not None:
            geojson_bcn["features"].append({
                "type": "Feature", "id": codi_mapa, 
                "properties": {"Nom_Barri": nom_barri}, "geometry": geom_obj
            })
        
    df_simulat['CODI_UNIC'] = df_simulat['CODI_UNIC'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    df_simulat['Nom_Barri'] = df_simulat['CODI_UNIC'].map(diccionari_barris)

    # ==============================================================================
    # 6. FILTRE PER BARRIS I SISTEMA DE PESTANYES (TABS)
    # ==============================================================================
    llista_barris = sorted(df_simulat['Nom_Barri'].dropna().unique())
    barri_seleccionat = st.selectbox(
        "🔍 Selecciona un barri per aïllar l'anàlisi:", 
        ["🌍 Tots els barris de Barcelona"] + list(llista_barris)
    )
    
    df_simulat['CODI_UNIC'] = df_simulat['CODI_UNIC'].astype(str)
    df_base['CODI_UNIC'] = df_base['CODI_UNIC'].astype(str)
    
    if barri_seleccionat != "🌍 Tots els barris de Barcelona":
        df_simulat = df_simulat[df_simulat['Nom_Barri'] == barri_seleccionat]
        df_base_filtrat = df_base[df_base['CODI_UNIC'].isin(df_simulat['CODI_UNIC'])]
        nivell_zoom = 13
        
        lats = [diccionari_centres[c][0] for c in df_simulat['CODI_UNIC'] if c in diccionari_centres]
        lons = [diccionari_centres[c][1] for c in df_simulat['CODI_UNIC'] if c in diccionari_centres]
        centre_mapa = {"lat": sum(lats)/len(lats), "lon": sum(lons)/len(lons)} if lats else {"lat": 41.38879, "lon": 2.15899}
    else:
        df_base_filtrat = df_base
        nivell_zoom = 11
        centre_mapa = {"lat": 41.38879, "lon": 2.1599}

    pestanya_mapa, pestanya_dades = st.tabs(["🗺️ Mapa de Vulnerabilitat", "📊 Anàlisi i Resultats"])

    # --- PESTANYA 1: MAPA ---
    with pestanya_mapa:
        tipus_mapa = st.radio(
            "Tipus de visualització del mapa:", 
            ["Vulnerabilitat Absoluta", "Impacte respecte CRB (Valor Absolut)", "Efecte de les Polítiques (What-If)"], 
            horizontal=True,
            help="1. Absoluta: Risc simulat. \n2. Impacte CRB: |CVI_Simulat - CVI_Original|. \n3. Efecte Polítiques: L'efecte directe (positiu o negatiu) d'haver mogut els controls lliscants."
        )
        
        if tipus_mapa == "Vulnerabilitat Absoluta":
            mantenir_escala = st.checkbox("🔒 Mantenir escala global", value=True) if barri_seleccionat != "🌍 Tots els barris de Barcelona" else True
            col_valor = 'CVI_Simulat'
            escala = "viridis"
            rang = [0, 80] if mantenir_escala else None 
            
        elif tipus_mapa == "Impacte respecte CRB (Valor Absolut)":
            col_valor = 'Impacte_Absolut_CRB'
            escala = "RdYlGn_r"
            limit_max = df_simulat['Impacte_Absolut_CRB'].max()
            limit_max = 1 if limit_max == 0 else limit_max
            rang = [0, limit_max] 
            
        else: # Efecte de les Polítiques (What-If)
            col_valor = 'Efecte_Politiques'
            escala = "RdYlGn_r"  # Divergent: Verd = baixa el CVI (millora), Vermell = puja (empitjora)
            limit_abs = max(abs(df_simulat['Efecte_Politiques'].min()), abs(df_simulat['Efecte_Politiques'].max()))
            limit_abs = 0.5 if limit_abs == 0 else limit_abs 
            rang = [-limit_abs, limit_abs] 

        fig = px.choropleth_mapbox(df_simulat, geojson=geojson_bcn, locations='CODI_UNIC', featureidkey='id', 
                                    color=col_valor, color_continuous_scale=escala, range_color=rang,
                                    mapbox_style="carto-positron", zoom=nivell_zoom, center=centre_mapa, 
                                    opacity=0.7, labels={col_valor: 'Punts CVI'})
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

    # --- PESTANYA 2: ANÀLISI I BOTÓ DE DESCÀRREGA ---
    with pestanya_dades:
        if barri_seleccionat == "🌍 Tots els barris de Barcelona":
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📈 Distribució Global (Simulació vs CRB)")
                df_hist = pd.DataFrame({'Vulnerabilitat (CVI)': pd.concat([df_base['CVI__2023-01-01'], df_simulat['CVI_Simulat']]),
                                        'Escenari': ['CRB Original'] * len(df_base) + ['Simulació'] * len(df_simulat)})
                st.plotly_chart(px.histogram(df_hist, x='Vulnerabilitat (CVI)', color='Escenari', barmode='overlay', opacity=0.75), use_container_width=True)
            with col2:
                st.subheader("🚨 Top 5 Seccions més Vulnerables")
                
                # Afegida la columna original i reanomenades per claredat
                top5 = df_simulat.nlargest(5, 'CVI_Simulat')[['Nom_Barri', 'CVI__2023-01-01', 'CVI_Simulat', 'Efecte_Politiques']]
                top5 = top5.rename(columns={'CVI__2023-01-01': 'CVI Original', 'Efecte_Politiques': 'Efecte Sliders'})
                st.dataframe(top5, hide_index=True, use_container_width=True)
        else:
            st.subheader(f"📊 Resum de l'Impacte: {barri_seleccionat}")
            
            df_barri = pd.merge(df_simulat[['CODI_UNIC', 'CVI_Simulat', 'Diferencia_vs_CRB', 'Efecte_Politiques']], 
                                df_base_filtrat[['CODI_UNIC', 'CVI__2023-01-01', 'CVI_Base_Escenari']], on='CODI_UNIC')
            
            mitjana_crb = df_barri['CVI__2023-01-01'].mean()
            mitjana_sim = df_barri['CVI_Simulat'].mean()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Mitjana CVI (CRB Original)", f"{mitjana_crb:.2f}")
            col2.metric("Mitjana CVI (Simulada actual)", f"{mitjana_sim:.2f}", delta=f"{(mitjana_sim - mitjana_crb):.2f} vs CRB", delta_color="inverse")
            col3.metric("Seccions analitzades", len(df_barri))
            
            st.markdown("---")
            st.write("Resum detallat per secció censal:")
            
            # Formategem la taula perquè es vegi neta incloent totes les dades
            taula_mostrar = df_barri[['CODI_UNIC', 'CVI__2023-01-01', 'CVI_Base_Escenari', 'CVI_Simulat', 'Diferencia_vs_CRB', 'Efecte_Politiques']].rename(
                columns={
                    'CVI__2023-01-01': 'CVI Original (CRB)', 
                    'CVI_Base_Escenari': 'Base del Model', 
                    'CVI_Simulat': 'Resultat Final', 
                    'Diferencia_vs_CRB': 'Delta vs CRB',
                    'Efecte_Politiques': 'Delta (Sliders)'
                }
            )
            st.dataframe(taula_mostrar, hide_index=True, use_container_width=True)
            
        # BOTÓ DE DESCÀRREGA
        st.markdown("---")
        st.subheader("📥 Exportació de Resultats")
        st.markdown("Descarrega l'estat actual per analitzar-lo a l'Excel:")
        
        # Preparem totes les columnes clau per exportar
        columnes_exportar = ['CODI_UNIC', 'Nom_Barri', 'CVI__2023-01-01', 'CVI_Simulat', 'Diferencia_vs_CRB', 'Impacte_Absolut_CRB', 'Efecte_Politiques'] + [col for col in columnes_X if col in df_simulat.columns]
        df_export = df_simulat[columnes_exportar].rename(columns={'CVI__2023-01-01': 'CVI_Original'})
        
        # Exportem en format ; per a Excel en espanyol/català
        csv_buffer = df_export.to_csv(index=False, sep=';', decimal=',').encode('utf-8')
        
        st.download_button(
            label="📄 Descarregar dades d'aquesta simulació (CSV)",
            data=csv_buffer,
            file_name=f"simulacio_{escenari_actual}_{barri_seleccionat.replace(' ', '_')}.csv",
            mime="text/csv"
        )