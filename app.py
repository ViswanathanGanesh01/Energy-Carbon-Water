import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.neighbors import KNeighborsRegressor
import duckdb
import os
import re
import json
import streamlit.components.v1 as components
from matplotlib.colors import LinearSegmentedColormap

# --- JOURNAL QUALITY SETTINGS ---
plt.rcParams['savefig.dpi'] = 200           
plt.rcParams['font.family'] = 'sans-serif' 

# --- CONFIGURATION & PATHS ---
PARQUET_ROOT = "results_parquet"
CLIMATE_STATIONS_CSV = "climate_zones.csv"
STATION_CARBON_MAPPING_CSV = "station_carbon_mapping.csv"

# Fixed Parameter
DEFAULT_IT_LOAD = 0.5

# Research Constants
DRY_ARCH_IDS = [4, 5, 6, 10, 11, 12]
ARCHS_AIR = [1, 2, 3, 4, 5, 6]
ARCHS_LIQUID = [7, 8, 9, 10, 11, 12]
COC = 5.0
DRIFT_RATE = 0.0002

# New Naming Scheme Mapping
ARCH_MAP = {
    1: "AXEW", 2: "ACEW", 3: "ACXW",
    4: "AXED", 5: "ACED", 6: "ACXD",
    7: "LXEW", 8: "LCEW", 9: "LCXW",
    10: "LXED", 11: "LCED", 12: "LCXD"
}
INV_ARCH_MAP = {v: k for k, v in ARCH_MAP.items()}

LIQ_SCENARIOS = {
    "W32": [{"max": 35.0, "score": 5}, {"max": 40.1, "score": 4}, {"max": 45.1, "score": 3}],
    "W40": [{"max": 40.1, "score": 5}, {"max": 35.0, "score": 5}, {"max": 45.1, "score": 4}],
    "W45": [{"max": 45.1, "score": 5}, {"max": 40.1, "score": 5}, {"max": 35.0, "score": 5}]
}

# --- CUSTOM COLORMAPS ---
thermal_colors = ["#d3d3d3", "#e5f5e0", "#a1d99b", "#74c476", "#31a354", "#006d2c"]
custom_thermal_cmap = LinearSegmentedColormap.from_list("thermal_grey_green", thermal_colors, N=256)
arch_colors = plt.cm.tab20(np.linspace(0, 1, 12))
air_arch_cmap = LinearSegmentedColormap.from_list("air_arch", arch_colors[:6], N=6)
liq_arch_cmap = LinearSegmentedColormap.from_list("liq_arch", arch_colors[6:], N=6)

# --- PAGE CONFIG ---
st.set_page_config(page_title="Energy-Carbon-Water", layout="wide", page_icon="🌍")

# REFINED COMPACT UI STYLING
st.markdown("""
    <style>
    .main .block-container { padding-top: 1rem !important; padding-bottom: 0rem !important; max-width: 98% !important; }
    .main [data-testid="stVerticalBlock"] { gap: 0.1rem !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 40px; font-size: 14px; padding: 0px 15px; }
    header[data-testid="stHeader"] { visibility: hidden; height: 0px; }
    footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

# --- UTILITIES ---
def parse_coordinate(coord_str):
    if pd.isna(coord_str): return None
    if isinstance(coord_str, (int, float)): return float(coord_str)
    s = str(coord_str).strip().upper()
    numeric_part = re.sub(r'[^\d\.-]', '', s)
    try:
        val = float(numeric_part)
        if 'S' in s or 'W' in s: val = -val
        return val
    except: return None

@st.cache_data
def get_virtual_table_data():
    if not os.path.exists(PARQUET_ROOT):
        st.error(f"📁 Data folder '{PARQUET_ROOT}' not found.")
        return pd.DataFrame()

    con = duckdb.connect(database=':memory:')
    sql_query = f"""
    WITH raw_data AS (
        SELECT 
            *,
            regexp_extract(filename, 'Architecture_(\\d+)', 1)::INT as ArchID,
            regexp_extract(filename, 'IT_Load_([\\d\\.]+)', 1)::DOUBLE as IT_Load,
            upper(replace(replace(regexp_extract(filename, '([^/\\\\]+)\\.parquet$', 1), '.parquet', ''), 'Climate', '')) as ClimateKey,
            "time" - lag("time") OVER (PARTITION BY filename ORDER BY "time") as step_duration
        FROM read_parquet('{PARQUET_ROOT}/*/*/*.parquet', filename=True)
    ),
    site_summaries AS (
        SELECT 
            ArchID, IT_Load, ClimateKey,
            sum(PUE * coalesce(step_duration, 600)) / sum(coalesce(step_duration, 600)) as PUE,
            sum((PumpCW * 0.00153 * (TCWRet - TCWSup) + 
                ((PumpCW * 0.00153 * (TCWRet - TCWSup) / ({COC} - 1)) - (PumpCW * {DRIFT_RATE})) + 
                (PumpCW * {DRIFT_RATE})) * coalesce(step_duration, 600)) as total_water_L,
            sum((IT_Load * 1000) * (coalesce(step_duration, 600) / 3600.0)) as total_it_energy_kwh,
            max(CASE WHEN ArchID <= 6 THEN TAirSup ELSE TCDUSup END) as MaxT,
            avg(CASE WHEN ValveChi = 0 AND ValveWSE = 1 THEN 1.0 ELSE 0.0 END) as FC,
            avg(CASE WHEN ValveChi = 1 AND ValveWSE = 1 THEN 1.0 ELSE 0.0 END) as PMC,
            avg(CASE WHEN ValveChi = 1 AND ValveWSE = 0 THEN 1.0 ELSE 0.0 END) as FMC
        FROM raw_data
        GROUP BY 1, 2, 3
    )
    SELECT *, (total_water_L / NULLIF(total_it_energy_kwh, 0)) as WUE FROM site_summaries
    """
    metrics_df = con.execute(sql_query).df()
    metrics_df['MaxT'] = metrics_df['MaxT'].apply(lambda x: x - 273.15 if x > 150 else x)
    
    try:
        z = pd.read_csv(CLIMATE_STATIONS_CSV)
        z['ClimateKey'] = z['Zone'].astype(str).str.strip().str.upper().str.replace("CLIMATE", "", regex=False).str.strip("_").str.strip()
        z['StationNumber'] = z['Station Number'].astype(str).str.strip().str.zfill(6)
        c = pd.read_csv(STATION_CARBON_MAPPING_CSV)
        c['StationNumber'] = c['StationID'].astype(str).str.strip().str.zfill(6)
        c['Lat'] = c['Lat'].apply(parse_coordinate); c['Long'] = c['Long'].apply(parse_coordinate)
        c = c.dropna(subset=['Lat', 'Long']).loc[c['Lat'] > -60]
        cef_col = next((col for col in c.columns if 'CEF' in col or 'CO2' in col), 'CEF_kgCO2_per_kWh')
        geo = pd.merge(z[['StationNumber', 'ClimateKey']], c[['StationNumber', 'Lat', 'Long', cef_col]], on='StationNumber')
        if cef_col != 'CEF_kgCO2_per_kWh': geo = geo.rename(columns={cef_col: 'CEF_kgCO2_per_kWh'})
        return pd.merge(metrics_df, geo.rename(columns={'Long': 'Lon'}), on='ClimateKey', how='inner')
    except Exception as e:
        st.error(f"Geographic Merge Error: {e}")
        return metrics_df

df = get_virtual_table_data()

@st.cache_data
def get_interpolated_grid(values, coords, res):
    valid_mask = ~np.isnan(coords).any(axis=1) & ~np.isnan(values)
    clean_coords, clean_values = coords[valid_mask], values[valid_mask]
    if len(clean_coords) < 3: return None, None, None
    lons = np.linspace(-179.9, 179.9, int(360 / res) + 1)
    lats = np.linspace(-60.0, 88.0, int((88.0 + 60.0) / res) + 1) 
    xx, yy = np.meshgrid(lons, lats)
    knn = KNeighborsRegressor(n_neighbors=3, weights='distance').fit(clean_coords, clean_values)
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    return xx, yy, Z

# --- UI SIDEBAR ---
with st.sidebar:
    st.header("Weights Control")
    w_t = st.slider("Thermal Compliance Weight", 0.0, 1.0, 0.7)
    w_p = st.slider("PUE Weight", 0.0, 1.0, 0.0)
    w_w = st.slider("WUE Weight", 0.0, 1.0, 0.1)
    w_c = st.slider("CUE Weight", 0.0, 1.0, 0.2)
    st.divider()
    liq_mode = st.selectbox("ASHRAE Liquid Mode", ["W32", "W40", "W45"])
    res = st.select_slider("Map Resolution", options=[5.0, 4.0, 3.0, 2.0, 1.0], value=5.0)

# --- APP LOGIC ---
if not df.empty:
    def static_score_row(row):
        if row['ArchID'] <= 6:
            bins = [{"max": 27.1, "score": 5}, {"max": 32.1, "score": 4}, {"max": 35.1, "score": 3}, {"max": 40.1, "score": 2}, {"max": 45.1, "score": 1}]
        else:
            bins = [{"max": 32.1, "score": 4}, {"max": 40.1, "score": 3}, {"max": 45.1, "score": 2}, {"max": 100.0, "score": 1}]
        for b in bins:
            if row['MaxT'] <= b['max']: return b['score']
        return 0

    df['Thermal Compliance'] = df.apply(static_score_row, axis=1)
    
    def dynamic_thermal_suitability(row, mode):
        if row['ArchID'] <= 6: return row['Thermal Compliance'] / 5.0
        bins = LIQ_SCENARIOS[mode]
        for b in bins:
            if row['MaxT'] <= b['max']: return b['score'] / 5.0
        return 0.0

    def norm_metric(s, inv=True):
        if s.max() == s.min(): return s * 0 + 1.0
        n = (s - s.min()) / (s.max() - s.min() + 1e-9)
        return 1.0 - n if inv else n

    df['n_pue'] = norm_metric(df['PUE'])
    df['n_wue'] = norm_metric(df['WUE'])
    df['CUE'] = df['PUE'] * df.get('CEF_kgCO2_per_kWh', 0.4)
    df['n_cue'] = norm_metric(df['CUE'])
    df['n_therm'] = df.apply(lambda r: dynamic_thermal_suitability(r, liq_mode), axis=1)
    df['Suitability'] = (df['n_therm']*w_t + df['n_pue']*w_p + df['n_wue']*w_w + df['n_cue']*w_c).clip(0, 1)

    filtered = df[df['IT_Load'] == DEFAULT_IT_LOAD]

    tab1, tab2, tab3 = st.tabs(["🌍 Global Analysis", "🖼️ Multi-Metric View", "📊 Architecture Comparison"])

    with tab1:
        st.subheader("Interactive 3D Global Suitability")
        
        # --- OPENGLOBUS INTEGRATION ---
        # 1. Prepare data for the 3D Globus
        globus_df = filtered.sort_values('Suitability', ascending=False).drop_duplicates('ClimateKey')
        globus_df['Architecture'] = globus_df['ArchID'].map(ARCH_MAP)
        data_to_send = globus_df[['Lat', 'Lon', 'Suitability', 'Architecture', 'ClimateKey', 'PUE', 'WUE']].dropna().to_dict(orient='records')
        
        try:
            with open("globus_component.html", "r", encoding='utf-8') as f:
                html_template = f.read()
            # Injecting data into the placeholder
            globus_html = html_template.replace("/* DATA_PLACEHOLDER */ []", json.dumps(data_to_send))
            components.html(globus_html, height=600, scrolling=False)
        except Exception as e:
            st.error(f"Could not load 3D Globus: {e}. Ensure 'globus_component.html' is in the root directory.")

        st.divider()
        st.subheader("Regional Comparison (Static Robinson Projection)")
        col_air, col_liq = st.columns(2)
        
        map_data_air = filtered[filtered['ArchID'].isin(ARCHS_AIR)].sort_values('Suitability', ascending=False).drop_duplicates('ClimateKey').dropna(subset=['Lon', 'Lat'])
        map_data_liq = filtered[filtered['ArchID'].isin(ARCHS_LIQUID)].sort_values('Suitability', ascending=False).drop_duplicates('ClimateKey').dropna(subset=['Lon', 'Lat'])

        with col_air:
            st.markdown("**Best Air-Cooled Architecture**")
            xx, yy, Z = get_interpolated_grid(map_data_air['ArchID'].values, map_data_air[['Lon', 'Lat']].values, res)
            if xx is not None:
                fig_a = plt.figure(figsize=(8, 5))
                ax = plt.axes(projection=ccrs.Robinson())
                ax.set_global()
                mesh = ax.pcolormesh(xx, yy, Z, cmap=air_arch_cmap, vmin=0.5, vmax=6.5, transform=ccrs.PlateCarree(), shading='nearest', rasterized=True)
                ax.add_feature(cfeature.OCEAN, facecolor='#eef7fa', zorder=2); ax.coastlines(resolution='110m', linewidth=0.4, zorder=4)
                cb = plt.colorbar(mesh, orientation='horizontal', pad=0.08, shrink=0.8)
                cb.set_ticks(range(1, 7)); cb.set_ticklabels([ARCH_MAP[i] for i in range(1, 7)], rotation=45)
                st.pyplot(fig_a); plt.close(fig_a)

        with col_liq:
            st.markdown("**Best Liquid-Cooled Architecture**")
            xx, yy, Z = get_interpolated_grid(map_data_liq['ArchID'].values, map_data_liq[['Lon', 'Lat']].values, res)
            if xx is not None:
                fig_l = plt.figure(figsize=(8, 5))
                ax = plt.axes(projection=ccrs.Robinson())
                ax.set_global()
                mesh = ax.pcolormesh(xx, yy, Z, cmap=liq_arch_cmap, vmin=6.5, vmax=12.5, transform=ccrs.PlateCarree(), shading='nearest', rasterized=True)
                ax.add_feature(cfeature.OCEAN, facecolor='#eef7fa', zorder=2); ax.coastlines(resolution='110m', linewidth=0.4, zorder=4)
                cb = plt.colorbar(mesh, orientation='horizontal', pad=0.08, shrink=0.8)
                cb.set_ticks(range(7, 13)); cb.set_ticklabels([ARCH_MAP[i] for i in range(7, 13)], rotation=45)
                st.pyplot(fig_l); plt.close(fig_l)

    with tab2:
        p_arch_name = st.selectbox("Choose Cooling Architecture", [f"{ARCH_MAP[i]}" for i in range(1, 13)], key="t2_arch")
        current_arch_id = INV_ARCH_MAP[p_arch_name]
        panel_df = filtered[filtered['ArchID'] == current_arch_id].dropna(subset=['Lon', 'Lat'])
        
        metrics_list = ['Suitability', 'Thermal Compliance', 'PUE', 'WUE', 'CUE', 'FC', 'PMC', 'FMC']
        meta_panel = {
            'Suitability': (plt.cm.Spectral, (0, 1), ''), 'Thermal Compliance': (custom_thermal_cmap, (0, 5), '(ASHRAE)'),
            'PUE': (plt.cm.RdYlGn_r, (1.1, 1.7), ''), 'WUE': (plt.cm.Blues, (0, 2.5), '(L/kWh)'),
            'CUE': (plt.cm.Purples, (0, 1.5), '(kgCO2/kWh)'), 'FC': (plt.cm.YlGn, (0, 1), '(Ratio)'),
            'PMC': (plt.cm.YlOrBr, (0, 1), '(Ratio)'), 'FMC': (plt.cm.OrRd, (0, 1), '(Ratio)')
        }

        if not panel_df.empty:
            for row_idx in range(4):
                cols = st.columns(2)
                for col_idx in range(2):
                    metric = metrics_list[row_idx * 2 + col_idx]
                    with cols[col_idx]:
                        fig_m = plt.figure(figsize=(6, 4))
                        ax = plt.axes(projection=ccrs.Robinson()); ax.set_global()
                        xx, yy, Z = get_interpolated_grid(panel_df[metric].values, panel_df[['Lon', 'Lat']].values, res)
                        if xx is not None:
                            cmap, v_range, unit = meta_panel[metric]
                            mesh = ax.pcolormesh(xx, yy, Z, cmap=cmap, vmin=v_range[0], vmax=v_range[1], transform=ccrs.PlateCarree(), shading='nearest', rasterized=True)
                            ax.add_feature(cfeature.OCEAN, facecolor='#eef7fa', zorder=2); ax.coastlines(resolution='110m', linewidth=0.3, zorder=4)
                            cb = plt.colorbar(mesh, orientation='horizontal', pad=0.08, shrink=0.8)
                            cb.set_label(f"{metric} {unit}")
                            st.pyplot(fig_m, use_container_width=True)
                        plt.close(fig_m)

    with tab3:
        st.subheader("Architecture Comparison")
        summary = filtered.groupby('ArchID')['Suitability'].mean().reset_index()
        summary['Architecture'] = summary['ArchID'].map(ARCH_MAP)
        st.bar_chart(summary.set_index('Architecture')['Suitability'])
        st.dataframe(filtered.sort_values('Suitability', ascending=False).head(20), use_container_width=True)
else:
    st.error("Data processing failed.")