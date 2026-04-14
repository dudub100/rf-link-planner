import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from geopy.distance import geodesic
import itur 
from astropy import units as u
from fpdf import FPDF
import tempfile
import os
import urllib.parse
from streamlit_js_eval import streamlit_js_eval, get_geolocation

# --- 1. CONFIG & SESSION STATE ---
st.set_page_config(layout="wide", page_title="RF Link Profiler Pro")

# Initialize persistent session state
if "lat_a" not in st.session_state: st.session_state.lat_a = 40.7128
if "lon_a" not in st.session_state: st.session_state.lon_a = -74.0060
if "h_a" not in st.session_state: st.session_state.h_a = 15.0
if "lat_b" not in st.session_state: st.session_state.lat_b = 40.7306
if "lon_b" not in st.session_state: st.session_state.lon_b = -73.9866
if "h_b" not in st.session_state: st.session_state.h_b = 20.0
if "env_temp" not in st.session_state: st.session_state.env_temp = 15.0
if "env_rh" not in st.session_state: st.session_state.env_rh = 50.0
if "gps_requested" not in st.session_state: st.session_state.gps_requested = False
if "pdf_data" not in st.session_state: st.session_state.pdf_data = None

# Deep-Linking Peer Check
params = st.query_params
if "peer_lat" in params:
    try:
        st.session_state.lat_b = float(params["peer_lat"])
        st.session_state.lon_b = float(params["peer_lon"])
        st.session_state.h_b = float(params["peer_h"])
        st.toast("✅ Peer Site B Location Loaded!")
    except: pass

# --- 2. HELPER FUNCTIONS ---

def fetch_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m"
        r = requests.get(url, timeout=5).json()
        return {"temp": float(r['current']['temperature_2m']), "rh": float(r['current']['relative_humidity_2m'])}
    except: return None

def get_ground_elevation(lat, lon):
    url = f"https://api.opentopodata.org/v1/srtm30m?locations={lat},{lon}"
    try:
        res = requests.get(url, timeout=5).json()
        return res['results'][0]['elevation']
    except: return 0.0

def get_elevation_profile(lat1, lon1, lat2, lon2, num_points=50):
    lats = np.linspace(lat1, lat2, num_points)
    lons = np.linspace(lon1, lon2, num_points)
    coords = list(zip(lats, lons))
    distances = [0.0]
    for i in range(1, len(coords)):
        dist = geodesic(coords[i-1], coords[i]).meters
        distances.append(distances[-1] + dist)

    loc_str = "|".join([f"{lat},{lon}" for lat, lon in coords])
    try:
        r = requests.get(f"https://api.opentopodata.org/v1/srtm30m?locations={loc_str}", timeout=10)
        if r.status_code == 200:
            elevs = [res['elevation'] for res in r.json().get('results', [])]
            return pd.DataFrame({"Distance (m)": distances, "Elevation (m)": elevs, "Lat": lats, "Lon": lons})
    except: return None

def calculate_itu_attenuation(lat, lon, d_km, f_ghz, tx_power, gain_a, gain_b, t_c, rh):
    if d_km <= 0: return pd.DataFrame()
    fsl = 92.4 + 20 * np.log10(d_km) + 20 * np.log10(f_ghz)
    e_s = 6.1121 * np.exp((17.502 * t_c) / (240.97 + t_c))
    e = e_s * (rh / 100.0)
    rho = 216.7 * (e / (t_c + 273.15))
    try:
        gas_loss = float(itur.models.itu676.gaseous_attenuation_terrestrial_path(
            r=d_km*u.km, f=f_ghz*u.GHz, el=0.0*u.deg, rho=rho*(u.g/u.m**3), P=1013.25*u.hPa, T=t_c*u.deg_C, mode='approx').value)
    except: gas_loss = 0.0
    rsl_clear = tx_power + gain_a + gain_b - fsl - gas_loss
    results = []
    for avail in [99.0, 99.9, 99.99, 99.999]:
        p = round(100.0 - avail, 3) 
        try: rain = float(itur.models.itu530.rain_attenuation(lat=lat, lon=lon, d=d_km*u.km, f=f_ghz*u.GHz, el=0.0*u.deg, p=p).value)
        except: rain = 0.0
        results.append({"Avail.": f"{avail}%", "FSL (dB)": f"{fsl:.1f}", "Atm. (dB)": f"{gas_loss:.2f}", "Rain (dB)": f"{rain:.1f}", "Clear RSL": f"{rsl_clear:.1f} dBm", "Faded RSL": f"{rsl_clear-rain:.1f} dBm"})
    return pd.DataFrame(results)

# --- 3. GPS HANDSHAKE ---
loc = get_geolocation()
if st.session_state.gps_requested and loc:
    st.session_state.lat_a = loc['coords']['latitude']
    st.session_state.lon_a = loc['coords']['longitude']
    alt = loc['coords']['altitude'] if loc['coords']['altitude'] else 0
    ground = get_ground_elevation(st.session_state.lat_a, st.session_state.lon_a)
    st.session_state.h_a = round(float(max(5.0, alt - ground)), 1)
    st.session_state.gps_requested = False
    st.rerun()

# --- 4. SIDEBAR ---
st.sidebar.title("📡 Field Tools")
click_target = st.sidebar.radio("Map Click Updates:", ["None", "Site A", "Site B"])

# GPS and Weather with descriptive names
if st.sidebar.button("📍 Set Site A to My Location"):
    st.session_state.gps_requested = True
    st.rerun()

if st.sidebar.button("☁️ Sync Local Weather"):
    w = fetch_weather(st.session_state.lat_a, st.session_state.lon_a)
    if w:
        st.session_state.env_temp, st.session_state.env_rh = w['temp'], w['rh']
        st.rerun()

st.sidebar.divider()

# WhatsApp Share - Improved visibility
curr_url = streamlit_js_eval(js_expressions="window.location.href", want_output=True, key="get_url")
if curr_url:
    base = curr_url.split("?")[0]
    s_url = f"{base}?peer_lat={st.session_state.lat_a}&peer_lon={st.session_state.lon_a}&peer_h={st.session_state.h_a}"
    wa = f"https://wa.me/?text={urllib.parse.quote('Connect to my RF link: ' + s_url)}"
    st.sidebar.markdown(f'''<a href="{wa}" target="_blank" style="text-decoration:none;"><div style="background-color:#25D366;color:white;padding:12px;border-radius:8px;text-align:center;font-weight:bold;">📲 Share Location via WhatsApp</div></a>''', unsafe_allow_html=True)
else:
    st.sidebar.info("⏳ Initializing Share Link...")

st.sidebar.divider()

# Site Details
st.sidebar.subheader("Site A (Green)")
st.session_state.lat_a = st.sidebar.number_input("Latitude A", value=float(st.session_state.lat_a), format="%.6f")
st.session_state.lon_a = st.sidebar.number_input("Longitude A", value=float(st.session_state.lon_a), format="%.6f")
st.session_state.h_a = st.sidebar.number_input("Antenna Height A (m)", value=float(st.session_state.h_a))

st.sidebar.subheader("Site B (Red)")
st.session_state.lat_b = st.sidebar.number_input("Latitude B", value=float(st.session_state.lat_b), format="%.6f")
st.session_state.lon_b = st.sidebar.number_input("Longitude B", value=float(st.session_state.lon_b), format="%.6f")
st.session_state.h_b = st.sidebar.number_input("Antenna Height B (m)", value=float(st.session_state.h_b))

st.sidebar.divider()

# RF Parameters
st.sidebar.subheader("RF Parameters")
freq = st.sidebar.number_input("Frequency (GHz)", value=15.0, min_value=0.1, step=0.1)
tx_pwr = st.sidebar.number_input("Transmit Power (dBm)", value=20.0, step=1.0)

wl = 0.3 / freq
diam_a = st.sidebar.number_input("Dish A Diameter (m)", value=0.6, step=0.1)
gain_a = 10 * np.log10(0.55 * (np.pi * diam_a / wl)**2)
st.sidebar.caption(f"Calculated Gain A: {gain_a:.1f} dBi")

diam_b = st.sidebar.number_input("Dish B Diameter (m)", value=0.6, step=0.1)
gain_b = 10 * np.log10(0.55 * (np.pi * diam_b / wl)**2)
st.sidebar.caption(f"Calculated Gain B: {gain_b:.1f} dBi")

st.sidebar.subheader("Atmosphere")
temp = st.sidebar.number_input("Temperature (°C)", value=float(st.session_state.env_temp), format="%.1f")
rh = st.sidebar.number_input("Relative Humidity (%)", value=float(st.session_state.env_rh), format="%.1f")

# --- 5. MAIN UI ---
st.title("RF Path Profiler & Link Budget Tool")
col1, col2 = st.columns([1, 1])

with col1:
    m = folium.Map(location=[st.session_state.lat_a, st.session_state.lon_a], zoom_start=12)
    folium.Marker([st.session_state.lat_a, st.session_state.lon_a], icon=folium.Icon(color="green"), popup="Site A").add_to(m)
    folium.Marker([st.session_state.lat_b, st.session_state.lon_b], icon=folium.Icon(color="red"), popup="Site B").add_to(m)
    folium.PolyLine([(st.session_state.lat_a, st.session_state.lon_a), (st.session_state.lat_b, st.session_state.lon_b)], color="blue").add_to(m)
    m_data = st_folium(m, height=500, width=700, key="rf_map")
    if m_data and m_data.get("last_clicked"):
        lat, lon = m_data["last_clicked"]["lat"], m_data["last_clicked"]["lng"]
        if click_target == "Site A": st.session_state.lat_a, st.session_state.lon_a = lat, lon; st.rerun()
        if click_target == "Site B": st.session_state.lat_b, st.session_state.lon_b = lat, lon; st.rerun()

with col2:
    if st.button("🚀 Calculate Link Parameters"):
        with st.spinner("Analyzing Terrain Profile..."):
            df = get_elevation_profile(st.session_state.lat_a, st.session_state.lon_a, st.session_state.lat_b, st.session_state.lon_b)
            if df is not None:
                dist = df.iloc[-1]["Distance (m)"]
                abs_a = df.iloc[0]["Elevation (m)"] + st.session_state.h_a
                abs_b = df.iloc[-1]["Elevation (m)"] + st.session_state.h_b
                df["LOS"] = np.linspace(abs_a, abs_b, len(df))
                df["F1"] = 17.32 * np.sqrt(((df["Distance (m)"]/1000) * ((dist-df["Distance (m)"])/1000)) / (freq * (dist/1000)))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["Distance (m)"], y=df["Elevation (m)"], fill='tozeroy', name='Terrain', line=dict(color='SaddleBrown')))
                fig.add_trace(go.Scatter(x=df["Distance (m)"], y=df["LOS"], name='LOS', line=dict(color='red', dash='dash')))
                fig.add_trace(go.Scatter(x=df["Distance (m)"], y=df["LOS"]-df["F1"], fill='tonexty', name='1st Fresnel', line=dict(color='rgba(0,0,255,0.1)')))
                st.plotly_chart(fig, use_container_width=True)
                
                df_att = calculate_itu_attenuation(st.session_state.lat_a, st.session_state.lon_a, dist/1000, freq, tx_pwr, gain_a, gain_b, temp, rh)
                st.dataframe(df_att, hide_index=True)
