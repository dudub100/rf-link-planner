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
import re
import streamlit.components.v1 as components
from streamlit_js_eval import get_geolocation

# --- 1. INITIAL SETUP (MUST BE FIRST) ---
st.set_page_config(layout="wide", page_title="RF Link Profiler Pro")

# Bulletproof URL Parameter Parser (Strips WhatsApp's weird trailing characters)
def get_safe_param(key, default):
    try:
        if key in st.query_params:
            val = st.query_params[key]
            if isinstance(val, list): val = val[0]
            # Strip EVERYTHING except numbers, decimals, and minus signs
            clean_val = re.sub(r'[^\d\.\-]', '', str(val))
            if clean_val: return float(clean_val)
    except Exception as e:
        pass
    return float(default)

# --- 2. SESSION STATE INITIALIZATION ---
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
if "peer_loaded" not in st.session_state: st.session_state.peer_loaded = False

# --- 3. ROBUST DEEP-LINKING ---
if "peer_lat" in st.query_params and not st.session_state.peer_loaded:
    st.session_state.lat_b = get_safe_param("peer_lat", st.session_state.lat_b)
    st.session_state.lon_b = get_safe_param("peer_lon", st.session_state.lon_b)
    st.session_state.h_b = get_safe_param("peer_h", st.session_state.h_b)
    st.session_state.peer_loaded = True
    st.toast("✅ Peer Location Synchronized!", icon="📡")

# --- 4. RF & ELEVATION HELPERS ---
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
        distances.append(distances[-1] + geodesic(coords[i-1], coords[i]).meters)

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

def generate_pdf_report(lat_a, lon_a, lat_b, lon_b, d_km, f_ghz, profile_img_path, df_att):
    class PDF(FPDF):
        def header(self):
            self.set_font("helvetica", "B", 16)
            self.cell(0, 10, "RF Link Path Profile & Budget Report", align="C", ln=True)
            self.line(10, 22, 200, 22); self.ln(5)
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("helvetica", "B", 12); pdf.cell(0, 10, "1. Link Overview", ln=True)
    pdf.set_font("helvetica", "", 10)
    pdf.cell(0, 8, f"Site A: {lat_a:.6f}, {lon_a:.6f} | Site B: {lat_b:.6f}, {lon_b:.6f}", ln=True)
    pdf.cell(0, 8, f"Frequency: {f_ghz} GHz | Path Distance: {d_km:.3f} km", ln=True)
    pdf.ln(5)
    pdf.set_font("helvetica", "B", 12); pdf.cell(0, 10, "2. Terrain & Fresnel Profile", ln=True)
    pdf.image(profile_img_path, x=10, w=190); pdf.ln(5)
    pdf.set_font("helvetica", "B", 12); pdf.cell(0, 10, "3. ITU-R Estimated Link Budget", ln=True)
    pdf.set_font("helvetica", "B", 8)
    for col in list(df_att.columns): pdf.cell(31, 8, col, border=1, align="C")
    pdf.ln()
    pdf.set_font("helvetica", "", 8)
    for row in df_att.itertuples(index=False):
        for item in row: pdf.cell(31, 8, str(item), border=1, align="C")
        pdf.ln()
    return bytes(pdf.output())

# --- 5. NATIVE WHATSAPP SHARE BUTTON ---
def render_whatsapp_button(lat, lon, h):
    html_code = f"""
    <a id="wa-link" href="#" target="_blank" style="text-decoration:none; display:block; background-color:#25D366; color:white; padding:12px; border-radius:8px; text-align:center; font-family:sans-serif; font-weight:bold;">
        📲 Share via WhatsApp
    </a>
    <script>
        // Native JS handles the URL instead of Python, preventing React crashes
        setTimeout(function() {{
            var base = window.location.href.split('?')[0];
            var shareUrl = base + "?peer_lat={lat}&peer_lon={lon}&peer_h={h}";
            var msg = encodeURIComponent("Connect to my RF link: " + shareUrl);
            document.getElementById('wa-link').href = "https://wa.me/?text=" + msg;
        }}, 200);
    </script>
    """
    components.html(html_code, height=60)

# --- 6. SIDEBAR TOOLS ---
st.sidebar.title("📡 Field Tools")
click_target = st.sidebar.radio("Map Click Updates:", ["None", "Site A", "Site B"])

if st.sidebar.button("📍 Set Site A to My Location"):
    st.session_state.gps_requested = True
    st.rerun()

loc_data = get_geolocation()
if st.session_state.gps_requested and loc_data:
    st.session_state.lat_a = float(loc_data['coords']['latitude'])
    st.session_state.lon_a = float(loc_data['coords']['longitude'])
    alt = loc_data['coords']['altitude'] if loc_data['coords']['altitude'] else 0
    ground = get_ground_elevation(st.session_state.lat_a, st.session_state.lon_a)
    st.session_state.h_a = round(float(max(5.0, alt - ground)), 1)
    st.session_state.gps_requested = False
    st.rerun()

if st.sidebar.button("☁️ Sync Local Weather"):
    w = fetch_weather(st.session_state.lat_a, st.session_state.lon_a)
    if w:
        st.session_state.env_temp, st.session_state.env_rh = w['temp'], w['rh']
        st.rerun()

st.sidebar.divider()

# Call the safe HTML WhatsApp button
render_whatsapp_button(st.session_state.lat_a, st.session_state.lon_a, st.session_state.h_a)

st.sidebar.divider()

# Manual Site Inputs
st.sidebar.subheader("Site A (Green)")
st.session_state.lat_a = st.sidebar.number_input("Lat A", value=float(st.session_state.lat_a), format="%.6f")
st.session_state.lon_a = st.sidebar.number_input("Lon A", value=float(st.session_state.lon_a), format="%.6f")
st.session_state.h_a = st.sidebar.number_input("Height A (m)", value=float(st.session_state.h_a))

st.sidebar.subheader("Site B (Red)")
st.session_state.lat_b = st.sidebar.number_input("Lat B", value=float(st.session_state.lat_b), format="%.6f")
st.session_state.lon_b = st.sidebar.number_input("Lon B", value=float(st.session_state.lon_b), format="%.6f")
st.session_state.h_b = st.sidebar.number_input("Height B (m)", value=float(st.session_state.h_b))

st.sidebar.divider()

st.sidebar.subheader("RF Parameters")
freq = st.sidebar.number_input("Frequency (GHz)", value=15.0, min_value=0.1)
tx_p = st.sidebar.number_input("TX Power (dBm)", value=20.0)
diam_a = st.sidebar.number_input("Dish A (m)", value=0.6)
gain_a = 10 * np.log10(0.55 * (np.pi * diam_a / (0.3/freq))**2)
diam_b = st.sidebar.number_input("Dish B (m)", value=0.6)
gain_b = 10 * np.log10(0.55 * (np.pi * diam_b / (0.3/freq))**2)

st.sidebar.subheader("Atmosphere")
temp = st.sidebar.number_input("Temp (°C)", value=float(st.session_state.env_temp), format="%.1f")
rh = st.sidebar.number_input("Humidity (%)", value=float(st.session_state.env_rh), format="%.1f")

# --- 7. MAIN DISPLAY ---
st.title("RF Path Profiler")
c1, c2 = st.columns([1, 1])

with c1:
    m = folium.Map(location=[float(st.session_state.lat_a), float(st.session_state.lon_a)], zoom_start=12)
    folium.Marker([float(st.session_state.lat_a), float(st.session_state.lon_a)], icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([float(st.session_state.lat_b), float(st.session_state.lon_b)], icon=folium.Icon(color="red")).add_to(m)
    folium.PolyLine([(float(st.session_state.lat_a), float(st.session_state.lon_a)), (float(st.session_state.lat_b), float(st.session_state.lon_b))], color="blue").add_to(m)
    m_data = st_folium(m, height=500, width=700, key="rf_map")
    
    if m_data and m_data.get("last_clicked"):
        lat, lon = m_data["last_clicked"]["lat"], m_data["last_clicked"]["lng"]
        if click_target == "Site A": st.session_state.lat_a, st.session_state.lon_a = lat, lon; st.rerun()
        if click_target == "Site B": st.session_state.lat_b, st.session_state.lon_b = lat, lon; st.rerun()

with c2:
    if st.button("🚀 Calculate Link"):
        with st.spinner("Analyzing..."):
            df = get_elevation_profile(st.session_state.lat_a, st.session_state.lon_a, st.session_state.lat_b, st.session_state.lon_b)
            if df is not None:
                dist = df.iloc[-1]["Distance (m)"]
                abs_a, abs_b = df.iloc[0]["Elevation (m)"]+st.session_state.h_a, df.iloc[-1]["Elevation (m)"]+st.session_state.h_b
                df["LOS"] = np.linspace(abs_a, abs_b, len(df))
                df["F1"] = 17.32 * np.sqrt(((df["Distance (m)"]/1000)*((dist-df["Distance (m)"])/1000))/(freq*(dist/1000)))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["Distance (m)"], y=df["Elevation (m)"], fill='tozeroy', name='Terrain'))
                fig.add_trace(go.Scatter(x=df["Distance (m)"], y=df["LOS"], name='LOS', line=dict(color='red', dash='dash')))
                fig.add_trace(go.Scatter(x=df["Distance (m)"], y=df["LOS"]-df["F1"], fill='tonexty', name='1st Fresnel'))
                st.plotly_chart(fig, use_container_width=True)
                
                df_att = calculate_itu_attenuation(st.session_state.lat_a, st.session_state.lon_a, dist/1000, freq, tx_p, gain_a, gain_b, temp, rh)
                st.dataframe(df_att, hide_index=True)
                
                with tempfile.TemporaryDirectory() as tmp:
                    img_path = os.path.join(tmp, "p.png")
                    fig.write_image(img_path)
                    st.session_state.pdf_data = generate_pdf_report(st.session_state.lat_a, st.session_state.lon_a, st.session_state.lat_b, st.session_state.lon_b, dist/1000, freq, img_path, df_att)

    if st.session_state.pdf_data:
        st.download_button("📄 Download PDF", st.session_state.pdf_data, "RF_Report.pdf")
