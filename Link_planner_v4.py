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
import urllib.parse
from streamlit_js_eval import get_geolocation

# --- 1. INITIAL SETUP ---
st.set_page_config(layout="wide", page_title="RF Link Profiler Pro")

def get_safe_param(key, default):
    try:
        val = st.query_params.get(key)
        if val:
            if isinstance(val, list): val = val[0]
            clean_val = re.sub(r'[^\d\.\-]', '', str(val))
            if clean_val: return float(clean_val)
    except: pass
    return float(default)

# Session State Initialization
defaults = {
    "lat_a": 40.7128, "lon_a": -74.0060, "h_a": 15.0,
    "lat_b": 40.7306, "lon_b": -73.9866, "h_b": 20.0,
    "env_temp": 15.0, "env_rh": 50.0, 
    "ch_bw": 56.0, "nf": 5.0, "max_qam": 4096,
    "gps_requested": False, "pdf_data": None, "peer_loaded": False
}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

# --- 2. DEEP-LINKING ---
if "peer_lat" in st.query_params and not st.session_state.peer_loaded:
    st.session_state.lat_b = get_safe_param("peer_lat", st.session_state.lat_b)
    st.session_state.lon_b = get_safe_param("peer_lon", st.session_state.lon_b)
    st.session_state.h_b = get_safe_param("peer_h", st.session_state.h_b)
    st.session_state.peer_loaded = True

# --- 3. HELPER FUNCTIONS ---
def get_ground_elevation(lat, lon):
    try:
        res = requests.get(f"https://api.opentopodata.org/v1/srtm30m?locations={lat},{lon}", timeout=5).json()
        return res['results'][0]['elevation']
    except: return 0.0

def get_elevation_profile(lat1, lon1, lat2, lon2, num_points=120):
    lats, lons = np.linspace(lat1, lat2, num_points), np.linspace(lon1, lon2, num_points)
    coords = list(zip(lats, lons))
    distances = [0.0]
    for i in range(1, len(coords)): distances.append(distances[-1] + geodesic(coords[i-1], coords[i]).meters)
    loc_str = "|".join([f"{lat},{lon}" for lat, lon in coords])
    try:
        r = requests.get(f"https://api.opentopodata.org/v1/srtm30m?locations={loc_str}", timeout=10)
        if r.status_code == 200:
            elevs = [res['elevation'] for res in r.json().get('results', [])]
            return pd.DataFrame({"Distance (m)": distances, "Elevation (m)": elevs, "Lat": lats, "Lon": lons})
    except: return None

def calculate_itu_capacity(lat, lon, d_km, f_ghz, tx_p, g_a, g_b, t_c, rh, bw_mhz, nf, max_qam):
    if d_km <= 0: return pd.DataFrame(), 0, 0
    fsl = 92.4 + 20 * np.log10(d_km) + 20 * np.log10(f_ghz)
    e_s = 6.1121 * np.exp((17.502 * t_c) / (240.97 + t_c))
    rho = 216.7 * ((e_s * (rh / 100.0)) / (t_c + 273.15))
    try:
        gas_loss = float(itur.models.itu676.gaseous_attenuation_terrestrial_path(
            r=d_km*u.km, f=f_ghz*u.GHz, el=0.0*u.deg, rho=rho*(u.g/u.m**3), P=1013.25*u.hPa, T=t_c*u.deg_C, mode='approx').value)
    except: gas_loss = 0.0
    
    rsl_clear = tx_p + g_a + g_b - fsl - gas_loss
    noise_floor = -174 + 10 * np.log10(bw_mhz * 1e6) + nf
    qam_impl_gap = 4.53 
    max_spec_eff = np.log2(max_qam)
    
    results = []
    for avail in [99.0, 99.9, 99.99, 99.999]:
        p = round(100.0 - avail, 3) 
        try: rain = float(itur.models.itu530.rain_attenuation(lat=lat, lon=lon, d=d_km*u.km, f=f_ghz*u.GHz, el=0.0*u.deg, p=p).value)
        except: rain = 0.0
        rsl = rsl_clear - rain
        snr = rsl - noise_floor
        eff_snr = snr - qam_impl_gap
        cap_mbps = bw_mhz * min(np.log2(1 + 10**(eff_snr/10)), max_spec_eff) if eff_snr > 0 else 0
        mse = -eff_snr if eff_snr > 0 else 0 
        
        results.append({"Avail.": f"{avail}%", "Rain(dB)": f"{rain:.1f}", "RSL(dBm)": f"{rsl:.1f}", "SNR(dB)": f"{snr:.1f}", "MSE(dB)": f"{mse:.1f}", "Cap(Mbps)": f"{cap_mbps:.0f}"})
    return pd.DataFrame(results), fsl + gas_loss, rsl_clear

def generate_pdf_report(params, profile_img_path, df_att):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", "B", 16); pdf.cell(0, 10, "RF Link Analysis Report", align="C", ln=True)
    pdf.line(10, 22, 200, 22); pdf.ln(5)
    
    pdf.set_font("helvetica", "B", 11); pdf.cell(0, 8, "1. Link Overview & Site Details", ln=True)
    pdf.set_font("helvetica", "", 9)
    pdf.cell(0, 6, f"Distance: {params['dist_km']:.3f} km | Status: {params['pdf_status']}", ln=True)
    pdf.cell(0, 6, f"Site A: {params['lat_a']:.5f}, {params['lon_a']:.5f} | H: {params['h_a']}m | Dish: {params['d_a']}m ({params['g_a']:.1f} dBi, BW: {params['bw_a']:.2f} deg)", ln=True)
    pdf.cell(0, 6, f"Site B: {params['lat_b']:.5f}, {params['lon_b']:.5f} | H: {params['h_b']}m | Dish: {params['d_b']}m ({params['g_b']:.1f} dBi, BW: {params['bw_b']:.2f} deg)", ln=True)
    
    pdf.ln(3); pdf.set_font("helvetica", "B", 11); pdf.cell(0, 8, "2. Reflection Analysis", ln=True)
    pdf.set_font("helvetica", "", 9)
    pdf.cell(0, 6, f"Total Reflection Suppression: {params['ref_disc']:.1f} dB", ln=True)
    pdf.cell(0, 6, f"Worst Case Destructive RSL: {params['rsl_dest']:.1f} dBm | Best Case Constructive: {params['rsl_const']:.1f} dBm", ln=True)
    
    pdf.ln(3); pdf.set_font("helvetica", "B", 11); pdf.cell(0, 8, "3. Terrain Profile", ln=True)
    pdf.image(profile_img_path, x=10, w=185); pdf.ln(5)
    
    pdf.set_font("helvetica", "B", 11); pdf.cell(0, 8, "4. Performance Table", ln=True)
    pdf.set_font("helvetica", "B", 8)
    for col in list(df_att.columns): pdf.cell(31, 8, col, border=1, align="C")
    pdf.ln()
    pdf.set_font("helvetica", "", 8)
    for row in df_att.itertuples(index=False):
        for item in row: pdf.cell(31, 8, str(item), border=1, align="C")
        pdf.ln()
    return bytes(pdf.output())

# --- 4. SIDEBAR ---
st.sidebar.title("📡 RF Link Planner")
click_target = st.sidebar.radio("Map Click:", ["None", "Site A", "Site B"])

if st.sidebar.button("📍 Use My GPS"):
    st.session_state.gps_requested = True
    st.rerun()

loc = get_geolocation()
if st.session_state.gps_requested and loc:
    st.session_state.lat_a, st.session_state.lon_a = float(loc['coords']['latitude']), float(loc['coords']['longitude'])
    st.session_state.h_a = round(float(max(5.0, (loc['coords']['altitude'] or 0) - get_ground_elevation(st.session_state.lat_a, st.session_state.lon_a))), 1)
    st.session_state.gps_requested = False; st.rerun()

st.sidebar.divider()
st.session_state.lat_a = st.sidebar.number_input("Lat A", value=float(st.session_state.lat_a), format="%.6f")
st.session_state.lon_a = st.sidebar.number_input("Lon A", value=float(st.session_state.lon_a), format="%.6f")
st.session_state.h_a = st.sidebar.number_input("Height A (m)", value=float(st.session_state.h_a))
st.session_state.lat_b = st.sidebar.number_input("Lat B", value=float(st.session_state.lat_b), format="%.6f")
st.session_state.lon_b = st.sidebar.number_input("Lon B", value=float(st.session_state.lon_b), format="%.6f")
st.session_state.h_b = st.sidebar.number_input("Height B (m)", value=float(st.session_state.h_b))

st.sidebar.divider()
freq = st.sidebar.number_input("Freq (GHz)", value=15.0)
tx_p = st.sidebar.number_input("TX Power (dBm)", value=20.0)
st.session_state.ch_bw = st.sidebar.number_input("BW (MHz)", value=float(st.session_state.ch_bw))
st.session_state.nf = st.sidebar.number_input("NF (dB)", value=float(st.session_state.nf))

qam_opts = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
st.session_state.max_qam = st.sidebar.selectbox("Max QAM", options=qam_opts, index=qam_opts.index(int(st.session_state.max_qam)), format_func=lambda x: "BPSK" if x==2 else f"{x}-QAM")

wl = 0.3 / freq
d_a = st.sidebar.number_input("Dish A (m)", value=0.6)
g_a = 10 * np.log10(0.55 * (np.pi * d_a / wl)**2); bw_a = 70 * (wl / d_a)
d_b = st.sidebar.number_input("Dish B (m)", value=0.6)
g_b = 10 * np.log10(0.55 * (np.pi * d_b / wl)**2); bw_b = 70 * (wl / d_b)

# --- 5. MAIN UI ---
st.title("RF Path Profiler & Link Capacity")
c1, c2 = st.columns([1, 1])

with c1:
    m = folium.Map(location=[float(st.session_state.lat_a), float(st.session_state.lon_a)], zoom_start=12)
    folium.Marker([st.session_state.lat_a, st.session_state.lon_a], icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([st.session_state.lat_b, st.session_state.lon_b], icon=folium.Icon(color="red")).add_to(m)
    folium.PolyLine([(st.session_state.lat_a, st.session_state.lon_a), (st.session_state.lat_b, st.session_state.lon_b)], color="blue").add_to(m)
    m_data = st_folium(m, height=450, width=650)
    if m_data and m_data.get("last_clicked"):
        if click_target == "Site A": st.session_state.lat_a, st.session_state.lon_a = m_data["last_clicked"]["lat"], m_data["last_clicked"]["lng"]; st.rerun()
        elif click_target == "Site B": st.session_state.lat_b, st.session_state.lon_b = m_data["last_clicked"]["lat"], m_data["last_clicked"]["lng"]; st.rerun()

with c2:
    if st.button("🚀 Analyze Link", type="primary", use_container_width=True):
        df = get_elevation_profile(st.session_state.lat_a, st.session_state.lon_a, st.session_state.lat_b, st.session_state.lon_b)
        if df is not None:
            dist = df.iloc[-1]["Distance (m)"]
            abs_a, abs_b = df.iloc[0]["Elevation (m)"]+st.session_state.h_a, df.iloc[-1]["Elevation (m)"]+st.session_state.h_b
            df["LOS"] = np.linspace(abs_a, abs_b, len(df))
            df["F1"] = 17.32 * np.sqrt(((df["Distance (m)"]/1000)*((dist-df["Distance (m)"])/1000))/(freq*(dist/1000)))
            
            is_obstructed = any(df["Elevation (m)"] > df["LOS"])
            ui_status = "❌ OBSTRUCTED" if is_obstructed else "✅ CLEAR"
            pdf_status = "OBSTRUCTED" if is_obstructed else "CLEAR"
            
            # Reflection Analysis
            x, y = df["Distance (m)"].values, df["Elevation (m)"].values
            dy, dx = np.gradient(y), np.gradient(x); dx[dx==0]=1e-6; slope_ang = np.arctan2(dy, dx)
            valid = (x > 0) & (x < dist)
            ang_a_ray = np.arctan2(abs_a - y, x); ang_b_ray = np.arctan2(abs_b - y, dist - x)
            diff = np.abs(ang_a_ray - ang_b_ray + 2 * slope_ang); diff[~valid] = np.inf
            idx = np.argmin(diff); ref_x, ref_y = x[idx], y[idx]
            
            off_a = abs(np.degrees(np.arctan2(abs_b-abs_a, dist)) - np.degrees(np.arctan2(ref_y-abs_a, ref_x)))
            off_b = abs(np.degrees(np.arctan2(abs_a-abs_b, dist)) - np.degrees(np.arctan2(ref_y-abs_b, dist-ref_x)))
            disc = min(12*(off_a/bw_a)**2, 25) + min(12*(off_b/bw_b)**2, 25)
            
            st.subheader(f"Path: {ui_status} | Distance: {dist/1000:.3f} km")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["Distance (m)"], y=df["Elevation (m)"], fill='tozeroy', name='Terrain', line=dict(color='SaddleBrown')))
            fig.add_trace(go.Scatter(x=df["Distance (m)"], y=df["LOS"], name='LOS', line=dict(color='red', dash='dash')))
            fig.add_trace(go.Scatter(x=df["Distance (m)"], y=df["LOS"]-df["F1"], fill='tonexty', name='1st Fresnel'))
            fig.add_trace(go.Scatter(x=[0, ref_x, dist], y=[abs_a, ref_y, abs_b], name='Reflection', line=dict(color='orange', dash='dot')))
            st.plotly_chart(fig, use_container_width=True)
            
            df_att, cl, rsl_c = calculate_itu_capacity(st.session_state.lat_a, st.session_state.lon_a, dist/1000, freq, tx_p, g_a, g_b, st.session_state.env_temp, st.session_state.env_rh, st.session_state.ch_bw, st.session_state.nf, st.session_state.max_qam)
            st.dataframe(df_att, hide_index=True)
            
            r_ratio = 10**(-disc/20)
            rsl_dest = rsl_c + 20*np.log10(max(1e-4, 1-r_ratio))
            rsl_const = rsl_c + 20*np.log10(1+r_ratio)
            
            params = {
                "dist_km": dist/1000, "pdf_status": pdf_status, "ref_disc": disc, "rsl_dest": rsl_dest, "rsl_const": rsl_const,
                "lat_a": st.session_state.lat_a, "lon_a": st.session_state.lon_a, "h_a": st.session_state.h_a, "d_a": d_a, "g_a": g_a, "bw_a": bw_a,
                "lat_b": st.session_state.lat_b, "lon_b": st.session_state.lon_b, "h_b": st.session_state.h_b, "d_b": d_b, "g_b": g_b, "bw_b": bw_b
            }
            with tempfile.TemporaryDirectory() as tmp:
                img = os.path.join(tmp, "p.png"); fig.write_image(img, width=800, height=400)
                st.session_state.pdf_data = generate_pdf_report(params, img, df_att)

    if st.session_state.pdf_data:
        st.download_button("📄 Download PDF", st.session_state.pdf_data, "RF_Report.pdf", type="primary")
