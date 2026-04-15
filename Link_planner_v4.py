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
    "ch_bw": 56.0, "nf": 5.0, # New Parameters
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
    st.toast("✅ Peer Location Synchronized!", icon="📡")

# --- 3. HELPER FUNCTIONS ---
def fetch_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m"
        r = requests.get(url, timeout=5).json()
        return {"temp": float(r['current']['temperature_2m']), "rh": float(r['current']['relative_humidity_2m'])}
    except: return None

def get_ground_elevation(lat, lon):
    try:
        res = requests.get(f"https://api.opentopodata.org/v1/srtm30m?locations={lat},{lon}", timeout=5).json()
        return res['results'][0]['elevation']
    except: return 0.0

def get_elevation_profile(lat1, lon1, lat2, lon2, num_points=100):
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

def calculate_itu_capacity(lat, lon, d_km, f_ghz, tx_power, gain_a, gain_b, t_c, rh, bw_mhz, nf):
    if d_km <= 0: return pd.DataFrame()
    fsl = 92.4 + 20 * np.log10(d_km) + 20 * np.log10(f_ghz)
    
    e_s = 6.1121 * np.exp((17.502 * t_c) / (240.97 + t_c))
    rho = 216.7 * ((e_s * (rh / 100.0)) / (t_c + 273.15))
    try:
        gas_loss = float(itur.models.itu676.gaseous_attenuation_terrestrial_path(
            r=d_km*u.km, f=f_ghz*u.GHz, el=0.0*u.deg, rho=rho*(u.g/u.m**3), P=1013.25*u.hPa, T=t_c*u.deg_C, mode='approx').value)
    except: gas_loss = 0.0
    
    clear_loss = fsl + gas_loss
    rsl_clear = tx_power + gain_a + gain_b - clear_loss
    noise_floor = -174 + 10 * np.log10(bw_mhz * 1e6) + nf
    
    # QAM Gap (~1.53 dB) + Implementation Loss (3 dB)
    qam_impl_gap = 4.53 
    
    results = []
    for avail in [99.0, 99.9, 99.99, 99.999]:
        p = round(100.0 - avail, 3) 
        try: rain = float(itur.models.itu530.rain_attenuation(lat=lat, lon=lon, d=d_km*u.km, f=f_ghz*u.GHz, el=0.0*u.deg, p=p).value)
        except: rain = 0.0
        
        rsl = rsl_clear - rain
        snr = rsl - noise_floor
        eff_snr = snr - qam_impl_gap
        
        # Shannon Capacity (Mbps)
        cap_mbps = (bw_mhz * 1e6 * np.log2(1 + 10**(eff_snr/10))) / 1e6 if eff_snr > 0 else 0
        mse = -eff_snr if eff_snr > 0 else 0 # Idealized MSE estimation
        
        results.append({
            "Avail.": f"{avail}%", 
            "Rain (dB)": f"{rain:.1f}", 
            "RSL (dBm)": f"{rsl:.1f}", 
            "SNR (dB)": f"{snr:.1f}",
            "MSE (dB)": f"{mse:.1f}",
            "Est. Cap (Mbps)": f"{cap_mbps:.1f}"
        })
    return pd.DataFrame(results), clear_loss, rsl_clear

def generate_pdf_report(params, profile_img_path, df_att):
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, "RF Link Path Profile & Budget Report", align="C", ln=True)
    pdf.line(10, 22, 200, 22); pdf.ln(5)
    
    # 1. Sites and Antennas
    pdf.set_font("helvetica", "B", 12); pdf.cell(0, 8, "1. Site & Antenna Parameters", ln=True)
    pdf.set_font("helvetica", "", 9)
    pdf.cell(0, 6, f"Frequency: {params['freq']} GHz | Channel BW: {params['bw']} MHz | Noise Figure: {params['nf']} dB", ln=True)
    pdf.cell(0, 6, f"Site A: {params['lat_a']:.5f}, {params['lon_a']:.5f} | H (AGL): {params['h_a']}m | Dish: {params['d_a']}m ({params['g_a']:.1f} dBi, BW: {params['bw_a']:.2f} deg)", ln=True)
    pdf.cell(0, 6, f"Site B: {params['lat_b']:.5f}, {params['lon_b']:.5f} | H (AGL): {params['h_b']}m | Dish: {params['d_b']}m ({params['g_b']:.1f} dBi, BW: {params['bw_b']:.2f} deg)", ln=True)
    pdf.cell(0, 6, f"Clear Sky Pathloss (FSL + Atm): {params['clear_loss']:.2f} dB | Clear Sky RSL: {params['rsl_clear']:.2f} dBm", ln=True)
    pdf.ln(3)

    # 2. Reflection
    pdf.set_font("helvetica", "B", 12); pdf.cell(0, 8, "2. Multipath Reflection Analysis", ln=True)
    pdf.set_font("helvetica", "", 9)
    if params['ref_disc'] < 10: pdf.set_text_color(220, 53, 69)
    elif params['ref_disc'] < 20: pdf.set_text_color(255, 153, 0)
    else: pdf.set_text_color(40, 167, 69)
    
    pdf.cell(0, 6, f"Total Antenna Suppression of Reflected Path: {params['ref_disc']:.1f} dB", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 6, f"Reflected Path Off-Boresight Angles: Site A = {params['ang_a']:.2f} deg | Site B = {params['ang_b']:.2f} deg", ln=True)
    pdf.cell(0, 6, f"Worst Case (Destructive) RSL: {params['rsl_dest']:.2f} dBm | Best Case (Constructive) RSL: {params['rsl_const']:.2f} dBm", ln=True)
    pdf.ln(3)

    # 3. Profile
    pdf.set_font("helvetica", "B", 12); pdf.cell(0, 8, "3. Terrain Profile", ln=True)
    pdf.image(profile_img_path, x=10, w=190); pdf.ln(5)

    # 4. Table
    pdf.set_font("helvetica", "B", 12); pdf.cell(0, 8, "4. ITU-R Link Budget & Capacity (Shannon w/ QAM Penalty)", ln=True)
    pdf.set_font("helvetica", "B", 8)
    col_widths = [20, 20, 25, 25, 25, 40]
    for col, w in zip(df_att.columns, col_widths): pdf.cell(w, 8, col, border=1, align="C")
    pdf.ln()
    pdf.set_font("helvetica", "", 8)
    for row in df_att.itertuples(index=False):
        for item, w in zip(row, col_widths): pdf.cell(w, 8, str(item), border=1, align="C")
        pdf.ln()
    return bytes(pdf.output())

# --- 4. SIDEBAR ---
st.sidebar.title("📡 Field Tools")
click_target = st.sidebar.radio("Map Click Updates:", ["None", "Site A", "Site B"])

col_gps, col_wea = st.sidebar.columns(2)
if col_gps.button("📍 GPS"): st.session_state.gps_requested = True; st.rerun()
if col_wea.button("☁️ Weather"):
    if w := fetch_weather(st.session_state.lat_a, st.session_state.lon_a):
        st.session_state.env_temp, st.session_state.env_rh = w['temp'], w['rh']
        st.rerun()

loc = get_geolocation()
if st.session_state.gps_requested and loc:
    st.session_state.lat_a, st.session_state.lon_a = float(loc['coords']['latitude']), float(loc['coords']['longitude'])
    st.session_state.h_a = round(float(max(5.0, (loc['coords']['altitude'] or 0) - get_ground_elevation(st.session_state.lat_a, st.session_state.lon_a))), 1)
    st.session_state.gps_requested = False; st.rerun()

st.sidebar.divider()
st.sidebar.subheader("Site Coordinates & Height (AGL)")
st.session_state.lat_a = st.sidebar.number_input("Lat A", value=float(st.session_state.lat_a), format="%.6f")
st.session_state.lon_a = st.sidebar.number_input("Lon A", value=float(st.session_state.lon_a), format="%.6f")
st.session_state.h_a = st.sidebar.number_input("Height A (m)", value=float(st.session_state.h_a))
st.session_state.lat_b = st.sidebar.number_input("Lat B", value=float(st.session_state.lat_b), format="%.6f")
st.session_state.lon_b = st.sidebar.number_input("Lon B", value=float(st.session_state.lon_b), format="%.6f")
st.session_state.h_b = st.sidebar.number_input("Height B (m)", value=float(st.session_state.h_b))

st.sidebar.divider()
st.sidebar.subheader("RF & System Parameters")
freq = st.sidebar.number_input("Freq (GHz)", value=15.0, min_value=0.1)
tx_p = st.sidebar.number_input("TX Power (dBm)", value=20.0)
st.session_state.ch_bw = st.sidebar.number_input("Channel BW (MHz)", value=float(st.session_state.ch_bw))
st.session_state.nf = st.sidebar.number_input("System Noise Figure (dB)", value=float(st.session_state.nf))

wl = 0.3 / freq
diam_a = st.sidebar.number_input("Dish A (m)", value=0.6)
gain_a = 10 * np.log10(0.55 * (np.pi * diam_a / wl)**2)
hpbw_a = 70.0 * (wl / diam_a)
st.sidebar.caption(f"Site A: {gain_a:.1f} dBi | BW: {hpbw_a:.2f}°")

diam_b = st.sidebar.number_input("Dish B (m)", value=0.6)
gain_b = 10 * np.log10(0.55 * (np.pi * diam_b / wl)**2)
hpbw_b = 70.0 * (wl / diam_b)
st.sidebar.caption(f"Site B: {gain_b:.1f} dBi | BW: {hpbw_b:.2f}°")

temp = st.sidebar.number_input("Temp (°C)", value=float(st.session_state.env_temp))
rh = st.sidebar.number_input("Humidity (%)", value=float(st.session_state.env_rh))

# --- 5. MAIN UI ---
st.title("RF Path & Link Capacity Profiler")
c1, c2 = st.columns([1, 1])

with c1:
    m = folium.Map(location=[float(st.session_state.lat_a), float(st.session_state.lon_a)], zoom_start=12)
    folium.Marker([st.session_state.lat_a, st.session_state.lon_a], icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([st.session_state.lat_b, st.session_state.lon_b], icon=folium.Icon(color="red")).add_to(m)
    folium.PolyLine([(st.session_state.lat_a, st.session_state.lon_a), (st.session_state.lat_b, st.session_state.lon_b)], color="blue").add_to(m)
    m_data = st_folium(m, height=450, width=700)
    if m_data and m_data.get("last_clicked"):
        if click_target == "Site A": st.session_state.lat_a, st.session_state.lon_a = m_data["last_clicked"]["lat"], m_data["last_clicked"]["lng"]; st.rerun()
        elif click_target == "Site B": st.session_state.lat_b, st.session_state.lon_b = m_data["last_clicked"]["lat"], m_data["last_clicked"]["lng"]; st.rerun()

with c2:
    if st.button("🚀 Analyze Link & Capacity", type="primary", use_container_width=True):
        with st.spinner("Calculating Geometry & Capacity..."):
            df = get_elevation_profile(st.session_state.lat_a, st.session_state.lon_a, st.session_state.lat_b, st.session_state.lon_b)
            if df is not None:
                dist = df.iloc[-1]["Distance (m)"]
                abs_a, abs_b = df.iloc[0]["Elevation (m)"]+st.session_state.h_a, df.iloc[-1]["Elevation (m)"]+st.session_state.h_b
                df["LOS"] = np.linspace(abs_a, abs_b, len(df))
                df["F1"] = 17.32 * np.sqrt(((df["Distance (m)"]/1000)*((dist-df["Distance (m)"])/1000))/(freq*(dist/1000)))
                
                # EXACT GEOMETRIC REFLECTION ALGORITHM
                x, y = df["Distance (m)"].values, df["Elevation (m)"].values
                dy, dx = np.gradient(y), np.gradient(x)
                dx[dx == 0] = 1e-6 
                slope_ang = np.arctan2(dy, dx)
                
                valid = (x > 0) & (x < dist)
                ang_a_ray = np.zeros_like(x); ang_b_ray = np.zeros_like(x)
                ang_a_ray[valid] = np.arctan2(abs_a - y[valid], x[valid])
                ang_b_ray[valid] = np.arctan2(abs_b - y[valid], dist - x[valid])
                
                diff = np.abs(ang_a_ray - ang_b_ray + 2 * slope_ang)
                diff[~valid] = np.inf
                idx_ref = np.argmin(diff)
                ref_x, ref_y = x[idx_ref], y[idx_ref]
                
                # Off-boresight angles (degrees)
                ang_los_a = np.degrees(np.arctan2(abs_b - abs_a, dist))
                ang_ref_a = np.degrees(np.arctan2(ref_y - abs_a, ref_x))
                off_bore_a = abs(ang_los_a - ang_ref_a)
                
                ang_los_b = np.degrees(np.arctan2(abs_a - abs_b, dist))
                ang_ref_b = np.degrees(np.arctan2(ref_y - abs_b, dist - ref_x))
                off_bore_b = abs(ang_los_b - ang_ref_b)
                
                # Antenna Discrimination
                disc_a = min(12.0 * (off_bore_a / hpbw_a)**2, 25.0)
                disc_b = min(12.0 * (off_bore_b / hpbw_b)**2, 25.0)
                tot_disc = disc_a + disc_b
                
                # Plotting
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', name='Terrain', line=dict(color='SaddleBrown')))
                fig.add_trace(go.Scatter(x=x, y=df["LOS"], name='LOS', line=dict(color='red', dash='dash')))
                fig.add_trace(go.Scatter(x=x, y=df["LOS"]-df["F1"], fill='tonexty', name='1st Fresnel', line=dict(color='rgba(0,0,255,0.1)')))
                fig.add_trace(go.Scatter(x=[0, ref_x, dist], y=[abs_a, ref_y, abs_b], name='Reflection Path', line=dict(color='orange', dash='dot')))
                fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Link Budget & Capacity
                df_att, clear_loss, rsl_clr = calculate_itu_capacity(
                    st.session_state.lat_a, st.session_state.lon_a, dist/1000, freq, tx_p, gain_a, gain_b, 
                    temp, rh, st.session_state.ch_bw, st.session_state.nf
                )
                
                # Constructive/Destructive Fade calculations (assuming rho ~ 1 for worst-case ground reflection)
                voltage_ratio = 10**(-tot_disc / 20)
                rsl_dest = rsl_clr + 20 * np.log10(max(1e-5, 1 - voltage_ratio))
                rsl_const = rsl_clr + 20 * np.log10(1 + voltage_ratio)
                
                # Display Results
                st.write(f"**Clear Sky Pathloss:** {clear_loss:.1f} dB | **Clear RSL:** {rsl_clr:.1f} dBm")
                st.dataframe(df_att, hide_index=True)
                
                if tot_disc < 10: st.error(f"🔴 Critical Multipath! Total Suppression: {tot_disc:.1f} dB (Destructive RSL drop to {rsl_dest:.1f} dBm)")
                elif tot_disc < 20: st.warning(f"🟡 Marginal Multipath. Total Suppression: {tot_disc:.1f} dB (Destructive RSL drop to {rsl_dest:.1f} dBm)")
                else: st.success(f"✅ Safe Multipath Suppression: {tot_disc:.1f} dB")
                
                # PDF Generation
                pdf_params = {
                    "freq": freq, "bw": st.session_state.ch_bw, "nf": st.session_state.nf,
                    "lat_a": st.session_state.lat_a, "lon_a": st.session_state.lon_a, "h_a": st.session_state.h_a,
                    "lat_b": st.session_state.lat_b, "lon_b": st.session_state.lon_b, "h_b": st.session_state.h_b,
                    "d_a": diam_a, "g_a": gain_a, "bw_a": hpbw_a,
                    "d_b": diam_b, "g_b": gain_b, "bw_b": hpbw_b,
                    "clear_loss": clear_loss, "rsl_clear": rsl_clr,
                    "ang_a": off_bore_a, "ang_b": off_bore_b, "ref_disc": tot_disc,
                    "rsl_dest": rsl_dest, "rsl_const": rsl_const
                }
                
                with tempfile.TemporaryDirectory() as tmp:
                    img_path = os.path.join(tmp, "p.png")
                    fig.write_image(img_path, width=900, height=400)
                    st.session_state.pdf_data = generate_pdf_report(pdf_params, img_path, df_att)

    if st.session_state.pdf_data:
        st.download_button("📄 Download Professional Report", st.session_state.pdf_data, "RF_Analysis.pdf", type="primary")
