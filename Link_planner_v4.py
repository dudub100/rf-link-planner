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
from streamlit_js_eval import streamlit_js_eval, get_geolocation

# --- 1. CONFIG & SESSION STATE ---
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

defaults = {
    "lat_a": 40.7128, "lon_a": -74.0060, "h_a": 15.0,
    "lat_b": 40.7306, "lon_b": -73.9866, "h_b": 20.0,
    "env_temp": 15.0, "env_rh": 50.0, 
    "ch_bw": 56.0, "nf": 5.0, "max_qam": 4096,
    "gps_requested": False, "pdf_data": None, "peer_loaded": False,
    "results_ready": False
}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

# --- 2. CORE ENGINEERING FUNCTIONS ---

# DUAL-API WEATHER FETCHER
def fetch_weather(lat, lon):
    # Attempt 1: Open-Meteo
    try:
        url_om = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m"
        r_om = requests.get(url_om, timeout=5)
        if r_om.status_code == 200:
            data = r_om.json()
            if 'current' in data:
                return {"temp": float(data['current']['temperature_2m']), "rh": float(data['current']['relative_humidity_2m'])}
    except: pass

    # Attempt 2: Fallback to Met.no
    try:
        url_met = f"https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={lat}&lon={lon}"
        headers = {'User-Agent': 'RFLinkPlanner/1.0 (github.com/dudub100)'} 
        r_met = requests.get(url_met, headers=headers, timeout=5)
        if r_met.status_code == 200:
            data = r_met.json()
            current = data['properties']['timeseries'][0]['data']['instant']['details']
            if 'air_temperature' in current and 'relative_humidity' in current:
                return {"temp": float(current['air_temperature']), "rh": float(current['relative_humidity'])}
    except: pass
    
    # If both fail
    return None

def get_elevation_profile(lat1, lon1, lat2, lon2, num_points=100):
    lats, lons = np.linspace(lat1, lat2, num_points), np.linspace(lon1, lon2, num_points)
    coords = list(zip(lats, lons))
    distances = [0.0]
    for i in range(1, len(coords)):
        distances.append(distances[-1] + geodesic(coords[i-1], coords[i]).meters)
    
    elevs = []
    batch_size = 30 
    for i in range(0, len(coords), batch_size):
        batch = coords[i:i+batch_size]
        loc_str = "|".join([f"{lat},{lon}" for lat, lon in batch])
        try:
            r = requests.get(f"https://api.opentopodata.org/v1/mapzen?locations={loc_str}", timeout=10)
            if r.status_code != 200:
                r = requests.get(f"https://api.opentopodata.org/v1/srtm30m?locations={loc_str}", timeout=10)
            if r.status_code == 200:
                elevs.extend([res['elevation'] for res in r.json().get('results', [])])
            else: return None
        except: return None
    return pd.DataFrame({"Distance (m)": distances, "Elevation (m)": elevs, "Lat": lats, "Lon": lons}) if len(elevs) == len(coords) else None

def calculate_diffraction_loss(df, freq_ghz):
    wl = 0.3 / freq_ghz
    dist_total = df.iloc[-1]["Distance (m)"]
    df["h_diff"] = df["Elevation (m)"] - df["LOS"]
    df["v_param"] = df["h_diff"] * np.sqrt((2 / wl) * (1 / (df["Distance (m)"] + 1e-6) + 1 / (dist_total - df["Distance (m)"] + 1e-6)))
    max_v = df["v_param"].max()
    return max(0.0, 6.9 + 20 * np.log10(np.sqrt((max_v - 0.1)**2 + 1) + max_v - 0.1)) if max_v > -0.78 else 0.0

def calculate_itu_capacity(lat, lon, d_km, f_ghz, tx_p, g_a, g_b, t_c, rh, bw_mhz, nf, max_qam, diff_loss):
    fsl = 92.4 + 20 * np.log10(d_km) + 20 * np.log10(f_ghz)
    e_s = 6.1121 * np.exp((17.502 * t_c) / (240.97 + t_c))
    rho = 216.7 * ((e_s * (rh / 100.0)) / (t_c + 273.15))
    try:
        gas_loss = float(itur.models.itu676.gaseous_attenuation_terrestrial_path(r=d_km*u.km, f=f_ghz*u.GHz, el=0.0*u.deg, rho=rho*(u.g/u.m**3), P=1013.25*u.hPa, T=t_c*u.deg_C, mode='approx').value)
    except: gas_loss = 0.0
    rsl_clear = tx_p + g_a + g_b - (fsl + gas_loss + diff_loss)
    noise_floor = -174 + 10 * np.log10(bw_mhz * 1e6) + nf
    eff_snr = (rsl_clear - noise_floor) - 4.53
    max_spec_eff = np.log2(max_qam)
    results = []
    for avail in [99.0, 99.9, 99.99, 99.999]:
        p = round(100.0 - avail, 3) 
        try: rain = float(itur.models.itu530.rain_attenuation(lat=lat, lon=lon, d=d_km*u.km, f=f_ghz*u.GHz, el=0.0*u.deg, p=p).value)
        except: rain = 0.0
        rsl = rsl_clear - rain
        snr_now = rsl - noise_floor
        eff_snr_now = snr_now - 4.53
        cap = bw_mhz * min(np.log2(1 + 10**(max(-5, eff_snr_now)/10)), max_spec_eff) if eff_snr_now > -10 else 0
        results.append({"Avail.": f"{avail}%", "Rain(dB)": f"{rain:.1f}", "RSL(dBm)": f"{rsl:.1f}", "SNR(dB)": f"{snr_now:.1f}", "MSE(dB)": f"{-eff_snr_now:.1f}", "Cap(Mbps)": f"{cap:.0f}"})
    return pd.DataFrame(results), fsl, gas_loss, rsl_clear

def generate_pdf_report(p, profile_img_path, df_att):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("helvetica", "B", 16); pdf.cell(0, 10, "RF Link Engineering Analysis", align="C", ln=True)
        pdf.line(10, 22, 200, 22); pdf.ln(5)
        pdf.set_font("helvetica", "B", 10); pdf.cell(0, 8, "1. Link Overview", ln=True)
        pdf.set_font("helvetica", "", 9)
        pdf.cell(0, 6, f"Distance: {p['dist_km']:.3f} km | Status: {p['pdf_status']} | Frequency: {p['freq']} GHz", ln=True)
        pdf.cell(0, 6, f"Site A: {p['lat_a']:.5f}, {p['lon_a']:.5f} | H: {p['h_a']}m | Site B: {p['lat_b']:.5f}, {p['lon_b']:.5f} | H: {p['h_b']}m", ln=True)
        pdf.ln(2); pdf.set_font("helvetica", "B", 10); pdf.cell(0, 8, "2. Budget & Reflection Data", ln=True)
        pdf.set_font("helvetica", "", 9)
        pdf.cell(90, 6, f"Clear Sky RSL: {p['rsl_clear']:.1f} dBm", ln=0); pdf.cell(0, 6, f"Diffraction Loss: {p['diff']:.1f} dB", ln=1)
        pdf.cell(90, 6, f"Multipath Disc: {p['ref_disc']:.1f} dB", ln=0); pdf.cell(0, 6, f"Worst Case RSL: {p['rsl_dest']:.1f} dBm", ln=1)
        if profile_img_path: pdf.image(profile_img_path, x=10, w=185); pdf.ln(5)
        pdf.set_font("helvetica", "B", 8)
        for col in df_att.columns: pdf.cell(31, 8, col, border=1, align="C")
        pdf.ln(); pdf.set_font("helvetica", "", 8)
        for row in df_att.itertuples(index=False):
            for item in row: pdf.cell(31, 8, str(item), border=1, align="C")
            pdf.ln()
        return bytes(pdf.output())
    except: return None

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.title("🛠️ RF Operations")
click_target = st.sidebar.radio("Map Click Selector:", ["None", "Site A", "Site B"])
c1, c2 = st.sidebar.columns(2)
if c1.button("📍 My GPS"): st.session_state.gps_requested = True; st.rerun()

if c2.button("☁️ Weather"):
    w = fetch_weather(st.session_state.lat_a, st.session_state.lon_a)
    if w:
        st.session_state.env_temp = w['temp']
        st.session_state.env_rh = w['rh']
        st.rerun()
    else:
        st.sidebar.error("⚠️ Both weather APIs failed. Manual entry required.")

loc = get_geolocation()
if st.session_state.gps_requested and loc:
    st.session_state.lat_a, st.session_state.lon_a = float(loc['coords']['latitude']), float(loc['coords']['longitude'])
    st.session_state.gps_requested = False; st.rerun()

curr_url = streamlit_js_eval(js_expressions="window.parent.location.href", want_output=True, key="get_url")
if curr_url:
    base = curr_url.split('?')[0]
    s_url = f"{base}?peer_lat={st.session_state.lat_a}&peer_lon={st.session_state.lon_a}&peer_h={st.session_state.h_a}"
    wa_link = f"https://wa.me/?text={urllib.parse.quote('Sync: ' + s_url)}"
    st.sidebar.markdown(f'''<a href="{wa_link}" target="_blank" style="text-decoration:none;"><div style="background-color:#25D366;color:white;padding:10px;border-radius:8px;text-align:center;font-weight:bold;">📲 Share with Peer</div></a>''', unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.subheader("Sites & Antenna Height (AGL)")
st.session_state.lat_a = st.sidebar.number_input("Lat A", value=float(st.session_state.lat_a), format="%.6f")
st.session_state.lon_a = st.sidebar.number_input("Lon A", value=float(st.session_state.lon_a), format="%.6f")
st.session_state.h_a = st.sidebar.number_input("Height A (m)", value=float(st.session_state.h_a))
st.session_state.lat_b = st.sidebar.number_input("Lat B", value=float(st.session_state.lat_b), format="%.6f")
st.session_state.lon_b = st.sidebar.number_input("Lon B", value=float(st.session_state.lon_b), format="%.6f")
st.session_state.h_b = st.sidebar.number_input("Height B (m)", value=float(st.session_state.h_b))

st.sidebar.divider()
freq = st.sidebar.number_input("Freq (GHz)", value=15.0); tx_p = st.sidebar.number_input("TX Power (dBm)", value=20.0)
st.session_state.ch_bw = st.sidebar.number_input("BW (MHz)", value=float(st.session_state.ch_bw))
q_ops = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
st.session_state.max_qam = st.sidebar.selectbox("Max QAM", q_ops, index=q_ops.index(int(st.session_state.max_qam)), format_func=lambda x: "BPSK" if x==2 else f"{x}-QAM")
st.session_state.nf = st.sidebar.number_input("Noise Figure (dB)", value=float(st.session_state.nf))

wl = 0.3 / freq
d_a = st.sidebar.number_input("Dish A (m)", value=0.6); g_a = 10 * np.log10(0.55 * (np.pi * d_a / wl)**2); bw_a = 70 * (wl / d_a)
d_b = st.sidebar.number_input("Dish B (m)", value=0.6); g_b = 10 * np.log10(0.55 * (np.pi * d_b / wl)**2); bw_b = 70 * (wl / d_b)

st.sidebar.divider()
st.session_state.env_temp = st.sidebar.number_input("Temp (°C)", value=float(st.session_state.env_temp))
st.session_state.env_rh = st.sidebar.number_input("Humidity (%)", value=float(st.session_state.env_rh))

# --- 4. MAIN UI ---
st.title("RF Path Profiler & Multipath Viewer")
col1, col2 = st.columns([1, 1])

with col1:
    m = folium.Map(location=[float(st.session_state.lat_a), float(st.session_state.lon_a)], zoom_start=12)
    folium.Marker([st.session_state.lat_a, st.session_state.lon_a], icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([st.session_state.lat_b, st.session_state.lon_b], icon=folium.Icon(color="red")).add_to(m)
    folium.PolyLine([(st.session_state.lat_a, st.session_state.lon_a), (st.session_state.lat_b, st.session_state.lon_b)], color="blue").add_to(m)
    m_data = st_folium(m, height=450, width=650, key="link_map")
    if m_data and m_data.get("last_clicked"):
        lat, lon = m_data["last_clicked"]["lat"], m_data["last_clicked"]["lng"]
        if click_target == "Site A": st.session_state.lat_a, st.session_state.lon_a = lat, lon; st.rerun()
        elif click_target == "Site B": st.session_state.lat_b, st.session_state.lon_b = lat, lon; st.rerun()

with col2:
    if st.button("🚀 Analyze Link", type="primary", use_container_width=True):
        st.session_state.results_ready = True

    if st.session_state.results_ready:
        with st.status("Analyzing Link Geometry...") as status:
            df = get_elevation_profile(st.session_state.lat_a, st.session_state.lon_a, st.session_state.lat_b, st.session_state.lon_b)
            if df is not None:
                dist = df.iloc[-1]["Distance (m)"]
                a_a, a_b = df.iloc[0]["Elevation (m)"]+st.session_state.h_a, df.iloc[-1]["Elevation (m)"]+st.session_state.h_b
                df["LOS"] = np.linspace(a_a, a_b, len(df))
                
                # FRESNEL ZONE
                df["F1"] = 17.32 * np.sqrt(((df["Distance (m)"]/1000)*((dist-df["Distance (m)"])/1000))/(freq*(dist/1000)))
                
                diff_loss = calculate_diffraction_loss(df, freq)
                is_obs = any(df["Elevation (m)"] > df["LOS"])
                
                # Reflection Search
                x, y = df["Distance (m)"].values, df["Elevation (m)"].values
                dy, dx = np.gradient(y), np.gradient(x); dx[dx==0]=1e-6; slope_ang = np.arctan2(dy, dx)
                ang_a_ray = np.arctan2(a_a - y, x); ang_b_ray = np.arctan2(a_b - y, dist - x)
                diff_trace = np.abs(ang_a_ray - ang_b_ray + 2 * slope_ang); diff_trace[~( (x>0) & (x<dist) )] = np.inf
                idx = np.argmin(diff_trace); ref_x, ref_y = x[idx], y[idx]
                off_a = abs(np.degrees(np.arctan2(a_b-a_a, dist)) - np.degrees(np.arctan2(ref_y-a_a, ref_x)))
                off_b = abs(np.degrees(np.arctan2(a_a-a_b, dist)) - np.degrees(np.arctan2(ref_y-a_b, dist-ref_x)))
                disc = min(12*(off_a/bw_a)**2, 25) + min(12*(off_b/bw_b)**2, 25)
                
                df_att, fsl, gas, rsl_clr = calculate_itu_capacity(st.session_state.lat_a, st.session_state.lon_a, dist/1000, freq, tx_p, g_a, g_b, st.session_state.env_temp, st.session_state.env_rh, st.session_state.ch_bw, st.session_state.nf, st.session_state.max_qam, diff_loss)
                r_ratio = 10**(-disc/20); rsl_dest = rsl_clr + 20*np.log10(max(1e-4, 1-r_ratio))
                
                status.update(label="Analysis Complete!", state="complete", expanded=False)
                st.subheader(f"{'❌ OBSTRUCTED' if is_obs else '✅ CLEAR'} | {dist/1000:.3f} km")
                st.markdown(f"**Clear Sky RSL:** `{rsl_clr:.1f} dBm` | **Worst-Case Multipath:** `{rsl_dest:.1f} dBm` | **Diffraction:** `{diff_loss:.1f} dB`")
                st.dataframe(df_att, hide_index=True)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', name='Terrain', line=dict(color='SaddleBrown')))
                fig.add_trace(go.Scatter(x=x, y=df["LOS"], name='LOS', line=dict(color='red', dash='dash')))
                fig.add_trace(go.Scatter(x=x, y=df["LOS"] - df["F1"], fill='tonexty', name='1st Fresnel (F1)', line=dict(color='rgba(0,0,255,0.2)')))
                fig.add_trace(go.Scatter(x=[0, ref_x, dist], y=[a_a, ref_y, a_b], name='Reflection', line=dict(color='orange', dash='dot')))
                fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=350)
                st.plotly_chart(fig, use_container_width=True)

                try:
                    with tempfile.TemporaryDirectory() as tmp:
                        img_p = os.path.join(tmp, "p.png"); fig.write_image(img_p)
                        st.session_state.pdf_data = generate_pdf_report({"dist_km": dist/1000, "pdf_status": "OBSTRUCTED" if is_obs else "CLEAR", "freq": freq, "rsl_clear": rsl_clr, "diff": diff_loss, "ref_disc": disc, "rsl_dest": rsl_dest, "lat_a": st.session_state.lat_a, "lon_a": st.session_state.lon_a, "h_a": st.session_state.h_a, "g_a": g_a, "lat_b": st.session_state.lat_b, "lon_b": st.session_state.lon_b, "h_b": st.session_state.h_b, "g_b": g_b, "fsl": fsl, "gas": gas}, img_p, df_att)
                except: st.session_state.pdf_data = generate_pdf_report({"dist_km": dist/1000, "pdf_status": "OBSTRUCTED" if is_obs else "CLEAR", "freq": freq, "rsl_clear": rsl_clr, "diff": diff_loss, "ref_disc": disc, "rsl_dest": rsl_dest, "lat_a": st.session_state.lat_a, "lon_a": st.session_state.lon_a, "h_a": st.session_state.h_a, "g_a": g_a, "lat_b": st.session_state.lat_b, "lon_b": st.session_state.lon_b, "h_b": st.session_state.h_b, "g_b": g_b, "fsl": fsl, "gas": gas}, None, df_att)

    if st.session_state.pdf_data:
        st.download_button("📄 Download PDF Report", st.session_state.pdf_data, "RF_Analysis.pdf", type="primary", use_container_width=True)
