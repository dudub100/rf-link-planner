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

st.set_page_config(layout="wide", page_title="Line of Sight Profiler")

# --- 1. SESSION STATE INITIALIZATION ---
if "site_a" not in st.session_state:
    st.session_state.site_a = {"lat": 40.7128, "lon": -74.0060, "h": 15.0} 
if "site_b" not in st.session_state:
    st.session_state.site_b = {"lat": 40.7306, "lon": -73.9866, "h": 20.0}
if "pdf_data" not in st.session_state:
    st.session_state.pdf_data = None

# --- 2. HELPER FUNCTIONS ---
def get_elevation_profile(lat1, lon1, lat2, lon2, num_points=50):
    lats = np.linspace(lat1, lat2, num_points)
    lons = np.linspace(lon1, lon2, num_points)
    coords = list(zip(lats, lons))
    distances = [0.0]
    for i in range(1, len(coords)):
        dist = geodesic(coords[i-1], coords[i]).meters
        distances.append(distances[-1] + dist)

    locations_str = "|".join([f"{lat},{lon}" for lat, lon in coords])
    url_opentopo = f"https://api.opentopodata.org/v1/srtm30m?locations={locations_str}"
    
    try:
        response1 = requests.get(url_opentopo, timeout=10)
        if response1.status_code == 200:
            results = response1.json().get('results', [])
            elevations = [res['elevation'] for res in results]
            if len(elevations) == num_points:
                return pd.DataFrame({"Distance (m)": distances, "Elevation (m)": elevations, "Lat": lats, "Lon": lons})
    except Exception as e:
        print(f"Elevation API error: {e}")
    return None

def calculate_itu_attenuation(lat, lon, d_km, f_ghz, tx_power, gain_a, gain_b, t_c, rh):
    if d_km <= 0: return pd.DataFrame()
    availabilities = [99.0, 99.5, 99.9, 99.95, 99.99, 99.995, 99.999]
    fsl = 92.4 + 20 * np.log10(d_km) + 20 * np.log10(f_ghz)
    
    e_s = 6.1121 * np.exp((17.502 * t_c) / (240.97 + t_c))
    e = e_s * (rh / 100.0)
    rho_calc = 216.7 * (e / (t_c + 273.15))
    
    d_val, f_val, rho_val = d_km * u.km, f_ghz * u.GHz, rho_calc * (u.g / u.m**3)
    P_val, T_val = 1013.25 * u.hPa, t_c * u.deg_C

    try:
        gas_loss = float(itur.models.itu676.gaseous_attenuation_terrestrial_path(r=d_val, f=f_val, el=0*u.deg, rho=rho_val, P=P_val, T=T_val, mode='approx').value)
    except: gas_loss = 0.0
        
    results = []
    rsl_clear = tx_power + gain_a + gain_b - fsl - gas_loss
    for avail in availabilities:
        p = round(100.0 - avail, 3) 
        try: rain_loss = float(itur.models.itu530.rain_attenuation(lat=lat, lon=lon, d=d_val, f=f_val, el=0*u.deg, p=p).value)
        except: rain_loss = 0.0
        results.append({"Avail.": f"{avail}%", "FSL (dB)": f"{fsl:.1f}", "Atm. (dB)": f"{gas_loss:.2f}", "Rain (dB)": f"{rain_loss:.1f}", "Clear RSL": f"{rsl_clear:.1f} dBm", "Faded RSL": f"{rsl_clear - rain_loss:.1f} dBm"})
    return pd.DataFrame(results)

def generate_pdf_report(site_a, site_b, d_km, f_ghz, map_img, prof_img, df_att, ref_data, ant_data):
    class PDF(FPDF):
        def header(self):
            self.set_font("helvetica", "B", 16)
            self.cell(0, 10, "RF Link Path Profile & Multipath Report", ln=True, align="C")
            self.line(10, 22, 200, 22); self.ln(10)
    
    pdf = PDF()
    pdf.add_page()
    
    # 1. Hardware & Link
    pdf.set_font("helvetica", "B", 12); pdf.cell(0, 8, "1. Link & Hardware Specifications", ln=True)
    pdf.set_font("helvetica", "", 10)
    pdf.cell(0, 6, f"Frequency: {f_ghz} GHz | Path Distance: {d_km:.3f} km", ln=True)
    pdf.cell(0, 6, f"Antenna A: {ant_data['dia_a']}m Diameter | Gain: {ant_data['gain_a']:.1f} dBi", ln=True)
    pdf.cell(0, 6, f"Antenna B: {ant_data['dia_b']}m Diameter | Gain: {ant_data['gain_b']:.1f} dBi", ln=True)
    pdf.ln(5)

    # 2. Multipath Analysis
    pdf.set_font("helvetica", "B", 12); pdf.cell(0, 8, "2. Multipath Reflection Analysis", ln=True)
    pdf.set_font("helvetica", "", 10)
    pdf.cell(0, 6, f"Nominal Suppression: {ref_data['total_disc']:.1f} dB", ln=True)
    pdf.cell(0, 6, f"Constructive Interference: +{ref_data['var_pos']:.2f} dB", ln=True)
    pdf.cell(0, 6, f"Destructive Interference: {ref_data['var_neg']:.2f} dB", ln=True)
    pdf.ln(3)
    
    # Tilt evaluation
    pdf.set_font("helvetica", "B", 10); pdf.cell(0, 6, "Tilt-Up Mitigation Evaluation (1.5dB Main Path Loss per side):", ln=True)
    pdf.set_font("helvetica", "", 10)
    pdf.cell(0, 6, f"New Multipath Suppression: {ref_data['tilted_disc']:.1f} dB", ln=True)
    pdf.cell(0, 6, f"Net Improvement (Suppression Gain - 3dB): {ref_data['net_improvement']:.1f} dB", ln=True)
    pdf.ln(5)

    pdf.image(prof_img, x=10, w=190); pdf.add_page()
    pdf.image(map_img, x=10, w=190); pdf.ln(10)
    
    pdf.set_font("helvetica", "B", 12); pdf.cell(0, 8, "3. ITU-R Estimated Link Budget", ln=True)
    pdf.set_font("helvetica", "B", 8)
    col_widths = [20, 25, 25, 25, 45, 45]
    for i, col in enumerate(df_att.columns): pdf.cell(col_widths[i], 8, col, border=1, align="C")
    pdf.ln()
    pdf.set_font("helvetica", "", 8)
    for row in df_att.itertuples(index=False):
        for i, item in enumerate(row): pdf.cell(col_widths[i], 8, str(item), border=1, align="C")
        pdf.ln()
    return bytes(pdf.output())

# --- 3. SIDEBAR ---
st.sidebar.title("Link Parameters")
env_temp = st.sidebar.number_input("Temperature (°C)", value=15.0)
env_humidity = st.sidebar.number_input("Humidity (%)", value=50.0)
frequency_ghz = st.sidebar.number_input("Frequency (GHz)", value=15.0)
tx_power = st.sidebar.number_input("TX Power (dBm)", value=20.0)

wavelength = 3e8 / (frequency_ghz * 1e9)

st.sidebar.subheader("Antenna A")
diameter_a = st.sidebar.number_input("Diameter A (m)", value=0.6)
gain_a = 10 * np.log10(0.55 * (np.pi * diameter_a / wavelength)**2)
hpbw_a = 70.0 * (wavelength / diameter_a)

st.sidebar.subheader("Antenna B")
diameter_b = st.sidebar.number_input("Diameter B (m)", value=0.6)
gain_b = 10 * np.log10(0.55 * (np.pi * diameter_b / wavelength)**2)
hpbw_b = 70.0 * (wavelength / diameter_b)

# --- 4. MAIN LAYOUT ---
st.title("Line of Sight & Multipath Analyzer")
col1, col2 = st.columns([1, 1])

with col1:
    m = folium.Map(location=[st.session_state.site_a["lat"], st.session_state.site_a["lon"]], zoom_start=12)
    folium.Marker([st.session_state.site_a["lat"], st.session_state.site_a["lon"]], icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([st.session_state.site_b["lat"], st.session_state.site_b["lon"]], icon=folium.Icon(color="red")).add_to(m)
    st_folium(m, height=400, width=600)

with col2:
    if st.button("Calculate Link & Analyze Multipath"):
        with st.spinner("Analyzing terrain and reflections..."):
            df_profile = get_elevation_profile(st.session_state.site_a["lat"], st.session_state.site_a["lon"], st.session_state.site_b["lat"], st.session_state.site_b["lon"])
            
            if df_profile is not None:
                dist = df_profile.iloc[-1]["Distance (m)"]
                abs_h_a = df_profile.iloc[0]["Elevation (m)"] + st.session_state.site_a["h"]
                abs_h_b = df_profile.iloc[-1]["Elevation (m)"] + st.session_state.site_b["h"]
                
                # Multipath Geometry
                x, y = df_profile["Distance (m)"].values, df_profile["Elevation (m)"].values
                a_a = np.arctan2(abs_h_a - y, x); a_b = np.arctan2(abs_h_b - y, dist - x)
                idx_ref = np.argmin(np.abs(a_a - a_b))
                
                angle_los = np.arctan2(abs_h_b - abs_h_a, dist)
                off_a = np.degrees(abs(angle_los - np.arctan2(y[idx_ref]-abs_h_a, x[idx_ref])))
                off_b = np.degrees(abs(np.arctan2(abs_h_a-abs_h_b, dist) - np.arctan2(y[idx_ref]-abs_h_b, dist-x[idx_ref])))
                
                # Nominal Suppression
                disc_a = min(12.0 * (off_a / hpbw_a)**2, 30.0)
                disc_b = min(12.0 * (off_b / hpbw_b)**2, 30.0)
                total_disc = disc_a + disc_b
                
                # 1. EVALUATE TILT-UP (1.5dB loss means tilting ~0.354 * HPBW)
                tilt_offset = np.sqrt(1.5 / 12.0)
                off_a_tilted = off_a + (tilt_offset * hpbw_a)
                off_b_tilted = off_b + (tilt_offset * hpbw_b)
                tilted_disc = min(12.0 * (off_a_tilted / hpbw_a)**2, 35.0) + min(12.0 * (off_b_tilted / hpbw_b)**2, 35.0)
                net_improvement = tilted_disc - total_disc - 3.0 # -3dB because both sides lose 1.5dB on main path

                # 2. RSL VARIANCE (RIPPLE)
                rho = 10**(-total_disc / 20.0)
                var_pos = 20 * np.log10(1 + rho)
                var_neg = 20 * np.log10(1 - rho)

                # --- PLOTTING ---
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', name='Terrain', line=dict(color='SaddleBrown')))
                fig.add_trace(go.Scatter(x=[0, dist], y=[abs_h_a, abs_h_b], mode='lines+markers', name='Direct Path', line=dict(color='red', dash='dash')))
                fig.add_trace(go.Scatter(x=[0, x[idx_ref], dist], y=[abs_h_a, y[idx_ref], abs_h_b], name='Reflected Path', line=dict(color='orange', dash='dot')))
                st.plotly_chart(fig, use_container_width=True)

                # --- RESULTS ---
                st.subheader("Multipath Analysis")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Suppression (Nominal)", f"{total_disc:.1f} dB")
                mc2.metric("RSL Ripple", f"+{var_pos:.2f} / {var_neg:.2f} dB")
                mc3.metric("Suppression (Tilted)", f"{tilted_disc:.1f} dB", f"{net_improvement:.1f} dB Net")
                
                if total_disc < 15:
                    st.error(f"Critical Multipath! Tilted suppression improves isolation by {net_improvement:.1f} dB (Net).")

                # --- PDF GENERATION ---
                df_attenuation = calculate_itu_attenuation(st.session_state.site_a["lat"], st.session_state.site_a["lon"], dist/1000, frequency_ghz, tx_power, gain_a, gain_b, env_temp, env_humidity)
                ref_data = {"total_disc": total_disc, "var_pos": var_pos, "var_neg": var_neg, "tilted_disc": tilted_disc, "net_improvement": net_improvement}
                ant_data = {"dia_a": diameter_a, "gain_a": gain_a, "dia_b": diameter_b, "gain_b": gain_b}
                
                with tempfile.TemporaryDirectory() as tmp:
                    p1 = os.path.join(tmp, "m.png"); p2 = os.path.join(tmp, "p.png")
                    fig.write_image(p2) # Reusing profile for demo; in real use you'd save map too
                    # To keep it simple for this script, we'll use the profile twice if map capture isn't setup
                    st.session_state.pdf_data = generate_pdf_report(st.session_state.site_a, st.session_state.site_b, dist/1000, frequency_ghz, p2, p2, df_attenuation, ref_data, ant_data)

if st.session_state.pdf_data:
    st.download_button("📄 Download Professional Report", st.session_state.pdf_data, "RF_Link_Analysis.pdf", "application/pdf")
