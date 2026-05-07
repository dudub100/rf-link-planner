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
        print(f"OpenTopoData failed: {e}")
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
        results.append({"Avail.": f"{avail}%", "Rain (dB)": f"{rain_loss:.1f}", "Clear RSL": f"{rsl_clear:.1f} dBm", "Faded RSL": f"{rsl_clear - rain_loss:.1f} dBm"})
    return pd.DataFrame(results)

def generate_pdf_report(site_a, site_b, d_km, f_ghz, map_img, prof_img, df_att, ref_data, ant_data):
    class PDF(FPDF):
        def header(self):
            self.set_font("helvetica", "B", 14); self.cell(0, 10, "RF Link Analysis Report", align="C", ln=True)
            self.line(10, 18, 200, 18); self.ln(5)
    
    pdf = PDF()
    pdf.add_page()
    
    # Antenna Sizes Section
    pdf.set_font("helvetica", "B", 12); pdf.cell(0, 8, "1. Hardware Specifications", ln=True)
    pdf.set_font("helvetica", "", 10)
    pdf.cell(0, 6, f"Antenna A: {ant_data['dia_a']}m Diameter | Gain: {ant_data['gain_a']:.1f} dBi", ln=True)
    pdf.cell(0, 6, f"Antenna B: {ant_data['dia_b']}m Diameter | Gain: {ant_data['gain_b']:.1f} dBi", ln=True)
    pdf.ln(4)

    # Multipath Section
    pdf.set_font("helvetica", "B", 12); pdf.cell(0, 8, "2. Multipath Analysis", ln=True)
    pdf.set_font("helvetica", "", 10)
    pdf.cell(0, 6, f"Nominal Multipath Suppression: {ref_data['total_disc']:.1f} dB", ln=True)
    pdf.cell(0, 6, f"Constructive/Destructive Variance: +/- {ref_data['variance']:.2f} dB", ln=True)
    
    if ref_data['tilted_disc'] > 0:
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(0, 6, f"Tilt-Up Optimization (1.5dB Main Path Loss per side):", ln=True)
        pdf.set_font("helvetica", "", 10)
        pdf.cell(0, 6, f" - New Multipath Suppression: {ref_data['tilted_disc']:.1f} dB", ln=True)
        pdf.cell(0, 6, f" - Net Improvement in S/R Ratio: {ref_data['tilt_improvement']:.1f} dB", ln=True)
    pdf.ln(5)

    pdf.image(prof_img, x=10, w=190); pdf.add_page()
    pdf.image(map_img, x=10, w=190); pdf.ln(5)
    
    pdf.set_font("helvetica", "B", 12); pdf.cell(0, 8, "3. Estimated Link Budget", ln=True)
    pdf.set_font("helvetica", "", 8)
    for col in df_att.columns: pdf.cell(45, 8, col, border=1)
    pdf.ln()
    for row in df_att.itertuples(index=False):
        for item in row: pdf.cell(45, 8, str(item), border=1)
        pdf.ln()
    return bytes(pdf.output())

# --- 3. SIDEBAR ---
st.sidebar.title("Link Parameters")
env_temp = st.sidebar.number_input("Temp (°C)", value=15.0)
env_humidity = st.sidebar.number_input("Humidity (%)", value=50.0)
frequency_ghz = st.sidebar.number_input("Frequency (GHz)", value=15.0)
tx_power = st.sidebar.number_input("TX Power (dBm)", value=20.0)

wavelength = 3e8 / (frequency_ghz * 1e9)
diameter_a = st.sidebar.number_input("Diameter A (m)", value=0.6)
gain_a = 10 * np.log10(0.55 * (np.pi * diameter_a / wavelength)**2)
hpbw_a = 70.0 * (wavelength / diameter_a)

diameter_b = st.sidebar.number_input("Diameter B (m)", value=0.6)
gain_b = 10 * np.log10(0.55 * (np.pi * diameter_b / wavelength)**2)
hpbw_b = 70.0 * (wavelength / diameter_b)

h_a = st.sidebar.number_input("Height A (m)", value=15.0)
h_b = st.sidebar.number_input("Height B (m)", value=20.0)

# --- 4. MAIN ---
if st.button("Run Full Analysis"):
    df_profile = get_elevation_profile(st.session_state.site_a["lat"], st.session_state.site_a["lon"], st.session_state.site_b["lat"], st.session_state.site_b["lon"])
    
    if df_profile is not None:
        dist = df_profile.iloc[-1]["Distance (m)"]
        abs_a, abs_b = df_profile.iloc[0]["Elevation (m)"] + h_a, df_profile.iloc[-1]["Elevation (m)"] + h_b
        
        # Multipath Geometry
        x, y = df_profile["Distance (m)"].values, df_profile["Elevation (m)"].values
        # Simple reflection point finder (lowest difference between arrival angles)
        a_a = np.arctan2(abs_a - y, x); a_b = np.arctan2(abs_b - y, dist - x)
        idx_ref = np.argmin(np.abs(a_a - a_b))
        
        off_a = np.degrees(abs(np.arctan2(abs_b-abs_a, dist) - np.arctan2(y[idx_ref]-abs_a, x[idx_ref])))
        off_b = np.degrees(abs(np.arctan2(abs_a-abs_b, dist) - np.arctan2(y[idx_ref]-abs_b, dist-x[idx_ref])))
        
        # Nominal Discrim
        disc_a = min(12.0 * (off_a / hpbw_a)**2, 35.0)
        disc_b = min(12.0 * (off_b / hpbw_b)**2, 35.0)
        total_disc = disc_a + disc_b
        
        # 1. Tilt Analysis (Shift boresight by finding angle for 1.5dB loss)
        # 1.5 = 12 * (tilt / hpbw)^2 => tilt = hpbw * sqrt(1.5/12)
        tilt_angle_a = hpbw_a * np.sqrt(1.5/12.0)
        tilt_angle_b = hpbw_b * np.sqrt(1.5/12.0)
        
        # New off-boresight is original off-boresight + tilt (moving away from ground)
        tilt_disc_a = min(12.0 * ((off_a + tilt_angle_a) / hpbw_a)**2, 35.0)
        tilt_disc_b = min(12.0 * ((off_b + tilt_angle_b) / hpbw_b)**2, 35.0)
        total_tilt_disc = tilt_disc_a + tilt_disc_b
        
        # 2. RSL Variance (Ripple)
        # Ratio of Reflected/Direct voltage: rho = 10^(-total_disc/20)
        rho = 10**(-total_disc / 20.0)
        var_pos = 20 * np.log10(1 + rho) # Constructive
        var_neg = 20 * np.log10(1 - rho) # Destructive
        
        st.subheader("Multipath & Tilt Analysis")
        c1, c2, c3 = st.columns(3)
        c1.metric("Nominal Suppression", f"{total_disc:.1f} dB")
        c2.metric("RSL Variance", f"{var_pos:.1f} / {var_neg:.1f} dB")
        c3.metric("Tilted Suppression", f"{total_tilt_disc:.1f} dB", f"{total_tilt_disc - total_disc - 3:.1f} dB Net")
        
        if total_disc < 15:
            st.warning("High Multipath Risk! Tilt-up suggested.")

        # PDF Data Prep
        df_att = calculate_itu_attenuation(40.7, -74.0, dist/1000, frequency_ghz, tx_power, gain_a, gain_b, env_temp, env_humidity)
        ref_data = {
            "total_disc": total_disc, "variance": var_pos, 
            "tilted_disc": total_tilt_disc, "tilt_improvement": (total_tilt_disc - total_disc - 3)
        }
        ant_data = {"dia_a": diameter_a, "gain_a": gain_a, "dia_b": diameter_b, "gain_b": gain_b}
        
        # Dummy plots for snippet
        fig = go.Figure(); fig.add_trace(go.Scatter(x=x, y=y))
        with tempfile.TemporaryDirectory() as tmp:
            p1 = os.path.join(tmp, "p1.png"); p2 = os.path.join(tmp, "p2.png")
            fig.write_image(p1); fig.write_image(p2)
            st.session_state.pdf_data = generate_pdf_report(None, None, dist/1000, frequency_ghz, p1, p2, df_att, ref_data, ant_data)

if st.session_state.pdf_data:
    st.download_button("Download Updated Report", st.session_state.pdf_data, "Link_Report_V2.pdf")
