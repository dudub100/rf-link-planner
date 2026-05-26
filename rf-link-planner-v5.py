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
from streamlit_geolocation import streamlit_geolocation
import urllib.parse

st.set_page_config(layout="wide", page_title="Line of Sight Profiler")

# --- 1. SESSION STATE INITIALIZATION ---
if "site_a" not in st.session_state:
    st.session_state.site_a = {"lat": 40.7128, "lon": -74.0060, "h": 15.0} 

if "site_b" not in st.session_state:
    # Check if a shared link provided Site B coordinates via URL Query Parameters
    query_params = st.query_params
    if "lat_b" in query_params and "lon_b" in query_params:
        st.session_state.site_b = {
            "lat": float(query_params["lat_b"]), 
            "lon": float(query_params["lon_b"]), 
            "h": 20.0
        }
    else:
        st.session_state.site_b = {"lat": 40.7306, "lon": -73.9866, "h": 20.0}

if "pdf_data" not in st.session_state:
    st.session_state.pdf_data = None

# Initialize session state for weather so inputs can be updated programmatically
if "env_temp" not in st.session_state:
    st.session_state.env_temp = 15.0
if "env_humidity" not in st.session_state:
    st.session_state.env_humidity = 50.0

# --- 2. HELPER FUNCTIONS ---
def get_elevation_profile(lat1, lon1, lat2, lon2, num_points=50):
    lats = np.linspace(lat1, lat2, num_points)
    lons = np.linspace(lon1, lon2, num_points)
    coords = list(zip(lats, lons))
    distances = [0.0]
    for i in range(1, len(coords)):
        dist = geodesic(coords[i-1], coords[i]).meters
        distances.append(distances[-1] + dist)

    # Attempt 1: OpenTopoData (SRTM 30m)
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
        print(f"OpenTopoData failed: {e}. Switching to fallback...")

    # Attempt 2: Open-Elevation (SRTM 90m)
    url_openelev = "https://api.open-elevation.com/api/v1/lookup"
    payload = {"locations": [{"latitude": lat, "longitude": lon} for lat, lon in coords]}
    
    try:
        response2 = requests.post(url_openelev, json=payload, timeout=15)
        if response2.status_code == 200:
            results = response2.json().get('results', [])
            elevations = [res['elevation'] for res in results]
            if len(elevations) == num_points:
                return pd.DataFrame({"Distance (m)": distances, "Elevation (m)": elevations, "Lat": lats, "Lon": lons})
    except Exception as e:
        st.error(f"Critical Failure: Both elevation servers timed out. Error: {e}")
        return None
        
    st.error("Elevation data could not be retrieved from either service.")
    return None

def calculate_itu_attenuation(lat, lon, d_km, f_ghz, tx_power, gain_a, gain_b, t_c, rh):
    if d_km <= 0: return pd.DataFrame()
    availabilities = [99.0, 99.5, 99.9, 99.95, 99.99, 99.995, 99.999]
    fsl = 92.4 + 20 * np.log10(d_km) + 20 * np.log10(f_ghz)
    
    # Saturation vapor pressure (hPa)
    e_s = 6.1121 * np.exp((17.502 * t_c) / (240.97 + t_c))
    # Actual vapor pressure (hPa)
    e = e_s * (rh / 100.0)
    # Water vapor density (g/m^3)
    rho_calc = 216.7 * (e / (t_c + 273.15))
    
    d_val = d_km * u.km
    f_val = f_ghz * u.GHz
    el_val = 0.0 * u.deg
    rho_val = rho_calc * (u.g / u.m**3) 
    P_val = 1013.25 * u.hPa
    T_val = t_c * u.deg_C  

    try:
        gas_loss_obj = itur.models.itu676.gaseous_attenuation_terrestrial_path(
            r=d_val, f=f_val, el=el_val, rho=rho_val, P=P_val, T=T_val, mode='approx'
        )
        gas_loss = float(gas_loss_obj.value)
    except Exception:
        gas_loss = 0.0
        
    results = []
    rsl_clear = tx_power + gain_a + gain_b - fsl - gas_loss

    for avail in availabilities:
        p = round(100.0 - avail, 3) 
        try:
            rain_loss_obj = itur.models.itu530.rain_attenuation(lat=lat, lon=lon, d=d_val, f=f_val, el=el_val, p=p)
            rain_loss = float(rain_loss_obj.value)
        except Exception:
            rain_loss = 0.0
            
        total_loss = fsl + gas_loss + rain_loss
        rsl_faded = rsl_clear - rain_loss

        results.append({
            "Avail.": f"{avail}%", "Outage": f"{p}%", "FSL (dB)": f"{fsl:.1f}", 
            "Atm. (dB)": f"{gas_loss:.2f}", "Rain (dB)": f"{rain_loss:.1f}", 
            "Total Loss": f"{total_loss:.1f}", "Clear RSL": f"{rsl_clear:.1f} dBm", "Faded RSL": f"{rsl_faded:.1f} dBm"
        })
    return pd.DataFrame(results)

def generate_pdf_report(site_a, site_b, d_km, f_ghz, map_img_path, profile_img_path, df_att, ref_data):
    class PDF(FPDF):
        def header(self):
            self.set_font("helvetica", "B", 16)
            self.cell(0, 10, "RF Link Path Profile & Budget Report", border=False, align="C", new_x="LMARGIN", new_y="NEXT")
            self.line(10, 22, 200, 22)
            self.ln(5)
        def footer(self):
            self.set_y(-15)
            self.set_font("helvetica", "I", 8)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    pdf = PDF()
    pdf.add_page()
    
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, "1. Link Parameters & Hardware", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 10)
    pdf.cell(0, 6, f"Frequency: {f_ghz} GHz", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Total Path Distance: {d_km:.3f} km", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Antenna A Size: {ref_data['dia_a']} m | Gain: {ref_data['gain_a']:.1f} dBi", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Antenna B Size: {ref_data['dia_b']} m | Gain: {ref_data['gain_b']:.1f} dBi", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Atmospheric Conditions: {ref_data['env_temp']} deg C | {ref_data['env_humidity']}% RH", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Clear Sky RSL: {ref_data['clear_rsl']}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, "2. Site Details", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 10)
    pdf.cell(95, 6, f"Site A: {site_a['lat']:.6f}, {site_a['lon']:.6f} | Height: {site_a['h']}m", new_x="RIGHT", new_y="TOP")
    pdf.cell(95, 6, f"Site B: {site_b['lat']:.6f}, {site_b['lon']:.6f} | Height: {site_b['h']}m", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, "3. Map Overview", new_x="LMARGIN", new_y="NEXT")
    pdf.image(map_img_path, x=15, w=180)
    pdf.ln(5)

    pdf.add_page()
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, "4. Line of Sight & Fresnel Zone Profile", new_x="LMARGIN", new_y="NEXT")
    pdf.image(profile_img_path, x=5, w=190)
    pdf.ln(5)

    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, "5. Multipath Reflection Analysis", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 10)
    
    tot_disc = ref_data["total_disc"]
    if tot_disc < 10.0:
        status = f"CRITICAL MULTIPATH: Nominal suppression is only {tot_disc:.1f} dB. Severe fading expected."
        pdf.set_text_color(220, 53, 69)
    elif 10.0 <= tot_disc < 20.0:
        status = f"MARGINAL MULTIPATH: Nominal suppression is {tot_disc:.1f} dB. Partial attenuation."
        pdf.set_text_color(255, 153, 0)
    else:
        status = f"CLEAR: Nominal suppression is {tot_disc:.1f} dB. Safely suppressed."
        pdf.set_text_color(40, 167, 69)

    pdf.multi_cell(0, 6, status, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    pdf.cell(95, 6, f"Site A Reflection Angle: {ref_data['ang_a']:.2f} deg (Suppression: {ref_data['disc_a']:.1f} dB)", new_x="RIGHT", new_y="TOP")
    pdf.cell(95, 6, f"Site B Reflection Angle: {ref_data['ang_b']:.2f} deg (Suppression: {ref_data['disc_b']:.1f} dB)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("helvetica", "B", 10)
    pdf.cell(0, 6, "Nominal RSL Multipath Variance (Ripple):", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 10)
    pdf.cell(0, 6, f"  Constructive Interference (Peak): +{ref_data['var_pos']:.2f} dB", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"  Destructive Interference (Fade): {ref_data['var_neg']:.2f} dB", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("helvetica", "B", 10)
    pdf.cell(0, 6, "Tilt-Up Mitigation Evaluation (1.5 dB Main Path Loss Per Side | 3.0 dB Total):", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 10)
    pdf.cell(0, 6, f"  New Absolute Multipath Suppression: {ref_data['tilted_disc']:.1f} dB", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"  Effective Suppression (Relative to dropped Main Path): {ref_data['effective_tilted_suppression']:.1f} dB", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"  Net Signal-to-Interference Improvement: {ref_data['net_improvement']:.1f} dB", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"  Tilted Constructive Interference (Peak): +{ref_data['tilted_var_pos']:.2f} dB", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"  Tilted Destructive Interference (Fade): {ref_data['tilted_var_neg']:.2f} dB", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, "6. ITU-R Estimated Link Budget", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "B", 8)
    col_widths = [16, 16, 18, 18, 18, 22, 40, 40]
    for col_name, width in zip(df_att.columns, col_widths):
        pdf.cell(width, 8, col_name, border=1, align="C")
    pdf.ln(8)
    
    pdf.set_font("helvetica", "", 8)
    for row in df_att.itertuples(index=False):
        for item, width in zip(row, col_widths):
            pdf.cell(width, 8, str(item), border=1, align="C")
        pdf.ln(8)
    return bytes(pdf.output())

# --- 3. SIDEBAR USER INTERFACE ---
st.sidebar.title("Link Parameters")
click_target = st.sidebar.radio("Map Click Updates:", ["None (View Only)", "Site A", "Site B"])

st.sidebar.divider()
st.sidebar.subheader("Atmospheric Conditions")

if st.sidebar.button("⛅ Fetch Weather at Site A"):
    try:
        lat = st.session_state.site_a["lat"]
        lon = st.session_state.site_a["lon"]
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m"
        res = requests.get(url, timeout=5).json()
        
        # Update session state with API results
        st.session_state.env_temp = float(res["current"]["temperature_2m"])
        st.session_state.env_humidity = float(res["current"]["relative_humidity_2m"])
        st.rerun() # Refresh the UI
    except Exception as e:
        st.sidebar.error("Failed to retrieve weather data.")

env_temp = st.sidebar.number_input("Temperature (°C)", key="env_temp", step=1.0)
env_humidity = st.sidebar.number_input("Relative Humidity (%)", key="env_humidity", min_value=0.0, max_value=100.0, step=1.0)

st.sidebar.divider()
st.sidebar.subheader("RF Parameters")
frequency_ghz = st.sidebar.number_input("Frequency (GHz)", value=15.0, min_value=0.1, step=0.1)
tx_power = st.sidebar.number_input("Transmit Power (dBm)", value=20.0, step=1.0)

c = 3e8
wavelength = c / (frequency_ghz * 1e9)

st.sidebar.subheader("Antenna A")
diameter_a = st.sidebar.number_input("Diameter A (m)", value=0.6, min_value=0.1, step=0.1)
gain_a = 10 * np.log10(0.55 * (np.pi * diameter_a / wavelength)**2)
hpbw_a = 70.0 * (wavelength / diameter_a)
st.sidebar.caption(f"Calculated Gain: **{gain_a:.1f} dBi**")
st.sidebar.caption(f"Calculated Beamwidth: **{hpbw_a:.2f}°**")

st.sidebar.subheader("Antenna B")
diameter_b = st.sidebar.number_input("Diameter B (m)", value=0.6, min_value=0.1, step=0.1)
gain_b = 10 * np.log10(0.55 * (np.pi * diameter_b / wavelength)**2)
hpbw_b = 70.0 * (wavelength / diameter_b)
st.sidebar.caption(f"Calculated Gain: **{gain_b:.1f} dBi**")
st.sidebar.caption(f"Calculated Beamwidth: **{hpbw_b:.2f}°**")

st.sidebar.divider()
st.sidebar.subheader("Site A")

# 1. Native GPS request component
st.sidebar.caption("Get Device GPS Location:")
geo_loc = streamlit_geolocation()
if geo_loc and geo_loc.get('latitude') is not None:
    if st.sidebar.button("📍 Apply GPS to Site A"):
        st.session_state.site_a["lat"] = geo_loc['latitude']
        st.session_state.site_a["lon"] = geo_loc['longitude']
        st.rerun()

lat_a = st.sidebar.number_input("Latitude A", value=st.session_state.site_a["lat"], format="%.6f")
lon_a = st.sidebar.number_input("Longitude A", value=st.session_state.site_a["lon"], format="%.6f")
h_a = st.sidebar.number_input("Antenna Height A (m)", value=st.session_state.site_a["h"], min_value=0.0)

st.sidebar.subheader("Site B")
lat_b = st.sidebar.number_input("Latitude B", value=st.session_state.site_b["lat"], format="%.6f")
lon_b = st.sidebar.number_input("Longitude B", value=st.session_state.site_b["lon"], format="%.6f")
h_b = st.sidebar.number_input("Antenna Height B (m)", value=st.session_state.site_b["h"], min_value=0.0)

st.session_state.site_a.update({"lat": lat_a, "lon": lon_a, "h": h_a})
st.session_state.site_b.update({"lat": lat_b, "lon": lon_b, "h": h_b})

# --- WhatsApp Share Button ---
st.sidebar.divider()
st.sidebar.subheader("🔗 Share Link")

# IMPORTANT: Change this URL to your actual deployed Streamlit app URL!
APP_BASE_URL = "https://your-app-name.streamlit.app" 

# Generate the app URL with Site A coordinates injected as Site B
share_url = f"{APP_BASE_URL}?lat_b={st.session_state.site_a['lat']}&lon_b={st.session_state.site_a['lon']}"
wa_message = f"Check out this Site A location! I've loaded it as Site B in the link planner here: {share_url}"
encoded_message = urllib.parse.quote(wa_message)
wa_link = f"https://wa.me/?text={encoded_message}"

st.sidebar.markdown(
    f"""
    <a href="{wa_link}" target="_blank" style="text-decoration: none;">
        <div style="background-color:#25D366; color:white; padding:10px 15px; border-radius:5px; text-align:center; font-weight:bold; cursor:pointer;">
            💬 Share to WhatsApp
        </div>
    </a>
    """,
    unsafe_allow_html=True
)

# --- 4. MAIN LAYOUT (MAP & PROFILE) ---
st.title("Line of Sight & Multipath Viewer")
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("OpenStreetMap")
    center_lat = (st.session_state.site_a["lat"] + st.session_state.site_b["lat"]) / 2
    center_lon = (st.session_state.site_a["lon"] + st.session_state.site_b["lon"]) / 2
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    folium.Marker([st.session_state.site_a["lat"], st.session_state.site_a["lon"]], popup="Site A", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([st.session_state.site_b["lat"], st.session_state.site_b["lon"]], popup="Site B", icon=folium.Icon(color="red")).add_to(m)
    folium.PolyLine(locations=[(st.session_state.site_a["lat"], st.session_state.site_a["lon"]), (st.session_state.site_b["lat"], st.session_state.site_b["lon"])], color="blue", weight=2.5).add_to(m)

    map_data = st_folium(m, height=500, width=700, key="map")
    if map_data and map_data.get("last_clicked"):
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lon = map_data["last_clicked"]["lng"]
        if click_target == "Site A":
            st.session_state.site_a.update({"lat": clicked_lat, "lon": clicked_lon})
            st.rerun() 
        elif click_target == "Site B":
            st.session_state.site_b.update({"lat": clicked_lat, "lon": clicked_lon})
            st.rerun() 

with col2:
    st.subheader("Link Profile & Attenuation")
    if st.button("Calculate Link Parameters"):
        with st.spinner("Calculating Multipath Risk and Link Budget..."):
            
            df_profile = get_elevation_profile(
                st.session_state.site_a["lat"], st.session_state.site_a["lon"],
                st.session_state.site_b["lat"], st.session_state.site_b["lon"]
            )
            
            if df_profile is not None and not df_profile.empty:
                elev_a = df_profile.iloc[0]["Elevation (m)"]
                elev_b = df_profile.iloc[-1]["Elevation (m)"]
                total_dist = df_profile.iloc[-1]["Distance (m)"]
                
                abs_h_a = elev_a + st.session_state.site_a["h"]
                abs_h_b = elev_b + st.session_state.site_b["h"]
                
                slope = (abs_h_b - abs_h_a) / total_dist if total_dist > 0 else 0
                fresnel_upper, fresnel_lower, fresnel_60_lower = [], [], []
                
                for index, row in df_profile.iterrows():
                    d1 = row["Distance (m)"]
                    d2 = total_dist - d1
                    los_elev = abs_h_a + (slope * d1)
                    f_radius = np.sqrt((wavelength * d1 * d2) / total_dist) if (d1 > 0 and d2 > 0) else 0
                    
                    fresnel_upper.append(los_elev + f_radius)
                    fresnel_lower.append(los_elev - f_radius)
                    fresnel_60_lower.append(los_elev - (0.6 * f_radius))
                
                df_profile["Fresnel_Upper"] = fresnel_upper
                df_profile["Fresnel_Lower"] = fresnel_lower
                df_profile["Fresnel_60_Lower"] = fresnel_60_lower

                # --- EXACT MULTIPATH REFLECTION CALCULATION ---
                x_vals = df_profile["Distance (m)"].values
                y_vals = df_profile["Elevation (m)"].values
                
                dy = np.gradient(y_vals)
                dx = np.gradient(x_vals)
                dx[dx == 0] = 1e-6 
                alpha_g = np.arctan2(dy, dx) 
                
                alpha_a = np.zeros_like(x_vals)
                alpha_b = np.zeros_like(x_vals)
                valid_idx = (x_vals > 0) & (x_vals < total_dist)
                
                alpha_a[valid_idx] = np.arctan2(abs_h_a - y_vals[valid_idx], x_vals[valid_idx])
                alpha_b[valid_idx] = np.arctan2(abs_h_b - y_vals[valid_idx], total_dist - x_vals[valid_idx])
                
                diff = np.abs(alpha_a - alpha_b + 2 * alpha_g)
                diff[~valid_idx] = np.inf
                
                grazing_a = alpha_a + alpha_g
                grazing_b = alpha_b - alpha_g
                diff[(grazing_a <= 0) | (grazing_b <= 0)] = np.inf
                
                idx_ref = np.argmin(diff)
                
                if np.isinf(diff[idx_ref]):
                    idx_ref = len(x_vals) // 2
                    
                ref_dist = x_vals[idx_ref]
                ref_elev = y_vals[idx_ref]
                
                angle_los_a = np.degrees(np.arctan2(abs_h_b - abs_h_a, total_dist))
                angle_ref_a = np.degrees(np.arctan2(ref_elev - abs_h_a, ref_dist))
                off_boresight_a = abs(angle_los_a - angle_ref_a)
                
                angle_los_b = np.degrees(np.arctan2(abs_h_a - abs_h_b, total_dist))
                angle_ref_b = np.degrees(np.arctan2(ref_elev - abs_h_b, total_dist - ref_dist))
                off_boresight_b = abs(angle_los_b - angle_ref_b)

                # --- CALCULATE ANTENNA DISCRIMINATION (dB) ---
                disc_a = min(12.0 * (off_boresight_a / hpbw_a)**2, 25.0)
                disc_b = min(12.0 * (off_boresight_b / hpbw_b)**2, 25.0)
                total_discrimination = disc_a + disc_b

                # --- RSL VARIANCE CALCULATION (NOMINAL) ---
                rho = 10**(-total_discrimination / 20.0)
                var_pos = 20 * np.log10(1 + rho)
                var_neg = 20 * np.log10(1 - rho)

                # --- TILT-UP EVALUATION & VARIANCE (3 dB Main Path Loss Total) ---
                tilt_angle_a = hpbw_a * np.sqrt(1.5 / 12.0)
                tilt_angle_b = hpbw_b * np.sqrt(1.5 / 12.0)
                
                tilted_off_a = off_boresight_a + tilt_angle_a
                tilted_off_b = off_boresight_b + tilt_angle_b
                
                tilted_disc_a = min(12.0 * (tilted_off_a / hpbw_a)**2, 25.0)
                tilted_disc_b = min(12.0 * (tilted_off_b / hpbw_b)**2, 25.0)
                tilted_total_disc = tilted_disc_a + tilted_disc_b
                
                effective_tilted_suppression = tilted_total_disc - 3.0
                net_improvement = effective_tilted_suppression - total_discrimination

                rho_tilted = 10**(-effective_tilted_suppression / 20.0)
                tilted_var_pos = 20 * np.log10(1 + rho_tilted)
                tilted_var_neg = 20 * np.log10(1 - rho_tilted)

                # --- 1. BUILD PROFILE CHART ---
                lowest_point = min(df_profile["Elevation (m)"].min(), min(fresnel_lower))
                highest_point = max(df_profile["Elevation (m)"].max(), max(fresnel_upper), abs_h_a, abs_h_b)
                
                fig_profile = go.Figure()
                fig_profile.add_trace(go.Scatter(x=df_profile["Distance (m)"], y=df_profile["Elevation (m)"], fill='tozeroy', mode='lines', line=dict(color='SaddleBrown'), name='Terrain'))
                fig_profile.add_trace(go.Scatter(x=df_profile["Distance (m)"], y=df_profile["Fresnel_Upper"], mode='lines', line=dict(color='rgba(0,0,255,0.2)'), showlegend=False))
                fig_profile.add_trace(go.Scatter(x=df_profile["Distance (m)"], y=df_profile["Fresnel_Lower"], fill='tonexty', mode='lines', fillcolor='rgba(0,0,255,0.1)', line=dict(color='rgba(0,0,255,0.2)'), name='1st Fresnel Zone'))
                fig_profile.add_trace(go.Scatter(x=df_profile["Distance (m)"], y=df_profile["Fresnel_60_Lower"], mode='lines', line=dict(color='purple', dash='dot'), name='60% Clearance'))
                fig_profile.add_trace(go.Scatter(x=[0, total_dist], y=[abs_h_a, abs_h_b], mode='lines+markers', line=dict(color='red', dash='dash'), marker=dict(size=8, color=['green', 'red']), name='Line of Sight'))
                fig_profile.add_trace(go.Scatter(x=[0, 0], y=[elev_a, abs_h_a], mode='lines', line=dict(color='black', width=4), name='Mast A'))
                fig_profile.add_trace(go.Scatter(x=[total_dist, total_dist], y=[elev_b, abs_h_b], mode='lines', line=dict(color='black', width=4), name='Mast B'))
                fig_profile.add_trace(go.Scatter(x=[0, ref_dist, total_dist], y=[abs_h_a, ref_elev, abs_h_b], mode='lines+markers', line=dict(color='orange', dash='dashdot', width=2), marker=dict(size=6, color='orange'), name='Reflected Path'))

                fig_profile.update_layout(
                    title=f"Distance: {total_dist/1000:.2f} km | Freq: {frequency_ghz} GHz",
                    xaxis_title="Distance (m)", yaxis_title="Elevation (m)",
                    yaxis=dict(range=[lowest_point - 15, highest_point + 15]),
                    margin=dict(l=0, r=0, t=40, b=0), hovermode="x unified"
                )
                st.plotly_chart(fig_profile, use_container_width=True)

                # --- 2. MULTIPATH WARNING BANNER ---
                if total_discrimination < 10.0:
                    st.error(f"🔴 **CRITICAL MULTIPATH:** Nominal suppression is only **{total_discrimination:.1f} dB**. Expect severe destructive fading.")
                elif 10.0 <= total_discrimination < 20.0:
                    st.warning(f"🟡 **MARGINAL MULTIPATH:** Nominal suppression is **{total_discrimination:.1f} dB**. Partial attenuation, expect signal ripple.")
                else:
                    st.success(f"✅ **CLEAR:** Nominal suppression is **{total_discrimination:.1f} dB**. Safely suppressed.")

                # --- 3. ADVANCED METRICS UI ---
                st.markdown("### Nominal Multipath Metrics")
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Nominal RSL Ripple (Peak)", f"+{var_pos:.2f} dB")
                col_m2.metric("Nominal RSL Ripple (Fade)", f"{var_neg:.2f} dB")
                col_m3.metric("Total Nominal Suppression", f"{total_discrimination:.1f} dB")

                st.markdown("### Tilted-Up Multipath Metrics (Accounts for 3dB Main Path Loss)")
                col_t1, col_t2, col_t3 = st.columns(3)
                col_t1.metric("Tilted RSL Ripple (Peak)", f"+{tilted_var_pos:.2f} dB")
                col_t2.metric("Tilted RSL Ripple (Fade)", f"{tilted_var_neg:.2f} dB")
                col_t3.metric("Effective Tilted Suppression", f"{effective_tilted_suppression:.1f} dB", f"{net_improvement:.1f} dB Net Gain")

                # --- 4. ITU TABLE ---
                d_km = total_dist / 1000.0
                st.markdown("### Estimated Path Attenuation & Receiver Level")
                df_attenuation = calculate_itu_attenuation(center_lat, center_lon, d_km, frequency_ghz, tx_power, gain_a, gain_b, env_temp, env_humidity)
                st.dataframe(df_attenuation, use_container_width=True, hide_index=True)

                # --- 5. PDF GENERATION ---
                
                # Fetch the Clear Sky RSL from the first row of the newly calculated DataFrame
                clear_rsl_val = df_attenuation.iloc[0]["Clear RSL"] if not df_attenuation.empty else "N/A"
                
                ref_data = {
                    "total_disc": total_discrimination,
                    "ang_a": off_boresight_a,
                    "disc_a": disc_a,
                    "ang_b": off_boresight_b,
                    "disc_b": disc_b,
                    "var_pos": var_pos,
                    "var_neg": var_neg,
                    "tilted_disc": tilted_total_disc,
                    "effective_tilted_suppression": effective_tilted_suppression,
                    "net_improvement": net_improvement,
                    "tilted_var_pos": tilted_var_pos,
                    "tilted_var_neg": tilted_var_neg,
                    "dia_a": diameter_a,
                    "gain_a": gain_a,
                    "dia_b": diameter_b,
                    "gain_b": gain_b,
                    "env_temp": env_temp,           # Injected Temperature
                    "env_humidity": env_humidity,   # Injected Humidity
                    "clear_rsl": clear_rsl_val      # Injected Clear Sky RSL
                }
                
                fig_map = go.Figure(go.Scattermapbox(mode="markers+lines", lon=[st.session_state.site_a["lon"], st.session_state.site_b["lon"]], lat=[st.session_state.site_a["lat"], st.session_state.site_b["lat"]], marker={'size': 12, 'color': ["green", "red"]}))
                fig_map.update_layout(mapbox={'style': "open-street-map", 'center': {'lon': center_lon, 'lat': center_lat}, 'zoom': 11}, margin={'l':0, 'r':0, 'b':0, 't':0}, showlegend=False)

                with tempfile.TemporaryDirectory() as tmp_dir:
                    map_path = os.path.join(tmp_dir, "map.png")
                    profile_path = os.path.join(tmp_dir, "profile.png")
                    fig_map.write_image(map_path, width=800, height=400)
                    fig_profile.write_image(profile_path, width=800, height=400)
                    
                    pdf_bytes = generate_pdf_report(st.session_state.site_a, st.session_state.site_b, d_km, frequency_ghz, map_path, profile_path, df_attenuation, ref_data)
                    st.session_state.pdf_data = pdf_bytes
                    
    if st.session_state.pdf_data:
        st.download_button(label="📄 Download Professional PDF Report", data=st.session_state.pdf_data, file_name="RF_Link_Report.pdf", mime="application/pdf")
