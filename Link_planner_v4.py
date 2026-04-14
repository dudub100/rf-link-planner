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

# Initialize default coordinates if not present
if "site_a" not in st.session_state:
    st.session_state.site_a = {"lat": 40.7128, "lon": -74.0060, "h": 15.0} 
if "site_b" not in st.session_state:
    st.session_state.site_b = {"lat": 40.7306, "lon": -73.9866, "h": 20.0}
if "pdf_data" not in st.session_state:
    st.session_state.pdf_data = None

# Peer Deep-Linking: Check if someone shared their location via URL
params = st.query_params
if "peer_lat" in params:
    try:
        st.session_state.site_b = {
            "lat": float(params["peer_lat"]),
            "lon": float(params["peer_lon"]),
            "h": float(params["peer_h"])
        }
        st.toast("✅ Peer location loaded as Site B!")
    except:
        pass

# --- 2. HELPER FUNCTIONS ---

def fetch_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m"
        response = requests.get(url, timeout=5).json()
        return {
            "temp": float(response['current']['temperature_2m']),
            "rh": float(response['current']['relative_humidity_2m'])
        }
    except Exception:
        return None

def get_ground_elevation(lat, lon):
    url = f"https://api.opentopodata.org/v1/srtm30m?locations={lat},{lon}"
    try:
        res = requests.get(url, timeout=5).json()
        return res['results'][0]['elevation']
    except:
        return 0.0

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
            return pd.DataFrame({"Distance (m)": distances, "Elevation (m)": elevations, "Lat": lats, "Lon": lons})
    except Exception:
        return None

def calculate_itu_attenuation(lat, lon, d_km, f_ghz, tx_power, gain_a, gain_b, t_c, rh):
    if d_km <= 0: return pd.DataFrame()
    availabilities = [99.0, 99.5, 99.9, 99.95, 99.99, 99.995, 99.999]
    fsl = 92.4 + 20 * np.log10(d_km) + 20 * np.log10(f_ghz)
    
    e_s = 6.1121 * np.exp((17.502 * t_c) / (240.97 + t_c))
    e = e_s * (rh / 100.0)
    rho_calc = 216.7 * (e / (t_c + 273.15))
    
    d_val, f_val = d_km * u.km, f_ghz * u.GHz
    rho_val = rho_calc * (u.g / u.m**3)
    
    try:
        gas_loss = float(itur.models.itu676.gaseous_attenuation_terrestrial_path(
            r=d_val, f=f_val, el=0.0*u.deg, rho=rho_val, P=1013.25*u.hPa, T=t_c*u.deg_C, mode='approx'
        ).value)
    except: gas_loss = 0.0
        
    results = []
    rsl_clear = tx_power + gain_a + gain_b - fsl - gas_loss

    for avail in availabilities:
        p = round(100.0 - avail, 3) 
        try:
            rain_loss = float(itur.models.itu530.rain_attenuation(lat=lat, lon=lon, d=d_val, f=f_val, el=0.0*u.deg, p=p).value)
        except: rain_loss = 0.0
        rsl_faded = rsl_clear - rain_loss
        results.append({
            "Avail.": f"{avail}%", "FSL (dB)": f"{fsl:.1f}", 
            "Atm. (dB)": f"{gas_loss:.2f}", "Rain (dB)": f"{rain_loss:.1f}", 
            "Clear RSL": f"{rsl_clear:.1f} dBm", "Faded RSL": f"{rsl_faded:.1f} dBm"
        })
    return pd.DataFrame(results)

# --- 3. SIDEBAR TOOLS ---
st.sidebar.title("🛠️ Field Operations")

# Map click target selection
click_target = st.sidebar.radio("Map Click Updates:", ["None", "Site A", "Site B"])

# GPS BUTTON
if st.sidebar.button("📍 Set Site A to My Location"):
    loc = get_geolocation()
    if loc:
        lat, lon = loc['coords']['latitude'], loc['coords']['longitude']
        alt = loc['coords']['altitude'] if loc['coords']['altitude'] else 0
        ground = get_ground_elevation(lat, lon)
        agl = float(max(5.0, alt - ground))
        st.session_state.site_a.update({"lat": lat, "lon": lon, "h": round(agl, 1)})
        st.rerun()
    else:
        st.sidebar.info("Requesting location... please click again if the browser just asked for permission.")

# WEATHER SYNC
if st.sidebar.button("☁️ Sync Local Weather"):
    w = fetch_weather(st.session_state.site_a["lat"], st.session_state.site_a["lon"])
    if w:
        st.session_state.env_temp, st.session_state.env_rh = w['temp'], w['rh']
        st.rerun()

st.sidebar.divider()

# WHATSAPP SHARING
current_url = streamlit_js_eval(js_expressions="window.location.href", want_output=True, key="get_url")
if current_url:
    base_url = current_url.split("?")[0]
    share_url = f"{base_url}?peer_lat={st.session_state.site_a['lat']}&peer_lon={st.session_state.site_a['lon']}&peer_h={st.session_state.site_a['h']}"
    wa_link = f"https://wa.me/?text={urllib.parse.quote('Connect to my site: ' + share_url)}"
    st.sidebar.markdown(f'''<a href="{wa_link}" target="_blank" style="text-decoration: none;"><div style="background-color: #25D366; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold;">📲 Share Site A with Peer</div></a>''', unsafe_allow_html=True)

st.sidebar.divider()

# SITE A INPUTS
st.sidebar.subheader("Site A (Green)")
lat_a = st.sidebar.number_input("Latitude A", value=float(st.session_state.site_a["lat"]), format="%.6f")
lon_a = st.sidebar.number_input("Longitude A", value=float(st.session_state.site_a["lon"]), format="%.6f")
h_a = st.sidebar.number_input("Height A (AGL)", value=float(st.session_state.site_a["h"]))
st.session_state.site_a.update({"lat": lat_a, "lon": lon_a, "h": h_a})

# SITE B INPUTS
st.sidebar.subheader("Site B (Red)")
lat_b = st.sidebar.number_input("Latitude B", value=float(st.session_state.site_b["lat"]), format="%.6f")
lon_b = st.sidebar.number_input("Longitude B", value=float(st.session_state.site_b["lon"]), format="%.6f")
h_b = st.sidebar.number_input("Height B (AGL)", value=float(st.session_state.site_b["h"]))
st.session_state.site_b.update({"lat": lat_b, "lon": lon_b, "h": h_b})

st.sidebar.divider()

# RF & ATMOSPHERE
temp = st.sidebar.number_input("Temp (°C)", value=float(st.session_state.get("env_temp", 15.0)), format="%.1f")
rh = st.sidebar.number_input("Humidity (%)", value=float(st.session_state.get("env_rh", 50.0)), format="%.1f")
freq = st.sidebar.number_input("Freq (GHz)", value=15.0)
pwr = st.sidebar.number_input("TX Power (dBm)", value=20.0)

# --- 4. MAIN LAYOUT ---
st.title("Line of Sight & Multipath Viewer")
col1, col2 = st.columns([1, 1])

with col1:
    m = folium.Map(location=[st.session_state.site_a["lat"], st.session_state.site_a["lon"]], zoom_start=13)
    folium.Marker([st.session_state.site_a["lat"], st.session_state.site_a["lon"]], icon=folium.Icon(color="green"), popup="Site A").add_to(m)
    folium.Marker([st.session_state.site_b["lat"], st.session_state.site_b["lon"]], icon=folium.Icon(color="red"), popup="Site B").add_to(m)
    folium.PolyLine([(st.session_state.site_a["lat"], st.session_state.site_a["lon"]), (st.session_state.site_b["lat"], st.session_state.site_b["lon"])], color="blue", weight=2).add_to(m)
    
    # Capture map clicks
    map_data = st_folium(m, height=500, width=700, key="link_map")
    
    if map_data and map_data.get("last_clicked"):
        new_lat = map_data["last_clicked"]["lat"]
        new_lon = map_data["last_clicked"]["lng"]
        if click_target == "Site A":
            st.session_state.site_a.update({"lat": new_lat, "lon": new_lon})
            st.rerun()
        elif click_target == "Site B":
            st.session_state.site_b.update({"lat": new_lat, "lon": new_lon})
            st.rerun()

with col2:
    if st.button("🚀 Calculate Link Parameters"):
        with st.spinner("Analyzing Path..."):
            df_prof = get_elevation_profile(st.session_state.site_a["lat"], st.session_state.site_a["lon"], st.session_state.site_b["lat"], st.session_state.site_b["lon"])
            if df_prof is not None:
                dist_m = df_prof.iloc[-1]["Distance (m)"]
                abs_a = df_prof.iloc[0]["Elevation (m)"] + st.session_state.site_a["h"]
                abs_b = df_prof.iloc[-1]["Elevation (m)"] + st.session_state.site_b["h"]
                
                df_prof["LOS"] = np.linspace(abs_a, abs_b, len(df_prof))
                df_prof["F1"] = 17.32 * np.sqrt(( (df_prof["Distance (m)"]/1000) * ((dist_m-df_prof["Distance (m)"])/1000) ) / (freq * (dist_m/1000)))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_prof["Distance (m)"], y=df_prof["Elevation (m)"], fill='tozeroy', name='Terrain', line=dict(color='brown')))
                fig.add_trace(go.Scatter(x=df_prof["Distance (m)"], y=df_prof["LOS"], name='LOS', line=dict(color='red', dash='dash')))
                fig.add_trace(go.Scatter(x=df_prof["Distance (m)"], y=df_prof["LOS"]-df_prof["F1"], name='1st Fresnel', line=dict(color='rgba(0,0,255,0.2)')))
                st.plotly_chart(fig, use_container_width=True)
                
                wl = 0.3 / freq
                g_a = 10 * np.log10(0.55 * (np.pi * 0.6 / wl)**2) # Default 0.6m dish
                df_att = calculate_itu_attenuation(st.session_state.site_a["lat"], st.session_state.site_a["lon"], dist_m/1000, freq, pwr, g_a, g_a, temp, rh)
                st.dataframe(df_att, hide_index=True)
