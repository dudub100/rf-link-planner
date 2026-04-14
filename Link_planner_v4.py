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

# Initialize default coordinates with unique keys for the UI components
if "lat_a" not in st.session_state: st.session_state.lat_a = 40.7128
if "lon_a" not in st.session_state: st.session_state.lon_a = -74.0060
if "h_a" not in st.session_state: st.session_state.h_a = 15.0

if "lat_b" not in st.session_state: st.session_state.lat_b = 40.7306
if "lon_b" not in st.session_state: st.session_state.lon_b = -73.9866
if "h_b" not in st.session_state: st.session_state.h_b = 20.0

if "gps_requested" not in st.session_state: st.session_state.gps_requested = False

# Peer Deep-Linking
params = st.query_params
if "peer_lat" in params:
    try:
        st.session_state.lat_b = float(params["peer_lat"])
        st.session_state.lon_b = float(params["peer_lon"])
        st.session_state.h_b = float(params["peer_h"])
        st.toast("✅ Peer location loaded as Site B!")
    except: pass

# --- 2. HELPER FUNCTIONS ---
def fetch_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m"
        response = requests.get(url, timeout=5).json()
        return {"temp": float(response['current']['temperature_2m']), "rh": float(response['current']['relative_humidity_2m'])}
    except: return None

def get_ground_elevation(lat, lon):
    url = f"https://api.opentopodata.org/v1/srtm30m?locations={lat},{lon}"
    try:
        res = requests.get(url, timeout=5).json()
        return res['results'][0]['elevation']
    except: return 0.0

# --- 3. GPS LOGIC (Outside Button Block) ---
# We call this every rerun. If we requested GPS and data finally exists, we update.
loc = get_geolocation()

if st.session_state.gps_requested and loc:
    lat = loc['coords']['latitude']
    lon = loc['coords']['longitude']
    alt = loc['coords']['altitude'] if loc['coords']['altitude'] else 0
    
    ground = get_ground_elevation(lat, lon)
    agl = float(max(5.0, alt - ground))
    
    # Update state directly
    st.session_state.lat_a = lat
    st.session_state.lon_a = lon
    st.session_state.h_a = round(agl, 1)
    
    # Reset the flag so it doesn't loop
    st.session_state.gps_requested = False
    st.rerun()

# --- 4. SIDEBAR ---
st.sidebar.title("🛠️ Field Operations")
click_target = st.sidebar.radio("Map Click Updates:", ["None", "Site A", "Site B"])

if st.sidebar.button("📍 Set Site A to My Location"):
    st.session_state.gps_requested = True
    st.rerun()

if st.sidebar.button("☁️ Sync Local Weather"):
    w = fetch_weather(st.session_state.lat_a, st.session_state.lon_a)
    if w:
        st.session_state.env_temp, st.session_state.env_rh = w['temp'], w['rh']
        st.rerun()

st.sidebar.divider()

# WHATSAPP SHARING
current_url = streamlit_js_eval(js_expressions="window.location.href", want_output=True, key="get_url")
if current_url:
    base_url = current_url.split("?")[0]
    share_url = f"{base_url}?peer_lat={st.session_state.lat_a}&peer_lon={st.session_state.lon_a}&peer_h={st.session_state.h_a}"
    wa_link = f"https://wa.me/?text={urllib.parse.quote('Sync your Site B with my coordinates: ' + share_url)}"
    st.sidebar.markdown(f'''<a href="{wa_link}" target="_blank" style="text-decoration: none;"><div style="background-color: #25D366; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold;">📲 Share Site A with Peer</div></a>''', unsafe_allow_html=True)

st.sidebar.divider()

# INPUTS (Bound to session state)
st.sidebar.subheader("Site A (Green)")
st.session_state.lat_a = st.sidebar.number_input("Lat A", value=st.session_state.lat_a, format="%.6f")
st.session_state.lon_a = st.sidebar.number_input("Lon A", value=st.session_state.lon_a, format="%.6f")
st.session_state.h_a = st.sidebar.number_input("Height A (AGL)", value=st.session_state.h_a)

st.sidebar.subheader("Site B (Red)")
st.session_state.lat_b = st.sidebar.number_input("Lat B", value=st.session_state.lat_b, format="%.6f")
st.session_state.lon_b = st.sidebar.number_input("Lon B", value=st.session_state.lon_b, format="%.6f")
st.session_state.h_b = st.sidebar.number_input("Height B (AGL)", value=st.session_state.h_b)

# --- 5. MAP & MAIN UI ---
st.title("Line of Sight & Multipath Viewer")
col1, col2 = st.columns([1, 1])

with col1:
    m = folium.Map(location=[st.session_state.lat_a, st.session_state.lon_a], zoom_start=13)
    folium.Marker([st.session_state.lat_a, st.session_state.lon_a], icon=folium.Icon(color="green"), popup="Site A").add_to(m)
    folium.Marker([st.session_state.lat_b, st.session_state.lon_b], icon=folium.Icon(color="red"), popup="Site B").add_to(m)
    folium.PolyLine([(st.session_state.lat_a, st.session_state.lon_a), (st.session_state.lat_b, st.session_state.lon_b)], color="blue").add_to(m)
    
    map_data = st_folium(m, height=500, width=700, key="link_map")
    
    # Handle Map Clicks
    if map_data and map_data.get("last_clicked"):
        new_lat, new_lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
        if click_target == "Site A":
            st.session_state.lat_a, st.session_state.lon_a = new_lat, new_lon
            st.rerun()
        elif click_target == "Site B":
            st.session_state.lat_b, st.session_state.lon_b = new_lat, new_lon
            st.rerun()

with col2:
    st.info("Click 'Calculate' to see the profile and link budget.")
    # (Rest of your calculation and Plotly logic goes here using st.session_state variables)
