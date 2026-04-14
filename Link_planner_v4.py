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

st.set_page_config(layout="wide", page_title="Line of Sight Profiler Pro")

# --- 1. SESSION STATE & DEEP LINKING ---
# Check if a peer shared their location with us via URL parameters
params = st.query_params

if "site_a" not in st.session_state:
    st.session_state.site_a = {"lat": 40.7128, "lon": -74.0060, "h": 15.0} 

# If the URL contains "peer" data, set it as Site B automatically
if "peer_lat" in params:
    st.session_state.site_b = {
        "lat": float(params.get("peer_lat")), 
        "lon": float(params.get("peer_lon")), 
        "h": float(params.get("peer_h"))
    }
elif "site_b" not in st.session_state:
    st.session_state.site_b = {"lat": 40.7306, "lon": -73.9866, "h": 20.0}

# --- 2. NEW HELPER FUNCTIONS ---

def fetch_weather(lat, lon):
    """Fetches real-time weather from Open-Meteo (Free, No Key)"""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m"
        response = requests.get(url, timeout=5).json()
        return {
            "temp": response['current']['temperature_2m'],
            "rh": response['current']['relative_humidity_2m']
        }
    except Exception:
        return None

def get_ground_elevation(lat, lon):
    """Reuse existing logic to get ground elevation for AGL calculation"""
    url_opentopo = f"https://api.opentopodata.org/v1/srtm30m?locations={lat},{lon}"
    try:
        res = requests.get(url_opentopo, timeout=5).json()
        return res['results'][0]['elevation']
    except:
        return 0.0

# --- [REST OF YOUR EXISTING HELPER FUNCTIONS (get_elevation_profile, calculate_itu_attenuation, etc.)] ---
# ... (Keep your existing functions here) ...

# --- 3. SIDEBAR USER INTERFACE ---
st.sidebar.title("📡 Field Tools")

# FEATURE 1: GET CURRENT LOCATION
if st.sidebar.button("📍 Set Site A to My Location"):
    loc = get_geolocation()
    if loc:
        my_lat = loc['coords']['latitude']
        my_lon = loc['coords']['longitude']
        my_alt = loc['coords']['altitude'] if loc['coords']['altitude'] else 0
        
        # Calculate AGL: (GPS Altitude) - (Ground Elevation)
        ground = get_ground_elevation(my_lat, my_lon)
        agl = max(2.0, my_alt - ground) # Default to 2m if calculation is weird
        
        st.session_state.site_a.update({"lat": my_lat, "lon": my_lon, "h": round(agl, 1)})
        st.sidebar.success(f"Site A Updated! (Ground: {ground:.1f}m, AGL: {agl:.1f}m)")
        st.rerun()
    else:
        st.sidebar.warning("Please allow location access in your browser.")

# FEATURE 2: AUTO-WEATHER
if st.sidebar.button("☁️ Sync Local Weather"):
    w_data = fetch_weather(st.session_state.site_a["lat"], st.session_state.site_a["lon"])
    if w_data:
        st.session_state.env_temp = w_data['temp']
        st.session_state.env_rh = w_data['rh']
        st.sidebar.success(f"Updated: {w_data['temp']}°C, {w_data['rh']}% RH")
    else:
        st.sidebar.error("Weather service unavailable.")

st.sidebar.divider()

# Input fields (using session state values)
env_temp = st.sidebar.number_input("Temperature (°C)", value=st.session_state.get("env_temp", 15.0), step=1.0)
env_humidity = st.sidebar.number_input("Relative Humidity (%)", value=st.session_state.get("env_rh", 50.0), min_value=0.0, max_value=100.0)

# --- FEATURE 3: SHARE TO PEER ---
st.sidebar.divider()
st.sidebar.subheader("Share with Technician")
current_url = streamlit_js_eval(js_expressions="window.location.href", want_output=True, key="get_url")

if current_url:
    # Clean the base URL (remove existing params)
    base_url = current_url.split("?")[0]
    share_params = {
        "peer_lat": st.session_state.site_a["lat"],
        "peer_lon": st.session_state.site_a["lon"],
        "peer_h": st.session_state.site_a["h"]
    }
    share_url = f"{base_url}?{urllib.parse.urlencode(share_params)}"
    wa_link = f"https://wa.me/?text={urllib.parse.quote('Connect to my RF Link: ' + share_url)}"
    
    st.sidebar.markdown(f'''<a href="{wa_link}" target="_blank" style="text-decoration: none;">
        <div style="background-color: #25D366; color: white; padding: 10px; border-radius: 5px; text-align: center;">
            Share Site A via WhatsApp
        </div></a>''', unsafe_allow_name=True)

# --- [REST OF YOUR EXISTING RF PARAMETERS & MAIN LAYOUT] ---
# ... (Continue with Frequency, Gain, and the Main Plotly/Map logic) ...
