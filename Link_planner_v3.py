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

    url = "https://api.open-elevation.com/api/v1/lookup"
    payload = {"locations": [{"latitude": lat, "longitude": lon} for lat, lon in coords]}
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            results = response.json().get('results', [])
            elevations = [res['elevation'] for res in results]
            return pd.DataFrame({"Distance (m)": distances, "Elevation (m)": elevations, "Lat": lats, "Lon": lons})
    except Exception as e:
        st.error(f"Failed to fetch elevation data: {e}")
        return None
    return None

def calculate_itu_attenuation(lat, lon, d_km, f_ghz, tx_power, gain_a, gain_b):
    if d_km <= 0: return pd.DataFrame()
    availabilities = [99.0, 99.5, 99.9, 99.95, 99.99, 99.995, 99.999]
    fsl = 92.4 + 20 * np.log10(d_km) + 20 * np.log10(f_ghz)
    
    d_val = d_km * u.km
    f_val = f_ghz * u.GHz
    el_val = 0.0 * u.deg
    rho_val = 7.5 * (u.g / u.m**3)
    P_val = 1013.25 * u.hPa
    T_val = 15.0 * u.deg_C

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

def generate_pdf_report(site_a, site_b, d_km, f_ghz, map_img_path, profile_img_path, df_att):
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
    pdf.cell(0, 8, "1. Link Parameters", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 10)
    pdf.cell(0, 6, f"Frequency: {f_ghz} GHz", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Total Path Distance: {d_km:.3f} km", new_x="LMARGIN", new_y="NEXT")
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
    pdf.cell(0, 8, "5. ITU-R Estimated Link Budget", new_x="LMARGIN", new_y="NEXT")
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
st.sidebar.subheader("RF Parameters")
frequency_ghz = st.sidebar.number_input("Frequency (GHz)", value=15.0, min_value=0.1, step=0.1)
tx_power = st.sidebar.number_input("Transmit Power (dBm)", value=20.0, step=1.0)

# Calculate Wavelength
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
lat_a = st.sidebar.number_input("Latitude A", value=st.session_state.site_a["lat"], format="%.6f")
lon_a = st.sidebar.number_input("Longitude A", value=st.session_state.site_a["lon"], format="%.6f")
h_a = st.sidebar.number_input("Antenna Height A (m)", value=st.session_state.site_a["h"], min_value=0.0)

st.sidebar.subheader("Site B")
lat_b = st.sidebar.number_input("Latitude B", value=st.session_state.site_b["lat"], format="%.6f")
lon_b = st.sidebar.number_input("Longitude B", value=st.session_state.site_b["lon"], format="%.6f")
h_b = st.sidebar.number_input("Antenna Height B (m)", value=st.session_state.site_b["h"], min_value=0.0)

st.session_state.site_a.update({"lat": lat_a, "lon": lon_a, "h": h_a})
st.session_state.site_b.update({"lat": lat_b, "lon": lon_b, "h": h_b})

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
        if click_target == "Site A" and (clicked_lat != st.session_state.site_a["lat"] or clicked_lon != st.session_state.site_a["lon"]):
            st.session_state.site_a.update({"lat": clicked_lat, "lon": clicked_lon})
            st.rerun() 
        elif click_target == "Site B" and (clicked_lat != st.session_state.site_b["lat"] or clicked_lon != st.session_state.site_b["lon"]):
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

                # --- MULTIPATH REFLECTION CALCULATION ---
                # 1. Approximate the flat reflection plane using minimum terrain elevation
                min_elev = df_profile["Elevation (m)"].min()
                h1_eff = abs_h_a - min_elev
                h2_eff = abs_h_b - min_elev
                
                # 2. Geometric reflection distance from Site A
                d_ref_ideal = total_dist * (h1_eff / (h1_eff + h2_eff)) if (h1_eff + h2_eff) > 0 else total_dist / 2
                
                # 3. Find closest actual terrain point to that distance
                idx_ref = (df_profile['Distance (m)'] - d_ref_ideal).abs().idxmin()
                ref_dist = df_profile.loc[idx_ref, 'Distance (m)']
                ref_elev = df_profile.loc[idx_ref, 'Elevation (m)']
                
                # 4. Angle Analysis (Degrees)
                angle_los_a = np.degrees(np.arctan2(abs_h_b - abs_h_a, total_dist))
                angle_ref_a = np.degrees(np.arctan2(ref_elev - abs_h_a, ref_dist))
                off_boresight_a = abs(angle_los_a - angle_ref_a)
                
                angle_los_b = np.degrees(np.arctan2(abs_h_a - abs_h_b, total_dist))
                angle_ref_b = np.degrees(np.arctan2(ref_elev - abs_h_b, total_dist - ref_dist))
                off_boresight_b = abs(angle_los_b - angle_ref_b)

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
                
                # ADD THE REFLECTED PATH TRACE
                fig_profile.add_trace(go.Scatter(x=[0, ref_dist, total_dist], y=[abs_h_a, ref_elev, abs_h_b], mode='lines+markers', line=dict(color='orange', dash='dashdot', width=2), marker=dict(size=6, color='orange'), name='Reflected Path'))

                fig_profile.update_layout(
                    title=f"Distance: {total_dist/1000:.2f} km | Freq: {frequency_ghz} GHz",
                    xaxis_title="Distance (m)", yaxis_title="Elevation (m)",
                    yaxis=dict(range=[lowest_point - 15, highest_point + 15]),
                    margin=dict(l=0, r=0, t=40, b=0), hovermode="x unified"
                )
                st.plotly_chart(fig_profile, use_container_width=True)

                # --- 2. MULTIPATH WARNING BANNER ---
                risk_a = off_boresight_a < (hpbw_a / 2)
                risk_b = off_boresight_b < (hpbw_b / 2)

                if risk_a or risk_b:
                    st.error("⚠️ **MULTIPATH WARNING:** The ground reflection path falls inside the main beam of the antenna. The signal is not sufficiently attenuated and may cause deep fading.")
                else:
                    st.success("✅ **CLEAR:** The ground reflection angle falls outside the main beam. Side-lobe suppression should adequately attenuate the multipath interference.")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**Site A Reflection Angle:** {off_boresight_a:.2f}° off-boresight")
                    st.markdown(f"*(Beamwidth limit: {hpbw_a/2:.2f}°)*")
                with col_b:
                    st.markdown(f"**Site B Reflection Angle:** {off_boresight_b:.2f}° off-boresight")
                    st.markdown(f"*(Beamwidth limit: {hpbw_b/2:.2f}°)*")

                # --- 3. ITU TABLE ---
                d_km = total_dist / 1000.0
                st.markdown("### Estimated Path Attenuation & Receiver Level")
                df_attenuation = calculate_itu_attenuation(center_lat, center_lon, d_km, frequency_ghz, tx_power, gain_a, gain_b)
                st.dataframe(df_attenuation, use_container_width=True, hide_index=True)

                # --- 4. PDF GENERATION ---
                fig_map = go.Figure(go.Scattermapbox(mode="markers+lines", lon=[st.session_state.site_a["lon"], st.session_state.site_b["lon"]], lat=[st.session_state.site_a["lat"], st.session_state.site_b["lat"]], marker={'size': 12, 'color': ["green", "red"]}))
                fig_map.update_layout(mapbox={'style': "open-street-map", 'center': {'lon': center_lon, 'lat': center_lat}, 'zoom': 11}, margin={'l':0, 'r':0, 'b':0, 't':0}, showlegend=False)

                with tempfile.TemporaryDirectory() as tmp_dir:
                    map_path = os.path.join(tmp_dir, "map.png")
                    profile_path = os.path.join(tmp_dir, "profile.png")
                    fig_map.write_image(map_path, width=800, height=400)
                    fig_profile.write_image(profile_path, width=800, height=400)
                    pdf_bytes = generate_pdf_report(st.session_state.site_a, st.session_state.site_b, d_km, frequency_ghz, map_path, profile_path, df_attenuation)
                    st.session_state.pdf_data = pdf_bytes
                    
    if st.session_state.pdf_data:
        st.download_button(label="📄 Download Professional PDF Report", data=st.session_state.pdf_data, file_name="RF_Link_Report.pdf", mime="application/pdf")
