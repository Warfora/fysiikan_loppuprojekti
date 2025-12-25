import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from math import radians, cos, sin, asin, sqrt
import folium
from streamlit_folium import st_folium

st.title("Fysiikan loppuprojekti")

# Askelmäärä ja suodatetun kiihtyvyysdatam kuvaaja 
st.write(f"Suodatetun kiihtyvyysdatam kuvaaja")

def butter_lowpass_filter(data, cutoff, nyq, order):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, nyq, order):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

df = pd.read_csv('assets/Linear Acceleration.csv')

data = df['Linear Acceleration y (m/s^2)']
T_tot = df['Time (s)'].max()
n = len(df['Time (s)'])

fs = n/T_tot
nyq = fs/2.0
order = 3
cutoff = 1/0.3

data_filt = butter_lowpass_filter(data, cutoff, nyq, order)

fig, ax = plt.subplots(figsize=(12,4))
ax.plot(df['Time (s)'], data, label='data')
ax.plot(df['Time (s)'], data_filt, label='suodatettu data')
ax.axis([0.5,T_tot,-10,10])
ax.grid()
ax.legend()
st.pyplot(fig)

jaksot = 0
for i in range(n-1):
    if data_filt[i]/data_filt[i+1] < 0:
        jaksot = jaksot + 1/2

st.write(f"Askelten määrä on: {jaksot}")

# Lasketaan nopeus ja matka 

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  
    return c * r

df_gps = pd.read_csv('assets/Location.csv')

df_gps['dist'] = np.zeros(len(df_gps)) 
df_gps['time_diff'] = np.zeros(len(df_gps))

for i in range(len(df_gps)-1):
    df_gps.loc[i,'dist'] = haversine(df_gps['Longitude (°)'][i], df_gps['Latitude (°)'][i], df_gps['Longitude (°)'][i+1], df_gps['Latitude (°)'][i+1])
    df_gps.loc[i,'time_diff'] = df_gps['Time (s)'][i+1] - df_gps['Time (s)'][i]

df_gps['velocity'] = df_gps['dist']/(df_gps['time_diff']/3600)
df_gps['tot_dist'] = np.cumsum(df_gps['dist'])

avg_speed = df_gps['velocity'].mean()
total_distance = df_gps['tot_dist'].iloc[-1]

st.write(f"Keskinopeus: {avg_speed:.4f} km/h")
st.write(f"Kokonaismatka: {total_distance:.4f} km")

# karttakuva 
st.write(f"Karttakuva")

start_lat = df_gps['Latitude (°)'].mean()
start_long = df_gps['Longitude (°)'].mean()
my_map = folium.Map(location=[start_lat, start_long], zoom_start=14)

folium.PolyLine(df_gps[['Latitude (°)', 'Longitude (°)']], color='blue', weight=3.5, opacity=1).add_to(my_map)

st_folium(my_map, width=700, height=500)