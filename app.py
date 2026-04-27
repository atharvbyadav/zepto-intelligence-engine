# ==========================================================
# 🚀 ZEPTO INTELLIGENCE ENGINE (ALL-IN-ONE)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import folium
from streamlit_folium import folium_static

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Zepto Intelligence Engine", layout="wide")

st.title("⚡ Zepto Hyperlocal Network Intelligence")

# -------------------------------
# SYNTHETIC DATA GENERATION (SMART)
# -------------------------------
@st.cache_data
def generate_data(n=150):
    np.random.seed(42)
    
    lat_center, lon_center = 18.5204, 73.8567  # Pune
    
    data = []
    for i in range(n):
        lat = lat_center + np.random.uniform(-0.08, 0.08)
        lon = lon_center + np.random.uniform(-0.08, 0.08)
        
        population = np.random.randint(20000, 120000)
        internet = np.random.uniform(0.6, 0.95)
        
        orders = population * internet * np.random.uniform(0.02, 0.08)
        
        data.append([lat, lon, population, internet, orders])
    
    df = pd.DataFrame(data, columns=[
        "Latitude", "Longitude", "Population",
        "Internet", "Monthly_Orders"
    ])
    
    return df

df = generate_data()

# -------------------------------
# SIDEBAR CONTROLS
# -------------------------------
st.sidebar.header("⚙️ Zepto Controls")

avg_speed = st.sidebar.slider("Rider Speed (km/h)", 10, 40, 20)
traffic = st.sidebar.slider("Traffic Multiplier", 1.0, 2.5, 1.5)
aov = st.sidebar.slider("Avg Order Value (₹)", 100, 1000, 300)
margin = st.sidebar.slider("Margin %", 10, 40, 20)/100
burn = st.sidebar.slider("Burn per Order (₹)", 10, 60, 25)
rider_eff = st.sidebar.slider("Orders per Rider/hr", 2, 10, 5)

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
df["Daily_Orders"] = df["Monthly_Orders"]/30

# Zepto-style demand signals
df["Order_Density"] = df["Monthly_Orders"] / 2
df["Night_Demand"] = df["Monthly_Orders"] * np.random.uniform(0.2,0.4,len(df))
df["Repeat"] = np.random.uniform(1.5,3.5,len(df))

# ZDSI SCORE
scaler = MinMaxScaler()
df[["d1","d2","d3"]] = scaler.fit_transform(
    df[["Order_Density","Night_Demand","Repeat"]]
)

df["ZDSI"] = 0.4*df["d1"] + 0.3*df["d2"] + 0.3*df["d3"]

# -------------------------------
# CLUSTERING (STORE PLACEMENT)
# -------------------------------
k = st.sidebar.slider("Number of Zepto MFCs", 3, 12, 6)

X = df[["Latitude","Longitude"]]

kmeans = KMeans(n_clusters=k, n_init=10)
df["Cluster"] = kmeans.fit_predict(X, sample_weight=df["Monthly_Orders"])

centers = kmeans.cluster_centers_

# -------------------------------
# DELIVERY TIME MODEL
# -------------------------------
def calc_time(lat1, lon1, lat2, lon2):
    dist = np.sqrt((lat1-lat2)**2 + (lon1-lon2)**2) * 111
    return (dist/avg_speed)*60*traffic

times = []
for i, row in df.iterrows():
    c = centers[row["Cluster"]]
    t = calc_time(row["Latitude"], row["Longitude"], c[0], c[1])
    times.append(t)

df["Delivery_Time"] = times

# SLA FILTER
df["Within_SLA"] = df["Delivery_Time"] <= 10

# -------------------------------
# ECONOMICS
# -------------------------------
df["Revenue"] = df["Monthly_Orders"] * aov
df["Profit"] = df["Revenue"]*margin - df["Monthly_Orders"]*burn

df["Hourly_Orders"] = df["Daily_Orders"]/24
df["Riders"] = df["Hourly_Orders"]/rider_eff

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview","🗺️ SLA Map","🧠 Optimization","📈 Forecast"
])

# -------------------------------
# TAB 1
# -------------------------------
with tab1:
    st.metric("Total Orders", int(df["Monthly_Orders"].sum()))
    st.metric("SLA Coverage %", round(df["Within_SLA"].mean()*100,2))
    st.metric("Total Profit ₹", int(df["Profit"].sum()))
    
    fig = px.scatter(df,
        x="Population",
        y="Monthly_Orders",
        color="ZDSI",
        size="Monthly_Orders",
        title="Demand vs Population"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# TAB 2 (MAP)
# -------------------------------
with tab2:
    m = folium.Map(location=[18.52,73.85], zoom_start=12)
    
    for _, r in df.iterrows():
        color = "green" if r["Within_SLA"] else "red"
        folium.CircleMarker(
            location=[r["Latitude"], r["Longitude"]],
            radius=5,
            color=color,
            fill=True
        ).add_to(m)
    
    for c in centers:
        folium.Marker(
            location=[c[0], c[1]],
            icon=folium.Icon(color="blue", icon="cube")
        ).add_to(m)
    
    folium_static(m)

# -------------------------------
# TAB 3 (OPTIMIZATION)
# -------------------------------
with tab3:
    cluster_stats = df.groupby("Cluster").agg({
        "Monthly_Orders":"sum",
        "Revenue":"sum",
        "Profit":"sum",
        "Riders":"sum"
    }).reset_index()
    
    st.dataframe(cluster_stats)
    
    fig = px.bar(cluster_stats,
        x="Cluster",
        y="Profit",
        title="Profit per MFC"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# TAB 4 (FORECAST)
# -------------------------------
with tab4:
    base = df["Daily_Orders"].sum()
    
    days = 120
    t = np.arange(days)
    
    trend = base*(1+0.001*t)
    season = base*0.2*np.sin(2*np.pi*t/7)
    noise = np.random.normal(0, base*0.1, days)
    
    demand = trend + season + noise
    
    X = t.reshape(-1,1)
    model = LinearRegression()
    model.fit(X, demand)
    
    pred = model.predict(X)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=demand, name="Actual"))
    fig.add_trace(go.Scatter(y=pred, name="Forecast"))
    
    st.plotly_chart(fig, use_container_width=True)
