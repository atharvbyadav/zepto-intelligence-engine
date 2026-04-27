# ==========================================================
# ⚡ ZEPTO INTELLIGENCE ENGINE (PURPLE THEME FINAL)
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
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Zepto Intelligence Engine",
    page_icon="⚡",
    layout="wide"
)

# -------------------------------
# PURPLE THEME + BRANDING
# -------------------------------
st.markdown("""
<style>
/* Background */
.main {
    background-color: #0E1117;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1A1C24;
}

/* Title */
.zepto-title {
    font-size: 42px;
    font-weight: 800;
    color: #6F2DBD;
}
.zepto-sub {
    font-size: 18px;
    color: #A78BFA;
    margin-bottom: 20px;
}

/* Metrics */
div[data-testid="stMetric"] {
    background-color: #1F1B2E;
    border: 1px solid #6F2DBD;
    border-radius: 10px;
    padding: 10px;
}

/* Buttons */
.stButton>button {
    background-color: #6F2DBD;
    color: white;
    border-radius: 8px;
}

/* Sliders */
.stSlider>div {
    color: #A78BFA;
}
</style>

<div class="zepto-title">⚡ Zepto Intelligence Engine</div>
<div class="zepto-sub">Hyperlocal 10-Minute Delivery Optimization System</div>
""", unsafe_allow_html=True)

# -------------------------------
# DATA GENERATION
# -------------------------------
@st.cache_data
def generate_data(n=150):
    np.random.seed(42)
    
    lat_center, lon_center = 18.5204, 73.8567
    
    data = []
    for _ in range(n):
        lat = lat_center + np.random.uniform(-0.08, 0.08)
        lon = lon_center + np.random.uniform(-0.08, 0.08)
        
        population = np.random.randint(20000, 120000)
        internet = np.random.uniform(0.6, 0.95)
        
        orders = population * internet * np.random.uniform(0.02, 0.08)
        
        data.append([lat, lon, population, internet, orders])
    
    return pd.DataFrame(data, columns=[
        "Latitude", "Longitude", "Population",
        "Internet", "Monthly_Orders"
    ])

df = generate_data()

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("⚙️ Zepto Controls")

avg_speed = st.sidebar.slider("Rider Speed (km/h)", 10, 40, 20)
traffic = st.sidebar.slider("Traffic Multiplier", 1.0, 2.5, 1.5)
aov = st.sidebar.slider("Avg Order Value (₹)", 100, 1000, 300)
margin = st.sidebar.slider("Margin %", 10, 40, 20) / 100
burn = st.sidebar.slider("Burn per Order (₹)", 10, 60, 25)
rider_eff = st.sidebar.slider("Orders per Rider/hr", 2, 10, 5)
k = st.sidebar.slider("Number of Zepto MFCs", 3, 12, 6)

# -------------------------------
# FEATURES
# -------------------------------
df["Daily_Orders"] = df["Monthly_Orders"] / 30
df["Order_Density"] = df["Monthly_Orders"] / 2
df["Night_Demand"] = df["Monthly_Orders"] * np.random.uniform(0.2, 0.4, len(df))
df["Repeat"] = np.random.uniform(1.5, 3.5, len(df))

scaler = MinMaxScaler()
df[["d1", "d2", "d3"]] = scaler.fit_transform(
    df[["Order_Density", "Night_Demand", "Repeat"]]
)

df["ZDSI"] = 0.4 * df["d1"] + 0.3 * df["d2"] + 0.3 * df["d3"]

# -------------------------------
# CLUSTERING
# -------------------------------
X = df[["Latitude", "Longitude"]]

kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
df["Cluster"] = kmeans.fit_predict(X, sample_weight=df["Monthly_Orders"])
centers = kmeans.cluster_centers_

# -------------------------------
# DELIVERY TIME (SAFE)
# -------------------------------
centers_df = pd.DataFrame(centers, columns=["c_lat", "c_lon"])
df = df.join(centers_df, on="Cluster")

df["Distance_km"] = np.sqrt(
    (df["Latitude"] - df["c_lat"])**2 +
    (df["Longitude"] - df["c_lon"])**2
) * 111

df["Delivery_Time"] = (df["Distance_km"] / avg_speed) * 60 * traffic
df["Within_SLA"] = df["Delivery_Time"] <= 10

# -------------------------------
# ECONOMICS
# -------------------------------
df["Revenue"] = df["Monthly_Orders"] * aov
df["Profit"] = df["Revenue"] * margin - df["Monthly_Orders"] * burn
df["Hourly_Orders"] = df["Daily_Orders"] / 24
df["Riders"] = df["Hourly_Orders"] / rider_eff

# -------------------------------
# TABS
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview", "🗺️ SLA Map", "🧠 Optimization", "📈 Forecast"
])

# -------------------------------
# TAB 1
# -------------------------------
with tab1:
    st.markdown("## 📊 Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("📦 Orders", f"{int(df['Monthly_Orders'].sum()):,}")
    col2.metric("⚡ SLA Coverage", f"{df['Within_SLA'].mean()*100:.1f}%")
    col3.metric("💰 Profit", f"₹{int(df['Profit'].sum()):,}")
    col4.metric("🏍 Riders", f"{int(df['Riders'].sum()):,}")

    fig = px.scatter(
        df,
        x="Population",
        y="Monthly_Orders",
        color="ZDSI",
        size="Monthly_Orders",
        title="Demand vs Population"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# TAB 2
# -------------------------------
with tab2:
    m = folium.Map(location=[18.52, 73.85], zoom_start=12)

    for _, r in df.iterrows():
        color = "green" if r["Within_SLA"] else "red"

        folium.CircleMarker(
            location=[r["Latitude"], r["Longitude"]],
            radius=6,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=f"""
            Orders: {int(r['Monthly_Orders'])}<br>
            Time: {r['Delivery_Time']:.1f} min
            """
        ).add_to(m)

    for c in centers:
        folium.Marker(
            location=[c[0], c[1]],
            icon=folium.Icon(color="blue", icon="cube")
        ).add_to(m)

    folium_static(m)

# -------------------------------
# TAB 3
# -------------------------------
with tab3:
    cluster_stats = df.groupby("Cluster").agg({
        "Monthly_Orders": "sum",
        "Revenue": "sum",
        "Profit": "sum",
        "Riders": "sum"
    }).reset_index()

    st.dataframe(cluster_stats)

    fig = px.bar(cluster_stats, x="Cluster", y="Profit")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# TAB 4
# -------------------------------
with tab4:
    base = df["Daily_Orders"].sum()

    days = 120
    t = np.arange(days)

    trend = base * (1 + 0.001 * t)
    season = base * 0.2 * np.sin(2 * np.pi * t / 7)
    noise = np.random.normal(0, base * 0.1, days)

    demand = trend + season + noise

    X = t.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, demand)

    pred = model.predict(X)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=demand, name="Actual"))
    fig.add_trace(go.Scatter(y=pred, name="Forecast"))

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# INSIGHTS
# -------------------------------
st.markdown("## 🧠 Key Insights")

sla = df["Within_SLA"].mean() * 100

if sla < 70:
    st.warning("⚠️ Low SLA coverage — increase MFC count or speed.")
else:
    st.success("✅ Strong SLA coverage.")

if df["Profit"].sum() < 0:
    st.error("💸 Network is burning cash.")
else:
    st.success("💰 Network is profitable.")
