import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
from sklearn.neighbors import KDTree

df = pd.read_csv("accident_data.csv")
school_zone = pd.read_csv("school_zone.csv")

df = df.dropna(subset=["위도", "경도"])
school_zone = school_zone.dropna(subset=["위도", "경도"])

accident_coords = df[["위도", "경도"]].values
school_coords = school_zone[["위도", "경도"]].values

tree = KDTree(school_coords, leaf_size=2)
distances, indices = tree.query(accident_coords, k=1)

within_100m = distances.flatten() <= 0.001

df_within = df[within_100m]
df_outside = df[~within_100m]

m = folium.Map(location=[df["위도"].mean(), df["경도"].mean()], zoom_start=12)

marker_cluster = MarkerCluster().add_to(m)
for lat, lon in zip(df["위도"], df["경도"]):
    folium.Marker(location=[lat, lon], icon=folium.Icon(color='red')).add_to(marker_cluster)

for lat, lon in zip(school_zone["위도"], school_zone["경도"]):
    folium.CircleMarker(location=[lat, lon], radius=6, color='blue', fill=True).add_to(m)

m.save("accident_school_zone_map.html")
