import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import json
import pandas as pd
import math



# --- Function to plot map and bar chart ---
def plot_charts(change_df, color_scale, map_title, bar_title):
    plot_df = change_df.copy()
    # st.write(plot_df)
    geo_key = st.session_state["geo_unit_col"]
    map_col = "Change"
    bar_col = "Change"

    GEO_COL = st.session_state["GEO_COL"]
    gdf = st.session_state["gdf"]

    # Merge shapefile with change data
    plot_df = gdf.merge(plot_df, left_on=GEO_COL, right_on=geo_key, how="left").copy()
    plot_df = plot_df[plot_df["Change"].notnull()]

    num_bars = len(plot_df)
    # st.write(num_bars)
    # 1. Get the geographic bounds
    minx, miny, maxx, maxy = gdf.total_bounds
    # st.write(minx,miny,maxx,maxy)

    # 2. Calculate the area of the bounding box
    width = abs(maxx - minx)
    height = abs(maxy - miny)

    # st.write(minx, miny, maxx, maxy, width, height)
    factor_padding = 0.05

    if width < 1.2 and height < 1.2:
        factor_padding = 0.001

    minx *= (1 - factor_padding)
    miny *= (1 - factor_padding)
    maxx *= (1 + factor_padding)
    maxy *= (1 + factor_padding)



    if (width <= height):
            minx -= (height - width) / 2
            maxx += (height - width) / 2


    else:
        miny -= (width - height)/2

    width = abs(maxx - minx)
    height = abs(maxy - miny)


    if width/height>1.68:
        minx -= (height - width) / 2
        maxx += (height - width) / 2
    width = abs(maxx - minx)
    height = abs(maxy - miny)

    center_lon = (minx + maxx) / 2
    center_lat = (miny + maxy) / 2

    # --- Choropleth Map ---
    map_fig = px.choropleth_mapbox(
        plot_df,
        geojson=plot_df.__geo_interface__,
        locations=plot_df.index,
        color=map_col,
        hover_name=geo_key,
        hover_data={map_col: True, bar_col: True},
        mapbox_style="white-bg",  # mapbox_style="carto-positron",
        opacity=0.7,
        color_continuous_scale=color_scale,
        labels={map_col: map_col},
        center={"lat": center_lat, "lon": center_lon},
        zoom=0  # overridden by bounds
    )

    # Add text labels
    map_fig.add_trace(go.Scattermapbox(
        lon=plot_df.geometry.centroid.x,
        lat=plot_df.geometry.centroid.y,
        mode='text',
        text=plot_df.apply(lambda row: f"{row[geo_key]}<br>{row[map_col]}", axis=1),
        textfont=dict(size=8, color='black'),
        hoverinfo='none'
    ))

    # Force map to fit bounds with dynamic padding
    factor_padding = 0.0001
    map_fig.update_mapboxes(
        bounds={
            "west": float(minx),
            "east": float(maxx),
            "south": float(miny),
            "north": float(maxy)
            #     lat/3 is also good estimate for maxy
        }
    )

    # map_fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    map_fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )

    # --- Bar Chart ---
    bar_fig = px.bar(
        plot_df.sort_values(bar_col, ascending=True),
        x=bar_col, y=geo_key, orientation='h',
        title=bar_title,
        labels={geo_key: 'District', bar_col: bar_col},
        color=bar_col,
        color_continuous_scale=color_scale,
        height=max(400, num_bars * 25),
        text=plot_df[bar_col].round(2)
    )
    bar_fig.update_traces(
        texttemplate="%{text}",
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(size=10, color="black"),
        cliponaxis=False
    )
    bar_fig.update_coloraxes(showscale=False)

    bar_fig.update_layout(
        # height=max(400, len(plot_df) * 25),
        margin=dict(l=0, r=0, t=50, b=0),
        yaxis=dict(tickfont=dict(size=10)),
        # height=num_bars*30,
    )

    selected_state = st.session_state["selected_state"]
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"District Map of {selected_state.title()}")
        with st.container():
            st.plotly_chart(map_fig)
    with col2:
        st.subheader("District Data")
        with st.container(height=400):
            st.plotly_chart(bar_fig)




