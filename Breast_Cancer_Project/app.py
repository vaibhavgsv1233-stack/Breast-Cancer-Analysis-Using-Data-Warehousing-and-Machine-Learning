
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(
    page_title="Breast Cancer Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Database location
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "04_Data_Warehouse" / "breast_cancer_warehouse.db"

st.title("Breast Cancer Data Warehouse Dashboard")
st.caption("Data Warehousing and Data Mining Project")

# Load warehouse data
@st.cache_data
def load_data():
    with sqlite3.connect(DB_PATH) as conn:
        query = """
        SELECT
            f.Fact_Key,
            f.Patient_Count,
            d.Class,
            t.Tumor_Size,
            t.Inv_Nodes,
            t.Deg_Malig,
            t.Node_Caps,
            c.Irradiat,
            p.Age,
            p.Breast
        FROM Fact_Breast_Cancer f
        JOIN Dim_Diagnosis d ON f.Diagnosis_Key = d.Diagnosis_Key
        JOIN Dim_Tumor t ON f.Tumor_Key = t.Tumor_Key
        JOIN Dim_Clinical c ON f.Clinical_Key = c.Clinical_Key
        JOIN Dim_Patient p ON f.Patient_Key = p.Patient_Key
        """
        return pd.read_sql_query(query, conn)

try:
    df = load_data()
except Exception as error:
    st.error(f"Could not load the warehouse: {error}")
    st.stop()

# Sidebar filters
st.sidebar.header("Dashboard Filters")

class_choice = st.sidebar.selectbox(
    "Diagnosis Class",
    ["All Classes", 0, 1]
)

min_tumor = int(df["Tumor_Size"].min())
max_tumor = int(df["Tumor_Size"].max())

tumor_range = st.sidebar.slider(
    "Tumor Size Range",
    min_value=min_tumor,
    max_value=max_tumor,
    value=(min_tumor, max_tumor)
)

malig_options = sorted(df["Deg_Malig"].dropna().unique().tolist())

malig_choice = st.sidebar.multiselect(
    "Malignancy Grade",
    options=malig_options,
    default=malig_options
)

# Apply filters
filtered = df.copy()

if class_choice != "All Classes":
    filtered = filtered[filtered["Class"] == class_choice]

filtered = filtered[
    filtered["Tumor_Size"].between(tumor_range[0], tumor_range[1])
]

filtered = filtered[filtered["Deg_Malig"].isin(malig_choice)]

# KPI metrics
st.subheader("Key Performance Indicators")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Matching Records", len(filtered))
col2.metric("Class 0", int((filtered["Class"] == 0).sum()))
col3.metric("Class 1", int((filtered["Class"] == 1).sum()))

if not filtered.empty:
    col4.metric("Avg Tumor Size", f"{filtered['Tumor_Size'].mean():.2f}")
    col5.metric("Avg Involved Nodes", f"{filtered['Inv_Nodes'].mean():.2f}")
else:
    col4.metric("Avg Tumor Size", "N/A")
    col5.metric("Avg Involved Nodes", "N/A")

st.divider()

if filtered.empty:
    st.warning("No records match the selected filters.")
else:
    # Charts
    left, right = st.columns(2)

    with left:
        st.subheader("Diagnosis Class Distribution")
        class_counts = filtered["Class"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        class_counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("Diagnosis Class")
        ax.set_ylabel("Number of Records")
        ax.tick_params(axis="x", rotation=0)
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Tumor Size Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(filtered["Tumor_Size"], bins=10)
        ax.set_xlabel("Tumor Size")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        plt.close(fig)

    left, right = st.columns(2)

    with left:
        st.subheader("Malignancy Grade Distribution")
        malig_counts = filtered["Deg_Malig"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        malig_counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("Malignancy Grade")
        ax.set_ylabel("Number of Records")
        ax.tick_params(axis="x", rotation=0)
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Average Involved Nodes by Class")
        avg_nodes = filtered.groupby("Class")["Inv_Nodes"].mean()
        fig, ax = plt.subplots(figsize=(6, 4))
        avg_nodes.plot(kind="bar", ax=ax)
        ax.set_xlabel("Diagnosis Class")
        ax.set_ylabel("Average Involved Nodes")
        ax.tick_params(axis="x", rotation=0)
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("View filtered records"):
        st.dataframe(filtered, use_container_width=True)

st.caption("Academic analytics demonstration — not a clinical diagnostic tool.")
