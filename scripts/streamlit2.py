import streamlit as st
import pandas as pd
from datetime import datetime
import os

st.set_page_config(page_title="Attendance Dashboard", layout="wide")

# Paths to your CSV files
ENTRY_CSV = "dataa/entry_attendance.csv"
EXIT_CSV = "dataa/exit_attendance.csv"
FINAL_CSV = r"C:\Users\abhi1\Desktop\BIOMETRIC\scripts\dataa\final_attendance3.csv"

# ------------------- Load CSVs Safely -------------------
@st.cache_data(ttl=10)
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

# ------------------- Main UI -------------------
st.title("📊 Biometric Attendance Dashboard")
refresh_interval = st.sidebar.slider("⏱ Auto Refresh Interval (seconds)", 5, 60, 10)

# Force refresh if button clicked
if st.sidebar.button("🔄 Manual Refresh"):
    st.cache_data.clear()

# Auto-refresh using JS trick
st.markdown(
    f"""
    <script>
        function autoRefresh() {{
            window.location.reload();
        }}
        setTimeout(autoRefresh, {refresh_interval * 1000});
    </script>
    """,
    unsafe_allow_html=True,
)

# ------------------- Load Data -------------------
entry_df = load_csv(ENTRY_CSV)
exit_df = load_csv(EXIT_CSV)
final_df = load_csv(FINAL_CSV)

# ------------------- Stats Summary -------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🟢 Present", final_df[final_df["Status"] == "Present"].shape[0])
with col2:
    st.metric("🔴 Absent", final_df[final_df["Status"] == "Absent"].shape[0])
with col3:
    st.metric("🚻 On Break", exit_df.shape[0] - final_df[final_df["Status"] == "Absent"].shape[0])

# ------------------- DataTables -------------------
st.subheader("📅 Entry Logs")
st.dataframe(entry_df, use_container_width=True)

st.subheader("🚪 Exit Logs")
st.dataframe(exit_df, use_container_width=True)

st.subheader("✅ Final Attendance")
st.dataframe(final_df.sort_values(by="Timestamp", ascending=False), use_container_width=True)

# ------------------- Export Option -------------------
with st.expander("📁 Export Final Attendance"):
    st.download_button(
        label="Download Final Attendance CSV",
        data=final_df.to_csv(index=False).encode("utf-8"),
        file_name="final_attendance.csv",
        mime="text/csv"
    )
