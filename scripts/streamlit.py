# import streamlit as st
# import pandas as pd
# from datetime import datetime

# st.set_page_config(page_title="Smart Attendance Dashboard", layout="wide")

# st.title("📊 Smart Attendance Dashboard")
# st.caption("Real-time view of class attendance (updated from face recognition system)")

# # Auto-refresh every 10 seconds
# st_autorefresh = st.empty()
# st_autorefresh.text("Refreshing...")  # Display a message while refreshing

# # Load CSVs
# try:
#     entry_df = pd.read_csv("dataa/entry_attendance.csv")
#     exit_df = pd.read_csv("dataa/exit_attendance.csv")
#     final_df = pd.read_csv("dataa/attendance.csv")
# except Exception as e:
#     st.error(f"Error loading files: {e}")
#     st.stop()

# # Convert time
# entry_df["Timestamp"] = pd.to_datetime(entry_df["Timestamp"])
# exit_df["Timestamp"] = pd.to_datetime(exit_df["Timestamp"])

# # Students in class
# students_in = []
# students_out = []

# for name in entry_df["Name"].unique():
#     last_entry = entry_df[entry_df["Name"] == name]["Timestamp"].max()
#     exits = exit_df[exit_df["Name"] == name]
#     last_exit = exits["Timestamp"].max() if not exits.empty else None

#     if last_exit is None or last_exit < last_entry:
#         students_in.append(name)
#     else:
#         students_out.append(name)

# # Layout
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("✅ Currently in Class")
#     if students_in:
#         for name in students_in:
#             st.success(f"🟢 {name}")
#     else:
#         st.info("No one is currently inside.")

# with col2:
#     st.subheader("🛑 Currently Out")
#     if students_out:
#         for name in students_out:
#             st.error(f"🔴 {name}")
#     else:
#         st.info("No students outside.")

# st.divider()
# st.subheader("📋 Final Attendance (Based on 1-hour rule + ML check)")

# if final_df.empty:
#     st.warning("Final attendance has not been marked yet.")
# else:
#     for i, row in final_df.iterrows():
#         name = row['Name']
#         status = row['Status']
#         if status.lower() == "present":
#             st.success(f"✅ {name} — {status}")
#         elif status.lower() == "absent":
#             st.error(f"❌ {name} — {status}")
#         else:
#             st.warning(f"⚠️ {name} — {status}")

# # Refresh every 10 seconds
# st_autorefresh.empty()
# st_autorefresh = st.empty()



import streamlit as st
import pandas as pd
from datetime import datetime

# Set the page configuration
st.set_page_config(page_title="Smart Attendance Dashboard", layout="wide")

# Dashboard Title
st.title("📊 Smart Attendance Dashboard")
st.caption("Real-time view of class attendance (updated from face recognition system)")

# Auto-refresh message
st_autorefresh = st.empty()
st_autorefresh.text("Refreshing...")  # Display a message while refreshing

# Load CSVs (make sure these files exist)
try:
    entry_df = pd.read_csv("dataa/entry_attendance.csv")
    exit_df = pd.read_csv("dataa/exit_attendance.csv")
    final_df = pd.read_csv("dataa/attendance.csv")
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# Convert time columns to datetime
entry_df["Timestamp"] = pd.to_datetime(entry_df["Timestamp"])
exit_df["Timestamp"] = pd.to_datetime(exit_df["Timestamp"])

# Get students inside and outside the class
students_in = []
students_out = []

for name in entry_df["Name"].unique():
    last_entry = entry_df[entry_df["Name"] == name]["Timestamp"].max()
    exits = exit_df[exit_df["Name"] == name]
    last_exit = exits["Timestamp"].max() if not exits.empty else None

    # Logic to check if the student is inside or outside
    if last_exit is None or last_exit < last_entry:
        students_in.append(name)
    else:
        students_out.append(name)

# Display results in two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("✅ Currently in Class")
    if students_in:
        for name in students_in:
            st.success(f"🟢 {name}")
    else:
        st.info("No one is currently inside.")

with col2:
    st.subheader("🛑 Currently Out")
    if students_out:
        for name in students_out:
            st.error(f"🔴 {name}")
    else:
        st.info("No students outside.")

# Add divider for separating sections
st.divider()

# Display final attendance
st.subheader("📋 Final Attendance (Based on 1-hour rule + ML check)")

if final_df.empty:
    st.warning("Final attendance has not been marked yet.")
else:
    for i, row in final_df.iterrows():
        name = row['Name']
        status = row['Status']
        if status.lower() == "present":
            st.success(f"✅ {name} — {status}")
        elif status.lower() == "absent":
            st.error(f"❌ {name} — {status}")
        else:
            st.warning(f"⚠️ {name} — {status}")

# Refresh every 10 seconds
st_autorefresh.empty()
st_autorefresh = st.empty()
