import csv
from datetime import datetime

def log_attendance(name):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_path = "C:\Users\abhi1\Desktop\BIOMETRIC\scripts\dataa\attendance.csv"
    
    try:
        with open(file_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            
            # Write header only if file is empty
            if file.tell() == 0:
                writer.writerow(["Name", "Timestamp"])
            
            writer.writerow([name, timestamp])
        
        print(f"✅ Logged: {name} at {timestamp}")
    
    except Exception as e:
        print(f"❌ Error writing to CSV: {e}")
