📷 Biometric Face Attendance System
By: Abhishek Singh
Vellore Institute of Technology, Chennai
📝 Introduction
Welcome to the Biometric Face Attendance System, a smart attendance solution powered by real-time face recognition. Designed for institutional use, this system uses dual-camera entry/exit tracking to log attendance intelligently, accounting for short breaks like washroom visits.

Built using Python, OpenCV, and Streamlit, this system provides accurate logs, visualizations, and flexibility — making it ideal for smart classrooms or gated environments.



🚀 Features
🧠 Face Recognition Attendance – No manual punching, only camera-based identity.

🎥 Dual Camera Logic – One for entry, one for exit. Tracks real-time movement.

⏱️ Short Break Handling – Exit < 5 minutes? Not marked as a full absence.

📊 Dashboard View – Visualize daily/weekly/monthly attendance through Streamlit.

🗃️ Auto Logs & Embeddings – JSON-based embeddings and CSV logs.

🔌 Mobile Camera Support – Use any phone camera via DroidCam.




🛠️ Technologies Used
Python – Core logic and scripting

OpenCV – Real-time video processing

face_recognition – Face detection and encoding

Streamlit – Interactive dashboard

DroidCam – For wireless phone-based camera streaming

NumPy / Pandas – For data manipulation and logs

🖥️ Installation & Setup
🔹 Step 1: Clone the Repository
Visit: https://github.com/your-username/biometric-attendance
Then clone using Git or download the ZIP.

🔹 Step 2: Setup Environment
Create a virtual environment

Install dependencies from requirements.txt
(Includes: opencv-python, face-recognition, streamlit, numpy, pandas)

🔹 Step 3: Add Face Data
Place face images inside dataa/faces/

Create folders by person name, add clear images inside each

Example:
dataa/faces/Abhishek/1.jpg, 2.jpg
dataa/faces/Sneha/1.jpg, 2.jpg

🔹 Step 4: Generate Embeddings
Run the script: scripts/extract_embeddings.py
This creates embeddings.json used for recognition.

🔹 Step 5: Start Attendance System
Run: scripts/ml4.py
This connects to two cameras and logs entry and exit.

🔹 Step 6: View Attendance Dashboard
Run: scripts/streamlit2.py
Open the Streamlit dashboard in your browser.

📱 Setting Up Cameras with DroidCam
Use your Android phone as a camera for the system.

Install DroidCam from Google Play:
https://play.google.com/store/apps/details?id=com.dev47apps.droidcam

Install DroidCam Client for Windows:
https://droidcam-client.en.uptodown.com/windows

Connect both devices via Wi-Fi

Your PC will now detect them as cameras (e.g., camera 0 and 1)

📁 Folder Structure
dataa/faces/ – Face image folders for each person

scripts/ – Core scripts (embedding generation, attendance logic, dashboard)

attendance.csv – Logs attendance

embeddings.json – Stores face encodings

results/, logs/, models/ – Support directories

.gitignore – Configured to skip sensitive/generated files

requirements.txt – Dependency list

📌 How It Works
User’s face is detected on entry/exit.

If the same person exits and returns within 5 minutes, it is ignored.

Long absences are logged in attendance.csv.

Dashboard displays attendance stats visually.

All camera operations happen via OpenCV with real-time recognition.

📂 .gitignore Essentials
Be sure your .gitignore includes:

faces/

embeddings.json

pycache/, *.pyc

.ipynb_checkpoints/

venv/

.env/, .vscode/

🤝 Contributing
Want to improve the project? We'd love your help!

Fork this repository

Create a branch (example: feature-enhancement)

Commit your changes

Push and open a pull request

📬 Contact
For questions, suggestions, or collaborations:

📧 Email: abhi11.sbsm@gmail.com
🔗 GitHub: Biometric Attendance System

