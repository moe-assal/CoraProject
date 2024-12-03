import threading
import CORA_UI
import subprocess

# Path to your main.py file
main_path = r"C:\\Users\\mersh\\OneDrive\\Desktop\\CoraProject\\main.py"

# Function to run CORA_UI.main
def run_cora_ui():
    CORA_UI.main()

# Function to run main.py using subprocess
def run_main_py():
    subprocess.run(["python", main_path])

# Create threads
cora_ui_thread = threading.Thread(target=run_cora_ui, name="CORA_UI_Thread")
main_py_thread = threading.Thread(target=run_main_py, name="Main_Py_Thread")

# Start threads
cora_ui_thread.start()
main_py_thread.start()
