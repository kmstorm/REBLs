import os
import shutil
import xnat
import zipfile
import sys

# Connect to XNAT
USER='username'
PASSWORD='password'

def download_xnat(data_folder):
    session = xnat.connect('http://localhost:xxxx', user=USER, password=PASSWORD)
    project = session.projects['REBL']

    # Download the data
    for i in range(len(project.subjects)):
        exp=project.subjects[i].experiments
        index = -1
        for j in range(len(exp)):
            SessionData = str(exp[j])
            index = j if SessionData.split()[0][1:3] == 'Mr' else index
            
        if index == -1:
            print(f"No MR data found for {project.subjects[i].data['label']}")
            continue

        exp[index].download(os.path.join(data_folder, f"{project.subjects[i].data['label']}.zip"))
        print(f"Downloaded {project.subjects[i].data['label']}.zip")

    # Disconnect from XNAT
    session.disconnect()
    return

# Unzip the data
def unzip_files(data_folder):
    for f in os.listdir(data_folder):
        if f.endswith(".zip"):
            zip_file = os.path.join(data_folder, f)
            output_folder = os.path.join(data_folder, os.path.splitext(f)[0])
        
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(output_folder)
            
            print(f"Unzipped {f}")
            # os.remove(zip_file)

# download_xnat("data")
# unzip_files("data")