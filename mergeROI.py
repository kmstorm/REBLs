import os
import shutil
import numpy as np
import pandas as pd
import nibabel as nib

path = 'your-path'

count = {
	"Name": [],
	"Number of lesion": []
}

def check_shape(data_arrays):
	shape = None
	for data in data_arrays:
		if shape is None: 
			shape = data.shape
		else:
			if data.shape != shape:
				return False
	return True

def mergeRT(patient_name, file_paths, output_path, delete = False):
	data_arrays = []
	shape = None
	for path in file_paths:
		img = nib.load(path)
		data = np.array(img.dataobj)
		data_arrays.append(data)

	if len(data_arrays)==0:
		print(f"{patient_name} don't have NifTi RTSTRUCT")
		return

	if not(check_shape(data_arrays)):
		print(f"{patient_name} having not valid RTSTRUCT")
		return

	intersection = np.maximum.reduce(data_arrays)

	ni_img = nib.Nifti1Image(intersection, affine = img.affine)
	if delete:
		for path in file_paths:
			os.remove(path)

	nib.save(ni_img,os.path.join(output_path,f"RTSTRUCT {patient_name}.nii"))
	print(f"{patient_name} proceed sucessful")

patients = os.listdir(path)
patients.sort()

for patient in patients:
	file_paths = []
	patient_path = os.path.join(path,patient)
	if (os.path.isdir(patient_path)):
		for file in os.listdir(patient_path):
				if file.startswith("RTSTRUCT") and (file.endswith(".nii") or file.endswith(".nii.gz")):
					filepath = os.path.join(patient_path, file)
					file_paths.append(filepath)
		
		count['Name'].append(patient)
		count['Number of lesion'].append(len(file_paths))
		mergeRT(patient, file_paths, patient_path, delete = True)
	else: 
		print(f"Skipping non-directory entry: {patient}")  # Informative message

# df = pd.DataFrame(count)
# df.to_csv('count.csv', index = False)