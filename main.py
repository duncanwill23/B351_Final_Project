import kagglehub
import shutil
import os

# Download latest version of the dataset
path = kagglehub.dataset_download("rounakbanik/the-movies-dataset")

# Create 'dataset' directory if it doesn't exist
dataset_folder = os.path.join(os.getcwd(), 'dataset')
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

# Extract CSV files to the "dataset" folder
for file_name in os.listdir(path):
    if file_name.endswith(".csv"):
        # Construct full file path
        source_file = os.path.join(path, file_name)
        destination_file = os.path.join(dataset_folder, file_name)
        shutil.copy(source_file, destination_file)

print(f"CSV files have been copied to the {dataset_folder} folder.")
