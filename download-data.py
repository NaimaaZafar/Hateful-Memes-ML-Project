import kagglehub
import shutil
import os

# Download dataset
path = kagglehub.dataset_download("marafey/hateful-memes-dataset")

# Get current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define destination directory inside script folder
dest_dir = os.path.join(script_dir, "hateful_memes")

# Copy dataset folder to script directory
if not os.path.exists(dest_dir):
    shutil.copytree(path, dest_dir)
    print(f"Dataset copied to: {dest_dir}")
else:
    print("Dataset already exists in the script directory.")

print("Path to dataset files:", dest_dir)
