import os
import shutil

# The folder where the .tif images are currently (wrong folder)
current_folder = r"C:\Users\SAHIL\Desktop\Land Usage AI\SEN-2 LULC"

# The folder where the .tif images should be moved (correct folder)
correct_folder = r"C:\Users\SAHIL\Desktop\Land Usage AI\SEN-2 LULC\train_masks"

# Make sure the correct destination exists
os.makedirs(correct_folder, exist_ok=True)

file_count = 0
duplicate_count = 0

# Walk through the entire current_folder and subfolders
for root, dirs, files in os.walk(current_folder):
    for file in files:
        if file.lower().endswith('.tif'):  # Only process .tif files
            source_path = os.path.join(root, file)
            dest_path = os.path.join(correct_folder, file)

            # If a file with the same name exists in the correct folder, rename it
            if os.path.exists(dest_path):
                base, ext = os.path.splitext(file)
                i = 1
                while os.path.exists(dest_path):
                    dest_path = os.path.join(correct_folder, f"{base}_{i}{ext}")
                    i += 1
                duplicate_count += 1

            shutil.move(source_path, dest_path)
            file_count += 1

print(f"✅ Moved {file_count} .tif files successfully.")
print(f"⚠️ Renamed {duplicate_count} duplicate files to avoid overwriting.")
