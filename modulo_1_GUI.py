# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 16:08:54 2023

@author: ferra
"""

import tkinter as tk
from tkinter import filedialog
from tkinter import PhotoImage
import os
import glob
import pandas as pd
import numpy as np
import rasterio
import time


from PIL import Image, ImageTk
# Global list to hold image references
global_images = []

def browse_train_val_csv():
    file_path = filedialog.askopenfilename(initialdir="/", title="Select Training Set CSV",
                                           filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    train_val_CSV_path_var.set(file_path)

def browse_s2_folder():
    folder_path = filedialog.askdirectory(initialdir="/", title="Select Sentinel-2 Folder")
    S2_folder_path_var.set(folder_path)

def browse_output_folder():
    folder_path = filedialog.askdirectory(initialdir="/", title="Select Output Folder")
    output_folder_path_var.set(folder_path)

def run_script(band_vars):
    output_folder_path = output_folder_path_var.get()
    train_val_CSV_path = train_val_CSV_path_var.get()
    S2_folder_path = S2_folder_path_var.get()

    selected_bands = [i for i, var in enumerate(band_vars, start=1) if var.get()]

    start_time = time.time()

    # Load the csv file into a pandas dataframe
    df = pd.read_csv(train_val_CSV_path)

        # Initialize an empty list to store the pixel values
    pixel_values = []
    class_labels_list = []
    
    # Loop over each sub-directory in the main directory
    for subdir in os.listdir(S2_folder_path):
        if os.path.isdir(os.path.join(S2_folder_path, subdir)):
            # Create a list of image paths in the current sub-directory
            image_paths = sorted(glob.glob(os.path.join(S2_folder_path, subdir) + "/*.tif"))
    
            # Get the geographic extent of the current sub-directory's tile
            with rasterio.open(image_paths[0]) as src:
                left, bottom, right, top = src.bounds
    
            # Filter the rows in the CSV file that correspond to the current sub-directory's tile
            sub_df = df[(df['X'] >= left) & (df['X'] <= right) & (df['Y'] >= bottom) & (df['Y'] <= top)]
            if sub_df.empty:
                continue
    
            # List to store the pixel values from each image
            image_pixel_values = []

            # Loop over all the images in the current sub-directory
            for image_path in image_paths:
                with rasterio.open(image_path) as src:
                    # Extract the pixel values for the given geographic coordinates for selected bands
                    selected_band_vals = [x for x in src.sample(zip(sub_df['X'].values, sub_df['Y'].values), indexes=selected_bands)]
                    # Convert the extracted pixel values to a Numpy array
                    selected_band_vals = np.array(selected_band_vals)

                    # Separate the SCL band (11th band) from other bands
                    scl_pixel_values = selected_band_vals[:, 10] #(since Python indexing starts at 0)
                    other_band_pixel_values = np.hstack([selected_band_vals[:, :10], selected_band_vals[:, 11:]])
    
                    # Stack the Numpy array into the list
                    image_pixel_values.append(other_band_pixel_values)
    
            # Create a mask (boolean array) to keep only non-cloudy pixels
            non_cloudy_mask = np.logical_and(scl_pixel_values != 3, scl_pixel_values != 8, scl_pixel_values != 9)
    
            # Apply the non-cloudy mask to the pixel values from other bands
            image_pixel_values = [vals[non_cloudy_mask] for vals in image_pixel_values]
            image_pixel_values = np.stack(image_pixel_values, axis=1)
    
            # Apply the non-cloudy mask to the class labels
            class_labels = sub_df['class'].values[non_cloudy_mask]
    
            # Append the current sub-directory's pixel values to the list
            pixel_values.append(image_pixel_values.reshape(image_pixel_values.shape[0], -1))
    
            # Append the class labels corresponding to the pixel values
            class_labels_list.append(class_labels)

    # Concatenate the list of Numpy arrays to create the final X array
    X = np.concatenate(pixel_values, axis=0)

    # Concatenate the list of class labels to create the final y array
    y = np.concatenate(class_labels_list, axis=0)

    # Print the shapes of X and y to verify that they have the same number of rows as the input CSV
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # Save the feature matrix and class labels as CSV files
    X_df = pd.DataFrame(X)
    X_df.to_csv(os.path.join(output_folder_path, "X.csv"), index=False)
    y_df = pd.DataFrame(y, columns=['Class'])
    y_df.to_csv(os.path.join(output_folder_path, "y.csv"), index=False)
   
    end_time = time.time()

    # Calculate the execution time
    execution_time = end_time - start_time
    
    # Generate a report file
    report_path = os.path.join(output_folder_path, "Module1_report_esecuzione.txt")
    with open(report_path, 'w') as report_file:
        report_file.write("Input files:\n")
        report_file.write(f"  - Training and validation CSV file: {train_val_CSV_path}\n")
        report_file.write(f"  - Sentinel-2 folder: {S2_folder_path}\n")
        report_file.write("\nOutput files:\n")
        report_file.write(f"  - X feature matrix: {os.path.join(output_folder_path, 'X.csv')}, size: {X.shape}\n")
        report_file.write(f"  - y class labels: {os.path.join(output_folder_path, 'y.csv')}, size: {y.shape}\n")
        report_file.write(f"\nNumber of features in X: {X.shape[1]}\n")
        # Calculate the class counts and percentages in the training/validation dataset
        class_counts = df['class'].value_counts(normalize=True)
        class_percentages = class_counts * 100
        # Write the class counts and percentages to the report file
        report_file.write("\nPoint class percentages in training dataset:\n")
        for class_name, class_percentage in class_percentages.items():
            report_file.write(f"  - {class_name}: {class_percentage:.2f}%\n")
        report_file.write(f"\nExecution time: {execution_time:.2f} seconds\n")




# Create the main window
root = tk.Tk()
root.title("Modulo-1")
root.geometry("700x700")  # Set the window size

# Try to load logo image, if not present skip
try:
    # Load the image using Pillow
    logo_image = Image.open("G:/codici_RF/codici/4EOSIAL/eosial_logo.png")
    logo_image = ImageTk.PhotoImage(logo_image)
    
    # Add the image to the global list to prevent garbage collection
    global_images.append(logo_image)

    # Create a label with the image
    logo_label = tk.Label(root, image=logo_image)
    logo_label.image = logo_image  # Keep a reference to prevent garbage collection
    logo_label.grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 0))

    # Add a label beside the image
    title_text = "Multitemporal\nSupervised Classifier\nfor Large-Scale AOIs"
    title_label = tk.Label(root, text=title_text, font=("Helvetica", 18, "bold"), anchor="w", justify="center", wraplength=300)
    title_label.grid(row=0, column=2, padx=(10, 0), pady=(10, 0))

except Exception as e:  # Catch all exceptions to see any potential issues
    print("Error loading the logo image:", e)

# Add a label beside the image
title_text = "Multitemporal\nSupervised Classifier\nfor Large-Scale AOIs"
title_label = tk.Label(root, text=title_text, font=("Helvetica", 18, "bold"), anchor="w", justify="center", wraplength=300)
title_label.grid(row=0, column=2, padx=(10, 0), pady=(10, 0))


# Add a description label
title = "Module-1"
title_label = tk.Label(root, text=title, font=("Helvetica", 14, "bold"))
title_label.grid(row=1, column=0, columnspan=3, padx=10, pady=(10, 0))

description1 = "Please select the input paths for Training Set CSV, Sentinel-2 Folder, and the Output Folder."
description1_label = tk.Label(root, text=description1, wraplength=550, justify='left')
description1_label.grid(row=2, column=0, columnspan=3, padx=10, pady=(0, 5))

description2 = "(note that the Sentinel-2 folder contains subfolders of multitemporal acquisitions for each adjacent tile)"
description2_label = tk.Label(root, text=description2, wraplength=550, justify='left')
description2_label.grid(row=3, column=0, columnspan=3, padx=10, pady=(0, 10))


    
# Add checkboxes for selecting bands
bands_no =13
band_vars = [tk.BooleanVar() for _ in range(bands_no)]  # Replace 12 with the number of bands
band_checkboxes = []
bands_label = tk.Label(root, text="Select Bands:")
bands_label.grid(row=4, column=0, sticky='e', padx=10, pady=5)

# Define the list of band names
band_names = ['ch.1 [B2]', 'ch.2 [B3]', 'ch.3 [B4]', 'ch.4 [B8]', 'ch.5 [B5]', 'ch.6 [B6]', 'ch.7 [B7]', 'ch.8 [B8A]', 'ch.9 [B11]', 'ch.10 [B12]', 'ch.11 [SCL] (mandatory for cloudy points filter)', 'ch.12 [VH] (Sentinel-1)', 'ch.13 [VV] (Sentinel-1)']

for i, band_name in enumerate(band_names):
    checkbox = tk.Checkbutton(root, text=f"{band_name}", variable=band_vars[i])
    checkbox.grid(row=4 + i // 2, column=i % 2 + 1, padx=10, pady=5, sticky='w')
    band_checkboxes.append(checkbox)

# activate the following instead if you want to name automatically the checkboxes
# for i in range(bands_no):  # Replace 12 with the number of bands
#     checkbox = tk.Checkbutton(root, text=f"Band {i + 1}", variable=band_vars[i])
#     checkbox.grid(row=4 + i // 2, column=i % 2 + 1, padx=10, pady=5, sticky='w')
#     band_checkboxes.append(checkbox)

# Add "Select All" checkbox
select_all_var = tk.BooleanVar()
select_all_checkbox = tk.Checkbutton(root, text="Select All", variable=select_all_var)
select_all_checkbox.grid(row=5, column=0, sticky='e', padx=10, pady=5)

def select_all_handler(*args):
    select_all = select_all_var.get()
    for var in band_vars:
        var.set(select_all)

select_all_var.trace("w", select_all_handler)

# # Add spacer rows
# spacer1 = tk.Label(root, text="")
# spacer1.grid(row=4, column=0)
# spacer2 = tk.Label(root, text="")
# spacer2.grid(row=6, column=0)
# spacer3 = tk.Label(root, text="")
# spacer3.grid(row=8, column=0)
# spacer3 = tk.Label(root, text="")
# spacer3.grid(row=10, column=0)

# Create input fields and buttons for each input path
train_val_csv_label = tk.Label(root, text="Training Set CSV:")
train_val_csv_label.grid(row=11, column=0, sticky='e')
train_val_CSV_path_var = tk.StringVar()
train_val_csv_entry = tk.Entry(root, textvariable=train_val_CSV_path_var)
train_val_csv_entry.grid(row=11, column=1)
train_val_csv_button = tk.Button(root, text="Browse", command=browse_train_val_csv)
train_val_csv_button.grid(row=11, column=2)

s2_folder_label = tk.Label(root, text="Sentinel-2 Folder:")
s2_folder_label.grid(row=13, column=0, sticky='e')
S2_folder_path_var = tk.StringVar()
s2_folder_entry = tk.Entry(root, textvariable=S2_folder_path_var)
s2_folder_entry.grid(row=13, column=1)
s2_folder_button = tk.Button(root, text="Browse", command=browse_s2_folder)
s2_folder_button.grid(row=13, column=2)

output_folder_label = tk.Label(root, text="Output Folder:")
output_folder_label.grid(row=15, column=0, sticky='e')
output_folder_path_var = tk.StringVar()
output_folder_entry = tk.Entry(root, textvariable=output_folder_path_var)
output_folder_entry.grid(row=15, column=1)
output_folder_button = tk.Button(root, text="Browse", command=browse_output_folder)
output_folder_button.grid(row=15, column=2)




# Create the "Run" button
run_button = tk.Button(root, text="Run", command=lambda: run_script(band_vars))
run_button.grid(row=17, column=1, pady=10, sticky='nsew')


# Add the end title
end_title = "Developed by Alvise Ferrari for EOSIAL (Scola di Ingegneria Aerospaziale - La Sapienza)"
end_label = tk.Label(root, text=end_title, font=("Helvetica", 8))
end_label.grid(row=18, column=0, columnspan=3, padx=10, pady=(10, 0))


# Run the main loop
root.mainloop()



