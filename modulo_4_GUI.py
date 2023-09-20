# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 16:58:48 2023

@author: ferra
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 16:45:54 2023

@author: ferra
"""


import os
import joblib
import pandas as pd
import numpy as np
import os
import rasterio
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pyproj import CRS, Transformer
import matplotlib.pyplot as plt
import glob
import pandas as pd
from tqdm import tqdm
#import multiprocessing
import gc
import time
from data_processing import process_batch
from parallel_processing import parallel_process_batch
from joblib import cpu_count

from PIL import Image, ImageTk
# Global list to hold image references
global_images = []

def load_trained_model(model_file_path):
    return joblib.load(model_file_path)

def preprocess_images(S2_image_paths, channel_indices):
    S2_image_list = []
    for i in range(len(S2_image_paths)):
        with rasterio.open(S2_image_paths[i]) as src:
            S2_image = src.read()

        # Select desired channels
        selected_channels = S2_image[channel_indices, :, :]

        n_bands, rows, cols = selected_channels.shape
        S2_image_list.append(np.transpose(selected_channels, (1, 2, 0)).reshape(-1, n_bands))
        # Close the rasterio file handle
        src.close()

    X_predict = np.concatenate(S2_image_list, axis=1)

    # Clean up memory
    del S2_image
    del S2_image_list

    return X_predict, rows, cols

def predict_class_labels(X_predict, clf, batch_size, n_cores):
    y_predict_proba = np.zeros((X_predict.shape[0], clf.n_classes_), dtype=np.float32)
    for i in tqdm(range(0, X_predict.shape[0], batch_size * n_cores)):
        j = min(i + batch_size * n_cores, X_predict.shape[0])
        X_batch_list = np.array_split(X_predict[i:j], n_cores)
        y_batch_list = parallel_process_batch(X_batch_list, clf, n_cores)
        y_batch = np.concatenate(y_batch_list)
        y_predict_proba[i:j] = y_batch

    return y_predict_proba

# def reshape_prediction(y_predict_proba, clf, pred_image):
#     y_predict_proba_reshaped = np.zeros((clf.n_classes_, pred_image.shape[0], pred_image.shape[1]), dtype=np.float32)
#     for i in range(clf.n_classes_):
#         y_predict_proba_reshaped[i, :, :] = y_predict_proba[:, i].reshape(pred_image.shape)
#     return y_predict_proba_reshaped

# def write_prediction_maps(output_folder_path, subdir, profile_proba, y_predict_proba_reshaped, profile_class, pred_image):
#     # Define the output file names for the predicted probability map and predicted class map
#     output_proba_filename = os.path.join(output_folder_path, subdir, "{}_predicted_proba.tif".format(subdir))
#     output_class_filename = os.path.join(output_folder_path, subdir, "{}_predicted_class.tif".format(subdir))

#     # Write the predicted probability map to a new GeoTIFF file
#     with rasterio.open(output_proba_filename, 'w', **profile_proba) as dst:
#         dst.write(y_predict_proba_reshaped.astype(dtype), range(1, clf.n_classes_ + 1))

#     # Write the predicted class map to a new GeoTIFF file
#     with rasterio.open(output_class_filename, 'w', **profile_class) as dst:
#         dst.write(pred_image.astype(rasterio.uint8), 1)

# def create_binary_mask(output_folder_path, subdir, profile_binary_mask, binary_mask):
#     # Define the output file name for the binary mask
#     output_binary_mask_filename = os.path.join(output_folder_path, subdir, "{}_binary_mask.tif".format(subdir))

#     # Write the binary mask to a new GeoTIFF file
#     with rasterio.open(output_binary_mask_filename, 'w', **profile_binary_mask) as dst:
#         dst.write(binary_mask, 1)
        
def start_processing():
    
    start_time = time.time()
    
    image_folder_path = image_folder_entry.get()
    model_file_path = model_file_entry.get()
    output_folder_path = output_folder_entry.get()
    
    # # File paths
    # S2_folder_path = os.path.join(main_folder_path, "normalized_mosaics_2021")
    # model_path = os.path.join(main_folder_path, "trained_model_2022/rf_model.joblib")
    # output_folder_path = os.path.join(main_folder_path, "output_normalized_mosaics_2021")

    # Load the trained model
    clf = load_trained_model(model_file_path)

    # Get the number of features the model was trained on
    n_features = clf.n_features_in_
    print("The model was trained with", n_features, "features")

    #Get the selected bands specified through the GUI
    channel_indices = list(map(int, bands_entry_var.get().split(',')))
    # Desired channel indices
    #channel_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]

    # Loop over each sub-directory in the main directory
    for subdir in os.listdir(image_folder_path):
        if os.path.isdir(os.path.join(image_folder_path, subdir)):
            # Create the output sub-directory inside the output folder
            output_subdir = os.path.join(output_folder_path, subdir)
            os.makedirs(output_subdir, exist_ok=True)
            # Create a list of image paths in the current sub-directory
            S2_image_paths = glob.glob(os.path.join(image_folder_path, subdir) + "/*.tif")

            # Preprocess the images
            X_predict, rows, cols = preprocess_images(S2_image_paths, channel_indices)
           
            # Get the selected batch size value from the GUI
            batch_size = int(batch_size_var.get())
            # Set the number of CPU cores to use
            n_cores = cpu_count()     # Set the number of CPU cores to use

            # Predict the class labels
            y_predict_proba = predict_class_labels(X_predict, clf, batch_size, n_cores)

            # Get the index of the highest probability for each pixel
            y_predict_index = np.argmax(y_predict_proba, axis=1)
            
            # Map the index to the predicted class label using clf.classes_
            y_predict = np.array([clf.classes_[i] for i in y_predict_index])
            
            # Reshape the prediction results to match the shape of the input images
            pred_image = np.reshape(y_predict, (rows, cols))

            # Creates a new 3D array y_predict_proba_reshaped with dimensions (rows, cols, n_classes), where rows and cols are the number of rows and columns in pred_image, 
            # respectively, and n_classes is the number of classes in the trained model.
            y_predict_proba_reshaped = np.zeros((clf.n_classes_, pred_image.shape[0], pred_image.shape[1]), dtype=np.float32)
            # reshape each column of 'y_predict_proba' into the shape of the image and add it as multiple channels in the y_predict_proba_reshaped 3D array
            for i in range(clf.n_classes_):
                y_predict_proba_reshaped[i, :, :] = y_predict_proba[:, i].reshape(pred_image.shape)


            # Open the input image and retrieve its metadata
            with rasterio.open(S2_image_paths[0]) as src:
                profile = src.profile
                transform = src.transform
            
            # Set the data type and nodata value of the output images
            dtype = rasterio.float32
            nodata = np.nan
            
            # Update the metadata to reflect the predicted probability map
            profile_proba = profile.copy()
            profile_proba.update(dtype=dtype, count=clf.n_classes_, nodata=nodata, compress='lzw')
            
            # Define the output file name for the predicted probability map
            output_proba_filename = os.path.join(output_subdir, "{}_predicted_proba.tif".format(subdir))
            
            # Write the predicted probability map to a new GeoTIFF file with the generated name
            with rasterio.open(output_proba_filename, 'w', **profile_proba) as dst:
                dst.write(y_predict_proba_reshaped.astype(dtype), range(1, clf.n_classes_ + 1))


            # Update the metadata to reflect the predicted class map
            profile_class = profile.copy()
            profile_class.update(dtype=rasterio.uint8, count=1, nodata=255, compress='lzw')

            # Define the output file name for the predicted class map
            output_class_filename = os.path.join(output_subdir, "{}_predicted_class.tif".format(subdir))

            # Write the predicted class map to a new GeoTIFF file with the generated name
            with rasterio.open(output_class_filename, 'w', **profile_class) as dst:
                dst.write(pred_image.astype(rasterio.uint8), 1)
                
                
            #############################################################################################
            # Binary mask for probability threshold
            
            # # Set a user-specified probability threshold
            # probability_threshold = 0.5  # Adjust this value based on your requirement
            # Get the probability threshold value from the GUI entry field
            probability_threshold = float(probability_threshold_entry.get())
            
            # Create a binary mask where at least one channel has a probability above the specified threshold
            binary_mask = np.any(y_predict_proba_reshaped > probability_threshold, axis=0).astype(rasterio.uint8)
            
            # Update the metadata to reflect the binary mask
            profile_binary_mask = profile.copy()
            profile_binary_mask.update(dtype=rasterio.uint8, count=1, nodata=0, compress='lzw')
            
            # Define the output file name for the binary mask
            output_binary_mask_filename = os.path.join(output_subdir, "{}_binary_mask.tif".format(subdir))
            
            # Write the binary mask to a new GeoTIFF file with the generated file name
            with rasterio.open(output_binary_mask_filename, 'w', **profile_binary_mask) as dst:
                dst.write(binary_mask, 1)
            ############################################################################################

            # Clean up memory
            del X_predict
            del y_predict
            del y_predict_proba
            del pred_image
            gc.collect()

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")

##---------------------GUI---------------------------

import os
import tkinter as tk
from tkinter import filedialog
from tkinter import PhotoImage

def select_image_folder():
    image_folder_path = filedialog.askdirectory(title="Select Image Folder")
    image_folder_entry.delete(0, tk.END)
    image_folder_entry.insert(tk.END, image_folder_path)

def select_model_file():
    model_file_path = filedialog.askopenfilename(title="Select Model File", filetypes=(("Joblib Files", "*.joblib"), ("All Files", "*.*")))
    model_file_entry.delete(0, tk.END)
    model_file_entry.insert(tk.END, model_file_path)

def select_output_folder():
    output_folder_path = filedialog.askdirectory(title="Select Output Folder")
    output_folder_entry.delete(0, tk.END)
    output_folder_entry.insert(tk.END, output_folder_path)
    

import tkinter as tk

# Create the main window
window = tk.Tk()
window.title("Modulo-4 GUI")
window.geometry("700x780")  # Set the window size

# Try to load logo image, if not present skip
try:
    # Load the image using Pillow
    logo_image = Image.open("G:/codici_RF/codici/4EOSIAL/eosial_logo.png")
    logo_image = ImageTk.PhotoImage(logo_image)
    
    # Add the image to the global list to prevent garbage collection
    global_images.append(logo_image)

    # Create a label with the image
    logo_label = tk.Label(window, image=logo_image)
    logo_label.image = logo_image  # Keep a reference to prevent garbage collection
    logo_label.grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 0))

    # Add a label beside the image
    title_text = "Multitemporal\nSupervised Classifier\nfor Large-Scale AOIs"
    title_label = tk.Label(window, text=title_text, font=("Helvetica", 18, "bold"), anchor="w", justify="center", wraplength=300)
    title_label.grid(row=0, column=2, padx=(10, 0), pady=(10, 0))

except Exception as e:  # Catch all exceptions to see any potential issues
    print("Error loading the logo image:", e)


# Add a description label
title = "Module-4"
title_label = tk.Label(window, text=title, font=("Helvetica", 14, "bold"))
title_label.grid(row=1, column=0, columnspan=4, padx=10, pady=(10, 0))

description1 = "Please select the input path for images or mosaics to classify, RF pre-trained model, and the Output Folder."
description1_label = tk.Label(window, text=description1, wraplength=550, justify='left')
description1_label.grid(row=2, column=0, columnspan=4, padx=10, pady=(0, 5))

description2 = "(note: select a pre-trained model that has been trained on a number of features (n.images x n.bands) equal to the dataset to be predicted)"
description2_label = tk.Label(window, text=description2, wraplength=550, justify='left')
description2_label.grid(row=3, column=0, columnspan=4, padx=10, pady=(0, 10))

# Add spacer rows
spacer1 = tk.Label(window, text="")
spacer1.grid(row=4, column=0)

# Create image folder selection widgets
image_folder_label = tk.Label(window, text="Image Folder:")
image_folder_label.grid(row=5, column=0, sticky='e', padx=10, pady=5)

image_folder_entry = tk.Entry(window, width=50)
image_folder_entry.grid(row=5, column=1, columnspan=2, padx=10, pady=5)

image_folder_button = tk.Button(window, text="Browse", command=select_image_folder)
image_folder_button.grid(row=5, column=3, padx=10, pady=5)

# Create model file selection widgets
model_file_label = tk.Label(window, text="Model File:")
model_file_label.grid(row=6, column=0, sticky='e', padx=10, pady=5)

model_file_entry = tk.Entry(window, width=50)
model_file_entry.grid(row=6, column=1, columnspan=2, padx=10, pady=5)

model_file_button = tk.Button(window, text="Browse", command=select_model_file)
model_file_button.grid(row=6, column=3, padx=10, pady=5)

# Create output folder selection widgets
output_folder_label = tk.Label(window, text="Output Folder:")
output_folder_label.grid(row=7, column=0, sticky='e', padx=10, pady=5)

output_folder_entry = tk.Entry(window, width=50)
output_folder_entry.grid(row=7, column=1, columnspan=2, padx=10, pady=5)

output_folder_button = tk.Button(window, text="Browse", command=select_output_folder)
output_folder_button.grid(row=7, column=3, padx=10, pady=5)


# Create a label and dropdown menu for batch_size
batch_size_label = tk.Label(window, text="batch_size:")
batch_size_label.grid(row=8, column=0, sticky='e', padx=10, pady=(20, 0))

batch_size_var = tk.StringVar(window)
batch_size_var.set("1000000")  # default value
batch_size_options = ["100000", "500000", "1000000", "1500000", "2000000"]
batch_size_menu = tk.OptionMenu(window, batch_size_var, *batch_size_options)
batch_size_menu.config(width=10)
batch_size_menu.grid(row=8, column=1, sticky='w', pady=(20, 0))

description3 = "batch-size selection (choose according to your memory availability)"
description3_label = tk.Label(window, text=description3, wraplength=400, justify='left')
description3_label.grid(row=8, column=2, columnspan=2, padx=(0, 10), pady=(20, 0), sticky='w')

# Add spacer row
spacer3 = tk.Label(window, text="")
spacer3.grid(row=9, column=0)

# Add a description label
bands_description = "Please enter the desired bands, separated by commas (e.g., 0,1,2). Default option selects all Sentinel-2 bands excluding 'SCL', plus VV and VH from Sentinel-1."
bands_description_label = tk.Label(window, text=bands_description, wraplength=550, justify='left')
bands_description_label.grid(row=10, column=0, columnspan=4, padx=10, pady=(10, 0))

# Create input field for desired bands
default_bands = "0,1,2,3,4,5,6,7,8,9,11,12"  # Set the default desired bands
bands_label = tk.Label(window, text="Desired bands:")
bands_label.grid(row=11, column=0, sticky='e', padx=10, pady=5)

bands_entry_var = tk.StringVar(value=default_bands)  # Set the default value
bands_entry = tk.Entry(window, textvariable=bands_entry_var)
bands_entry.grid(row=11, column=1, columnspan=2, padx=10, pady=5)


# Create a label and entry field for probability threshold
probability_threshold_label_text = "Probability Threshold (binary mask):"
probability_threshold_label = tk.Label(window, text=probability_threshold_label_text, wraplength=150)
probability_threshold_label.grid(row=12, column=0, sticky='e', padx=10, pady=5)

default_probability_threshold = 0.5  # Set the default probability threshold value

probability_threshold_entry = tk.Entry(window)
probability_threshold_entry.insert(tk.END, default_probability_threshold)  # Set the default value
probability_threshold_entry.grid(row=12, column=1, columnspan=2, padx=10, pady=5)



# Create the "Run" button
run_button = tk.Button(window, text="Run", command=start_processing)
run_button.grid(row=13, column=1, columnspan=2, pady=10, sticky='nsew')


# Add the end title
end_title = "Developed by Alvise Ferrari for EOSIAL (Scola di Ingegneria Aerospaziale - La Sapienza)"
end_title_label = tk.Label(window, text=end_title, font=("Helvetica", 8), anchor="center")
end_title_label.grid(row=14, column=0, columnspan=4, pady=(20, 10))


# Start the GUI event loop
window.mainloop()
