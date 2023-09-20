# -*- coding: utf-8 -*-
"""
Created on Tue May  2 19:40:01 2023

@author: ferra
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 11:59:36 2023

@author: ferra
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import time
from datetime import datetime
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, PhotoImage

from PIL import Image, ImageTk
# Global list to hold image references
global_images = []


def browse_X_csv():
    file_path = filedialog.askopenfilename(initialdir="/", title="Select Training Set CSV",
                                           filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    X_CSV_path_var.set(file_path)
    
def browse_y_csv():
    file_path = filedialog.askopenfilename(initialdir="/", title="Select Training Set CSV",
                                           filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    y_CSV_path_var.set(file_path)

# def browse_joblib_model():
#     file_path = filedialog.askopenfilename(initialdir="/", title="Select Pre-trained Model",
#                                            filetypes=(("Joblib files", "*.joblib"), ("All files", "*.*")))
#     joblib_model_path_var.set(file_path)
    
def browse_output_folder():
    folder_path = filedialog.askdirectory(initialdir="/", title="Select Output Folder")
    output_folder_path_var.set(folder_path)
    
    

def run_script():
    output_folder_path = output_folder_path_var.get()
    X_CSV_path = X_CSV_path_var.get()
    y_CSV_path = y_CSV_path_var.get()
    
    start_time = time.time()


    # Read the feature matrix and class label array from CSV files
    X_df = pd.read_csv(X_CSV_path)
    y_df = pd.read_csv(y_CSV_path)

    # Convert dataframes to numpy arrays
    X = X_df.to_numpy()
    y = y_df['Class'].to_numpy()

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    # Get the selected hyperparameter values from the GUI
    n_estimators = int(n_estimators_var.get())
    max_features = max_features_var.get()
    if max_features == "None":
        max_features = None
    
    # Train the random forest classifier using the selected hyperparameters
    clf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, class_weight='balanced', random_state=2)
    clf.fit(X_train, y_train)

    # Save the trained model
    model_path = os.path.join(output_folder_path_var.get(), "rf_model.joblib")
    joblib.dump(clf, model_path)

    # Evaluate the performance of the trained model on the testing data
    y_pred = clf.predict(X_test)

    # Compute several common classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    # Print the computed metrics
    print("Accuracy: {:.3f}%".format(accuracy * 100))
    print("Precision: {:.3f}".format(precision))
    print("Recall: {:.3f}".format(recall))
    print("F1 score: {:.3f}".format(f1))
    print("Confusion Matrix:")
    print(cm)

    # Print a classification report
    target_names = np.unique(y)
    target_names = target_names.astype(str)
    report = classification_report(y_test, y_pred, target_names=target_names, digits=3)
    print(report)

    # Save requested information to a text file
    execution_time = time.time() - start_time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_file = os.path.join(output_folder_path_var.get(), "Module2_report_esecuzione.txt")

    with open(output_file, "w") as file:
        file.write("Time and date of execution: {}\n".format(current_time))
        file.write("Execution time: {:.2f} seconds\n".format(execution_time))
        file.write("Name of the inputs: X.csv, y.csv\n")
        file.write("Name of the outputs: rf_model_lazio_v3.joblib\n")


        
        # Save classifier hyperparameters
        file.write("\nClassifier Hyperparameters:\n")
        file.write("n_estimators: {}\n".format(clf.n_estimators))
        file.write("max_features: {}\n".format(clf.max_features))
        file.write("class_weight: {}\n".format(clf.class_weight))
        
        # Save classification report
        file.write("\nClassification Report:\n")
        file.write(report)
        file.write("\nConfusion Matrix:\n")
        file.write(np.array2string(cm, separator=', '))

    print("Execution summary saved to: {}".format(output_file))
    
    
    
# Create the main window
root = tk.Tk()
root.title("Modulo-2")
root.geometry("600x700")  # Set the window size

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

except tk.TclError as e:
    # Handle the case when the image file is not found
    print("Logo image not found:", e)

# Add a label beside the image
title_text = "Multitemporal\nSupervised Classifier\nfor Large-Scale AOIs"
title_label = tk.Label(root, text=title_text, font=("Helvetica", 20, "bold"), anchor="w", justify="center", wraplength=300)
title_label.grid(row=0, column=2, padx=(10, 0), pady=(10, 0))


# Add a description label
title = "Module-2"
title_label = tk.Label(root, text=title, font=("Helvetica", 14, "bold"))
title_label.grid(row=1, column=0, columnspan=3, padx=10, pady=(10, 0))

description1 = "Please select the input paths for X (input features) CSV, y (input features) CSV, and the Output Folder."
description1_label = tk.Label(root, text=description1, wraplength=550, justify='left')
description1_label.grid(row=2, column=0, columnspan=3, padx=10, pady=(0, 5))

description2 = "(note: The inputs X and y are created using Module-1)"
description2_label = tk.Label(root, text=description2, wraplength=550, justify='left')
description2_label.grid(row=3, column=0, columnspan=3, padx=10, pady=(0, 10))

description3 = "Random-Forest hyperparameters selection:"
description3_label = tk.Label(root, text=description3, wraplength=550, justify='left')
description3_label.grid(row=11, column=0, columnspan=3, padx=10, pady=(0, 10))

# Add spacer rows
spacer1 = tk.Label(root, text="")
spacer1.grid(row=4, column=0)
spacer2 = tk.Label(root, text="")
spacer2.grid(row=6, column=0)
spacer3 = tk.Label(root, text="")
spacer3.grid(row=8, column=0)
spacer4 = tk.Label(root, text="")
spacer4.grid(row=10, column=0)





# Create input fields and buttons for each input path
X_val_CSV_label = tk.Label(root, text="X input features CSV:")
X_val_CSV_label.grid(row=5, column=0, sticky='e')
X_CSV_path_var = tk.StringVar()
X_val_CSV_entry = tk.Entry(root, textvariable=X_CSV_path_var)
X_val_CSV_entry.grid(row=5, column=1)
X_val_CSV_button = tk.Button(root, text="Browse", command=browse_X_csv)
X_val_CSV_button.grid(row=5, column=2)

y_val_CSV_label = tk.Label(root, text="y output classes CSV:")
y_val_CSV_label.grid(row=7, column=0, sticky='e')
y_CSV_path_var = tk.StringVar()
y_val_CSV_entry = tk.Entry(root, textvariable=y_CSV_path_var)
y_val_CSV_entry.grid(row=7, column=1)
y_val_CSV_button = tk.Button(root, text="Browse", command=browse_y_csv)
y_val_CSV_button.grid(row=7, column=2)

output_folder_label = tk.Label(root, text="Output Folder:")
output_folder_label.grid(row=9, column=0, sticky='e')
output_folder_path_var = tk.StringVar()
output_folder_entry = tk.Entry(root, textvariable=output_folder_path_var)
output_folder_entry.grid(row=9, column=1)
output_folder_button = tk.Button(root, text="Browse", command=browse_output_folder)
output_folder_button.grid(row=9, column=2)

# Create a label and dropdown menu for n_estimators
n_estimators_label = tk.Label(root, text="n_estimators:")
n_estimators_label.grid(row=12, column=0, sticky='e', pady=(20, 0))
n_estimators_var = tk.StringVar(root)
n_estimators_var.set("500")  # default value
n_estimators_options = ["100", "500", "1000"]
n_estimators_menu = tk.OptionMenu(root, n_estimators_var, *n_estimators_options)
n_estimators_menu.grid(row=12, column=1, sticky='w', pady=(20, 0))

# Create a label and dropdown menu for max_features
max_features_label = tk.Label(root, text="max_features:")
max_features_label.grid(row=13, column=0, sticky='e')
max_features_var = tk.StringVar(root)
max_features_var.set("sqrt")  # default value
max_features_options = ["sqrt", "log2", "None"]
max_features_menu = tk.OptionMenu(root, max_features_var, *max_features_options)
max_features_menu.grid(row=13, column=1, sticky='w')


# Create the "Run" button
run_button = tk.Button(root, text="Run", command=run_script)
run_button.grid(row=13, column=2, pady=10, sticky='nsew')

# Add the end title
end_title = "Developed by Alvise Ferrari for EOSIAL (Scola di Ingegneria Aerospaziale - La Sapienza)"
end_label = tk.Label(root, text=end_title, font=("Helvetica", 8))
end_label.grid(row=14, column=0, columnspan=3, padx=10, pady=(10, 0))

# Run the main loop
root.mainloop()
