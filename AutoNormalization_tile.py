# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 14:04:42 2023

@author: ferra

## Code Description

The provided Python code is used to normalize specified bands in Sentinel-2 satellite images and save the normalized images as new GeoTIFF files. It utilizes the `rasterio` library for reading and writing raster data and the `numpy` library for numerical operations.

## Usage
1. Set the `image_folder_path` variable to the path of the folder containing the Sentinel-2 satellite images that need to be normalized.
2. Set the `output_folder_path` variable to the desired path where the normalized images will be saved. The script will create the output folder if it doesn't already exist.
3. Specify the bands that need to be normalized by modifying the `specified_bands` list. The list contains the band indices (0-based) of the bands to be normalized.
4. Run the script. It will loop over each sub-directory in the main directory specified by `image_folder_path` and process the images in each sub-directory.
5. The script opens each GeoTIFF image, reads the image data, normalizes the specified bands, and substitutes the 11th band from the original image. The normalization process calculates the mean and standard deviation of the band's data, normalizes the data by subtracting the mean and dividing by the standard deviation, and scales the values between 0 and 65535.
6. The script copies the metadata from the original image, updates the necessary metadata fields (e.g., data type, band count, compression), and updates the band names (descriptions) in the metadata.
7. It creates a new GeoTIFF file with the normalized image data and updated metadata. The normalized image is saved in the `output_folder_path` with the same directory structure as the input images. The normalized images are named with the suffix '_normalized' appended to the original filenames.

## Dependencies

The code requires the following dependencies:

- `os` module: Provides a way to interact with the operating system.
- `glob` module: Facilitates the searching of files using pattern matching.
- `rasterio` library: Enables reading and writing raster data in various formats.
- `numpy` library: Provides support for large, multi-dimensional arrays and mathematical functions.

Ensure that these dependencies are installed in your Python environment before running the code.

## Inputs and Format
The code expects the input images to be in the GeoTIFF format. GeoTIFF is a commonly used file format for geospatial raster data that combines raster imagery with metadata, such as geographic information and coordinate reference system (CRS) details. It is widely employed in the fields of remote sensing and Geographic Information Systems (GIS).
The `image_folder_path` variable should be set to the path of the folder containing the Sentinel-2 satellite images that require normalization. The folder structure is assumed to include sub-directories within the main directory, with each sub-directory containing the individual image files. The code recursively processes all sub-directories within the specified main directory.
The input images should correspond to Sentinel-2 Level-2A products, which are atmospherically corrected and processed data. Each image is expected to consist of multiple bands representing different spectral channels, such as Red, Green, Blue, and Near-Infrared.
The code utilizes the `rasterio` library to open and read the GeoTIFF images. This library provides efficient and convenient functionalities for working with raster datasets, including access to the raster data array and metadata.
The normalized output images are saved in the GeoTIFF format as well. They are created in the `output_folder_path`, which should be set to the desired path where the normalized images will be saved. The script automatically creates the output folder if it doesn't already exist, ensuring the preservation of the same directory structure as the input images.
During the process of saving the normalized images, the code updates the metadata of each image, including the data type, band count, and compression settings. Additionally, the band names (descriptions) from the original images are preserved in the metadata of the normalized images. This allows software applications like QGIS to recognize and display the band names correctly when opening the normalized images.


###############################

The provided code appears to be a Python script that normalizes specified bands in Sentinel-2 satellite images and saves the normalized images as new GeoTIFF files. 
Below is a breakdown of the code:

The script starts with some metadata information and a description of its purpose and usage.
It imports necessary libraries: os for interacting with the operating system, glob for file searching, rasterio for reading and writing raster data, and numpy for numerical operations.
It defines the paths to the input folder (image_folder_path) containing the Sentinel-2 satellite images and the output folder (output_folder_path) where the normalized images will be saved.
The script creates the output folder if it doesn't already exist using the os.makedirs() function.
It specifies the bands to be normalized by modifying the specified_bands list, which contains the 0-based indices of the bands to be normalized.
The script then loops over each sub-directory in the main directory specified by image_folder_path.
Within each sub-directory, it creates a list of image paths using the glob.glob() function.
It then loops over each image path and performs the following operations:
Defines the output sub-directory path within the output_folder_path and creates the sub-directory if it doesn't exist.
Opens the GeoTIFF image using rasterio.open() and reads the image data.
Retrieves the band names from the original image using the src.descriptions property.
Normalizes the specified bands and substitutes the 11th band from the original image.
Copies the metadata from the original image into the profile variable.
Updates necessary metadata fields, such as data type, band count, and compression settings.
Updates the band names (descriptions) in the metadata.
Creates a new GeoTIFF file with the normalized image data and updated metadata using rasterio.open() in write mode.
The script iterates through all the images in each sub-directory, normalizes them, and saves the normalized images with the suffix "_normalized" appended to the original filenames in the corresponding output sub-directories.
"""


import os
import glob
import rasterio
import numpy as np

image_folder_path = "E:/rf_img/2019"
output_folder_path = "E:/rf_img/normalized_mosaics_2019"

# Create the output folder if it doesn't exist
os.makedirs(output_folder_path, exist_ok=True)

specified_bands = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]  # Specify the bands you want to normalize

# Loop over each sub-directory in the main directory
for subdir in os.listdir(image_folder_path):
    if os.path.isdir(os.path.join(image_folder_path, subdir)):
        # Create a list of image paths in the current sub-directory
        image_paths = sorted(glob.glob(os.path.join(image_folder_path, subdir) + "/*.tif"))

        # Loop over each image path
        for image_path in image_paths:
            # Define the output sub-directory path
            output_subdir = os.path.join(output_folder_path, subdir)
            os.makedirs(output_subdir, exist_ok=True)

            # Open the GeoTIFF image
            with rasterio.open(image_path, 'r') as src:
                # Read the image data
                image = src.read()

                # Get the band names from the original image
                band_names = src.descriptions

                # Normalize specified bands and substitute the 11th band from the original image
                normalized_image = np.zeros_like(image, dtype=np.uint16)
                for band in range(image.shape[0]):
                    if band in specified_bands:
                        band_data = image[band, :, :]
                        mean = np.mean(band_data)
                        std = np.std(band_data)
                        normalized_band = (band_data - mean) / std
                        normalized_band = np.interp(normalized_band, (normalized_band.min(), normalized_band.max()), (0, 65535))
                        normalized_image[band, :, :] = normalized_band.astype(np.uint16)
                    else:
                        normalized_image[band, :, :] = image[band, :, :]

                # Copy the metadata from the original image
                profile = src.profile

                # Update the necessary metadata fields
                profile.update(
                    dtype=rasterio.uint16,
                    count=image.shape[0],
                    compress='lzw'
                )

                # Update the band names in the metadata
                profile['descriptions'] = band_names

                # Create a new GeoTIFF file with the normalized image data and updated metadata
                output_filename = os.path.basename(image_path).replace('.tif', '_normalized.tif')
                output_path = os.path.join(output_subdir, output_filename)
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(normalized_image)
