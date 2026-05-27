"""
Image preprocessing utilities using PIL for OCR optimization.

This module provides functions to rotate, crop, and segment images
before performing OCR operations.
"""

import os
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

def rotate_image(image_path, degrees=270):
    """
    Rotates the image clockwise. 
    The uploaded image is 90 degrees counter-clockwise, 
    so rotating 270 degrees counter-clockwise (or 90 clockwise) fixes it.
    """
    print(f"Rotating image by {degrees} degrees...")
    with Image.open(image_path) as img:
        return img.rotate(degrees, expand=True)

def find_deskew_angle(img):
    """
    Tests minor angle variations around the center to find the exact angle
    where the notebook gutter spine is perfectly vertical.
    """
    print("Calculating precise deskew angle using gutter line analysis...")
    
    # Work with a downscaled, grayscale edge map for speed and stability
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    
    width, height = edges.size
    # Focus only on the middle 20% width slice where the gutter resides
    center_left = int(width * 0.4)
    center_right = int(width * 0.6)
    gutter_zone = edges.crop((center_left, 0, center_right, height))
    
    best_angle = 0
    max_peak_value = 0
    
    # Test angles from -5 to +5 degrees in 0.5-degree steps
    angles_to_test = np.arange(-5, 5.5, 0.5)
    
    for angle in angles_to_test:
        # Rotate the edge slice
        rotated_slice = gutter_zone.rotate(angle, resample=Image.BICUBIC, expand=False)
        slice_data = np.array(rotated_slice)
        
        # Sum columns horizontally to find the intensity profile
        col_sums = np.sum(slice_data, axis=0)
        
        # The angle where the spine is straightest will yield the highest, sharpest peak
        current_max_peak = np.max(col_sums)
        
        if current_max_peak > max_peak_value:
            max_peak_value = current_max_peak
            best_angle = angle
            
    print(f"-> Detected alignment skew: {best_angle} degrees.")
    return best_angle

def crop_and_split_pipeline(img, output_dir="output_segments"):
    """
    Deskews the image, crops tightly to the notebook edges first, 
    and then splits the isolated notebook exactly in half.
    """
    # Step 1: Fix alignment
    fine_angle = find_deskew_angle(img)
    corrected_img = img.rotate(fine_angle, resample=Image.BICUBIC, expand=True)
    
    # Step 2: Generate edge map of the straight image to locate boundaries
    width, height = corrected_img.size
    gray = corrected_img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_data = np.array(edges)
    
    # Step 3: Compute vertical and horizontal projection profiles
    col_sums = np.sum(edge_data, axis=0)
    row_sums = np.sum(edge_data, axis=1)
    
    # Noise threshold filter (ignore faint background noise/shading changes)
    threshold_y = np.quantile(row_sums, 0.90)
    threshold_x = np.quantile(col_sums, 0.90)
    
    # Find outer bounds
    significant_rows = np.where(row_sums > threshold_y)[0]
    significant_cols = np.where(col_sums > threshold_x)[0]
    
    # Set coordinates with slight padding safety margins
    top_y = max(0, significant_rows[0] - 10) if len(significant_rows) > 0 else 0
    bottom_y = min(height, significant_rows[-1] + 10) if len(significant_rows) > 0 else height
    left_x = max(0, significant_cols[0] - 50) if len(significant_cols) > 0 else 0
    right_x = min(width, significant_cols[-1] -50) if len(significant_cols) > 0 else width
    
    print(f"-> Cropping notebook bounding box: Left={left_x}, Top={top_y}, Right={right_x}, Bottom={bottom_y}")
    
    # Step 4: Perform the hard crop to isolate the notebook from the scanner bed
    notebook_box = (left_x, top_y, right_x, bottom_y)
    cropped_notebook = corrected_img.crop(notebook_box)
    
    # Step 5: Split the isolated notebook directly down the middle
    notebook_width, notebook_height = cropped_notebook.size
    midpoint = notebook_width // 2
    print(f"-> Splitting isolated notebook at clean midpoint X: {midpoint}")
    
    left_box = (0, 0, midpoint, notebook_height)
    right_box = (midpoint, 0, notebook_width, notebook_height)
    
    left_page = cropped_notebook.crop(left_box)
    right_page = cropped_notebook.crop(right_box)
    
    # Step 6: Save output files
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    left_path = os.path.join(output_dir, "left_page_final.jpg")
    right_path = os.path.join(output_dir, "right_page_final.jpg")
    
    left_page.save(left_path, "JPEG", quality=95)
    right_page.save(right_path, "JPEG", quality=95)
    
    print(f"Saved cleanly isolated pages:\n - {left_path}\n - {right_path}")

def pipeline(image_path):
    # Base 90 degree flip
    base_rotated = rotate_image(image_path)
    
    # Clean, isolated crop and split
    crop_and_split_pipeline(base_rotated)


# --- Execution ---
if __name__ == "__main__":
    # Replace with your actual file path
    input_path = "/Users/jon/Projects/ebird-format/field-notes"
    input_filename = f"{input_path}/input_test2.jpg" 

    if os.path.exists(input_filename):
         pipeline(input_filename)
    else:
        print(f"Error: Could not find {input_filename}. Please place it in the same directory.")

