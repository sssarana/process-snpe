import os
import glob
import numpy as np
import cv2

# Define the input and output directories
input_dir = '/home/images100'  # Directory containing your 100 images
output_dir = '/home/img_outputs'  # Directory where you want to save the raw binary files
output_txt = '/home/inputs.txt'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Open the text file in write mode
with open(output_txt, 'w') as f:
    # Process each image in the input directory
    for img_path in glob.glob(os.path.join(input_dir, '*.jpg')):  # Assuming all images are .jpg
        # Load and resize the image
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (640, 640))

        # Convert to float32 and normalize
        img_float = img_resized.astype(np.float32)
        img_normalized = img_float / 255.0

        # Create a unique output filename based on the input filename
        base_name = os.path.basename(img_path)
        output_filename = os.path.splitext(base_name)[0] + '.raw'
        output_filepath = os.path.join(output_dir, output_filename)

        # Save as raw binary file
        img_normalized.tofile(output_filepath)

        # Write the absolute path of the output file to the text file
        f.write(output_filepath + '\n')
