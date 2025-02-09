This code supports both 24-bit (RGB) and 32-bit (RGBA) images.

Requirements
Python 3.x
OpenCV (opencv-python package)
NumPy (numpy package)
You can install the required packages using pip:


pip install opencv-python numpy


The folder contains several 32 bit and 24 bit images. 


input_path = "rev.png"
Change "rev.png" to the name of your image file, including its extension.

Run the Script: Execute the script from your terminal or command prompt:

python rgbvadf.py

Choose Image Format: If your image has an alpha channel (RGBA), you will be prompted to choose whether to save the output as a 32-bit image (with alpha) or a 24-bit image (without alpha).

Output: The reconstructed image will be saved as reconstructed.png in the same directory.

Metrics
After processing, the script will print the following metrics:

Reconstructed size: Size of the reconstructed image in kilobytes (KB).
PSNR: Peak Signal-to-Noise Ratio, which measures the quality of the reconstructed image.
Mean Absolute Difference: Average pixel-wise difference between the original and reconstructed images.
