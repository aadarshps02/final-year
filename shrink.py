
import cv2  
import numpy as np  
import struct  
from math import ceil  
import os
# kindly read the readme text file before using this code 

BITS_PER_VALUE = 6  
BITS_PER_BYTE = 8  
IMAGE_TYPE_24BIT = 0  
IMAGE_TYPE_32BIT = 1  

def find_msv_position(val):
    
    val_int = int(val)  
    return 7 if val_int == 0 else 7 - (val_int.bit_length() - 1)  
def vadf_compress_channel(channel, k_bits=3):
    
    if not 1 <= k_bits <= 3:
        raise ValueError("k_bits must be between 1 and 3")  
    compressed_bits = []  # List to hold compressed bit values
    mask = (1 << k_bits) - 1  

    
    for val in channel.flatten():
        pos = find_msv_position(val)  
        available_bits = 7 - pos  

        if available_bits <= 0:
            data = 0  
        else:
            shift = available_bits - k_bits  
            if shift > 0:
                round_bit = (val >> (shift - 1)) & 1  
                data = (val >> shift) & mask  
                data += round_bit  
                if data > mask:  
                    data = mask
            else:
                data = (val << abs(shift)) & mask  

       
        packed = (pos << 3) | data
        compressed_bits.extend([(packed >> (5 - i)) & 1 for i in range(6)])  
    num_bits = len(compressed_bits)  
    num_bytes = ceil(num_bits / BITS_PER_BYTE)  
    compressed = bytearray(num_bytes)  
    
    for i in range(num_bytes):
        byte = 0
        for j in range(BITS_PER_BYTE):
            idx = i * BITS_PER_BYTE + j
            if idx < num_bits:
                byte |= compressed_bits[idx] << (7 - j)  
        compressed[i] = byte  

    return compressed, num_bits 

def vadf_decompress_channel(compressed, original_shape, k_bits=3, num_bits=None):
   
    height, width = original_shape  
    total_pixels = height * width  
    bit_array = [] 
    for byte in compressed:
        bit_array.extend([(byte >> (7 - i)) & 1 for i in range(BITS_PER_BYTE)])  
    
    if num_bits:
        bit_array = bit_array[:num_bits]  

    reconstructed = []  
    for i in range(0, len(bit_array), BITS_PER_VALUE):
        if i + BITS_PER_VALUE > len(bit_array):
            break
        
        bits = bit_array[i:i + BITS_PER_VALUE] 
        packed = sum(bit << (5 - j) for j, bit in enumerate(bits)) 
        pos = packed >> 3  
        data = packed & 0x07  

        if pos == 7:
            val = 0 
        else:
            shift = max(0, 7 - pos - k_bits)  
            val = (1 << (7 - pos)) | (data << shift) 
        reconstructed.append(np.uint8(val))  
    
    return np.array(reconstructed[:total_pixels], dtype=np.uint8).reshape(original_shape)  

def compress_image(img, k_bits=3):
    
    channels = cv2.split(img)  # Split the image into its color channels ie r g b
    num_channels = len(channels)  # Get the number of channels
    
    if num_channels not in (3, 4):
        raise ValueError("Unsupported number of channels (must be 3 or 4)")  
    
    image_type = IMAGE_TYPE_32BIT if num_channels == 4 else IMAGE_TYPE_24BIT  
    header = struct.pack('!HHBB', img.shape[1], img.shape[0], k_bits, image_type)     
    compressed_channels = [] 
    bits_info = []  
    for channel in channels:
        compressed, bits = vadf_compress_channel(channel, k_bits)  # Compress each channel
        compressed_channels.append(compressed)  # Store compressed channel
        bits_info.append(bits) 
    
    bits_format = '!IIII' if num_channels == 4 else '!III'  # Format for bits info based on channel count
    header += struct.pack(bits_format, *bits_info)  # Append bits info to header
    
    return header + b''.join(compressed_channels)  # Return the complete compressed data

def decompress_image(data):
    """
    Returns:
        A 3D numpy array representing the decompressed image.
    """
    try:
        base_header = data[:6]  # Extracts the header information
        width, height, k_bits, image_type = struct.unpack('!HHBB', base_header)  # Unpack header data
        num_channels = 4 if image_type == IMAGE_TYPE_32BIT else 3  # Determine number of channels based on image type
        
        bits_format = '!IIII' if num_channels == 4 else '!III'  
        bits_data = data[6:6 + struct.calcsize(bits_format)] 
        bits_info = list(struct.unpack(bits_format, bits_data))  
        
        data_ptr = 6 + struct.calcsize(bits_format)  
        
        reconstructed_channels = []  
        for bits in bits_info:
            byte_size = ceil(bits / BITS_PER_BYTE) 
            channel_data = data[data_ptr:data_ptr + byte_size] 
            data_ptr += byte_size 
            
            # Decompress the channel and store it
            channel = vadf_decompress_channel(channel_data, (height, width), k_bits, bits)
            reconstructed_channels.append(channel)  
        
        return cv2.merge(reconstructed_channels)  # Merge channels and return the image
    
    except Exception as e:
        print(f"Decompression error: {str(e)}")  # Handle errors during decompression
        return None  # Return None if an error occurs

def mean_absolute_difference(original, reconstructed):
    
    return np.mean(np.abs(original.astype(int) - reconstructed.astype(int)))  

if __name__ == "__main__":
    input_path = "lena.png"  # input image
    reconstructed_path = "reconstructed.png"  # Path for output image

    try:
        # Load image with alpha channel if present
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED) 
        if img is None:
            raise ValueError("Failed to load image")  
        
        # 4 alpha channel
        if img.shape[2] == 4:  # Check if image has 4 channels 
            choice = input("Input image has alpha channel. Save output as:\n"
                           "1. 32-bit with alpha\n"
                           "2. 24-bit without alpha\n"
                           "Choose (1/2): ").strip()
            if choice == "2":
                img = img[:, :, :3]  # Remove alpha channel if user chooses 24-bit
        
       
        compressed_data = compress_image(img)  
        
    
        reconstructed = decompress_image(compressed_data)  
        
        del compressed_data  
        
        if reconstructed is not None:
            cv2.imwrite(reconstructed_path, reconstructed)  # Save the reconstructed image
            
            # Calculate metrics
            psnr = cv2.PSNR(img, reconstructed) if img.shape == reconstructed.shape else -1  # Calculate PSNR
            mad = mean_absolute_difference(img, reconstructed)  # Calculate MAD
            
            print(f"\nReconstructed size: {os.path.getsize(reconstructed_path)/1024:.1f} KB")  
            print(f"PSNR: {psnr:.2f} dB")  # peak singal noise ratio value
            print(f"Mean Absolute Difference: {mad:.2f}")  
            
    except Exception as e:
        print(f"Error: {str(e)}")  
