import os
import cv2
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path


def process_image(image_path, save_path, metadata_path, seed=None):
    # Set random seed for reproducibility
    if seed is None:
        seed = np.random.randint(0, 2**32 - 1)
    rng = np.random.RandomState(seed)
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return
    
    # Get original dimensions
    h, w = img.shape[:2]
    
    # Calculate scaling factor to make short side 1024
    scale = 1024 / min(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Resize image
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Calculate valid ranges for random crop
    y_max = new_h - 1024
    x_max = new_w - 1024
    
    # Random crop coordinates
    y_start = rng.randint(0, max(1, y_max + 1))
    x_start = rng.randint(0, max(1, x_max + 1))
    
    # Perform crop
    cropped = resized[y_start:y_start + 1024, x_start:x_start + 1024]
    
    # Save the cropped image
    cv2.imwrite(save_path, cropped)
    
    # Save metadata
    metadata = {
        'original_size': {'height': h, 'width': w},
        'resized_size': {'height': new_h, 'width': new_w},
        'crop_coordinates': {
            'y_start': y_start,
            'x_start': x_start,
            'height': 1024,
            'width': 1024
        },
        'random_seed': seed,
        'scale_factor': scale
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent = 2)


def crop_from_metadata(image_path, save_path, metadata_path):
    """Reproduce crop using saved metadata."""
    # Read metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return False
    
    # Resize image using saved scale factor
    scale = metadata['scale_factor']
    new_h = metadata['resized_size']['height']
    new_w = metadata['resized_size']['width']
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Get crop coordinates from metadata
    coords = metadata['crop_coordinates']
    y_start = coords['y_start']
    x_start = coords['x_start']
    height = coords['height']
    width = coords['width']
    
    # Perform crop
    cropped = resized[y_start:y_start + height, x_start:x_start + width]
    
    # Save the cropped image
    return cv2.imwrite(save_path, cropped)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process images with random or metadata-based cropping')
    parser.add_argument('--mode', choices=['random', 'metadata'], default='random',
                       help='Cropping mode: random for new crops, metadata to reproduce crops')
    parser.add_argument('--images-dir', type=str, required=True, help='Input images directory')
    parser.add_argument('--save-dir', type=str, required=True, help='Output directory for cropped images')
    parser.add_argument('--metadata-dir', type=str, required=True, help='Directory for metadata files')
    
    args = parser.parse_args()
    
    # Create save directories if they don't exist
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.metadata_dir, exist_ok=True)
    
    # Process all images
    image_files = [f for f in os.listdir(args.images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(args.images_dir, image_file)
        save_path = os.path.join(args.save_dir, image_file)
        metadata_path = os.path.join(args.metadata_dir, Path(image_file).stem + '.json')
        
        if args.mode == 'random':
            process_image(input_path, save_path, metadata_path)
        else:  # metadata mode
            if os.path.exists(metadata_path):
                success = crop_from_metadata(input_path, save_path, metadata_path)
                if not success:
                    print(f"Failed to process {image_file}")
            else:
                print(f"No metadata found for {image_file}")

    print("Processing complete!")

if __name__ == "__main__":
    main()
