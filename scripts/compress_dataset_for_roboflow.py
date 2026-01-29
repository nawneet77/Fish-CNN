"""
Compress dataset images to ensure EACH image is under 20 MB for RoboFlow upload.

This script takes all images from the dataset folders and ensures each individual
image is under 20 MB, compressing only those that exceed the limit.
"""

import os
from pathlib import Path
from PIL import Image
import shutil
from tqdm import tqdm


def get_file_size_mb(file_path):
    """Get file size in MB."""
    return file_path.stat().st_size / (1024 * 1024)


def compress_image_to_target(img_path, output_path, max_size_mb=19.5):
    """
    Compress a single image to be under the target size.
    
    Args:
        img_path: Path to input image
        output_path: Path to save compressed image
        max_size_mb: Maximum size in MB (default 19.5 to leave buffer)
    
    Returns:
        tuple: (success, original_size_mb, final_size_mb)
    """
    original_size_mb = get_file_size_mb(img_path)
    
    try:
        with Image.open(img_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'LA', 'P'):
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                rgb_img.paste(img, mask=img.split()[-1] if len(img.split()) > 3 else None)
                img = rgb_img
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # If original is already under the limit, just copy with slight optimization
            if original_size_mb < max_size_mb:
                img.save(output_path, 'JPEG', quality=95, optimize=True)
                final_size_mb = get_file_size_mb(output_path)
                return True, original_size_mb, final_size_mb
            
            # Image is too large, need to compress
            # Start with quality 85 and progressively reduce if needed
            quality = 85
            max_dimension = max(img.size)
            
            # Try compressing with different quality levels
            for attempt in range(5):
                # Resize if dimension is very large
                temp_img = img.copy()
                if max_dimension > 4000:
                    scale = 4000 / max_dimension
                    new_size = tuple(int(dim * scale) for dim in temp_img.size)
                    temp_img = temp_img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Save with current quality
                temp_img.save(output_path, 'JPEG', quality=quality, optimize=True)
                current_size_mb = get_file_size_mb(output_path)
                
                # Check if we're under the limit
                if current_size_mb < max_size_mb:
                    return True, original_size_mb, current_size_mb
                
                # Reduce quality for next attempt
                quality -= 10
                max_dimension = int(max_dimension * 0.85)
                
                if quality < 40:
                    # Last resort: more aggressive resizing
                    scale = 0.7
                    new_size = tuple(int(dim * scale) for dim in img.size)
                    temp_img = img.resize(new_size, Image.Resampling.LANCZOS)
                    temp_img.save(output_path, 'JPEG', quality=40, optimize=True)
                    final_size_mb = get_file_size_mb(output_path)
                    return True, original_size_mb, final_size_mb
            
            final_size_mb = get_file_size_mb(output_path)
            return True, original_size_mb, final_size_mb
            
    except Exception as e:
        print(f"\nError processing {img_path.name}: {e}")
        return False, original_size_mb, 0


def compress_images(source_folders, output_folder, max_size_mb=19.5):
    """
    Process images from source folders, ensuring each is under max_size_mb.
    
    Args:
        source_folders: List of folder paths containing images
        output_folder: Path to save processed images
        max_size_mb: Maximum size per image in MB (default 19.5 to leave buffer)
    """
    # Create output folder
    output_path = Path(output_folder)
    if output_path.exists():
        print(f"Removing existing output folder: {output_folder}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all image paths
    all_images = []
    for folder in source_folders:
        folder_path = Path(folder)
        if folder_path.exists():
            images = list(folder_path.glob('*.jpg'))
            all_images.extend(images)
            print(f"Found {len(images)} images in {folder}")
    
    if not all_images:
        print("No images found!")
        return
    
    total_images = len(all_images)
    print(f"\nTotal images to process: {total_images}")
    print(f"Maximum size per image: {max_size_mb:.1f} MB")
    
    # Process each image
    print("\nProcessing images...")
    compressed_count = 0
    copied_count = 0
    failed_count = 0
    total_original_size = 0
    total_final_size = 0
    oversized_images = []
    
    for img_path in tqdm(all_images, desc="Processing images"):
        output_file = output_path / img_path.name
        success, orig_size, final_size = compress_image_to_target(img_path, output_file, max_size_mb)
        
        if success:
            total_original_size += orig_size
            total_final_size += final_size
            
            if orig_size > max_size_mb:
                compressed_count += 1
                if final_size > max_size_mb:
                    oversized_images.append((img_path.name, final_size))
            else:
                copied_count += 1
        else:
            failed_count += 1
    
    # Final report
    print(f"\n{'='*70}")
    print(f"Processing Complete!")
    print(f"{'='*70}")
    print(f"Total images processed: {total_images}")
    print(f"  - Images that needed compression: {compressed_count}")
    print(f"  - Images copied (already under limit): {copied_count}")
    if failed_count > 0:
        print(f"  - Failed: {failed_count}")
    print(f"\nOriginal total size: {total_original_size:.2f} MB")
    print(f"Final total size: {total_final_size:.2f} MB")
    print(f"Space saved: {total_original_size - total_final_size:.2f} MB")
    
    if oversized_images:
        print(f"\n⚠ Warning: {len(oversized_images)} image(s) still exceed {max_size_mb} MB:")
        for name, size in oversized_images[:10]:  # Show first 10
            print(f"  - {name}: {size:.2f} MB")
        if len(oversized_images) > 10:
            print(f"  ... and {len(oversized_images) - 10} more")
    else:
        print(f"\n✓ All images are under {max_size_mb} MB!")
    
    print(f"\nOutput folder: {output_folder}")
    print(f"{'='*70}")


def main():
    # Define source folders
    base_path = Path(__file__).parent.parent
    source_folders = [
        base_path / "Dataset images new",
        base_path / "Dataset images new 2",
        base_path / "Dataset images new 3"
    ]
    
    # Check if folders exist
    existing_folders = [f for f in source_folders if f.exists()]
    if not existing_folders:
        print("Error: No dataset folders found!")
        return
    
    # Define output folder
    output_folder = base_path / "Dataset_compressed_for_roboflow"
    
    print("Fish Dataset Compression Tool")
    print("="*70)
    print(f"Source folders:")
    for folder in existing_folders:
        print(f"  - {folder.name}")
    print(f"Output folder: {output_folder.name}")
    print(f"Max size per image: 19.5 MB (buffer for RoboFlow's 20 MB limit)")
    print("="*70)
    
    # Compress images
    compress_images(existing_folders, output_folder, max_size_mb=19.5)
    
    print("\nYou can now upload the compressed dataset to RoboFlow!")


if __name__ == "__main__":
    main()
