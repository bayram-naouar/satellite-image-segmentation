import os
import cv2
from PIL import Image
import numpy as np
import albumentations as A
from tqdm import tqdm
import yaml

def augment_and_save(image_dir, mask_dir, augmented_image_dir, augmented_mask_dir, num_augmentations=5, image_size=(128, 128)):
    """
    Perform augmentation and save the results.
    
    Args:
        image_dir (str): Path to the directory containing the original images.
        mask_dir (str): Path to the directory containing the segmentation masks.
        augmented_image_dir (str): Directory to save augmented images.
        augmented_mask_dir (str): Directory to save augmented masks.
        num_augmentations (int): Number of augmented images to generate per original image.
        image_size (tuple): Size to resize images and masks to.
    """
    # Create directories if they don't exist
    os.makedirs(augmented_image_dir, exist_ok=True)
    os.makedirs(augmented_mask_dir, exist_ok=True)

    transform = A.Compose([
        A.RandomCrop(width=image_size[0], height=image_size[1]),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        A.Blur(blur_limit=3)
    ])

    image_filenames = sorted(os.listdir(image_dir))
    mask_filenames = sorted(os.listdir(mask_dir))

    for img_file, mask_file in tqdm(zip(image_filenames, mask_filenames), total=len(image_filenames)):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)
        
        if not os.path.exists(img_path):
            print(f"Warning: {img_file} not found, skipping.")
            continue
        if not os.path.exists(mask_path):
            print(f"Warning: {mask_file} not found, skipping.")
            continue

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"\nWarning: Loading {img_path} as PIL Image.")
            img = Image.open(img_path)
            img = np.array(img)
        if mask is None:
            print(f"\nWarning: Loading {mask_path} as PIL Image.")
            mask = Image.open(mask_path)
            mask = np.array(mask)

        # Resize image and mask to the desired size
        img = cv2.resize(img, image_size)
        mask = cv2.resize(mask, image_size)

        for i in range(num_augmentations):
            augmented = transform(image=img, mask=mask)
            augmented_image = augmented['image']
            augmented_mask = augmented['mask']

            # Save augmented images and masks
            augmented_image_filename = f"{os.path.splitext(img_file)[0]}_aug_{i}.jpg"
            augmented_mask_filename = f"{os.path.splitext(mask_file)[0]}_aug_{i}.png"
            
            cv2.imwrite(os.path.join(augmented_image_dir, augmented_image_filename), augmented_image)
            cv2.imwrite(os.path.join(augmented_mask_dir, augmented_mask_filename), augmented_mask)

if __name__ == "__main__":
    config = yaml.safe_load(open("config.yaml"))
    image_dir = config["data"]["image_dir"]
    mask_dir = config["data"]["mask_dir"]
    augmented_image_dir = config["data"]["augmented_image_dir"]
    augmented_mask_dir = config["data"]["augmented_mask_dir"]
    
    augment_and_save(image_dir, mask_dir, augmented_image_dir, augmented_mask_dir)
