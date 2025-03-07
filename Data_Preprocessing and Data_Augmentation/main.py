import os
import cv2
import numpy as np
import albumentations as A

# Path to the main directory containing subfolders
MAIN_DIR = r"C:\Users\qures\OneDrive\Desktop\Dataset - Copy (2)"

# Image preprocessing (resize + normalization)
def preprocess_image(image, target_size=(256, 256)):
    image = cv2.resize(image, target_size)  # Resize to target size
    image = image.astype(np.float32) / 255.0  # Normalize pixel values (0 to 1)
    return image

# Augmentation pipeline
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.4),
    A.Rotate(limit=30, p=0.5),
    A.GaussianBlur(p=0.3),
    A.RandomResizedCrop((256, 256), scale=(0.8, 1.0), ratio=(0.75, 1.33), always_apply=False, p=0.5),
    A.ColorJitter(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, p=0.5),
    A.RandomGamma(p=0.3),
    A.ISONoise(p=0.3),
    A.HueSaturationValue(p=0.3)
])

# Function to generate different rotated views
def generate_rotated_views(image, num_views=12):
    height, width = image.shape[:2]
    rotated_images = []
    for i in range(num_views):
        angle = (360 / num_views) * i
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        rotated_images.append(rotated_image)
    return rotated_images

# Process each subfolder
for folder_name in os.listdir(MAIN_DIR):
    folder_path = os.path.join(MAIN_DIR, folder_name)
    if not os.path.isdir(folder_path):
        continue
    
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images.sort()  # Sort files to maintain order

    # Step 1: Rename original images
    for idx, img_name in enumerate(images):
        old_path = os.path.join(folder_path, img_name)
        new_name = f"{folder_name}_{idx}_original.jpg"
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
    
    # Step 2: Process images after renaming
    img_counter = 0
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]  # Reload after renaming

    for img_name in images:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Skipping {img_name} (Unreadable image)")
            continue
        
        # Preprocess and save original image (overwriting)
        img = preprocess_image(img)
        img_uint8 = (img * 255).astype(np.uint8)
        cv2.imwrite(img_path, img_uint8, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        img_counter += 1
        
        # Generate and save rotated images
        for rotated_img in generate_rotated_views(img_uint8, num_views=12):
            cv2.imwrite(os.path.join(folder_path, f"{folder_name}_{img_counter}_rotated.jpg"), rotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            img_counter += 1
        
        # Generate and save augmented images
        for _ in range(5):
            augmented_img = augmentation(image=img_uint8)['image']
            cv2.imwrite(os.path.join(folder_path, f"{folder_name}_{img_counter}_augmented.jpg"), augmented_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            img_counter += 1

print("✅ Data preprocessing, augmentation, and viewpoint generation completed successfully!")
