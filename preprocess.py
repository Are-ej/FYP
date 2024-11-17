import cv2
import os
from albumentations import Compose, HorizontalFlip, Rotate, RandomBrightnessContrast
from matplotlib import pyplot as plt

# Preprocessing functions
def resize_image(image, size=(416, 416)):
    return cv2.resize(image, size)

def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_img = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2BGR)

def reduce_noise(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def augment_image(image):
    transform = Compose([
        HorizontalFlip(p=0.5),
        Rotate(limit=15, p=0.5),
        RandomBrightnessContrast(p=0.5),
    ])
    augmented = transform(image=image)
    return augmented["image"]

def normalize(image):
    return image / 255.0

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = resize_image(image)
    image = enhance_contrast(image)
    image = reduce_noise(image)
    image = augment_image(image)
    image = normalize(image)
    return image

def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Loop through each category folder in the main dataset directory
    for category_folder in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category_folder)
        output_category_path = os.path.join(output_dir, category_folder)
        
        if not os.path.exists(output_category_path):
            os.makedirs(output_category_path)
        
        # Loop through each image in the category folder
        for img_file in os.listdir(category_path):
            img_path = os.path.join(category_path, img_file)
            processed_image = preprocess_image(img_path)
            
            # Save processed image
            output_path = os.path.join(output_category_path, img_file)
            cv2.imwrite(output_path, (processed_image * 255).astype("uint8"))  # Convert to 0-255 range for saving

# Run the function for all images
input_directory = "dataset"
output_directory = "preprocessed_dataset"
process_directory(input_directory, output_directory)


def display_images(original_path, processed_path):
    # Ensure the paths are properly formatted
    if not os.path.exists(original_path):
        print(f"Original image not found at {original_path}")
        return
    if not os.path.exists(processed_path):
        print(f"Processed image not found at {processed_path}")
        return

    # Load the images
    original = cv2.imread(original_path)
    processed = cv2.imread(processed_path)

    if original is None:
        print("Error loading original image. Check if the path is correct or if the file is corrupted.")
        return
    if processed is None:
        print("Error loading processed image. Check if the image was saved correctly.")
        return

    # Display the images
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Processed Image")
    ax[1].axis("off")
    plt.show()
# Using raw strings (recommended solution)
display_images(
    r"D:\fyp_pre\dataset\type1\acne-closed-comedo-1.jpg",
    r"D:\fyp_pre\preprocessed_dataset\type1\acne-closed-comedo-1.jpg"
)
