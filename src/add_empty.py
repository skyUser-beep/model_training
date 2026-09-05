import os
import shutil #managing file and directories
import random


# CONFIGURATION
SOURCE_DIR = r"../dataset/empty_frames"
TRAIN_IMAGES = r"biscuit_dataset\images\train"
VAL_IMAGES = r"biscuit_dataset\images\val"
TRAIN_LABELS = r"biscuit_dataset\labels\train"
VAL_LABELS = r"biscuit_dataset\labels\val"

# 80% train / 20% validation
TRAIN_RATIO = 0.80
# Reproducible split
random.seed(42)

# CREATE DIRECTORIES
os.makedirs(TRAIN_IMAGES, exist_ok=True)
os.makedirs(VAL_IMAGES, exist_ok=True)
os.makedirs(TRAIN_LABELS, exist_ok=True)
os.makedirs(VAL_LABELS, exist_ok=True)

# FIND EMPTY-CONVEYOR IMAGES
images = [
    f for f in os.listdir(SOURCE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
if not images:
    print("ERROR: No images found in:")
    print(SOURCE_DIR)
    exit()
images.sort()
random.shuffle(images)

# SPLIT
split_index = int(len(images) * TRAIN_RATIO)
train_images = images[:split_index]
val_images = images[split_index:]
print("=" * 60)
print("EMPTY CONVEYOR DATASET")
print("=" * 60)
print(f"Total empty images : {len(images)}")
print(f"Train              : {len(train_images)}")
print(f"Validation         : {len(val_images)}")
print()

# COPY EMPTY IMAGES + CREATE EMPTY LABELS
def add_empty_images(image_list, image_destination, label_destination):
    for image_name in image_list:
        source_image = os.path.join(SOURCE_DIR,image_name)
        # Prefix prevents collision with existing images
        new_name = "empty_" + image_name
        destination_image = os.path.join(image_destination,new_name)
        # Corresponding YOLO label
        label_name = os.path.splitext(new_name)[0] + ".txt"
        destination_label = os.path.join(label_destination,label_name)
        # Copy image
        shutil.copy2(
            source_image,
            destination_image
        )
        # Create EMPTY YOLO label
        # Empty conveyor = zero objects
        with open(destination_label,"w",encoding="utf-8"):
            pass
        print(f"Added: {new_name}")
# ADD TRAIN DATA
print("Adding training images...")
add_empty_images(train_images,TRAIN_IMAGES, TRAIN_LABELS)
print()

# ADD VALIDATION DATA
print("Adding validation images...")
add_empty_images(val_images,VAL_IMAGES,VAL_LABELS)
# FINISHED
print()
print("=" * 60)
print("EMPTY DATA ADDED SUCCESSFULLY")
print("=" * 60)
print(f"Training empty images   : {len(train_images)}")
print(f"Validation empty images : {len(val_images)}")
print()
print("Important:")
print("Empty images have EMPTY .txt label files.")
print("They are NOT a separate 'empty' class.")
print()
print("Next step: verify the dataset, then fine-tune.")