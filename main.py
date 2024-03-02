import numpy as np
from ultralytics import YOLO
from PIL import Image
import argparse
import os

# Setup argparse for command line arguments
parser = argparse.ArgumentParser(description='Process images to detect and classify heroes.')
parser.add_argument('--input_folder', type=str, help='Path to the folder containing test images')
parser.add_argument('--output_file', type=str, help='Path to the output file where results will be saved')

args = parser.parse_args()

# Load models
model = YOLO('models/det.pt')  # Load a custom detection model
cls_model = YOLO('models/classify.pt')  # Load a custom classification model

output_dir = os.path.dirname(args.output_file)
if output_dir and not os.path.exists(output_dir):  # Check if output_dir is not an empty string
    os.makedirs(output_dir)

with open(args.output_file, 'w') as output_file:
    for img_filename in os.listdir(args.input_folder):
        try:
            img_path = os.path.join(args.input_folder, img_filename)
            org_img = Image.open(img_path)

            # Get the dimensions of the image
            width, height = org_img.size

            # Define the coordinates for the area to crop
            # Left, Top, Right, Bottom
            crop_area = (0, 0, width/2, height)

            # Crop the image
            cropped_img = org_img.crop(crop_area)
            # Predict with the model
            results = model(cropped_img, imgsz=640,save=False)  # predict on an image
            for result in results:
                if len(result.boxes) > 0:
                    # Find the box with the smallest 'left' value
                    smallest_left_box = min(result.boxes, key=lambda box: np.array(box.xyxy.cpu(), dtype=int).squeeze()[0])
                    left, top, right, bottom = np.array(smallest_left_box.xyxy.cpu(), dtype=int).squeeze()
                    # Crop the image based on the box coordinates
                    img_array = np.array(cropped_img)
                    # Crop the image using array slicing
                    cropped_array = img_array[top:bottom, left:right]
                    # Convert the cropped array back to a PIL Image
                    cropped_img_cls = Image.fromarray(cropped_array)
                    cls_results = cls_model(source=cropped_img_cls, imgsz=128,save=False)  # predict on the cropped image
                    hero_name = cls_results[0].names[cls_results[0].probs.top1]
                    # Write the output
                    output_file.write(f"{img_filename} {hero_name}\n")
                else:
                    output_file.write(f"{img_filename} There is no hero in the image\n")
        except Exception as e:
            print(str(e))
            continue
