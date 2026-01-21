import cv2
import numpy as np
import os
import glob

# ================= CONFIGURATION =================
# Update path if necessary
INPUT_FOLDER = r"D:\productize_tech_assignment\task_2\input-images" 
OUTPUT_FOLDER = r"D:\productize_tech_assignment\task_2\task_2_output"

# Sensitivity settings
# Increase MIN_CONTOUR_AREA if it detects too much small noise (leaves, wind)
# Decrease if it misses small objects
MIN_CONTOUR_AREA = 100
# Threshold for what counts as a "difference" in pixel intensity (0-255)
DIFF_THRESHOLD = 30     
# =================================================

def detect_changes():
    # Create output directory
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Get all "Before" images (files that do NOT contain '~2')
    # We assume 'X.jpg' is before and 'X~2.jpg' is after.
    all_files = glob.glob(os.path.join(INPUT_FOLDER, "*.jpg"))
    before_images = [f for f in all_files if "~2" not in f and "-3" not in f]
    
    print(f"Found {len(before_images)} pairs to process.")

    for before_path in before_images:
        filename = os.path.basename(before_path)
        name_no_ext = os.path.splitext(filename)[0]
        
        # Construct the expected "After" filename (X~2.jpg)
        after_filename = f"{name_no_ext}~2.jpg"
        after_path = os.path.join(INPUT_FOLDER, after_filename)
        
        # Construct Output filename (X-3.jpg)
        output_filename = f"{name_no_ext}-3.jpg"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        # Check if pair exists
        if not os.path.exists(after_path):
            print(f"Skipping {filename}: Missing pair {after_filename}")
            continue

        print(f"Processing pair: {filename} vs {after_filename}")

        # 1. Load Images
        img_before = cv2.imread(before_path)
        img_after = cv2.imread(after_path)

        if img_before is None or img_after is None:
            print("Error reading images.")
            continue

        # 2. Convert to Grayscale
        # Color isn't usually necessary for change detection and adds noise
        gray_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
        gray_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)

        # 3. Compute Absolute Difference
        # This calculates |Before - After| for every pixel
        diff = cv2.absdiff(gray_before, gray_after)

        # 4. Apply Thresholding
        # If pixel difference > 30, convert to White (255), else Black (0)
        _, thresh = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

        # 5. Morphological Operations (Noise Cleaning)
        # Dilate helps merge nearby pixels (e.g., a car might have shiny parts 
        # that look like the background, splitting the car in two. Dilation fixes this.)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        
        # 6. Find Contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Prepare output image (Copy of 'After' image to draw on)
        img_annotated = img_after.copy()
        
        objects_found = 0
        
        for contour in contours:
            # 7. Filter out small noise
            if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                continue

            objects_found += 1
            
            # Get Bounding Box coordinates
            (x, y, w, h) = cv2.boundingRect(contour)
            
            # Draw Rectangle on the "After" image
            # Color: Yellow (0, 255, 255), Thickness: 2
            cv2.rectangle(img_annotated, (x, y), (x + w, y + h), (0, 255, 255), 2)
            
            # Optional: Add label
            # cv2.putText(img_annotated, "Missing", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 8. Save Output
        cv2.imwrite(output_path, img_annotated)
        print(f"  -> Detected {objects_found} changes. Saved to {output_filename}")

if __name__ == "__main__":
    detect_changes()