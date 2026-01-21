import cv2
import numpy as np
import os
import glob
import shutil

# ================= CONFIGURATION =================
INPUT_DIR = r"D:\productize_tech_assignment\Task 1\input-images"
OUTPUT_DIR = r"D:\productize_tech_assignment\Task 1\task_1_output"

# Hyperparameters — tune these and rerun to find best for your dataset
K_FOR_KNN = 30                   # Number of nearest neighbors for knnMatch. Higher (e.g., 5) allows stricter filtering variants.
LOWE_RATIO = 0.65                # Lowe's ratio threshold (0.6-0.8). Lower = stricter = fewer but higher-quality matches = often higher inlier ratio.
RANSAC_THRESHOLD = 22.0          # Reprojection error threshold in pixels. Higher = more inliers but risk of distortion (10-15 good for noisy thermal).
MIN_MATCH_COUNT = 30             # Minimum good matches to attempt alignment. Lower if you want to try even sparse cases.
NFEATURES = 25000                # Max SIFT keypoints. High value (as you tested) for dense coverage and more matches.
PHASE_PREALIGN_THRESHOLD = 50    # If good matches < this, trigger phase correlation pre-align (less needed with high features).
CLAHE_CLIP_LIMIT = 2.5           # CLAHE clip limit for thermal contrast. 2.0-4.0; higher = stronger enhancement.
# =================================================

def load_images(rgb_path, thermal_path):
    img_rgb = cv2.imread(rgb_path)
    img_thermal = cv2.imread(thermal_path)
    return img_rgb, img_thermal

def phase_correlation_init(gray1, gray2):
    """Quick FFT-based translation estimate (returns shift x,y)."""
    h, w = gray1.shape
    fh, fw = cv2.getOptimalDFTSize(h), cv2.getOptimalDFTSize(w)
    
    f1 = np.fft.fft2(gray1, (fh, fw))
    f2 = np.fft.fft2(gray2, (fh, fw))
    corr = np.fft.ifft2(f1 * np.conj(f2))
    corr = np.abs(np.fft.fftshift(corr))
    
    sy, sx = np.unravel_index(np.argmax(corr), corr.shape)
    sy -= fh // 2
    sx -= fw // 2
    
    return sx, sy

def enhance_thermal(gray_thermal):
    """CLAHE for thermal contrast boost."""
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))
    return clahe.apply(gray_thermal)

def get_best_affine_transform(img1_gray, img2_gray, img_thermal_full):
    """
    SIFT-affine with high features, configurable params, CLAHE, and optional phase pre-align.
    """
    img2_enhanced = enhance_thermal(img2_gray)
    
    thermal_to_use = img_thermal_full
    gray_thermal_to_use = img2_enhanced
    
    sift = cv2.SIFT_create(nfeatures=NFEATURES, nOctaveLayers=3, contrastThreshold=0.04)
    
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    if des1 is None or len(kp1) < 3:
        print("  Warning: Insufficient RGB features.")
        return None
    
    attempts = [gray_thermal_to_use, cv2.bitwise_not(gray_thermal_to_use)]
    
    best_matches = []
    best_kp2 = None
    best_num_good = 0
    
    for therm_attempt_gray in attempts:
        kp2, des2 = sift.detectAndCompute(therm_attempt_gray, None)
        if des2 is None or len(kp2) < 3:
            continue

        index_params = dict(algorithm=1, trees=3)
        search_params = dict(checks=30)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        try:
            matches = flann.knnMatch(des1, des2, k=K_FOR_KNN)
        except Exception:
            continue

        good = []
        for m in matches:
            if len(m) >= 2 and m[0].distance < LOWE_RATIO * m[1].distance:
                good.append(m[0])
        
        if len(good) > best_num_good:
            best_num_good = len(good)
            best_matches = good
            best_kp2 = kp2

    print(f"  Matches found: {best_num_good}")

    if len(best_matches) >= MIN_MATCH_COUNT:
        src_pts = np.float32([best_kp2[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)

        if best_num_good < PHASE_PREALIGN_THRESHOLD:
            print(f"  Low matches (<{PHASE_PREALIGN_THRESHOLD}) — applying phase correlation pre-align.")
            tx, ty = phase_correlation_init(img1_gray, img2_enhanced)
            M_pre = np.float32([[1, 0, tx], [0, 1, ty]])
            height, width = img_thermal_full.shape[:2]
            pre_warped_thermal = cv2.warpAffine(img_thermal_full, M_pre, (width, height))
            pre_gray = cv2.cvtColor(pre_warped_thermal, cv2.COLOR_BGR2GRAY)
            pre_enhanced = enhance_thermal(pre_gray)
            
            # Re-detect on pre-aligned
            attempts_pre = [pre_enhanced, cv2.bitwise_not(pre_enhanced)]
            for therm_pre in attempts_pre:
                kp2_pre, des2_pre = sift.detectAndCompute(therm_pre, None)
                if des2_pre is None:
                    continue
                matches_pre = flann.knnMatch(des1, des2_pre, k=K_FOR_KNN)
                good_pre = [m[0] for m in matches_pre if len(m) >= 2 and m[0].distance < LOWE_RATIO * m[1].distance]
                if len(good_pre) > best_num_good:
                    best_num_good = len(good_pre)
                    best_matches = good_pre
                    best_kp2 = kp2_pre
                    thermal_to_use = pre_warped_thermal
            
            print(f"  After pre-align: {best_num_good} matches")
            src_pts = np.float32([best_kp2[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp1[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)

        M, inliers = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESHOLD)
        inlier_count = np.sum(inliers.ravel() == 1) if inliers is not None else 0
        inlier_ratio = inlier_count / len(best_matches) if best_num_good > 0 else 0
        print(f"  Affine: {len(best_matches)} matches, {inlier_count} inliers (ratio: {inlier_ratio:.2f})")
        
        height, width = img1_gray.shape
        aligned_thermal = cv2.warpAffine(thermal_to_use, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
        return aligned_thermal
    
    print(f"  Failed: Only {best_num_good} matches (need {MIN_MATCH_COUNT}+)")
    return None

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    raw_list = glob.glob(os.path.join(INPUT_DIR, "*_Z.JPG")) + glob.glob(os.path.join(INPUT_DIR, "*_Z.jpg"))
    rgb_files = sorted(list(set(raw_list)))
    
    print(f"Found {len(rgb_files)} RGB images in input folder.")

    for rgb_path in rgb_files:
        filename = os.path.basename(rgb_path)
        
        if "_Z.JPG" in filename:
            thermal_filename = filename.replace("_Z.JPG", "_T.JPG")
            at_filename = filename.replace("_Z.JPG", "_AT.JPG")
            prefix = filename.replace("_Z.JPG", "")
        else:
            thermal_filename = filename.replace("_Z.jpg", "_T.jpg")
            at_filename = filename.replace("_Z.jpg", "_AT.jpg")
            prefix = filename.replace("_Z.jpg", "")

        thermal_path = os.path.join(INPUT_DIR, thermal_filename)
        
        if not os.path.exists(thermal_path):
            print(f"Skipping {prefix}: Thermal file {thermal_filename} not found.")
            continue

        print(f"Processing: {prefix}")

        pair_folder = os.path.join(OUTPUT_DIR, prefix)
        os.makedirs(pair_folder, exist_ok=True)

        shutil.copy2(rgb_path, os.path.join(pair_folder, filename))
        # shutil.copy2(thermal_path, os.path.join(pair_folder, thermal_filename))

        img_rgb, img_thermal = load_images(rgb_path, thermal_path)
        
        if img_rgb is None or img_thermal is None:
            print(f"  Error reading images for {prefix}")
            continue

        gray_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        gray_thermal = cv2.cvtColor(img_thermal, cv2.COLOR_BGR2GRAY)

        aligned_thermal = get_best_affine_transform(gray_rgb, gray_thermal, img_thermal)

        if aligned_thermal is not None:
            save_path = os.path.join(pair_folder, at_filename)
            cv2.imwrite(save_path, aligned_thermal)
            print(f"  -> Success: Saved {at_filename}")
        else:
            print(f"  -> FAILED: Could not align {prefix}")

if __name__ == "__main__":
    main()