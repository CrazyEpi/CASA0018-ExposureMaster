import os
import random
import rawpy
import numpy as np
from PIL import Image
import exifread
from fractions import Fraction

# CONFIG
INPUT_DIR = "E:\Pic\\2025-8-Jap-2-011130"
OUTPUT_DIR = "E:\\UCL\\CASA0018-ML\\Dataset_Part3"
LOG_FILE_PATH = os.path.join(OUTPUT_DIR, "processed_log.txt")


def get_exif_info(file_path):
    # Extract ISO and Shutter Speed to filter out night images
    with open(file_path, 'rb') as f:
        tags = exifread.process_file(f, details=False)
        iso_tag = tags.get('EXIF ISOSpeedRatings') or tags.get('Image ISOSpeedRatings')
        iso = int(str(iso_tag)) if iso_tag else None
        exp_tag = tags.get('EXIF ExposureTime')
        exposure_time = float(Fraction(str(exp_tag))) if exp_tag else None
    return iso, exposure_time


def get_next_folder_index(output_dir):
    # Scan the folder to accumulate folder index
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        return 1

    # Find all folders
    existing_folders = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f)) and f.isdigit()]
    if not existing_folders:
        return 1

    indices = [int(f) for f in existing_folders]
    return max(indices) + 1


def build_sice_style_dataset(input_dir, output_dir):
    # Configurations like SICE dataset
    # SICE Data Rule:1.jpg(-3EV), 2.jpg(-2EV), 3.jpg(-1EV), 4.jpg(0EV), 5.jpg(+1EV), 6.jpg(+2EV), 7.jpg(+3EV)
    ev_map = {
        -3: "1.jpg", -2: "2.jpg", -1: "3.jpg",
        0: "4.jpg",
        +1: "5.jpg", +2: "6.jpg", +3: "7.jpg"
    }
    BASE_CALIBRATION = 1
    MAX_ISO = 800
    MAX_SHUTTER_TIME = 1.0 / 30.0
    SAMPLE_RATIO = 0.10

    # 1. Check log to avoid processing same photo
    processed_files = set()
    if os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, 'r') as f:
            processed_files = set(line.strip() for line in f)

    # 2. Scan and Filter
    print(f"Started Scanning: {input_dir}")
    valid_photos = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.nef'):
            if filename in processed_files:
                continue  # Skip processed ones

            file_path = os.path.join(input_dir, filename)
            try:
                iso, exp_time = get_exif_info(file_path)
                if iso and exp_time:
                    if iso <= MAX_ISO and exp_time <= MAX_SHUTTER_TIME:
                        valid_photos.append(filename)
                    else:
                        print(f"  [Filter] {filename}: ISO: {iso} / Shutter: {exp_time:.4f}s No Match!!!")
            except Exception as e:
                print(f"  [Error] Unable to Read {filename}: {e}")

    total_valid = len(valid_photos)
    if total_valid == 0:
        print("No matching unprocessed photos found")
        return

    # 3. Random sample
    sample_size = max(1, int(total_valid * SAMPLE_RATIO))
    selected_photos = random.sample(valid_photos, sample_size)
    print(f"\nTotal Filtered Photo: {total_valid} ,Random Sampled: {sample_size} ")

    # 4. Get dataset folder number index
    current_idx = get_next_folder_index(output_dir)

    # 5. Exposure adjustments
    for filename in selected_photos:
        raw_path = os.path.join(input_dir, filename)
        sample_folder = os.path.join(output_dir, str(current_idx))
        os.makedirs(sample_folder, exist_ok=True)

        print(f"\n[{current_idx}] Processing: {filename}")

        try:
            with rawpy.imread(raw_path) as raw:
                # Get Linear RGB Data
                rgb_linear = raw.postprocess(
                    gamma=(1, 1), no_auto_bright=True, use_camera_wb=True, output_bps=16
                ).astype(np.float32)

                for ev, target_name in ev_map.items():
                    total_ev_shift = BASE_CALIBRATION + ev
                    multiplier = 2.0 ** total_ev_shift

                    # Adjust Exposure and Transform to sRGB
                    rgb_shifted = np.clip(rgb_linear * multiplier, 0, 65535.0)
                    rgb_gamma = np.power(rgb_shifted / 65535.0, 1 / 2.2) * 255.0
                    rgb_8bit = np.clip(rgb_gamma, 0, 255).astype(np.uint8)

                    # Save Pic
                    save_path = os.path.join(sample_folder, target_name)
                    Image.fromarray(rgb_8bit).save(save_path, quality=95)
                    print(f"    Generated: {target_name} (EV {ev:+})")

            # Write txt log
            with open(LOG_FILE_PATH, 'a') as f:
                f.write(filename + '\n')

            current_idx += 1

        except Exception as e:
            print(f"  [Failed] When processing {filename} got wrong: {e}")

    print(
        f"\nFinished!!! Start Folder Index: {get_next_folder_index(output_dir) - sample_size}, End Folder Index: {current_idx - 1}")

if __name__ == "__main__":
    build_sice_style_dataset(INPUT_DIR, OUTPUT_DIR)