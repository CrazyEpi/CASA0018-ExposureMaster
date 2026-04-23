import os
import random
import numpy as np
from PIL import Image

# CONFIG
INPUT_DIR = "E:\\UCL\\CASA0018-ML\\WIDER"
OUTPUT_DIR = "E:\\UCL\\CASA0018-ML\\Dataset_Part5"
LOG_FILE_PATH = os.path.join(OUTPUT_DIR, "processed_log_wider.txt")
NUM_SAMPLES = 160


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


def build_wider_exposure_dataset(input_dir, output_dir, sample_size_limit):
    # Configurations like SICE dataset
    # SICE Data Rule (Simulated for JPEG): 1.jpg(-2EV), 2.jpg(-1EV), 3.jpg(0EV), 4.jpg(+1EV), 5.jpg(+2EV)
    ev_map = {
        -2: "1.jpg", -1: "2.jpg",
        0: "3.jpg",
        +1: "4.jpg", +2: "5.jpg"
    }

    # 1. Check log to avoid processing same photo
    processed_files = set()
    if os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, 'r') as f:
            processed_files = set(line.strip() for line in f)

    # 2. Scan and Filter
    print(f"Started Scanning: {input_dir}")
    valid_photos = []
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg')):
                rel_path = os.path.relpath(os.path.join(root, filename), input_dir)
                if rel_path not in processed_files:
                    valid_photos.append(rel_path)

    total_valid = len(valid_photos)
    if total_valid == 0:
        print("No matching unprocessed photos found")
        return

    # 3. Random sample
    sample_size = min(sample_size_limit, total_valid)
    selected_photos = random.sample(valid_photos, sample_size)
    print(f"\nTotal Filtered Photo: {total_valid} ,Random Sampled: {sample_size} ")

    # 4. Get dataset folder number index
    current_idx = get_next_folder_index(output_dir)

    # 5. Exposure adjustments
    for filename in selected_photos:
        full_path = os.path.join(input_dir, filename)
        sample_folder = os.path.join(output_dir, str(current_idx))
        os.makedirs(sample_folder, exist_ok=True)

        print(f"\n[{current_idx}] Processing: {filename}")

        try:
            with Image.open(full_path) as img:
                img = img.convert('RGB')
                img_array = np.array(img).astype(np.float32)

                # Get Linear RGB Data (Simulated linearization for JPEG)
                rgb_linear = np.power(img_array / 255.0, 2.2)

                for ev, target_name in ev_map.items():
                    multiplier = 2.0 ** ev

                    # Adjust Exposure and Transform to sRGB
                    rgb_shifted = np.clip(rgb_linear * multiplier, 0, 1.0)
                    rgb_gamma = np.power(rgb_shifted, 1 / 2.2) * 255.0
                    rgb_8bit = np.clip(rgb_gamma, 0, 255).astype(np.uint8)

                    # Save Pic
                    save_path = os.path.join(sample_folder, target_name)
                    Image.fromarray(rgb_8bit).save(save_path, quality=90)
                    print(f"    Generated: {target_name} (EV {ev:+})")

            # Write txt log
            with open(LOG_FILE_PATH, 'a') as f:
                f.write(filename + '\n')

            current_idx += 1

        except Exception as e:
            print(f"  [Failed] When processing {filename} got wrong: {e}")

    print(f"\nFinished!!! Start Folder Index: {get_next_folder_index(output_dir) - sample_size}, End Folder Index: {current_idx - 1}")

if __name__ == "__main__":
    build_wider_exposure_dataset(INPUT_DIR, OUTPUT_DIR, NUM_SAMPLES)