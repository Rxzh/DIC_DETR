# generate_dataset.py
# This script pre-generates the entire dataset.
# Run this file *once* before you start training.

import numpy as np
from skimage.draw import line
from skimage.transform import radon
import warnings
from time import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# Import the configuration
from config import config

warnings.filterwarnings("ignore", category=UserWarning, module='skimage')

# --- 1. Generation Function (Copied from your script) ---

def generate_radon_sample(size=128, num_lines=1, theta_res=128):
    """
    Generate a synthetic Radon-transform training sample. (Optimized)
    """
    theta = np.linspace(0., 180., theta_res, endpoint=False)
    peak_map = np.zeros((size, theta_res), dtype=np.float32)
    r_image = np.zeros((size, theta_res), dtype=np.float32)

    for _ in range(num_lines):
        x0, y0 = np.random.randint(0, size, 2)
        x1, y1 = np.random.randint(0, size, 2)
        single_line = np.zeros((size, size))
        rr, cc = line(y0, x0, y1, x1)
        single_line[rr.clip(0, size - 1), cc.clip(0, size - 1)] = 1

        r_line = radon(single_line, theta=theta, circle=True)
        max_idx = np.unravel_index(np.argmax(r_line), r_line.shape)
        peak_map[max_idx] = 1.0
        r_image += r_line

    r_min = r_image.min()
    r_max = r_image.max()
    if r_max - r_min > 1e-6:
        r_image = (r_image - r_min) / (r_max - r_min)
    else:
        r_image = np.zeros_like(r_image) 

    # Return as (H, W, 1) to be consistent
    return r_image[..., np.newaxis], peak_map[..., np.newaxis]


# --- 2. Worker Function for Multiprocessing ---

def generate_and_save_sample(sample_index, save_dir, cfg):
    """
    A single worker's task: generate one sample and save it to disk.
    """
    try:
        # Generate the data
        num_lines = np.random.randint(cfg.MIN_LINES, cfg.MAX_LINES + 1)
        r_image, peak_map = generate_radon_sample(
            size=cfg.IMG_SIZE,
            num_lines=num_lines,
            theta_res=cfg.THETA_RES
        )
        
        # Define the save path
        file_path = save_dir / f"sample_{sample_index:07d}.npz"
        
        # Save as a compressed .npz file
        # This stores both arrays in one file, efficiently
        np.savez_compressed(
            file_path,
            r_image=r_image,
            peak_map=peak_map
        )
        return None # Success
    except Exception as e:
        return f"Error generating sample {sample_index}: {e}"

# --- 3. Main Orchestration Function ---

def create_dataset_split(split_name, total_samples, save_dir, cfg):
    """
    Generates all samples for a given split (e.g., 'train' or 'val').
    """
    print(f"\n--- Generating {split_name} split ---")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # --- This is the restart logic ---
    # 1. Find all files that *already* exist
    existing_files = set(save_dir.glob("sample_*.npz"))
    existing_indices = set()
    for f in existing_files:
        try:
            # Extract the index number from the filename
            idx = int(f.stem.split('_')[1])
            existing_indices.add(idx)
        except Exception:
            pass # Ignore malformed filenames

    # 2. Define all indices we *want*
    required_indices = set(range(total_samples))
    
    # 3. Find the indices we *need to generate*
    indices_to_generate = sorted(list(required_indices - existing_indices))
    
    if not indices_to_generate:
        print(f"'{split_name}' split is already complete ({total_samples} samples found).")
        return

    print(f"Found {len(existing_indices)} existing samples.")
    print(f"Need to generate {len(indices_to_generate)} new samples.")

    # --- Multiprocessing Setup ---
    # Use functools.partial to "pre-fill" the arguments
    # that are the same for every worker.
    worker_func = partial(generate_and_save_sample, save_dir=save_dir, cfg=cfg)
    
    num_processes = min(cpu_count(), cfg.NUM_WORKERS, len(indices_to_generate))
    print(f"Starting generation with {num_processes} processes...")
    
    with Pool(processes=num_processes) as pool:
        # Use imap_unordered for efficiency (results come in any order)
        # Use tqdm for a nice progress bar
        results = list(tqdm(
            pool.imap_unordered(worker_func, indices_to_generate),
            total=len(indices_to_generate),
            desc=f"Generating {split_name} samples"
        ))
    
    # Report any errors
    errors = [r for r in results if r is not None]
    if errors:
        print("\n--- Errors Occurred ---")
        for err in errors[:10]: # Print first 10 errors
            print(err)
        if len(errors) > 10:
            print(f"...and {len(errors) - 10} more.")
    
    print(f"\nFinished generating {split_name} split.")

# --- 4. Main Execution ---

if __name__ == '__main__':
    t_start = time()
    
    print(f"Starting dataset generation...")
    print(f"Saving data to: {config.DATA_DIR.resolve()}")
    
    # Generate Training Set
    train_dir = config.DATA_DIR / "train"
    create_dataset_split(
        split_name="train",
        total_samples=config.TOTAL_TRAIN_SAMPLES,
        save_dir=train_dir,
        cfg=config
    )
    
    # Generate Validation Set
    val_dir = config.DATA_DIR / "val"
    create_dataset_split(
        split_name="val",
        total_samples=config.TOTAL_VAL_SAMPLES,
        save_dir=val_dir,
        cfg=config
    )
    
    t_end = time()
    print(f"\nTotal generation time: {t_end - t_start:.2f}s")