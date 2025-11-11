# train.py
# Main training script
# Now reads from the pre-generated dataset.

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DetrImageProcessor
from PIL import Image
import warnings
from time import time
from pathlib import Path

# Import the shared configuration
from config import config

warnings.filterwarnings("ignore", category=UserWarning, module='skimage')


# --- 1. New Dataset Definition ---

class PreGeneratedRadonDataset(Dataset):
    """
    A custom PyTorch Dataset that loads pre-generated .npz samples
    from disk and formats them for DETR.
    """
    def __init__(self, data_dir, processor, box_width, box_height, class_id):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.box_width = box_width
        self.box_height = box_height
        self.class_id = class_id

        # Find all the .npz files in the directory
        self.file_list = sorted(list(self.data_dir.glob("sample_*.npz")))
        
        if not self.file_list:
            raise FileNotFoundError(
                f"No 'sample_*.npz' files found in {self.data_dir}."
                f"Did you run 'generate_dataset.py' first?"
            )
        
        # Get dimensions from the first sample (assume all are the same)
        with np.load(self.file_list[0]) as data:
            r_image = data['r_image']
            self.img_h, self.img_w = r_image.shape[:2]
            print(f"Dataset at {data_dir} found {len(self.file_list)} samples.")
            print(f"Detected sample dimensions: (H={self.img_h}, W={self.img_w})")


    def __len__(self):
        # The length is the number of .npz files we found
        return len(self.file_list)

    def __getitem__(self, idx):
        # 1. Get the file path
        file_path = self.file_list[idx]
        
        # 2. Load the data from the .npz file
        # Use 'with' to ensure the file is closed properly
        with np.load(file_path) as data:
            r_image = data['r_image']   # Shape (H, W, 1)
            peak_map = data['peak_map'] # Shape (H, W, 1)

        # 3. Convert image to PIL RGB format
        # (This is identical to your old dataset's logic)
        r_image_norm = (r_image.squeeze() * 255).astype(np.uint8)
        pil_image = Image.fromarray(r_image_norm, mode='L').convert("RGB")
        
        # 4. Convert peak_map to DETR target format
        # (This is identical to your old dataset's logic)
        rho_indices, theta_indices = np.where(peak_map.squeeze() > 0)
        
        boxes = []
        class_labels = []

        for rho, theta in zip(rho_indices, theta_indices):
            x_coord = int(theta)
            y_coord = int(rho)
            
            px_width = self.box_width
            px_height = self.box_height
            
            px_center_x = x_coord + 0.5 
            px_center_y = y_coord + 0.5
            
            center_x = px_center_x / self.img_w
            center_y = px_center_y / self.img_h
            width = px_width / self.img_w
            height = px_height / self.img_h
            
            boxes.append([center_x, center_y, width, height])
            class_labels.append(self.class_id)

        # 5. Pre-process the image
        # (This is identical to your old dataset's logic)
        encoding = self.processor(images=pil_image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() 
        
        # 6. Create the final target dictionary
        # (This is identical to your old dataset's logic)
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'class_labels': torch.tensor(class_labels, dtype=torch.int64)
        }
        
        # The output format is *exactly* the same as before
        return pixel_values, target

# --- 2. Collate Function (Unchanged) ---

def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    # We still need the processor here to pad the batched tensors
    encoding = processor.pad(pixel_values, return_tensors="pt")
    
    labels = [item[1] for item in batch] 
    
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = labels
    return batch

# --- 3. Main Data Loading Setup ---

print(f"Loading processor: {config.PROCESSOR_NAME}")
# Define 'processor' globally so collate_fn can access it
processor = DetrImageProcessor.from_pretrained(config.PROCESSOR_NAME)

print("Creating training dataset...")
train_dir = config.DATA_DIR / "train"
train_dataset = PreGeneratedRadonDataset(
    data_dir=train_dir,
    processor=processor,
    box_width=config.BOX_WIDTH_PX,  
    box_height=config.BOX_HEIGHT_PX,
    class_id=config.CLASS_ID
)

print("Creating validation dataset...")
val_dir = config.DATA_DIR / "val"
val_dataset = PreGeneratedRadonDataset(
    data_dir=val_dir,
    processor=processor,
    box_width=config.BOX_WIDTH_PX,  
    box_height=config.BOX_HEIGHT_PX,
    class_id=config.CLASS_ID
)

print("Creating dataloaders...")
train_dataloader = DataLoader(
    train_dataset, 
    collate_fn=collate_fn, 
    batch_size=config.BATCH_SIZE, 
    shuffle=True,
    num_workers=config.NUM_WORKERS,
    pin_memory=True
)
val_dataloader = DataLoader(
    val_dataset, 
    collate_fn=collate_fn, 
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS, 
    pin_memory=True
)

print("\nData loading pipeline successfully created.")

# --- 4. Test Block (Unchanged) ---

if __name__ == '__main__':

    print("\nFetching one batch from train_dataloader...")
    try:
        t0 = time()
        batch = next(iter(train_dataloader))
        t1 = time()
        
        # This will now be *extremely* fast
        print(f'Batch sampled in {t1-t0:.4f}s') 
        print("Batch keys:", batch.keys())
        print("Pixel values shape:", batch['pixel_values'].shape)
        print("Pixel mask shape:", batch['pixel_mask'].shape)

        print("\nBatch 'labels' content (list of target dicts):")
        print(f"  Length of list: {len(batch['labels'])} (should match batch size)")

        if len(batch['labels']) > 0:
            first_target = batch['labels'][0]
            print("\n  Target for first sample in batch:")
            print(f"    Keys: {first_target.keys()}")
            print(f"    Number of peaks (boxes): {len(first_target['boxes'])}")
            print(f"    'boxes' tensor shape: {first_target['boxes'].shape}")
            if len(first_target['boxes']) > 0:
                print(f"    Example box: {first_target['boxes'][0]}")

    except Exception as e:
        print(f"\nAn error occurred while fetching a batch: {e}")