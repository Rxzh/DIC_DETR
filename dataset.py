import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from PIL import Image


def get_collate_fn(processor):
    def collate_fn(batch):
        pixel_values = [item[0] for item in batch]
        encoding = processor.pad(pixel_values, return_tensors="pt")
        
        labels = [item[1] for item in batch] 
        
        batch = {}
        batch['pixel_values'] = encoding['pixel_values']
        batch['pixel_mask'] = encoding['pixel_mask']
        batch['labels'] = labels
        return batch
    return collate_fn


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
        file_path = self.file_list[idx]
        with np.load(file_path) as data:
            r_image = data['r_image']   # (H, W, 1)
            peak_map = data['peak_map'] # (H, W, 1)

        r_image_norm = (r_image.squeeze() * 255).astype(np.uint8)
        pil_image = Image.fromarray(r_image_norm, mode='L').convert("RGB")
        
        # Convert peak_map to DETR target format
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

        encoding = self.processor(images=pil_image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() 
        
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'class_labels': torch.tensor(class_labels, dtype=torch.int64)
        }

        return pixel_values, target