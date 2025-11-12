import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from PIL import Image
import tarfile
import io


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
    from a directory of sharded .tar archives and formats them for DETR.
    
    The 'data_dir' should point to a directory containing .tar files
    (e.g., 'train/' or 'val/' from the 'dataset_tars' output).
    """
    def __init__(self, data_dir, processor, box_width, box_height, class_id):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.box_width = box_width
        self.box_height = box_height
        self.class_id = class_id

        # Find all the .tar files in the directory
        tar_files = sorted(list(self.data_dir.glob("*.tar")))
        if not tar_files:
            raise FileNotFoundError(
                f"No '*.tar' files found in {self.data_dir}. "
                f"Did you run 'create_tar_archives.py' and point 'data_dir' to the correct split folder?"
            )
        
        # Build a map of (tar_file_path, npz_member_name)
        # This map is our 'virtual' file list
        self.file_map = []
        print(f"Building file map from {len(tar_files)} tar archives in {self.data_dir}...")
        for tar_path in tar_files:
            try:
                with tarfile.open(tar_path, 'r') as tar:
                    # Get a sorted list of .npz files within this tar archive
                    npz_members = sorted(
                        [m.name for m in tar.getmembers() if m.isfile() and m.name.endswith('.npz')]
                    )
                    for member_name in npz_members:
                        # Store the path to the tar file and the name of the file inside it
                        self.file_map.append((str(tar_path), member_name))
            except tarfile.ReadError:
                print(f"Warning: Could not read {tar_path}, skipping.")
        
        if not self.file_map:
            raise FileNotFoundError(
                f"No 'sample_*.npz' files found inside the .tar archives in {self.data_dir}."
            )
        
        # Get dimensions from the first sample (assume all are the same)
        try:
            first_tar_path, first_member_name = self.file_map[0]
            with tarfile.open(first_tar_path, 'r') as tar:
                member = tar.getmember(first_member_name)
                f = tar.extractfile(member)
                if f is None:
                    raise IOError(f"Could not extract {first_member_name} from {first_tar_path}")
                data_bytes = io.BytesIO(f.read())
                
            with np.load(data_bytes) as data:
                r_image = data['r_image']
                self.img_h, self.img_w = r_image.shape[:2]
        except Exception as e:
            raise IOError(f"Failed to read first sample to get dimensions: {e}")

        print(f"Dataset at {data_dir} found {len(self.file_map)} samples across {len(tar_files)} tar archives.")
        print(f"Detected sample dimensions: (H={self.img_h}, W={self.img_w})")

    def __len__(self):
        # The length is the total number of .npz files we found
        return len(self.file_map)

    def __getitem__(self, idx):
        # Get the tar file path and the internal .npz file name
        tar_path_str, member_name = self.file_map[idx]
        
        try:
            # Open the correct tar file
            with tarfile.open(tar_path_str, 'r') as tar:
                # Get the specific member (file)
                member = tar.getmember(member_name)
                # Extract it into an in-memory file-like object
                f = tar.extractfile(member)
                if f is None:
                    raise IOError(f"Could not extract {member_name} from {tar_path_str}")
                
                # Read the file data into a BytesIO buffer
                data_bytes = io.BytesIO(f.read())
        except Exception as e:
            print(f"Error opening/reading {tar_path_str} or member {member_name}: {e}")
            raise e

        # Load the .npz data from the in-memory buffer
        with np.load(data_bytes) as data:
            r_image = data['r_image']   # (H, W, 1)
            peak_map = data['peak_map'] # (H, W, 1)

        # --- The rest of your processing logic is identical ---
        
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