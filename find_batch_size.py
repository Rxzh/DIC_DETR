import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
import time
import sys
from pathlib import Path



import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line
from skimage.transform import radon
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DetrImageProcessor
from PIL import Image
import warnings
from train import *


MODEL_PATH = Path("pretrained_weights/facebook-detr-resnet-50") 



def check_max_batch_size():

    if not torch.cuda.is_available():
        print("CUDA not available. This script requires a GPU.")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Testing on: {torch.cuda.get_device_name(device)}")

    # Load model and processor
    try:
        processor = DetrImageProcessor.from_pretrained(MODEL_PATH)
        model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have an internet connection and 'transformers' is installed.")
        sys.exit(1)


    print("Creating training dataset...")
    train_dir = config.DATA_DIR / "train"
    train_dataset = PreGeneratedRadonDataset(
        data_dir=train_dir,
        processor=processor,
        box_width=config.BOX_WIDTH_PX,  
        box_height=config.BOX_HEIGHT_PX,
        class_id=config.CLASS_ID
    )

        
    model.to(device)
    model.train() # Set to train mode

    batch_size = 1
    max_successful_bs = 0

    print("Starting search... Will increment batch size until OOM.")
    print("-" * 30)

    while True:
        print(f"Testing batch size: {batch_size}", end="")
        start_time = time.time()

        try:

            train_dataloader = DataLoader(
                train_dataset, 
                collate_fn=collate_fn, 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )


            batch = next(iter(train_dataloader))
            
            # Move batch to GPU
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs['loss']
            loss.backward()

            del outputs, loss, batch
            torch.cuda.empty_cache()
            
            end_time = time.time()
            print(f" ... OK (Time: {end_time - start_time:.2f}s)")
            
            max_successful_bs = batch_size
            batch_size += 1

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "out-of-memory" in str(e).lower() or "oom" in str(e).lower():
                print(f" ... FAILED (OOM)")
                print("\n" + "=" * 30)
                print(f"Maximum possible batch size: {max_successful_bs}")
                print("=" * 30)
                break
            else:
                print(f" ... FAILED (Unknown Error: {e})")
                raise e
        except Exception as e:
            print(f" ... FAILED (Non-Runtime Error: {e})")
            raise e


if __name__ == "__main__":
    check_max_batch_size()