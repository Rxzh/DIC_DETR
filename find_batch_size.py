import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
import time
import sys
from pathlib import Path


TEST_IMAGE_H = 800 
TEST_IMAGE_W = 800 


# Number of boxes/labels to simulate per image.
NUM_DUMMY_BOXES = 100

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

            # Create B dummy images (list of tensors)
            # Using tensors is easier here.
            dummy_images = [
                torch.rand(3, TEST_IMAGE_H, TEST_IMAGE_W) 
                for _ in range(batch_size)
            ]
            
            # Create B dummy targets
            # The base DETR model expects COCO classes (0-90)
            dummy_targets = []
            for _ in range(batch_size):
                dummy_targets.append({
                    'boxes': torch.rand(NUM_DUMMY_BOXES, 4), # [cx, cy, w, h] format
                    'class_labels': torch.randint(0, 91, (NUM_DUMMY_BOXES,)) 
                })

            # creates 'pixel_values', 'pixel_mask' and formats labels
            batch = processor(
                images=dummy_images, 
                annotations=dummy_targets, 
                return_tensors="pt"
            )
            
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